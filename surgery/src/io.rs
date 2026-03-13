//! Memory-mapped I/O and merge orchestration for safetensors files.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use safetensors::Dtype;

use crate::config::{AdapterConfig, BiasMode};
use crate::merge::{bytes_to_f32, f32_to_bytes, merge_bias, merge_lora};
use crate::names::build_name_mapping;
use crate::{MergeStats, Result, SurgeryError};

#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    /// Absolute byte offset (not relative to data section).
    data_start: usize,
    data_end: usize,
}

struct MappedSafetensors {
    mmap: Mmap,
    /// Ordered by appearance in the file.
    tensors: Vec<(String, TensorInfo)>,
}

impl MappedSafetensors {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            SurgeryError::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {}", path.display(), e),
            ))
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            SurgeryError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to mmap {}: {}", path.display(), e),
            ))
        })?;

        let (header_len, metadata) = SafeTensors::read_metadata(&mmap).map_err(|e| {
            SurgeryError::Safetensors(format!("failed to parse safetensors header: {e}"))
        })?;
        let data_offset = 8 + header_len;

        let mut tensors: Vec<(String, TensorInfo)> = Vec::new();
        for (name, info) in metadata.tensors() {
            let (start, end) = info.data_offsets;
            tensors.push((
                name.to_string(),
                TensorInfo {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data_start: data_offset + start,
                    data_end: data_offset + end,
                },
            ));
        }

        tensors.sort_by_key(|(_, info)| info.data_start);

        Ok(Self { mmap, tensors })
    }

    fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        for (n, info) in &self.tensors {
            if n == name {
                return Ok(&self.mmap[info.data_start..info.data_end]);
            }
        }
        Err(SurgeryError::TensorNotFound {
            name: name.to_string(),
            location: "file".to_string(),
        })
    }

    fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        for (n, info) in &self.tensors {
            if n == name {
                return Ok(info);
            }
        }
        Err(SurgeryError::TensorNotFound {
            name: name.to_string(),
            location: "file".to_string(),
        })
    }
}

/// Returns the serialized header bytes and the total data section size.
fn build_output_header(tensors: &[(String, TensorInfo)]) -> Result<(Vec<u8>, usize)> {
    let mut header_map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut current_offset: usize = 0;

    for (name, info) in tensors {
        let elem_size = dtype_byte_size(info.dtype)?;
        let num_elements: usize = info.shape.iter().product();
        let byte_size = num_elements * elem_size;

        let tensor_entry = serde_json::json!({
            "dtype": dtype_to_string(info.dtype),
            "shape": info.shape,
            "data_offsets": [current_offset, current_offset + byte_size],
        });
        header_map.insert(name.clone(), tensor_entry);
        current_offset += byte_size;
    }

    let header_json = serde_json::to_string(&header_map)?;
    Ok((header_json.into_bytes(), current_offset))
}

fn dtype_byte_size(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F16 | Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        _ => Err(SurgeryError::UnsupportedDtype {
            name: "unknown".to_string(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

fn dtype_to_string(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::F32 => "F32",
        Dtype::F64 => "F64",
        Dtype::I32 => "I32",
        Dtype::I64 => "I64",
        Dtype::U8 => "U8",
        Dtype::I8 => "I8",
        Dtype::BOOL => "BOOL",
        Dtype::U16 => "U16",
        Dtype::U32 => "U32",
        Dtype::U64 => "U64",
        Dtype::I16 => "I16",
        _ => "F32",
    }
}

/// Processes one tensor at a time, keeping memory bounded regardless of model size.
pub fn merge_and_write(
    base_model_path: &Path,
    adapter_weights_path: &Path,
    config: &AdapterConfig,
    output_path: &Path,
    progress: Option<&dyn Fn(usize, usize)>,
) -> Result<MergeStats> {
    let base = MappedSafetensors::open(base_model_path)?;
    let adapter = MappedSafetensors::open(adapter_weights_path)?;

    let base_names: Vec<&str> = base.tensors.iter().map(|(n, _)| n.as_str()).collect();
    let adapter_names: Vec<&str> = adapter.tensors.iter().map(|(n, _)| n.as_str()).collect();

    let name_mapping = build_name_mapping(
        &base_names,
        &adapter_names,
        config.target_modules(),
        config.modules_to_save(),
    )?;

    let total_tensors = base.tensors.len();

    let output_tensor_list: Vec<(String, TensorInfo)> = base
        .tensors
        .iter()
        .map(|(name, info)| (name.clone(), info.clone()))
        .collect();
    let (header_bytes, _total_data_size) = build_output_header(&output_tensor_list)?;

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    let header_len = header_bytes.len() as u64;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header_bytes)?;

    let mut stats = MergeStats {
        tensors_copied: 0,
        tensors_merged: 0,
        tensors_replaced: 0,
        biases_merged: 0,
    };

    for (i, (name, info)) in base.tensors.iter().enumerate() {
        if let Some(progress) = &progress {
            progress(i, total_tensors);
        }

        if let Some(adapter_tensor_name) = name_mapping.replacement(name) {
            let adapter_data = adapter.tensor_data(adapter_tensor_name)?;
            writer.write_all(adapter_data)?;
            stats.tensors_replaced += 1;
            continue;
        }

        if let Some((lora_a_name, lora_b_name)) = name_mapping.lora_pair(name) {
            let base_data = &base.mmap[info.data_start..info.data_end];
            let lora_a_info = adapter.tensor_info(lora_a_name)?;
            let lora_b_info = adapter.tensor_info(lora_b_name)?;
            let lora_a_data = adapter.tensor_data(lora_a_name)?;
            let lora_b_data = adapter.tensor_data(lora_b_name)?;

            let base_f32 = bytes_to_f32(base_data, info.dtype, &info.shape, name)?;
            let a_f32 = bytes_to_f32(
                lora_a_data,
                lora_a_info.dtype,
                &lora_a_info.shape,
                lora_a_name,
            )?;
            let b_f32 = bytes_to_f32(
                lora_b_data,
                lora_b_info.dtype,
                &lora_b_info.shape,
                lora_b_name,
            )?;

            let merged = merge_lora(
                &base_f32,
                &a_f32,
                &b_f32,
                config.scaling(),
                config.fan_in_fan_out(),
            );

            let merged_bytes = f32_to_bytes(&merged, info.dtype)?;
            writer.write_all(&merged_bytes)?;
            stats.tensors_merged += 1;
            // All temporaries (base_f32, a_f32, b_f32, merged, merged_bytes) dropped here
            continue;
        }

        if matches!(config.bias(), BiasMode::LoraOnly | BiasMode::All) {
            if let Some(adapter_bias_name) = name_mapping.bias_source(name) {
                let base_data = &base.mmap[info.data_start..info.data_end];
                let adapter_bias_info = adapter.tensor_info(adapter_bias_name)?;
                let adapter_bias_data = adapter.tensor_data(adapter_bias_name)?;

                // Bias tensors may be 1D; reshape to [1, N] for Array2
                let base_shape = if info.shape.len() == 1 {
                    vec![1, info.shape[0]]
                } else {
                    info.shape.clone()
                };
                let adapter_shape = if adapter_bias_info.shape.len() == 1 {
                    vec![1, adapter_bias_info.shape[0]]
                } else {
                    adapter_bias_info.shape.clone()
                };

                let base_f32 = bytes_to_f32(base_data, info.dtype, &base_shape, name)?;
                let adapter_f32 = bytes_to_f32(
                    adapter_bias_data,
                    adapter_bias_info.dtype,
                    &adapter_shape,
                    adapter_bias_name,
                )?;

                let merged = merge_bias(&base_f32, &adapter_f32);
                let merged_bytes = f32_to_bytes(&merged, info.dtype)?;
                writer.write_all(&merged_bytes)?;
                stats.biases_merged += 1;
                continue;
            }
        }

        let data = &base.mmap[info.data_start..info.data_end];
        writer.write_all(data)?;
        stats.tensors_copied += 1;
    }

    if let Some(progress) = &progress {
        progress(total_tensors, total_tensors);
    }

    writer.flush()?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    use half::f16;
    use safetensors::tensor::serialize;

    fn write_safetensors(path: &Path, tensors: Vec<(&str, Vec<f32>, Vec<usize>, Dtype)>) {
        let mut data_map: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> = Vec::new();
        for (name, values, shape, dtype) in tensors {
            let bytes = match dtype {
                Dtype::F16 => values
                    .iter()
                    .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                    .collect(),
                Dtype::F32 => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
                _ => panic!("unsupported dtype in test helper"),
            };
            data_map.push((name.to_string(), bytes, shape, dtype));
        }

        let tensor_views: Vec<_> = data_map
            .iter()
            .map(|(name, data, shape, dtype)| {
                (
                    name.as_str(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let serialized = serialize(tensor_views, &None).unwrap();
        std::fs::write(path, serialized).unwrap();
    }

    #[test]
    fn end_to_end_merge() {
        let dir = tempfile::tempdir().unwrap();

        let base_path = dir.path().join("base.safetensors");
        write_safetensors(
            &base_path,
            vec![
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    vec![1.0, 0.0, 0.0, 1.0],
                    vec![2, 2],
                    Dtype::F16,
                ),
                (
                    "model.embed_tokens.weight",
                    vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    vec![3, 2],
                    Dtype::F16,
                ),
            ],
        );

        let adapter_path = dir.path().join("adapter_model.safetensors");
        write_safetensors(
            &adapter_path,
            vec![
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![1.0, 1.0],
                    vec![1, 2],
                    Dtype::F16,
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![1.0, 1.0],
                    vec![2, 1],
                    Dtype::F16,
                ),
            ],
        );

        let config_json = r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#;
        let config = crate::config::AdapterConfig::from_json(config_json).unwrap();

        let output_path = dir.path().join("merged.safetensors");

        let stats =
            merge_and_write(&base_path, &adapter_path, &config, &output_path, None).unwrap();

        assert_eq!(stats.tensors_merged, 1);
        assert_eq!(stats.tensors_copied, 1);
        assert_eq!(stats.tensors_replaced, 0);

        let output = MappedSafetensors::open(&output_path).unwrap();
        assert_eq!(output.tensors.len(), 2);

        let embed_info = output.tensor_info("model.embed_tokens.weight").unwrap();
        assert_eq!(embed_info.shape, vec![3, 2]);

        let q_proj_info = output
            .tensor_info("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q_proj_info.shape, vec![2, 2]);
        let q_proj_data = output
            .tensor_data("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let q_proj_f32 = crate::merge::bytes_to_f32(
            q_proj_data,
            q_proj_info.dtype,
            &q_proj_info.shape,
            "q_proj",
        )
        .unwrap();

        // base [[1,0],[0,1]] + 1.0 * B@A [[1,1],[1,1]] = [[2,1],[1,2]]
        assert!((q_proj_f32[[0, 0]] - 2.0).abs() < 0.01);
        assert!((q_proj_f32[[0, 1]] - 1.0).abs() < 0.01);
        assert!((q_proj_f32[[1, 0]] - 1.0).abs() < 0.01);
        assert!((q_proj_f32[[1, 1]] - 2.0).abs() < 0.01);
    }
}
