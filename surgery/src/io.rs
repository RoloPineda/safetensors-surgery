//! Memory-mapped I/O and merge orchestration for safetensors files.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use safetensors::Dtype;
use serde::Deserialize;

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
    index: HashMap<String, usize>,
    /// The `__metadata__` key from the safetensors header, if present.
    metadata: Option<HashMap<String, String>>,
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
                name,
                TensorInfo {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data_start: data_offset + start,
                    data_end: data_offset + end,
                },
            ));
        }

        tensors.sort_by_key(|(_, info)| info.data_start);

        let index: HashMap<String, usize> = tensors
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();

        let metadata = metadata.metadata().clone();

        Ok(Self {
            mmap,
            tensors,
            index,
            metadata,
        })
    }

    fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self.tensor_info(name)?;
        Ok(&self.mmap[info.data_start..info.data_end])
    }

    fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        match self.index.get(name) {
            Some(&idx) => Ok(&self.tensors[idx].1),
            None => Err(SurgeryError::TensorNotFound {
                name: name.to_string(),
                location: "file".to_string(),
            }),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ShardIndex {
    weight_map: BTreeMap<String, String>,
}

impl ShardIndex {
    fn from_path(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let index: ShardIndex = serde_json::from_str(&contents)?;
        if index.weight_map.is_empty() {
            return Err(SurgeryError::ShardingError(
                "weight_map is empty in index.json".to_string(),
            ));
        }
        Ok(index)
    }

    /// Returns the unique shard filenames in sorted order.
    fn shard_filenames(&self) -> Vec<String> {
        let mut names: Vec<String> = self.weight_map.values().cloned().collect();
        names.sort();
        names.dedup();
        names
    }
}

/// Resolved base model source — either a single file or a sharded set.
pub(crate) enum BaseModelSource {
    SingleFile(PathBuf),
    Sharded { dir: PathBuf, index: ShardIndex },
}

/// Resolves a base model path to either a single file or sharded model.
///
/// - If `path` is a file ending in `.safetensors`, uses it directly.
/// - If `path` is a directory containing `model.safetensors`, uses that file.
/// - If `path` is a directory containing `model.safetensors.index.json`, uses sharded mode.
pub(crate) fn resolve_base_model(path: &Path) -> Result<BaseModelSource> {
    if path.is_file() {
        return Ok(BaseModelSource::SingleFile(path.to_path_buf()));
    }

    if path.is_dir() {
        let single = path.join("model.safetensors");
        if single.is_file() {
            return Ok(BaseModelSource::SingleFile(single));
        }

        let index_path = path.join("model.safetensors.index.json");
        if index_path.is_file() {
            let index = ShardIndex::from_path(&index_path)?;
            return Ok(BaseModelSource::Sharded {
                dir: path.to_path_buf(),
                index,
            });
        }

        return Err(SurgeryError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "directory '{}' contains neither model.safetensors nor model.safetensors.index.json",
                path.display()
            ),
        )));
    }

    Err(SurgeryError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("base model path '{}' does not exist", path.display()),
    )))
}

/// Computes dry-run info by inspecting the base model and adapter without merging.
pub(crate) fn compute_dry_run_info(
    base_model_path: &Path,
    adapter_weights_path: &Path,
    config: &AdapterConfig,
) -> Result<crate::DryRunInfo> {
    let source = resolve_base_model(base_model_path)?;
    let adapter = MappedSafetensors::open(adapter_weights_path)?;
    let adapter_names: Vec<&str> = adapter.tensors.iter().map(|(n, _)| n.as_str()).collect();

    let (base_names, tensor_infos, is_sharded, shard_count) = match &source {
        BaseModelSource::SingleFile(path) => {
            let base = MappedSafetensors::open(path)?;
            let names: Vec<String> = base.tensors.iter().map(|(n, _)| n.clone()).collect();
            let infos: Vec<TensorInfo> = base.tensors.iter().map(|(_, i)| i.clone()).collect();
            (names, infos, false, 1)
        }
        BaseModelSource::Sharded { dir, index } => {
            let mut names = Vec::new();
            let mut infos = Vec::new();
            let shard_filenames = index.shard_filenames();
            let shard_count = shard_filenames.len();
            for shard_filename in &shard_filenames {
                let shard = MappedSafetensors::open(&dir.join(shard_filename))?;
                for (name, info) in &shard.tensors {
                    names.push(name.clone());
                    infos.push(info.clone());
                }
            }
            (names, infos, true, shard_count)
        }
    };

    let base_name_refs: Vec<&str> = base_names.iter().map(|s| s.as_str()).collect();
    let name_mapping = build_name_mapping(
        &base_name_refs,
        &adapter_names,
        config.target_modules(),
        config.modules_to_save(),
    )?;

    let mut lora_target_count = 0;
    let mut replacement_count = 0;
    let mut bias_merge_count = 0;
    let mut estimated_bytes: u64 = 0;

    for (name, info) in base_names.iter().zip(tensor_infos.iter()) {
        let elem_size = dtype_byte_size(info.dtype).unwrap_or(0);
        let num_elements: usize = info.shape.iter().product();
        estimated_bytes += (num_elements * elem_size) as u64;

        if name_mapping.replacement(name).is_some() {
            replacement_count += 1;
        } else if name_mapping.is_lora_target(name) {
            lora_target_count += 1;
        } else if name_mapping.bias_source(name).is_some() {
            bias_merge_count += 1;
        }
    }

    let base_tensor_count = base_names.len();
    let passthrough_count =
        base_tensor_count - lora_target_count - replacement_count - bias_merge_count;

    Ok(crate::DryRunInfo {
        base_tensor_count,
        lora_target_count,
        replacement_count,
        bias_merge_count,
        passthrough_count,
        estimated_output_bytes: estimated_bytes,
        is_sharded,
        shard_count,
    })
}

/// Returns the serialized header bytes and the total data section size.
fn build_output_header(
    tensors: &[(String, TensorInfo)],
    metadata: Option<&HashMap<String, String>>,
) -> Result<(Vec<u8>, usize)> {
    let mut header_map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut current_offset: usize = 0;

    if let Some(meta) = metadata {
        header_map.insert(
            "__metadata__".to_string(),
            serde_json::to_value(meta).map_err(SurgeryError::Json)?,
        );
    }

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

/// Processes a single tensor: merge, replace, bias-merge, or pass-through.
fn process_tensor(
    name: &str,
    info: &TensorInfo,
    base_data: &[u8],
    adapter: &MappedSafetensors,
    config: &AdapterConfig,
    name_mapping: &crate::names::NameMapping,
    writer: &mut BufWriter<File>,
) -> Result<TensorAction> {
    if let Some(adapter_tensor_name) = name_mapping.replacement(name) {
        let adapter_data = adapter.tensor_data(adapter_tensor_name)?;
        writer.write_all(adapter_data)?;
        return Ok(TensorAction::Replaced);
    }

    if let Some((lora_a_name, lora_b_name)) = name_mapping.lora_pair(name) {
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
        return Ok(TensorAction::Merged);
    }

    if matches!(config.bias(), BiasMode::LoraOnly | BiasMode::All) {
        if let Some(adapter_bias_name) = name_mapping.bias_source(name) {
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
            return Ok(TensorAction::BiasMerged);
        }
    }

    writer.write_all(base_data)?;
    Ok(TensorAction::Copied)
}

enum TensorAction {
    Copied,
    Merged,
    Replaced,
    BiasMerged,
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
    let (header_bytes, _total_data_size) =
        build_output_header(&output_tensor_list, base.metadata.as_ref())?;

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

        let base_data = &base.mmap[info.data_start..info.data_end];
        let action = process_tensor(
            name,
            info,
            base_data,
            &adapter,
            config,
            &name_mapping,
            &mut writer,
        )?;
        apply_action(&mut stats, action);
    }

    if let Some(progress) = &progress {
        progress(total_tensors, total_tensors);
    }

    writer.flush()?;
    Ok(stats)
}

fn apply_action(stats: &mut MergeStats, action: TensorAction) {
    match action {
        TensorAction::Copied => stats.tensors_copied += 1,
        TensorAction::Merged => stats.tensors_merged += 1,
        TensorAction::Replaced => stats.tensors_replaced += 1,
        TensorAction::BiasMerged => stats.biases_merged += 1,
    }
}

/// Merges a sharded base model with a LoRA adapter, producing sharded output.
pub(crate) fn merge_sharded(
    base_dir: &Path,
    index: &ShardIndex,
    adapter_weights_path: &Path,
    config: &AdapterConfig,
    output_dir: &Path,
    progress: Option<&dyn Fn(usize, usize)>,
) -> Result<MergeStats> {
    let adapter = MappedSafetensors::open(adapter_weights_path)?;

    let shard_filenames = index.shard_filenames();

    let all_base_names: Vec<&str> = index.weight_map.keys().map(|s| s.as_str()).collect();
    let total_tensors = all_base_names.len();

    let base_name_refs: Vec<&str> = all_base_names;
    let adapter_names: Vec<&str> = adapter.tensors.iter().map(|(n, _)| n.as_str()).collect();

    let name_mapping = build_name_mapping(
        &base_name_refs,
        &adapter_names,
        config.target_modules(),
        config.modules_to_save(),
    )?;

    fs::create_dir_all(output_dir)?;

    let mut stats = MergeStats {
        tensors_copied: 0,
        tensors_merged: 0,
        tensors_replaced: 0,
        biases_merged: 0,
    };
    let mut tensors_processed: usize = 0;

    for shard_filename in &shard_filenames {
        let shard_path = base_dir.join(shard_filename);
        let shard = MappedSafetensors::open(&shard_path)?;

        let output_shard_path = output_dir.join(shard_filename);

        let shard_tensor_list: Vec<(String, TensorInfo)> = shard
            .tensors
            .iter()
            .map(|(name, info)| (name.clone(), info.clone()))
            .collect();
        let (header_bytes, _) = build_output_header(&shard_tensor_list, shard.metadata.as_ref())?;

        let output_file = File::create(&output_shard_path)?;
        let mut writer = BufWriter::new(output_file);

        let header_len = header_bytes.len() as u64;
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header_bytes)?;

        for (name, info) in &shard.tensors {
            if let Some(progress) = &progress {
                progress(tensors_processed, total_tensors);
            }

            let base_data = &shard.mmap[info.data_start..info.data_end];
            let action = process_tensor(
                name,
                info,
                base_data,
                &adapter,
                config,
                &name_mapping,
                &mut writer,
            )?;
            apply_action(&mut stats, action);
            tensors_processed += 1;
        }

        writer.flush()?;
    }

    if let Some(progress) = &progress {
        progress(total_tensors, total_tensors);
    }

    write_output_index(index, output_dir)?;

    Ok(stats)
}

fn write_output_index(index: &ShardIndex, output_dir: &Path) -> Result<()> {
    let output_index = serde_json::json!({
        "metadata": {},
        "weight_map": index.weight_map,
    });
    let json = serde_json::to_string_pretty(&output_index)?;
    fs::write(output_dir.join("model.safetensors.index.json"), json)?;
    Ok(())
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
        fs::write(path, serialized).unwrap();
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

    #[test]
    fn sharded_merge() {
        let dir = tempfile::tempdir().unwrap();
        let base_dir = dir.path().join("base");
        let adapter_dir = dir.path().join("adapter");
        let output_dir = dir.path().join("output");
        fs::create_dir_all(&base_dir).unwrap();
        fs::create_dir_all(&adapter_dir).unwrap();

        // Shard 1: q_proj
        write_safetensors(
            &base_dir.join("model-00001-of-00002.safetensors"),
            vec![(
                "model.layers.0.self_attn.q_proj.weight",
                vec![1.0, 0.0, 0.0, 1.0],
                vec![2, 2],
                Dtype::F16,
            )],
        );

        // Shard 2: embed_tokens (non-target, pass-through)
        write_safetensors(
            &base_dir.join("model-00002-of-00002.safetensors"),
            vec![(
                "model.embed_tokens.weight",
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                vec![3, 2],
                Dtype::F16,
            )],
        );

        let index_json = serde_json::json!({
            "metadata": {},
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors"
            }
        });
        fs::write(
            base_dir.join("model.safetensors.index.json"),
            serde_json::to_string(&index_json).unwrap(),
        )
        .unwrap();

        write_safetensors(
            &adapter_dir.join("adapter_model.safetensors"),
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
        let index = ShardIndex::from_path(&base_dir.join("model.safetensors.index.json")).unwrap();

        let stats = merge_sharded(
            &base_dir,
            &index,
            &adapter_dir.join("adapter_model.safetensors"),
            &config,
            &output_dir,
            None,
        )
        .unwrap();

        assert_eq!(stats.tensors_merged, 1);
        assert_eq!(stats.tensors_copied, 1);

        let shard1 =
            MappedSafetensors::open(&output_dir.join("model-00001-of-00002.safetensors")).unwrap();
        assert_eq!(shard1.tensors.len(), 1);
        let q_data = shard1
            .tensor_data("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let q_info = shard1
            .tensor_info("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let q_f32 =
            crate::merge::bytes_to_f32(q_data, q_info.dtype, &q_info.shape, "q_proj").unwrap();
        assert!((q_f32[[0, 0]] - 2.0).abs() < 0.01);
        assert!((q_f32[[1, 1]] - 2.0).abs() < 0.01);

        let shard2 =
            MappedSafetensors::open(&output_dir.join("model-00002-of-00002.safetensors")).unwrap();
        assert_eq!(shard2.tensors.len(), 1);
        assert!(shard2.tensor_info("model.embed_tokens.weight").is_ok());

        let out_index =
            ShardIndex::from_path(&output_dir.join("model.safetensors.index.json")).unwrap();
        assert_eq!(out_index.weight_map.len(), 2);
        assert_eq!(
            out_index.weight_map["model.layers.0.self_attn.q_proj.weight"],
            "model-00001-of-00002.safetensors"
        );
    }

    #[test]
    fn resolve_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");
        fs::write(&file_path, b"dummy").unwrap();

        match resolve_base_model(&file_path).unwrap() {
            BaseModelSource::SingleFile(p) => assert_eq!(p, file_path),
            BaseModelSource::Sharded { .. } => panic!("expected single file"),
        }
    }

    #[test]
    fn resolve_directory_single_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), b"dummy").unwrap();

        match resolve_base_model(dir.path()).unwrap() {
            BaseModelSource::SingleFile(p) => {
                assert_eq!(p, dir.path().join("model.safetensors"));
            }
            BaseModelSource::Sharded { .. } => panic!("expected single file"),
        }
    }

    #[test]
    fn resolve_directory_sharded() {
        let dir = tempfile::tempdir().unwrap();
        let index = serde_json::json!({
            "metadata": {},
            "weight_map": { "a": "shard-00001.safetensors" }
        });
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        match resolve_base_model(dir.path()).unwrap() {
            BaseModelSource::SingleFile(_) => panic!("expected sharded"),
            BaseModelSource::Sharded { dir: d, index: idx } => {
                assert_eq!(d, dir.path());
                assert_eq!(idx.shard_filenames(), vec!["shard-00001.safetensors"]);
            }
        }
    }

    #[test]
    fn resolve_empty_directory_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_base_model(dir.path()).is_err());
    }
}
