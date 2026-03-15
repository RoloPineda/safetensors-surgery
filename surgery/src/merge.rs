//! LoRA merge math: dtype conversion and weight merging.

use std::io::Write;

use half::{bf16, f16};
use ndarray::{s, Array2};
use safetensors::Dtype;

use crate::{Result, SurgeryError};

/// Converts raw tensor bytes to an `Array2<f32>`, upcasting from the storage dtype.
pub fn bytes_to_f32(
    bytes: &[u8],
    dtype: Dtype,
    shape: &[usize],
    name: &str,
) -> Result<Array2<f32>> {
    if shape.len() != 2 {
        return Err(SurgeryError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![0, 0],
            got: shape.to_vec(),
        });
    }
    let (rows, cols) = (shape[0], shape[1]);
    let expected_elements = rows * cols;

    let values: Vec<f32> = match dtype {
        Dtype::F16 => {
            if bytes.len() != expected_elements * 2 {
                return Err(SurgeryError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![expected_elements * 2],
                    got: vec![bytes.len()],
                });
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                .collect()
        }
        Dtype::BF16 => {
            if bytes.len() != expected_elements * 2 {
                return Err(SurgeryError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![expected_elements * 2],
                    got: vec![bytes.len()],
                });
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                .collect()
        }
        Dtype::F32 => {
            if bytes.len() != expected_elements * 4 {
                return Err(SurgeryError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![expected_elements * 4],
                    got: vec![bytes.len()],
                });
            }
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }
        _ => {
            return Err(SurgeryError::UnsupportedDtype {
                name: name.to_string(),
                dtype: format!("{dtype:?}"),
            });
        }
    };

    Array2::from_shape_vec((rows, cols), values).map_err(|_| SurgeryError::ShapeMismatch {
        name: name.to_string(),
        expected: vec![rows, cols],
        got: shape.to_vec(),
    })
}

/// Converts an `Array2<f32>` back to raw bytes, downcasting to the storage dtype.
pub fn f32_to_bytes(array: &Array2<f32>, dtype: Dtype) -> Result<Vec<u8>> {
    let values = array.as_slice().ok_or_else(|| {
        SurgeryError::Safetensors("array is not contiguous in memory".to_string())
    })?;

    match dtype {
        Dtype::F16 => Ok(values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect()),
        Dtype::BF16 => Ok(values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect()),
        Dtype::F32 => Ok(values.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err(SurgeryError::UnsupportedDtype {
            name: "output".to_string(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

/// Converts raw tensor bytes to an `Array2<f64>`, upcasting from the storage dtype.
pub fn bytes_to_f64(
    bytes: &[u8],
    dtype: Dtype,
    shape: &[usize],
    name: &str,
) -> Result<Array2<f64>> {
    if shape.len() != 2 {
        return Err(SurgeryError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![0, 0],
            got: shape.to_vec(),
        });
    }
    let (rows, cols) = (shape[0], shape[1]);
    let expected_elements = rows * cols;

    let values: Vec<f64> = match dtype {
        Dtype::F16 => {
            if bytes.len() != expected_elements * 2 {
                return Err(SurgeryError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![expected_elements * 2],
                    got: vec![bytes.len()],
                });
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f64())
                .collect()
        }
        Dtype::BF16 => {
            if bytes.len() != expected_elements * 2 {
                return Err(SurgeryError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![expected_elements * 2],
                    got: vec![bytes.len()],
                });
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]).to_f64())
                .collect()
        }
        Dtype::F32 => {
            if bytes.len() != expected_elements * 4 {
                return Err(SurgeryError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![expected_elements * 4],
                    got: vec![bytes.len()],
                });
            }
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
                .collect()
        }
        _ => {
            return Err(SurgeryError::UnsupportedDtype {
                name: name.to_string(),
                dtype: format!("{dtype:?}"),
            });
        }
    };

    Array2::from_shape_vec((rows, cols), values).map_err(|_| SurgeryError::ShapeMismatch {
        name: name.to_string(),
        expected: vec![rows, cols],
        got: shape.to_vec(),
    })
}

/// Merges LoRA weights into base tensor bytes with f64-accurate matmul.
///
/// When `low_memory` is false (default): materializes the full delta matrix
/// in f64, then fuses the add+convert pass row by row. Faster but uses more
/// memory (~1.88GB for the largest tensors in a 70B model).
///
/// When `low_memory` is true: tiles the matmul in 512-row chunks using
/// `bytes_to_f32` and `f32_to_bytes` per tile. Uses roughly 150MB per tile
/// instead of the full delta, at the cost of ~1.5x slower speed.
///
/// Both paths accumulate the matmul in f64; the final base+delta addition
/// is done in f32 (low-memory) or f32 per-element (non-low-memory).
///
/// When `fan_in_fan_out` is true, the delta (lora_B @ lora_A) is transposed
/// before adding to the base weight, matching PEFT's behavior for Conv1D/GPT-2
/// layers.
// Justified: streaming function needs all merge parameters plus I/O target;
// a parameter struct would obscure the data flow for a single call site.
#[allow(clippy::too_many_arguments)]
pub fn streaming_lora_merge_write(
    base_bytes: &[u8],
    lora_a: &Array2<f64>,
    lora_b: &Array2<f64>,
    scaling: f64,
    fan_in_fan_out: bool,
    base_shape: &[usize],
    dtype: Dtype,
    low_memory: bool,
    writer: &mut impl Write,
) -> Result<()> {
    if base_shape.len() != 2 {
        return Err(SurgeryError::ShapeMismatch {
            name: "base tensor".to_string(),
            expected: vec![0, 0],
            got: base_shape.to_vec(),
        });
    }
    let rows = base_shape[0];
    let cols = base_shape[1];

    let elem_size = match dtype {
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::F32 => 4,
        _ => {
            return Err(SurgeryError::UnsupportedDtype {
                name: "output".to_string(),
                dtype: format!("{dtype:?}"),
            });
        }
    };
    let row_bytes = cols * elem_size;
    let expected_bytes = rows * row_bytes;
    if base_bytes.len() != expected_bytes {
        return Err(SurgeryError::ShapeMismatch {
            name: "base tensor bytes".to_string(),
            expected: vec![expected_bytes],
            got: vec![base_bytes.len()],
        });
    }

    // Validate LoRA matrix dimensions against base tensor.
    // When fan_in_fan_out is true, the base weight is stored as [in_features, out_features]
    // and the delta = (B @ A)^T = A^T @ B^T has shape [in_features, out_features].
    // B is [out_features, r], A is [r, in_features].
    let (expected_b_rows, expected_a_cols) = if fan_in_fan_out {
        // Base is [in_features, out_features]; delta = (B @ A)^T = A^T @ B^T
        // B is [out_features, r], A is [r, in_features]
        (cols, rows)
    } else {
        // Base is [out_features, in_features]; delta = B @ A
        // B is [out_features, r], A is [r, in_features]
        (rows, cols)
    };
    if lora_b.nrows() != expected_b_rows {
        return Err(SurgeryError::ShapeMismatch {
            name: "lora_B rows vs base".to_string(),
            expected: vec![expected_b_rows],
            got: vec![lora_b.nrows()],
        });
    }
    if lora_a.ncols() != expected_a_cols {
        return Err(SurgeryError::ShapeMismatch {
            name: "lora_A cols vs base".to_string(),
            expected: vec![expected_a_cols],
            got: vec![lora_a.ncols()],
        });
    }
    if lora_b.ncols() != lora_a.nrows() {
        return Err(SurgeryError::ShapeMismatch {
            name: "LoRA matmul inner dimension".to_string(),
            expected: vec![lora_b.ncols()],
            got: vec![lora_a.nrows()],
        });
    }

    if low_memory {
        const TILE_ROWS: usize = 512;
        let mut start = 0;
        while start < rows {
            let end = (start + TILE_ROWS).min(rows);
            let tile_rows = end - start;

            let delta_f64 = if fan_in_fan_out {
                // delta = (B @ A)^T = A^T @ B^T; tile over rows of A^T (= cols of A).
                // A^T[start..end, :] is A[:, start..end]^T, shape [tile_rows, r].
                // Result: [tile_rows, out_features] = [tile_rows, cols] (since cols = out_features for fan_in_fan_out base).
                let tile_a_t = lora_a.slice(s![.., start..end]).t().to_owned();
                tile_a_t.dot(&lora_b.t()) * scaling
            } else {
                let tile_b = lora_b.slice(s![start..end, ..]).to_owned();
                tile_b.dot(lora_a) * scaling
            };
            let delta_f32 = delta_f64.mapv(|v| v as f32);

            let base_offset = start * row_bytes;
            let base_tile_bytes = &base_bytes[base_offset..base_offset + tile_rows * row_bytes];
            let mut base_tile = bytes_to_f32(base_tile_bytes, dtype, &[tile_rows, cols], "tile")?;

            base_tile += &delta_f32;

            let out_bytes = f32_to_bytes(&base_tile, dtype)?;
            writer.write_all(&out_bytes)?;

            start = end;
        }
    } else {
        let delta_f64 = if fan_in_fan_out {
            // delta = (B @ A)^T = A^T @ B^T, shape [in_features, out_features]
            lora_a.t().dot(&lora_b.t()) * scaling
        } else {
            lora_b.dot(lora_a) * scaling
        };

        let delta_shape = delta_f64.shape();
        if delta_shape[0] != rows || delta_shape[1] != cols {
            return Err(SurgeryError::ShapeMismatch {
                name: "LoRA delta".to_string(),
                expected: vec![rows, cols],
                got: delta_shape.to_vec(),
            });
        }

        // Ensure contiguous layout after potential transpose.
        let delta_owned;
        let delta = match delta_f64.as_slice() {
            Some(s) => s,
            None => {
                delta_owned = delta_f64.as_standard_layout().into_owned();
                delta_owned.as_slice().ok_or_else(|| {
                    SurgeryError::Safetensors("delta array is not contiguous in memory".to_string())
                })?
            }
        };

        let mut out_buf: Vec<u8> = vec![0u8; row_bytes];

        for i in 0..rows {
            let base_row = &base_bytes[i * row_bytes..(i + 1) * row_bytes];
            let delta_row = &delta[i * cols..(i + 1) * cols];

            match dtype {
                Dtype::F16 => {
                    for ((in_chunk, &d), out_chunk) in base_row
                        .chunks_exact(2)
                        .zip(delta_row)
                        .zip(out_buf.chunks_exact_mut(2))
                    {
                        let base_val = f16::from_le_bytes([in_chunk[0], in_chunk[1]]).to_f32();
                        let merged = base_val + d as f32;
                        out_chunk.copy_from_slice(&f16::from_f32(merged).to_le_bytes());
                    }
                }
                Dtype::BF16 => {
                    for ((in_chunk, &d), out_chunk) in base_row
                        .chunks_exact(2)
                        .zip(delta_row)
                        .zip(out_buf.chunks_exact_mut(2))
                    {
                        let base_val = bf16::from_le_bytes([in_chunk[0], in_chunk[1]]).to_f32();
                        let merged = base_val + d as f32;
                        out_chunk.copy_from_slice(&bf16::from_f32(merged).to_le_bytes());
                    }
                }
                Dtype::F32 => {
                    for ((in_chunk, &d), out_chunk) in base_row
                        .chunks_exact(4)
                        .zip(delta_row)
                        .zip(out_buf.chunks_exact_mut(4))
                    {
                        let base_val = f32::from_le_bytes([
                            in_chunk[0],
                            in_chunk[1],
                            in_chunk[2],
                            in_chunk[3],
                        ]);
                        let merged = base_val + d as f32;
                        out_chunk.copy_from_slice(&merged.to_le_bytes());
                    }
                }
                _ => {
                    return Err(SurgeryError::UnsupportedDtype {
                        name: "output".to_string(),
                        dtype: format!("{dtype:?}"),
                    });
                }
            }
            writer.write_all(&out_buf)?;
        }
    }
    Ok(())
}

/// Converts an `Array2<f64>` to raw bytes, downcasting to the storage dtype.
pub fn f64_to_bytes(array: &Array2<f64>, dtype: Dtype) -> Result<Vec<u8>> {
    let values = array.as_slice().ok_or_else(|| {
        SurgeryError::Safetensors("array is not contiguous in memory".to_string())
    })?;

    match dtype {
        Dtype::F16 => Ok(values
            .iter()
            .flat_map(|&v| f16::from_f64(v).to_le_bytes())
            .collect()),
        Dtype::BF16 => Ok(values
            .iter()
            .flat_map(|&v| bf16::from_f64(v).to_le_bytes())
            .collect()),
        Dtype::F32 => Ok(values
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect()),
        _ => Err(SurgeryError::UnsupportedDtype {
            name: "output".to_string(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

/// Adds adapter bias values to base bias values element-wise.
pub fn merge_bias(base_bias: &Array2<f32>, adapter_bias: &Array2<f32>) -> Result<Array2<f32>> {
    if base_bias.shape() != adapter_bias.shape() {
        return Err(SurgeryError::ShapeMismatch {
            name: "bias".to_string(),
            expected: base_bias.shape().to_vec(),
            got: adapter_bias.shape().to_vec(),
        });
    }
    Ok(base_bias + adapter_bias)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_f16() {
        let original = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let bytes = f32_to_bytes(&original, Dtype::F16).unwrap();
        let recovered = bytes_to_f32(&bytes, Dtype::F16, &[2, 3], "test").unwrap();
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.01, "f16 roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_bf16() {
        let original = Array2::from_shape_vec((2, 2), vec![1.0, -2.0, 0.5, 3.0]).unwrap();
        let bytes = f32_to_bytes(&original, Dtype::BF16).unwrap();
        let recovered = bytes_to_f32(&bytes, Dtype::BF16, &[2, 2], "test").unwrap();
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.05, "bf16 roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_f32() {
        let original = Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let bytes = f32_to_bytes(&original, Dtype::F32).unwrap();
        let recovered = bytes_to_f32(&bytes, Dtype::F32, &[2, 2], "test").unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn merge_bias_basic() {
        let base = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let adapter = Array2::from_shape_vec((1, 4), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let result = merge_bias(&base, &adapter).unwrap();
        assert!((result[[0, 0]] - 1.1).abs() < 1e-6);
        assert!((result[[0, 3]] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn unsupported_dtype_errors() {
        let bytes = vec![0u8; 16];
        let err = bytes_to_f32(&bytes, Dtype::I32, &[2, 2], "test_tensor").unwrap_err();
        assert!(err.to_string().contains("unsupported dtype"));
    }

    #[test]
    fn bytes_to_f32_wrong_length_errors() {
        // Shape says 2x2 (4 elements = 16 bytes for F32), but only give 8 bytes
        let bytes = vec![0u8; 8];
        let err = bytes_to_f32(&bytes, Dtype::F32, &[2, 2], "test_tensor").unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn bytes_to_f32_1d_shape_errors() {
        let bytes = vec![0u8; 8];
        let err = bytes_to_f32(&bytes, Dtype::F32, &[2], "test_tensor").unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn f64_to_bytes_bf16_roundtrip() {
        let array = Array2::from_shape_vec((1, 4), vec![1.0_f64, 2.0, -3.0, 0.5]).unwrap();
        let bytes = f64_to_bytes(&array, Dtype::BF16).unwrap();
        // Should produce 8 bytes (4 elements * 2 bytes each)
        assert_eq!(bytes.len(), 8);
        // Read back via bytes_to_f32
        let recovered = bytes_to_f32(&bytes, Dtype::BF16, &[1, 4], "test").unwrap();
        assert!((recovered[[0, 0]] - 1.0).abs() < 0.05);
        assert!((recovered[[0, 1]] - 2.0).abs() < 0.05);
        assert!((recovered[[0, 2]] - (-3.0)).abs() < 0.05);
        assert!((recovered[[0, 3]] - 0.5).abs() < 0.05);
    }
}
