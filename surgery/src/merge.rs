//! LoRA merge math: dtype conversion and weight merging.

use std::io::Write;

use half::{bf16, f16};
use ndarray::{s, Array2};
use safetensors::Dtype;

use crate::{Result, SurgeryError};

/// Generates a `bytes -> Array2<T>` conversion function for a given numeric type.
/// Each invocation produces a public function that decodes F16, BF16, or F32 storage
/// bytes into the target type, with shape and length validation.
macro_rules! impl_bytes_to_array {
    ($(#[doc = $doc:expr])* $fn_name:ident -> $out_ty:ty, f16: $f16_conv:expr, bf16: $bf16_conv:expr, f32: $f32_conv:expr) => {
        $(#[doc = $doc])*
        pub fn $fn_name(
            bytes: &[u8],
            dtype: Dtype,
            shape: &[usize],
            name: &str,
        ) -> Result<Array2<$out_ty>> {
            if shape.len() != 2 {
                return Err(SurgeryError::ShapeMismatch {
                    name: format!("{name} (expected 2D)"),
                    expected: vec![2],
                    got: vec![shape.len()],
                });
            }
            let (rows, cols) = (shape[0], shape[1]);
            let expected_elements = rows * cols;

            let (elem_size, converter): (usize, Box<dyn Fn(&[u8]) -> $out_ty>) = match dtype {
                Dtype::F16 => (2, Box::new($f16_conv)),
                Dtype::BF16 => (2, Box::new($bf16_conv)),
                Dtype::F32 => (4, Box::new($f32_conv)),
                _ => {
                    return Err(SurgeryError::UnsupportedDtype {
                        name: name.to_string(),
                        dtype: format!("{dtype:?}"),
                    });
                }
            };

            let expected_bytes = expected_elements * elem_size;
            if bytes.len() != expected_bytes {
                return Err(SurgeryError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![expected_bytes],
                    got: vec![bytes.len()],
                });
            }

            let values: Vec<$out_ty> = bytes.chunks_exact(elem_size).map(|c| converter(c)).collect();

            Array2::from_shape_vec((rows, cols), values).map_err(|_| SurgeryError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![rows, cols],
                got: shape.to_vec(),
            })
        }
    };
}

/// Generates an `Array2<T> -> bytes` conversion function for a given numeric type.
/// Each invocation produces a public function that encodes array values into F16,
/// BF16, or F32 storage bytes.
macro_rules! impl_array_to_bytes {
    ($(#[doc = $doc:expr])* $fn_name:ident <- $in_ty:ty, f16: $f16_conv:expr, bf16: $bf16_conv:expr, f32: $f32_conv:expr) => {
        $(#[doc = $doc])*
        pub fn $fn_name(array: &Array2<$in_ty>, dtype: Dtype) -> Result<Vec<u8>> {
            let values = array.as_slice().ok_or_else(|| {
                SurgeryError::Safetensors("array is not contiguous in memory".to_string())
            })?;

            let converter: Box<dyn Fn(&$in_ty) -> Vec<u8>> = match dtype {
                Dtype::F16 => Box::new($f16_conv),
                Dtype::BF16 => Box::new($bf16_conv),
                Dtype::F32 => Box::new($f32_conv),
                _ => {
                    return Err(SurgeryError::UnsupportedDtype {
                        name: "output".to_string(),
                        dtype: format!("{dtype:?}"),
                    });
                }
            };

            Ok(values.iter().flat_map(|v| converter(v)).collect())
        }
    };
}

impl_bytes_to_array!(
    /// Converts raw tensor bytes to an `Array2<f32>`, upcasting from the storage dtype.
    bytes_to_f32 -> f32,
    f16: |c: &[u8]| f16::from_le_bytes([c[0], c[1]]).to_f32(),
    bf16: |c: &[u8]| bf16::from_le_bytes([c[0], c[1]]).to_f32(),
    f32: |c: &[u8]| f32::from_le_bytes([c[0], c[1], c[2], c[3]])
);

impl_array_to_bytes!(
    /// Converts an `Array2<f32>` back to raw bytes, downcasting to the storage dtype.
    f32_to_bytes <- f32,
    f16: |v: &f32| f16::from_f32(*v).to_le_bytes().to_vec(),
    bf16: |v: &f32| bf16::from_f32(*v).to_le_bytes().to_vec(),
    f32: |v: &f32| v.to_le_bytes().to_vec()
);

impl_bytes_to_array!(
    /// Converts raw tensor bytes to an `Array2<f64>`, upcasting from the storage dtype.
    bytes_to_f64 -> f64,
    f16: |c: &[u8]| f16::from_le_bytes([c[0], c[1]]).to_f64(),
    bf16: |c: &[u8]| bf16::from_le_bytes([c[0], c[1]]).to_f64(),
    f32: |c: &[u8]| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64
);

impl_array_to_bytes!(
    /// Converts an `Array2<f64>` to raw bytes, downcasting to the storage dtype.
    f64_to_bytes <- f64,
    f16: |v: &f64| f16::from_f64(*v).to_le_bytes().to_vec(),
    bf16: |v: &f64| bf16::from_f64(*v).to_le_bytes().to_vec(),
    f32: |v: &f64| (*v as f32).to_le_bytes().to_vec()
);

/// Computes `W_base + (alpha / r) * (lora_B @ lora_A)`.
///
/// When `fan_in_fan_out` is true, the delta is transposed: `(B @ A).T`.
pub fn merge_lora(
    base: &Array2<f32>,
    lora_a: &Array2<f32>,
    lora_b: &Array2<f32>,
    scaling: f32,
    fan_in_fan_out: bool,
) -> Result<Array2<f32>> {
    if lora_b.ncols() != lora_a.nrows() {
        return Err(SurgeryError::ShapeMismatch {
            name: "LoRA matmul inner dimension".to_string(),
            expected: vec![lora_b.ncols()],
            got: vec![lora_a.nrows()],
        });
    }

    let delta = if fan_in_fan_out {
        lora_b.dot(lora_a).t().to_owned()
    } else {
        lora_b.dot(lora_a)
    };

    if delta.nrows() != base.nrows() || delta.ncols() != base.ncols() {
        return Err(SurgeryError::ShapeMismatch {
            name: "LoRA delta vs base".to_string(),
            expected: vec![base.nrows(), base.ncols()],
            got: vec![delta.nrows(), delta.ncols()],
        });
    }

    Ok(base + &(delta * scaling))
}

/// Merges LoRA weights into base tensor bytes with f64-accurate matmul.
///
/// When `low_memory` is false (default): materializes the full delta matrix
/// in f64, then fuses the add+convert pass row by row. Faster but peak memory
/// scales with the largest tensor dimension.
///
/// When `low_memory` is true: tiles the matmul in fixed-size chunks using
/// `bytes_to_f32` and `f32_to_bytes` per tile. Bounded memory at the cost
/// of slower speed due to per-tile allocation overhead.
///
/// Both paths accumulate the matmul in f64 and perform the final base+delta
/// addition in f64 before downcasting, preserving numerical accuracy.
// 9 parameters needed: base data, both LoRA matrices, scaling, fan_in_fan_out,
// shape, dtype, memory mode, and writer. A struct would obscure the call site.
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
            name: "base tensor (expected 2D)".to_string(),
            expected: vec![2],
            got: vec![base_shape.len()],
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

    // Validate LoRA dimensions before matmul to return a clean error
    // instead of panicking inside ndarray's dot().
    // B is [out_features, rank], A is [rank, in_features].
    // Without fan_in_fan_out: delta = B @ A, shape [out_features, in_features] = [rows, cols].
    // With fan_in_fan_out: delta = (B @ A).T, base is stored as [in_features, out_features].
    let (expected_b_rows, expected_a_cols) = if fan_in_fan_out {
        (cols, rows)
    } else {
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
            name: "LoRA matmul inner dimension (rank)".to_string(),
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

            // Without fan_in_fan_out: rows [start..end] of B @ A = B[start..end, :] @ A.
            // With fan_in_fan_out: rows [start..end] of (B @ A).T = cols [start..end] of B @ A
            //   = B @ A[:, start..end], then transpose.
            let delta_f64 = if fan_in_fan_out {
                let tile_a = lora_a.slice(s![.., start..end]);
                (lora_b.dot(&tile_a) * scaling).t().to_owned()
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
            (lora_b.dot(lora_a) * scaling)
                .t()
                .as_standard_layout()
                .into_owned()
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

        let delta = delta_f64.as_slice().ok_or_else(|| {
            SurgeryError::Safetensors("delta array is not contiguous in memory".to_string())
        })?;

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
                        let merged = (base_val as f64 + d) as f32;
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
                        let merged = (base_val as f64 + d) as f32;
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
                        let merged = (base_val as f64 + d) as f32;
                        out_chunk.copy_from_slice(&merged.to_le_bytes());
                    }
                }
                _ => {
                    return Err(SurgeryError::UnsupportedDtype {
                        name: "merge output".to_string(),
                        dtype: format!("{dtype:?}"),
                    });
                }
            }
            writer.write_all(&out_buf)?;
        }
    }
    Ok(())
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
    fn merge_lora_basic() {
        let base = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let lora_a = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
        let lora_b = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
        let scaling = 2.0;

        let merged = merge_lora(&base, &lora_a, &lora_b, scaling, false).unwrap();

        // delta = [[1],[1]] @ [[1,1,1]] = [[1,1,1],[1,1,1]]
        // merged = base + 2.0 * delta = [[3,2,2],[2,3,2]]
        assert_eq!(merged[[0, 0]], 3.0);
        assert_eq!(merged[[0, 1]], 2.0);
        assert_eq!(merged[[0, 2]], 2.0);
        assert_eq!(merged[[1, 0]], 2.0);
        assert_eq!(merged[[1, 1]], 3.0);
        assert_eq!(merged[[1, 2]], 2.0);
    }

    #[test]
    fn merge_lora_fan_in_fan_out() {
        let base = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let lora_a = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let lora_b = Array2::from_shape_vec((2, 1), vec![3.0, 4.0]).unwrap();
        let scaling = 1.0;

        let merged = merge_lora(&base, &lora_a, &lora_b, scaling, true).unwrap();

        // B @ A = [[3],[4]] @ [[1,2]] = [[3,6],[4,8]]
        // (B @ A).T = [[3,4],[6,8]]
        // merged = base + delta = [[4,4],[6,9]]
        assert_eq!(merged[[0, 0]], 4.0);
        assert_eq!(merged[[0, 1]], 4.0);
        assert_eq!(merged[[1, 0]], 6.0);
        assert_eq!(merged[[1, 1]], 9.0);
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
    fn merge_lora_scaling_factor() {
        // Verify scaling = alpha/r is correctly applied
        let base = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let lora_a = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let lora_b = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
        // scaling = alpha/r = 16/8 = 2.0
        let merged = merge_lora(&base, &lora_a, &lora_b, 2.0, false).unwrap();
        // delta = [[1,1],[1,1]], merged = 0 + 2.0 * delta = [[2,2],[2,2]]
        assert_eq!(merged[[0, 0]], 2.0);
        assert_eq!(merged[[1, 1]], 2.0);
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
