//! LoRA merge math: dtype conversion and weight merging.

use half::{bf16, f16};
use ndarray::Array2;
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

/// Computes `W_base + (alpha / r) * (lora_B @ lora_A)`.
///
/// When `fan_in_fan_out` is true, `lora_A` is transposed before the matmul.
#[must_use]
pub fn merge_lora(
    base: &Array2<f32>,
    lora_a: &Array2<f32>,
    lora_b: &Array2<f32>,
    scaling: f32,
    fan_in_fan_out: bool,
) -> Array2<f64> {
    let base_f64 = base.mapv(|v| v as f64);
    let a_f64 = lora_a.mapv(|v| v as f64);
    let b_f64 = lora_b.mapv(|v| v as f64);

    let delta = if fan_in_fan_out {
        b_f64.dot(&a_f64.t())
    } else {
        b_f64.dot(&a_f64)
    };

    base_f64 + &(delta * scaling as f64)
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
#[must_use]
pub fn merge_bias(base_bias: &Array2<f32>, adapter_bias: &Array2<f32>) -> Array2<f32> {
    base_bias + adapter_bias
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

        let merged = merge_lora(&base, &lora_a, &lora_b, scaling, false);

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
        let lora_a = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let lora_b = Array2::from_shape_vec((2, 1), vec![3.0, 4.0]).unwrap();
        let scaling = 1.0;

        let merged = merge_lora(&base, &lora_a, &lora_b, scaling, true);

        // A^T = [[1, 2]], B @ A^T = [[3,6],[4,8]]
        // merged = base + delta = [[4,6],[4,9]]
        assert_eq!(merged[[0, 0]], 4.0);
        assert_eq!(merged[[0, 1]], 6.0);
        assert_eq!(merged[[1, 0]], 4.0);
        assert_eq!(merged[[1, 1]], 9.0);
    }

    #[test]
    fn merge_bias_basic() {
        let base = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let adapter = Array2::from_shape_vec((1, 4), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let result = merge_bias(&base, &adapter);
        assert!((result[[0, 0]] - 1.1).abs() < 1e-6);
        assert!((result[[0, 3]] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn unsupported_dtype_errors() {
        let bytes = vec![0u8; 16];
        let err = bytes_to_f32(&bytes, Dtype::I32, &[2, 2], "test_tensor").unwrap_err();
        assert!(err.to_string().contains("unsupported dtype"));
    }
}
