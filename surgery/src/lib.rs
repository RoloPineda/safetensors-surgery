//! safetensors-surgery: merge PEFT LoRA adapters into base models with bounded memory.

pub mod config;
pub mod io;
pub mod merge;
pub mod names;

use std::path::Path;

use thiserror::Error;

/// Errors that can occur during safetensors merge operations.
#[derive(Debug, Error)]
pub enum SurgeryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("safetensors error: {0}")]
    Safetensors(String),

    #[error("tensor '{name}' not found in {location}")]
    TensorNotFound {
        name: String,
        /// E.g. "base model", "adapter".
        location: String,
    },

    #[error("shape mismatch for tensor '{name}': expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error(
        "unsupported dtype '{dtype}' for tensor '{name}'; only F16, BF16, and F32 are supported"
    )]
    UnsupportedDtype { name: String, dtype: String },

    #[error("invalid adapter config: {0}")]
    InvalidConfig(String),

    #[error("adapter targets module '{module}' but no matching tensor found in adapter file")]
    MissingAdapterTensor { module: String },

    #[error("sharding error: {0}")]
    ShardingError(String),
}

/// Result type alias using `SurgeryError`.
pub type Result<T> = std::result::Result<T, SurgeryError>;

/// Statistics returned after a merge operation completes.
#[derive(Debug, Clone)]
pub struct MergeStats {
    pub tensors_copied: usize,
    pub tensors_merged: usize,
    pub tensors_replaced: usize,
    pub biases_merged: usize,
}

/// Merges a LoRA adapter into a base model, writing the result to the output path.
///
/// Memory usage is bounded by one tensor at a time regardless of model size.
pub fn merge_adapter(
    base_model_path: &Path,
    adapter_path: &Path,
    output_path: &Path,
    progress: Option<&dyn Fn(usize, usize)>,
) -> Result<MergeStats> {
    let adapter_config_path = adapter_path.join("adapter_config.json");
    let adapter_config = config::AdapterConfig::from_path(&adapter_config_path)?;

    let adapter_weights_path = adapter_path.join("adapter_model.safetensors");

    io::merge_and_write(
        base_model_path,
        &adapter_weights_path,
        &adapter_config,
        output_path,
        progress,
    )
}
