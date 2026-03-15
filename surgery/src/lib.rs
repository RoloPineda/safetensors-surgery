//! safetensors-surgery: merge PEFT LoRA adapters into base models with bounded memory.

pub mod config;
pub mod io;
pub mod merge;
pub mod names;
extern crate blas_src;

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
    TensorNotFound { name: String, location: String },

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

pub type Result<T> = std::result::Result<T, SurgeryError>;

/// Statistics returned after a merge operation completes.
#[derive(Debug, Clone)]
pub struct MergeStats {
    pub tensors_copied: usize,
    pub tensors_merged: usize,
    pub tensors_replaced: usize,
    pub biases_merged: usize,
}

/// Pre-merge analysis returned by [`dry_run_info`].
#[derive(Debug, Clone)]
pub struct DryRunInfo {
    pub base_tensor_count: usize,
    pub lora_target_count: usize,
    pub replacement_count: usize,
    pub bias_merge_count: usize,
    pub passthrough_count: usize,
    pub estimated_output_bytes: u64,
    pub is_sharded: bool,
    pub shard_count: usize,
}

/// Inspects a base model and adapter without writing output.
pub fn dry_run_info(base_model_path: &Path, adapter_path: &Path) -> Result<DryRunInfo> {
    let adapter_config_path = adapter_path.join("adapter_config.json");
    let adapter_config = config::AdapterConfig::from_path(&adapter_config_path)?;
    let adapter_weights_path = adapter_path.join("adapter_model.safetensors");

    io::compute_dry_run_info(base_model_path, &adapter_weights_path, &adapter_config)
}

/// Merges a LoRA adapter into a base model, writing the result to the output path.
///
/// When `low_memory` is true, the merge uses tiled computation to reduce peak
/// memory at the cost of speed. When false, the full delta matrix is materialized
/// for faster merging.
///
/// `base_model_path` can be:
/// - A `.safetensors` file (single-file model)
/// - A directory containing `model.safetensors` (single-file model)
/// - A directory containing `model.safetensors.index.json` (sharded model)
///
/// For single-file models, `output_path` is the output `.safetensors` file.
/// For sharded models, `output_path` is the output directory (shards + index.json).
pub fn merge_adapter(
    base_model_path: &Path,
    adapter_path: &Path,
    output_path: &Path,
    low_memory: bool,
    progress: Option<&dyn Fn(usize, usize)>,
) -> Result<MergeStats> {
    let adapter_config_path = adapter_path.join("adapter_config.json");
    let adapter_config = config::AdapterConfig::from_path(&adapter_config_path)?;

    let adapter_weights_path = adapter_path.join("adapter_model.safetensors");

    match io::resolve_base_model(base_model_path)? {
        io::BaseModelSource::SingleFile(base_file) => io::merge_and_write(
            &base_file,
            &adapter_weights_path,
            &adapter_config,
            output_path,
            low_memory,
            progress,
        ),
        io::BaseModelSource::Sharded { dir, index } => io::merge_sharded(
            &dir,
            &index,
            &adapter_weights_path,
            &adapter_config,
            output_path,
            low_memory,
            progress,
        ),
    }
}
