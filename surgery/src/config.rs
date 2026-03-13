//! Parsing and validation of PEFT adapter configuration files.

use std::path::Path;

use serde::Deserialize;

use crate::{Result, SurgeryError};

/// Controls how bias tensors are handled during merge.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BiasMode {
    /// No bias tensors in the adapter.
    None,
    /// Bias tensors for target modules only.
    LoraOnly,
    /// Bias tensors for all modules.
    All,
}

/// Parsed PEFT adapter configuration from `adapter_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct AdapterConfig {
    r: u32,
    lora_alpha: f32,
    target_modules: Vec<String>,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default = "default_bias")]
    bias: BiasMode,
    #[serde(default)]
    modules_to_save: Option<Vec<String>>,
    peft_type: String,
}

fn default_bias() -> BiasMode {
    BiasMode::None
}

impl AdapterConfig {
    /// Parses and validates from a file path. Rejects non-LORA peft types.
    pub fn from_path(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }

    /// Parses and validates from a JSON string. Rejects non-LORA peft types.
    pub fn from_json(json: &str) -> Result<Self> {
        let config: AdapterConfig = serde_json::from_str(json)?;
        if config.peft_type != "LORA" {
            return Err(SurgeryError::InvalidConfig(format!(
                "unsupported peft_type '{}', expected 'LORA'",
                config.peft_type
            )));
        }
        if config.r == 0 {
            return Err(SurgeryError::InvalidConfig(
                "rank (r) must be greater than zero".to_string(),
            ));
        }
        Ok(config)
    }

    /// Returns the LoRA rank.
    pub fn rank(&self) -> u32 {
        self.r
    }

    /// Returns the LoRA alpha value.
    pub fn alpha(&self) -> f32 {
        self.lora_alpha
    }

    /// Returns the scaling coefficient `alpha / r`.
    #[must_use]
    pub fn scaling(&self) -> f32 {
        self.lora_alpha / self.r as f32
    }

    /// Returns the target module names for LoRA application.
    pub fn target_modules(&self) -> &[String] {
        &self.target_modules
    }

    /// Returns whether Conv1D-style transposition is enabled.
    pub fn fan_in_fan_out(&self) -> bool {
        self.fan_in_fan_out
    }

    /// Returns the bias handling mode.
    pub fn bias(&self) -> &BiasMode {
        &self.bias
    }

    /// Returns modules whose weights are fully replaced (not low-rank merged).
    pub fn modules_to_save(&self) -> Option<&[String]> {
        self.modules_to_save.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_config() {
        let json = r#"{
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "modules_to_save": null,
            "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
            "peft_type": "LORA"
        }"#;
        let config = AdapterConfig::from_json(json).unwrap();
        assert_eq!(config.rank(), 8);
        assert_eq!(config.alpha(), 16.0);
        assert_eq!(config.scaling(), 2.0);
        assert_eq!(config.target_modules(), &["q_proj", "v_proj"]);
        assert!(!config.fan_in_fan_out());
        assert_eq!(config.bias(), &BiasMode::None);
        assert!(config.modules_to_save().is_none());
    }

    #[test]
    fn parse_config_with_lora_only_bias() {
        let json = r#"{
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj"],
            "bias": "lora_only",
            "peft_type": "LORA"
        }"#;
        let config = AdapterConfig::from_json(json).unwrap();
        assert_eq!(config.bias(), &BiasMode::LoraOnly);
        assert!(!config.fan_in_fan_out());
    }

    #[test]
    fn reject_non_lora_peft_type() {
        let json = r#"{
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "peft_type": "PREFIX_TUNING"
        }"#;
        let err = AdapterConfig::from_json(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("PREFIX_TUNING"));
        assert!(msg.contains("LORA"));
    }

    #[test]
    fn reject_zero_rank() {
        let json = r#"{
            "r": 0,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "peft_type": "LORA"
        }"#;
        let err = AdapterConfig::from_json(json).unwrap_err();
        assert!(err.to_string().contains("rank"));
    }

    #[test]
    fn parse_with_modules_to_save() {
        let json = r#"{
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "modules_to_save": ["lm_head"],
            "peft_type": "LORA"
        }"#;
        let config = AdapterConfig::from_json(json).unwrap();
        assert_eq!(config.modules_to_save().unwrap(), &["lm_head"]);
    }
}
