//! Tensor name mapping between base model and PEFT adapter conventions.

use std::collections::{HashMap, HashSet};

use crate::{Result, SurgeryError};

#[derive(Debug)]
pub struct NameMapping {
    /// base_name -> (lora_A adapter name, lora_B adapter name)
    lora_targets: HashMap<String, (String, String)>,
    /// base_name -> adapter name, for full module replacements
    replacements: HashMap<String, String>,
    /// base bias name -> adapter bias name
    biases: HashMap<String, String>,
}

const ADAPTER_PREFIX: &str = "base_model.model.";

impl NameMapping {
    pub fn lora_pair(&self, base_name: &str) -> Option<&(String, String)> {
        self.lora_targets.get(base_name)
    }

    pub fn is_lora_target(&self, base_name: &str) -> bool {
        self.lora_targets.contains_key(base_name)
    }

    pub fn replacement(&self, base_name: &str) -> Option<&str> {
        self.replacements.get(base_name).map(|s| s.as_str())
    }

    pub fn bias_source(&self, base_name: &str) -> Option<&str> {
        self.biases.get(base_name).map(|s| s.as_str())
    }
}

/// Builds a name mapping by iterating adapter tensor names, stripping the
/// `base_model.model.` prefix and `.lora_A.weight` / `.lora_B.weight` suffixes,
/// and matching against base model tensor names.
pub fn build_name_mapping(
    base_tensor_names: &[&str],
    adapter_tensor_names: &[&str],
    target_modules: &[String],
    modules_to_save: Option<&[String]>,
) -> Result<NameMapping> {
    let base_set: HashSet<&str> = base_tensor_names.iter().copied().collect();

    let mut lora_a_map: HashMap<String, String> = HashMap::new();
    let mut lora_b_map: HashMap<String, String> = HashMap::new();
    let mut replacements: HashMap<String, String> = HashMap::new();
    let mut biases: HashMap<String, String> = HashMap::new();

    for &adapter_name in adapter_tensor_names {
        let stripped = match adapter_name.strip_prefix(ADAPTER_PREFIX) {
            Some(s) => s,
            None => {
                eprintln!(
                    "warning: adapter tensor '{}' does not start with '{}', skipping",
                    adapter_name, ADAPTER_PREFIX
                );
                continue;
            }
        };

        if let Some(base_part) = stripped.strip_suffix(".lora_A.weight") {
            let base_weight = format!("{base_part}.weight");
            if base_set.contains(base_weight.as_str()) {
                lora_a_map.insert(base_weight, adapter_name.to_string());
            } else {
                eprintln!(
                    "warning: adapter tensor '{}' has no matching base tensor '{}'",
                    adapter_name, base_weight
                );
            }
        } else if let Some(base_part) = stripped.strip_suffix(".lora_B.weight") {
            let base_weight = format!("{base_part}.weight");
            if base_set.contains(base_weight.as_str()) {
                lora_b_map.insert(base_weight, adapter_name.to_string());
            } else {
                eprintln!(
                    "warning: adapter tensor '{}' has no matching base tensor '{}'",
                    adapter_name, base_weight
                );
            }
        } else if let Some(save_modules) = modules_to_save {
            for module in save_modules {
                if stripped.contains(module.as_str()) && base_set.contains(stripped) {
                    replacements.insert(stripped.to_string(), adapter_name.to_string());
                }
            }
        }

        if stripped.ends_with(".bias") && base_set.contains(stripped) {
            biases.insert(stripped.to_string(), adapter_name.to_string());
        }
    }

    let mut lora_targets: HashMap<String, (String, String)> = HashMap::new();
    for (base_name, a_name) in &lora_a_map {
        if let Some(b_name) = lora_b_map.get(base_name) {
            lora_targets.insert(base_name.clone(), (a_name.clone(), b_name.clone()));
        } else {
            eprintln!(
                "warning: found lora_A for '{}' but no matching lora_B",
                base_name
            );
        }
    }

    for module in target_modules {
        let has_match = lora_targets
            .keys()
            .any(|base_name| base_name.contains(module.as_str()));
        if !has_match {
            return Err(SurgeryError::MissingAdapterTensor {
                module: module.clone(),
            });
        }
    }

    Ok(NameMapping {
        lora_targets,
        replacements,
        biases,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_name_mapping() {
        let base_names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.embed_tokens.weight",
        ];
        let adapter_names = vec![
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
        ];
        let target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];

        let mapping =
            build_name_mapping(&base_names, &adapter_names, &target_modules, None).unwrap();

        assert!(mapping.is_lora_target("model.layers.0.self_attn.q_proj.weight"));
        assert!(mapping.is_lora_target("model.layers.0.self_attn.v_proj.weight"));
        assert!(!mapping.is_lora_target("model.layers.0.self_attn.k_proj.weight"));
        assert!(!mapping.is_lora_target("model.embed_tokens.weight"));

        let (a, b) = mapping
            .lora_pair("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(
            a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        );
        assert_eq!(
            b,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        );
    }

    #[test]
    fn missing_target_module_errors() {
        let base_names = vec!["model.layers.0.self_attn.q_proj.weight"];
        let adapter_names = vec![
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        ];
        let target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];

        let err =
            build_name_mapping(&base_names, &adapter_names, &target_modules, None).unwrap_err();
        assert!(err.to_string().contains("v_proj"));
    }

    #[test]
    fn module_replacement_mapping() {
        let base_names = vec!["model.layers.0.self_attn.q_proj.weight", "lm_head.weight"];
        let adapter_names = vec![
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.lm_head.weight",
        ];
        let target_modules = vec!["q_proj".to_string()];
        let modules_to_save = vec!["lm_head".to_string()];

        let mapping = build_name_mapping(
            &base_names,
            &adapter_names,
            &target_modules,
            Some(&modules_to_save),
        )
        .unwrap();

        assert_eq!(
            mapping.replacement("lm_head.weight"),
            Some("base_model.model.lm_head.weight")
        );
    }
}
