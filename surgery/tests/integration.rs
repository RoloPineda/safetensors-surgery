use safetensors::tensor::{serialize, SafeTensors, TensorView};
use safetensors::Dtype;
use std::fs;
use std::path::Path;

/// Writes a safetensors file from f32 values.
fn write_f32_safetensors(path: &Path, tensors: &[(&str, &[f32], Vec<usize>)]) {
    let byte_vecs: Vec<(String, Vec<u8>)> = tensors
        .iter()
        .map(|(name, values, _)| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (name.to_string(), bytes)
        })
        .collect();

    let views: Vec<(&str, TensorView)> = tensors
        .iter()
        .zip(byte_vecs.iter())
        .map(|((name, _, shape), (_, bytes))| {
            (
                *name,
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();

    let serialized = serialize(views, &None).unwrap();
    fs::write(path, serialized).unwrap();
}

/// Reads a tensor from a safetensors file and returns its f32 values.
fn read_f32_tensor(path: &Path, name: &str) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    let tensors = SafeTensors::deserialize(&data).unwrap();
    let view = tensors.tensor(name).unwrap();
    assert_eq!(view.dtype(), Dtype::F32);
    view.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn merge_two_4x4_lora_targets() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    // Base model: two 4×4 identity matrices
    #[rustfmt::skip]
    let q_base: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let v_base: [f32; 16] = [
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
        0.0, 0.0, 0.0, 2.0,
    ];

    write_f32_safetensors(
        &base_path,
        &[
            (
                "model.layers.0.self_attn.q_proj.weight",
                &q_base,
                vec![4, 4],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                &v_base,
                vec![4, 4],
            ),
        ],
    );

    // ── Adapter: rank=2, alpha=4, scaling = alpha/r = 2.0 ──
    //
    // q_proj lora_A [2, 4]:  [[1, 0, 0, 0],
    //                         [0, 1, 0, 0]]
    // q_proj lora_B [4, 2]:  [[1, 0],
    //                         [0, 1],
    //                         [0, 0],
    //                         [0, 0]]
    //
    // B @ A = [[1, 0, 0, 0],
    //          [0, 1, 0, 0],
    //          [0, 0, 0, 0],
    //          [0, 0, 0, 0]]
    //
    // q_merged = q_base + 2.0 * (B @ A)
    //          = [[3, 0, 0, 0],
    //             [0, 3, 0, 0],
    //             [0, 0, 1, 0],
    //             [0, 0, 0, 1]]

    #[rustfmt::skip]
    let q_lora_a: [f32; 8] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ];
    #[rustfmt::skip]
    let q_lora_b: [f32; 8] = [
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
    ];

    // v_proj lora_A [2, 4]:  [[0, 0, 1, 0],
    //                         [0, 0, 0, 1]]
    // v_proj lora_B [4, 2]:  [[0, 0],
    //                         [0, 0],
    //                         [1, 0],
    //                         [0, 1]]
    //
    // B @ A = [[0, 0, 0, 0],
    //          [0, 0, 0, 0],
    //          [0, 0, 1, 0],
    //          [0, 0, 0, 1]]
    //
    // v_merged = v_base + 2.0 * (B @ A)
    //          = [[2, 0, 0, 0],
    //             [0, 2, 0, 0],
    //             [0, 0, 4, 0],
    //             [0, 0, 0, 4]]

    #[rustfmt::skip]
    let v_lora_a: [f32; 8] = [
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let v_lora_b: [f32; 8] = [
        0.0, 0.0,
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    ];

    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &q_lora_a,
                vec![2, 4],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &q_lora_b,
                vec![4, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
                &v_lora_a,
                vec![2, 4],
            ),
            (
                "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
                &v_lora_b,
                vec![4, 2],
            ),
        ],
    );

    let config = r#"{
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["q_proj", "v_proj"],
        "fan_in_fan_out": false,
        "bias": "none",
        "peft_type": "LORA"
    }"#;
    fs::write(adapter_dir.join("adapter_config.json"), config).unwrap();

    let stats = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None).unwrap();

    assert_eq!(stats.tensors_merged, 2);
    assert_eq!(stats.tensors_copied, 0);
    assert_eq!(stats.tensors_replaced, 0);
    assert_eq!(stats.biases_merged, 0);

    let output_data = fs::read(&output_path).unwrap();
    let output_tensors = SafeTensors::deserialize(&output_data).unwrap();
    assert_eq!(output_tensors.len(), 2);

    let q_merged = read_f32_tensor(&output_path, "model.layers.0.self_attn.q_proj.weight");
    #[rustfmt::skip]
    let q_expected: [f32; 16] = [
        3.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for (i, (got, want)) in q_merged.iter().zip(q_expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "q_proj[{i}]: got {got}, expected {want}"
        );
    }

    let v_merged = read_f32_tensor(&output_path, "model.layers.0.self_attn.v_proj.weight");
    #[rustfmt::skip]
    let v_expected: [f32; 16] = [
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 4.0,
    ];
    for (i, (got, want)) in v_merged.iter().zip(v_expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "v_proj[{i}]: got {got}, expected {want}"
        );
    }

    let q_view = output_tensors
        .tensor("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q_view.shape(), &[4, 4]);
    assert_eq!(q_view.dtype(), Dtype::F32);

    let v_view = output_tensors
        .tensor("model.layers.0.self_attn.v_proj.weight")
        .unwrap();
    assert_eq!(v_view.shape(), &[4, 4]);
    assert_eq!(v_view.dtype(), Dtype::F32);
}

/// Writes a safetensors file with an explicit dtype per tensor.
fn write_typed_safetensors(path: &Path, tensors: &[(&str, &[f32], Vec<usize>, Dtype)]) {
    let byte_vecs: Vec<(String, Vec<u8>)> = tensors
        .iter()
        .map(|(name, values, _, dtype)| {
            let bytes: Vec<u8> = match dtype {
                Dtype::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                Dtype::F16 => values
                    .iter()
                    .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                    .collect(),
                Dtype::BF16 => values
                    .iter()
                    .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                    .collect(),
                _ => panic!("unsupported dtype in test helper"),
            };
            (name.to_string(), bytes)
        })
        .collect();

    let views: Vec<(&str, TensorView)> = tensors
        .iter()
        .zip(byte_vecs.iter())
        .map(|((name, _, shape, dtype), (_, bytes))| {
            (
                *name,
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();

    let serialized = serialize(views, &None).unwrap();
    fs::write(path, serialized).unwrap();
}

/// Writes an adapter_config.json to the given directory.
fn write_adapter_config(dir: &Path, config_json: &str) {
    fs::write(dir.join("adapter_config.json"), config_json).unwrap();
}

/// Reads a tensor's raw bytes and dtype from a safetensors file.
fn read_tensor_info(path: &Path, name: &str) -> (Vec<u8>, Dtype, Vec<usize>) {
    let data = fs::read(path).unwrap();
    let tensors = SafeTensors::deserialize(&data).unwrap();
    let view = tensors.tensor(name).unwrap();
    let dtype = view.dtype();
    let shape = view.shape().to_vec();
    let raw = view.data().to_vec();
    (raw, dtype, shape)
}

/// Reads a tensor from a safetensors file, converting any supported dtype to f32 values.
fn read_tensor_as_f32(path: &Path, name: &str) -> Vec<f32> {
    let (raw, dtype, _shape) = read_tensor_info(path, name);
    match dtype {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => panic!("unsupported dtype {:?}", dtype),
    }
}

#[test]
fn merge_with_passthrough_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    #[rustfmt::skip]
    let q_base: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let embed: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

    write_f32_safetensors(
        &base_path,
        &[
            (
                "model.layers.0.self_attn.q_proj.weight",
                &q_base,
                vec![4, 4],
            ),
            ("model.embed_tokens.weight", &embed, vec![2, 3]),
        ],
    );

    // Adapter only targets q_proj: rank=1, alpha=1, scaling=1.0
    // lora_A [1, 4]: [[1, 0, 0, 0]]
    // lora_B [4, 1]: [[1], [0], [0], [0]]
    // B @ A = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    // merged = identity + 1.0 * delta
    let lora_a: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    let lora_b: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 4],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![4, 1],
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let stats = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None).unwrap();

    assert_eq!(stats.tensors_copied, 1);
    assert_eq!(stats.tensors_merged, 1);

    // Verify embed_tokens is byte-identical
    let base_data = fs::read(&base_path).unwrap();
    let base_tensors = SafeTensors::deserialize(&base_data).unwrap();
    let base_embed = base_tensors.tensor("model.embed_tokens.weight").unwrap();

    let output_data = fs::read(&output_path).unwrap();
    let output_tensors = SafeTensors::deserialize(&output_data).unwrap();
    let output_embed = output_tensors.tensor("model.embed_tokens.weight").unwrap();

    assert_eq!(base_embed.data(), output_embed.data());
}

#[test]
fn merge_with_bias_lora_only() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    // Base: 2x2 identity weight + 1D bias [1.0, 2.0, 3.0, 4.0]
    #[rustfmt::skip]
    let q_base: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let bias_base: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    // Write base with weight (2x2) and bias (1D shape [4])
    write_typed_safetensors(
        &base_path,
        &[
            (
                "model.layers.0.self_attn.q_proj.weight",
                &q_base,
                vec![2, 2],
                Dtype::F32,
            ),
            (
                "model.layers.0.self_attn.q_proj.bias",
                &bias_base,
                vec![1, 4],
                Dtype::F32,
            ),
        ],
    );

    // Adapter: lora_A/B for q_proj + bias
    let lora_a: [f32; 2] = [1.0, 0.0];
    let lora_b: [f32; 2] = [1.0, 0.0];
    let adapter_bias: [f32; 4] = [0.5, 0.5, 0.5, 0.5];

    write_typed_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
                Dtype::F32,
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
                Dtype::F32,
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.bias",
                &adapter_bias,
                vec![1, 4],
                Dtype::F32,
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "fan_in_fan_out": false,
            "bias": "lora_only",
            "peft_type": "LORA"
        }"#,
    );

    let stats = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None).unwrap();

    assert_eq!(stats.tensors_merged, 1);
    assert_eq!(stats.biases_merged, 1);

    // Verify bias is summed: base [1,2,3,4] + adapter [0.5,0.5,0.5,0.5] = [1.5,2.5,3.5,4.5]
    let bias_values = read_tensor_as_f32(&output_path, "model.layers.0.self_attn.q_proj.bias");
    let expected = [1.5, 2.5, 3.5, 4.5];
    for (i, (got, want)) in bias_values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "bias[{i}]: got {got}, expected {want}"
        );
    }
}

#[test]
fn merge_with_modules_to_save() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    #[rustfmt::skip]
    let q_base: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let lm_head_base: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    write_f32_safetensors(
        &base_path,
        &[
            (
                "model.layers.0.self_attn.q_proj.weight",
                &q_base,
                vec![2, 2],
            ),
            ("lm_head.weight", &lm_head_base, vec![2, 2]),
        ],
    );

    // Adapter: lora for q_proj + full replacement for lm_head
    let lora_a: [f32; 2] = [1.0, 0.0];
    let lora_b: [f32; 2] = [1.0, 0.0];
    let lm_head_replacement: [f32; 4] = [9.0, 9.0, 9.0, 9.0];

    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
            ),
            (
                "base_model.model.lm_head.weight",
                &lm_head_replacement,
                vec![2, 2],
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "modules_to_save": ["lm_head"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let stats = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None).unwrap();

    assert_eq!(stats.tensors_merged, 1);
    assert_eq!(stats.tensors_replaced, 1);

    let lm_head_values = read_f32_tensor(&output_path, "lm_head.weight");
    for (i, val) in lm_head_values.iter().enumerate() {
        assert!(
            (val - 9.0).abs() < 1e-6,
            "lm_head[{i}]: got {val}, expected 9.0"
        );
    }
}

#[test]
fn merge_bf16_preserves_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    // Base: 2x2 identity in BF16
    #[rustfmt::skip]
    let q_base: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

    write_typed_safetensors(
        &base_path,
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            &q_base,
            vec![2, 2],
            Dtype::BF16,
        )],
    );

    // Adapter in BF16: rank=1, alpha=1, scaling=1.0
    // lora_A [1, 2]: [[1, 1]]
    // lora_B [2, 1]: [[1], [1]]
    // B @ A = [[1,1],[1,1]]
    // merged = [[1,0],[0,1]] + 1.0 * [[1,1],[1,1]] = [[2,1],[1,2]]
    let lora_a: [f32; 2] = [1.0, 1.0];
    let lora_b: [f32; 2] = [1.0, 1.0];

    write_typed_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
                Dtype::BF16,
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
                Dtype::BF16,
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let stats = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None).unwrap();
    assert_eq!(stats.tensors_merged, 1);

    // Verify output dtype is BF16
    let (_, dtype, _) = read_tensor_info(&output_path, "model.layers.0.self_attn.q_proj.weight");
    assert_eq!(dtype, Dtype::BF16);

    // Verify values are approximately correct
    let values = read_tensor_as_f32(&output_path, "model.layers.0.self_attn.q_proj.weight");
    let expected = [2.0, 1.0, 1.0, 2.0];
    for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.1,
            "bf16 merged[{i}]: got {got}, expected {want}"
        );
    }
}

#[test]
fn merge_f16_preserves_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    #[rustfmt::skip]
    let q_base: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

    write_typed_safetensors(
        &base_path,
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            &q_base,
            vec![2, 2],
            Dtype::F16,
        )],
    );

    let lora_a: [f32; 2] = [1.0, 1.0];
    let lora_b: [f32; 2] = [1.0, 1.0];

    write_typed_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
                Dtype::F16,
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
                Dtype::F16,
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let stats = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None).unwrap();
    assert_eq!(stats.tensors_merged, 1);

    let (_, dtype, _) = read_tensor_info(&output_path, "model.layers.0.self_attn.q_proj.weight");
    assert_eq!(dtype, Dtype::F16);

    let values = read_tensor_as_f32(&output_path, "model.layers.0.self_attn.q_proj.weight");
    let expected = [2.0, 1.0, 1.0, 2.0];
    for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.01,
            "f16 merged[{i}]: got {got}, expected {want}"
        );
    }
}

#[test]
fn merge_with_fan_in_fan_out() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    // Base: 2x2 identity
    #[rustfmt::skip]
    let q_base: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

    write_f32_safetensors(
        &base_path,
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            &q_base,
            vec![2, 2],
        )],
    );

    // With fan_in_fan_out=true:
    // lora_A shape [2, 1]: [[1], [2]]  -> A^T = [[1, 2]]
    // lora_B shape [2, 1]: [[3], [4]]
    // B @ A^T = [[3],[4]] @ [[1,2]] = [[3,6],[4,8]]
    // scaling = alpha/r = 1/1 = 1.0
    // merged = [[1,0],[0,1]] + 1.0 * [[3,6],[4,8]] = [[4,6],[4,9]]
    let lora_a: [f32; 2] = [1.0, 2.0];
    let lora_b: [f32; 2] = [3.0, 4.0];

    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![2, 1],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "fan_in_fan_out": true,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let stats = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None).unwrap();
    assert_eq!(stats.tensors_merged, 1);

    let values = read_f32_tensor(&output_path, "model.layers.0.self_attn.q_proj.weight");
    #[rustfmt::skip]
    let expected: [f32; 4] = [4.0, 6.0, 4.0, 9.0];
    for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "fan_in_fan_out[{i}]: got {got}, expected {want}"
        );
    }
}

#[test]
fn dry_run_reports_correct_counts() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    fs::create_dir_all(&adapter_dir).unwrap();

    // Base: 3 tensors (2 LoRA targets + 1 passthrough)
    #[rustfmt::skip]
    let identity: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let embed: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

    write_f32_safetensors(
        &base_path,
        &[
            (
                "model.layers.0.self_attn.q_proj.weight",
                &identity,
                vec![2, 2],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                &identity,
                vec![2, 2],
            ),
            ("model.embed_tokens.weight", &embed, vec![2, 3]),
        ],
    );

    // Adapter with lora for both q_proj and v_proj
    let lora_a: [f32; 2] = [1.0, 0.0];
    let lora_b: [f32; 2] = [1.0, 0.0];

    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
            ),
            (
                "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj", "v_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let info = surgery::dry_run_info(&base_path, &adapter_dir).unwrap();

    assert_eq!(info.base_tensor_count, 3);
    assert_eq!(info.lora_target_count, 2);
    assert_eq!(info.passthrough_count, 1);
    assert_eq!(info.replacement_count, 0);
    assert_eq!(info.bias_merge_count, 0);
    assert!(!info.is_sharded);
    assert_eq!(info.shard_count, 1);
}

#[test]
fn merge_with_progress_callback() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    // Base: 2 tensors
    #[rustfmt::skip]
    let identity: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let embed: [f32; 4] = [0.1, 0.2, 0.3, 0.4];

    write_f32_safetensors(
        &base_path,
        &[
            (
                "model.layers.0.self_attn.q_proj.weight",
                &identity,
                vec![2, 2],
            ),
            ("model.embed_tokens.weight", &embed, vec![2, 2]),
        ],
    );

    let lora_a: [f32; 2] = [1.0, 0.0];
    let lora_b: [f32; 2] = [1.0, 0.0];

    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let call_count = std::sync::atomic::AtomicUsize::new(0);
    let progress = |_current: usize, _total: usize| {
        call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    };

    surgery::merge_adapter(&base_path, &adapter_dir, &output_path, Some(&progress)).unwrap();

    // Progress fires once per tensor (2) + once for the final call = 3
    let count = call_count.load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(count, 3, "expected 3 progress callbacks, got {count}");
}

#[test]
fn merge_sharded_model() {
    let dir = tempfile::tempdir().unwrap();
    let base_dir = dir.path().join("base");
    let adapter_dir = dir.path().join("adapter");
    let output_dir = dir.path().join("output");
    fs::create_dir_all(&base_dir).unwrap();
    fs::create_dir_all(&adapter_dir).unwrap();

    // Shard 1: q_proj
    #[rustfmt::skip]
    let q_base: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    write_f32_safetensors(
        &base_dir.join("model-00001-of-00002.safetensors"),
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            &q_base,
            vec![2, 2],
        )],
    );

    // Shard 2: embed_tokens (passthrough)
    let embed: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    write_f32_safetensors(
        &base_dir.join("model-00002-of-00002.safetensors"),
        &[("model.embed_tokens.weight", &embed, vec![2, 3])],
    );

    // Write shard index
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

    // Adapter targets q_proj
    let lora_a: [f32; 2] = [1.0, 1.0];
    let lora_b: [f32; 2] = [1.0, 1.0];

    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let stats = surgery::merge_adapter(&base_dir, &adapter_dir, &output_dir, None).unwrap();

    assert_eq!(stats.tensors_merged, 1);
    assert_eq!(stats.tensors_copied, 1);

    // Verify output has both shards and index
    assert!(output_dir.join("model-00001-of-00002.safetensors").exists());
    assert!(output_dir.join("model-00002-of-00002.safetensors").exists());
    assert!(output_dir.join("model.safetensors.index.json").exists());

    // Verify shard 1 merged correctly
    let merged_values = read_f32_tensor(
        &output_dir.join("model-00001-of-00002.safetensors"),
        "model.layers.0.self_attn.q_proj.weight",
    );
    // B @ A = [[1],[1]] @ [[1,1]] = [[1,1],[1,1]]
    // merged = identity + 1.0 * [[1,1],[1,1]] = [[2,1],[1,2]]
    let expected = [2.0, 1.0, 1.0, 2.0];
    for (i, (got, want)) in merged_values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "sharded q_proj[{i}]: got {got}, expected {want}"
        );
    }

    // Verify shard 2 passthrough is byte-identical
    let base_data = fs::read(base_dir.join("model-00002-of-00002.safetensors")).unwrap();
    let base_tensors = SafeTensors::deserialize(&base_data).unwrap();
    let base_embed = base_tensors.tensor("model.embed_tokens.weight").unwrap();

    let output_data = fs::read(output_dir.join("model-00002-of-00002.safetensors")).unwrap();
    let output_tensors = SafeTensors::deserialize(&output_data).unwrap();
    let output_embed = output_tensors.tensor("model.embed_tokens.weight").unwrap();

    assert_eq!(base_embed.data(), output_embed.data());
}

#[test]
fn merge_errors_on_missing_adapter_config() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    #[rustfmt::skip]
    let q_base: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    write_f32_safetensors(
        &base_path,
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            &q_base,
            vec![2, 2],
        )],
    );

    // Write adapter weights but no config
    let lora_a: [f32; 2] = [1.0, 0.0];
    let lora_b: [f32; 2] = [1.0, 0.0];
    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
            ),
        ],
    );

    let result = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None);
    assert!(result.is_err());
}

#[test]
fn merge_errors_when_no_targets_match() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    let output_path = dir.path().join("merged.safetensors");
    fs::create_dir_all(&adapter_dir).unwrap();

    #[rustfmt::skip]
    let q_base: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    write_f32_safetensors(
        &base_path,
        &[(
            "model.layers.0.self_attn.q_proj.weight",
            &q_base,
            vec![2, 2],
        )],
    );

    // Adapter has lora for q_proj but config targets "nonexistent_module"
    let lora_a: [f32; 2] = [1.0, 0.0];
    let lora_b: [f32; 2] = [1.0, 0.0];
    write_f32_safetensors(
        &adapter_dir.join("adapter_model.safetensors"),
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &lora_a,
                vec![1, 2],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                &lora_b,
                vec![2, 1],
            ),
        ],
    );

    write_adapter_config(
        &adapter_dir,
        r#"{
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["nonexistent_module"],
            "fan_in_fan_out": false,
            "bias": "none",
            "peft_type": "LORA"
        }"#,
    );

    let result = surgery::merge_adapter(&base_path, &adapter_dir, &output_path, None);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("nonexistent_module"),
        "error should mention the module name, got: {err_msg}"
    );
}
