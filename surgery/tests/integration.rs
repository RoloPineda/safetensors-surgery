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
            (*name, TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap())
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
