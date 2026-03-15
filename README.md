# safetensors-surgery

[![CI](https://github.com/RoloPineda/safetensors-surgery/actions/workflows/ci.yml/badge.svg)](https://github.com/RoloPineda/safetensors-surgery/actions/workflows/ci.yml)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE)


`safetensors-surgery` merges PEFT LoRA adapters into safetensors base models using memory-mapped I/O.
## Why this tool

Merging a LoRA adapter into a base model with PEFT's `merge_and_unload` requires loading the entire model into memory. For Mistral-7B that's 28GB of RAM. For Llama3-70B it's well over 100GB. mergekit needs less (13GB for Llama3-70B) but still scales linearly with model size. If you train on a rented GPU and want to merge on your laptop, you're stuck. You either keep the GPU running, merge in the cloud, or serve with the adapter applied at runtime (slower inference).

`safetensors-surgery` solves this by memory-mapping the base model and processing one tensor at a time. Llama3-70B merges in 6.4GB with `--low-memory`.

### Tradeoffs

**Advantages:** 2x less memory than mergekit on 70B models, f64-accurate merge results (100% of elements within 1 ULP of the f64 reference), 100% bit-identical passthrough on non-LoRA tensors, no Python or PyTorch dependency at runtime.

**Downsides:** On very large models (70B+), surgery is slower on wall-clock time than mergekit due to f64 matmul overhead on large projection matrices.

## Contents

- [Why This Tool](#why-this-tool)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [Supported Formats](#supported-formats)
- [Adapter Config Options](#adapter-config-options)
- [Current Known Limitations](#current-known-limitations)
- [Benchmarks](#benchmarks)

## Installation

### CLI (from source)

```sh
git clone https://github.com/RoloPineda/safetensors-surgery.git
cd safetensors-surgery
cargo install --path cli
```

## Usage

### CLI

```sh
# Single-file base model
safetensors-surgery merge \
  --base-model ./Mistral-7B-v0.1 \
  --adapter ./my-lora-adapter \
  --output ./merged-model.safetensors

# Sharded base model (output is a directory with shards + index.json)
safetensors-surgery merge \
  --base-model ./Qwen2.5-14B \
  --adapter ./my-lora-adapter \
  --output ./merged-model

# Low-memory mode (tiled merge, uses less RAM at the cost of speed)
safetensors-surgery merge \
  --base-model ./Llama-3-70B \
  --adapter ./my-lora-adapter \
  --output ./merged-model \
  --low-memory

# Preview what will happen without writing anything
safetensors-surgery merge \
  --base-model ./Mistral-7B-v0.1 \
  --adapter ./my-lora-adapter \
  --output ./merged-model.safetensors \
  --dry-run
```


## How It Works

The base model is memory-mapped, allowing the OS to stream weights from disk rather than loading the entire model into memory. For each LoRA target tensor, the base weight, lora_A, and lora_B are read, the merge `base + (alpha / r) * (B @ A)` is computed with f64 matmul accumulation, and the final base+delta addition is performed in f64 before downcasting to the output dtype. This preserves numerical accuracy that would be lost with f32 intermediate computation. Non-LoRA tensors are byte-copied from the mmap to the output without allocation or conversion. For sharded models, each shard is opened and closed independently.

Two merge paths are available. The default path materializes the full delta matrix in f64 for maximum speed. The `--low-memory` path tiles the matmul in fixed-size chunks, bounding peak memory at the cost of slower speed due to per-tile allocation overhead.

## Performance

![Peak memory usage](docs/benchmark_memory.png)

![Merge time](docs/benchmark_time.png)

Note: PEFT caused OOM error during the Llama3-70B run and hence the numbers show as 0 for memory and time. 

Benchmarked on an AMD Ryzen 9 9950X3D, 128GB DDR5, 1TB NVMe. Median of 3 runs. Full methodology and reproduction instructions in [`benchmarks/`](benchmarks/).


| Model | Tool | Peak RSS | Time | Accuracy | Passthrough |
|:---|:---|---:|---:|:---:|:---:|
| **OPT-350M** (fp16) | surgery | **660 MB** | **0.40s** | Equivalent | 100% |
| | PEFT | 2,098 MB | 2.60s | Equivalent | 100% |
| | mergekit | 1,997 MB | 2.57s | Equivalent | 100% |
| **TinyLlama-1.1B** (bf16) | surgery | **2,193 MB** | **1.10s** | Equivalent | 100% |
| | PEFT | 5,029 MB | 3.10s | Equivalent | 100% |
| | mergekit | 7,022 MB | 3.00s | Equivalent | 100% |
| **Mistral-7B** (bf16) | surgery | **9,856 MB** | 12.97s | Equivalent | 100% |
| | PEFT | 28,493 MB | 11.98s | Equivalent | 100% |
| | mergekit | 14,668 MB | **8.56s** | Equivalent | 100% |
| **Llama3-70B** (bf16) | surgery | 11,779 MB | 365.77s | Equivalent | 100% |
| | surgery --low-memory | **6,403 MB** | 640.36s | Equivalent | 100% |
| | mergekit | 13,083 MB | **180.57s** | Divergent | 100% |

**Peak RSS:** Maximum resident set size. Lower is better.

**Accuracy:** Measured using per-element ULP (unit in the last place) distance against a Python f64 reference merge. "Equivalent" means 100% of LoRA-merged elements are within 1 ULP of the f64 reference. "Divergent" means some elements exceed the output dtype's representable precision boundary.

**Passthrough:** Percentage of non-LoRA tensors bit-identical to the original base model.

## Supported Formats

**Base models:** Single-file or sharded safetensors in fp16, bf16, or fp32. Must be unquantized.

**Adapters:** PEFT-format LoRA (`adapter_config.json` + `adapter_model.safetensors`). This covers adapters trained with PEFT, Axolotl, Unsloth, and LLaMA-Factory, which all save through PEFT's serialization.

**Output:** Safetensors matching the input layout (single-file or sharded). The original dtype and `__metadata__` header are preserved.

## Adapter Config Options

These fields are read from `adapter_config.json`:

| Field | Required | Notes |
|:---|:---:|:---|
| `peft_type` | Yes | Must be `"LORA"` |
| `r` | Yes | LoRA rank |
| `lora_alpha` | Yes | Scaling factor (merge uses `alpha / r`) |
| `target_modules` | Yes | Which layers have LoRA weights |
| `fan_in_fan_out` | No | Transpose for Conv1D layers (GPT-2 style models) |
| `bias` | No | `"none"` (default), `"lora_only"`, or `"all"` |
| `modules_to_save` | No | Full modules replaced entirely, not low-rank merged |

## Current Limitations

**Quantized base models not supported.** The base model must be unquantized fp16, bf16, or fp32 safetensors. GPTQ, AWQ, and bitsandbytes 4-bit models cannot be used.

**Speed on very large models.** Surgery is faster or competitive through 7B. At 70B scale, f64 matmul on large projection matrices makes surgery slower than mergekit on wall-clock time. The --low-memory flag trades more time for less memory, which is useful when RAM is the constraint rather than speed.

**No output dtype conversion.** The output always matches the base model's dtype. Converting a bf16 merge to fp16 requires a separate tool.

**No architecture validation.** Surgery matches tensors by name. Pointing it at the wrong base model produces a valid but meaningless safetensors file without warning.

**MLX adapters not supported.** Apple's MLX framework uses a different LoRA format (`lora_a`/`lora_b` naming, transposed shapes). Convert to PEFT format first.

**Single adapter only.** Multi-adapter merge methods (TIES, DARE, SLERP) are not supported. One LoRA at a time.

## Benchmarks

See [`benchmarks/`](benchmarks/) for reproduction instructions. Download test models and run the comparison harness with:

```sh
uv run benchmarks/download_data.py --models all
uv run benchmarks/compare.py --models opt-350m tinyllama-1.1b mistral-7b --runs 3
```

To benchmark the low-memory mode alongside the default:

```sh
uv run benchmarks/compare.py --models mistral-7b --tools surgery surgery_low_mem mergekit --runs 3
```

Hardware varies. If you run benchmarks on your machine, consider sharing the results (CPU model, RAM, storage type, and the `benchmarks/results.json` file) so the community can see performance across different setups.

## License

Mozilla Public License 2.0 (MPL-2.0). See [`LICENSE`](LICENSE).
