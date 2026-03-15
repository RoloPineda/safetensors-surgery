# Benchmarks

Compare safetensors-surgery against PEFT `merge_and_unload`, PEFT low-memory mode, and mergekit.

Measures three things per tool: peak memory (RSS), wall-clock time, and merge accuracy against a Python f64 reference.

## Quick start

Requires: Linux (uses `/usr/bin/time -v` for RSS measurement), [uv](https://docs.astral.sh/uv/), Rust toolchain.

```bash
# 1. Download the smallest test model (~700MB)
uv run benchmarks/download_data.py --models opt-350m

# 2. Run the benchmark
uv run benchmarks/compare.py --models opt-350m --runs 1
```

That's it. uv reads dependency metadata from the scripts and handles Python version and packages automatically. Results print to stdout and save to `benchmarks/results.html` (interactive chart), `benchmarks/results.png`, and `benchmarks/results.json`.

## Full benchmark

Download all four models and run with 3 iterations per tool:

```bash
uv run benchmarks/download_data.py --models all
uv run benchmarks/compare.py --models all --runs 3
```

## Picking which models to benchmark

Not everyone has 140GB of free disk. Pick what fits:

| Model | Download | Base dtype | Notes |
|---|---|---|---|
| `opt-350m` | ~700MB | fp16 | Fast, good for sanity checks |
| `tinyllama-1.1b` | ~2.5GB | bf16 | Tests bf16 accuracy path |
| `mistral-7b` | ~15GB | bf16 | Single-file, 7B scale |
| `llama3-70b` | ~124GB | bf16 | Sharded, 70B scale, requires significant disk |

```bash
# Download only what you need
uv run benchmarks/download_data.py --models opt-350m tinyllama-1.1b

# See what's already downloaded
uv run benchmarks/download_data.py --list
```

## Options

```
benchmarks/compare.py options:
  --models MODEL [MODEL ...]   Which models to benchmark
  --tools TOOL [TOOL ...]      Which tools (surgery, surgery_low_mem, peft, peft_low_mem, mergekit)
  --runs N                     Iterations per tool (default: 3, reports median)
  --output PATH                Output chart path (default: benchmarks/results.png)
  --drop-caches                Drop OS page cache before each run (requires passwordless sudo)
```

Run only specific tools:

```bash
# Compare default and low-memory surgery modes
uv run benchmarks/compare.py --models mistral-7b --tools surgery surgery_low_mem --runs 3

# Compare surgery against PEFT only
uv run benchmarks/compare.py --models opt-350m --tools surgery peft --runs 1
```

## Without uv

If you prefer pip:

```bash
pip install plotly kaleido torch peft transformers huggingface-hub mergekit safetensors

python3 benchmarks/download_data.py --models opt-350m
python3 benchmarks/compare.py --models opt-350m --runs 1
```

## What it measures

**Peak RSS (MB):** Maximum resident set size of the process. Each tool runs in an isolated subprocess so measurements don't contaminate each other.

**Time (s):** Wall-clock time including both merge and save-to-disk. Median of N runs. Includes process startup for all tools (Rust binary startup for surgery, Python interpreter startup for PEFT/mergekit).

**Max ULP:** Maximum ULP (unit in the last place) distance of any element in any LoRA target tensor, compared to a Python f64 reference merge computed on the fly. Lower is better. A max ULP of 0 means bit-identical to the reference after downcast; 1 means the adjacent representable value.

**Mean ULP:** Average ULP distance across all elements of all LoRA target tensors. Reported to 6 decimal places to distinguish near-perfect tools.

**Within 1 ULP (%):** Percentage of all LoRA-merged elements that are within 1 ULP of the f64 reference. 100% means every element is either exact or off by the smallest representable step.

**Passthrough:** Percentage of non-LoRA tensors that are bit-identical to the original base model. Should always be 100% for all tools.

## Accuracy methodology

For each LoRA target tensor, the harness computes a Python f64 reference: `base_f64 + scaling * (B_f64 @ A_f64)`, where all operands are upcast to float64 before computation. This reference is then downcast to the output dtype (bf16/f16/f32) for comparison. ULP distance is computed using the IEEE 754 sign-magnitude integer trick, which maps floating-point values to integers such that adjacent representable floats differ by 1.

The reference is computed per-tensor in 512-row chunks to avoid materializing the full model in memory simultaneously.

## Sharing results

If you run benchmarks, I'd appreciate if you share your results. Include:

- CPU model (e.g., `lscpu | grep "Model name"`)
- RAM amount
- Storage type (NVMe, SATA SSD, HDD)
- Which models you tested
- The full terminal output or the `benchmarks/results.json` file

Open an issue or discussion with the results. Benchmark diversity across different hardware helps everyone understand real-world performance.