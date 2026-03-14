# Benchmarks

Compare safetensors-surgery against PEFT `merge_and_unload`, PEFT low-memory mode, and mergekit.

Measures three things per tool: peak memory (RSS), wall-clock time, and merge accuracy against an f64 gold standard.

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

Download all three models (~18GB total) and run with 3 iterations per tool:

```bash
uv run benchmarks/download_data.py --models all
uv run benchmarks/compare.py --models opt-350m tinyllama-1.1b mistral-7b --runs 3
```

## Picking which models to benchmark

Not everyone has 18GB of free disk. Pick what fits:

| Model | Download | Base dtype | Notes |
|---|---|---|---|
| `opt-350m` | ~700MB | fp16 | Fast, good for sanity checks |
| `tinyllama-1.1b` | ~2.5GB | bf16 | Tests bf16 accuracy path |
| `mistral-7b` | ~15GB | bf16 | Single-file, 7B scale |

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
  --tools TOOL [TOOL ...]      Which tools (surgery, peft, peft_low_mem, mergekit)
  --runs N                     Iterations per tool (default: 3, reports median)
  --output PATH                Output chart path (default: benchmarks/results.png)
```

Run only specific tools:

```bash
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

**LoRA max_err:** Maximum absolute error of any single element in any LoRA target tensor, compared to the f64 gold standard (merge computed in f64, downcast once to the base model's original dtype).

**LoRA RMS:** Mean RMS error across all LoRA target tensors. Lower is better.

**Passthrough:** Percentage of non-LoRA tensors that are bit-identical to the original base model. Surgery achieves 100% because it copies mmap byte ranges directly. PEFT scores lower because it loads all tensors into PyTorch and re-saves them, introducing fp16/bf16 roundtrip noise even on weights it never modified.

## Sharing results

If you run benchmarks, we'd appreciate if you share your results. Include:

- CPU model (e.g., `lscpu | grep "Model name"`)
- RAM amount
- Storage type (NVMe, SATA SSD, HDD)
- Which models you tested
- The full terminal output or the `benchmarks/results.json` file

Open an issue or discussion with the results. Benchmark diversity across different hardware helps everyone understand real-world performance.