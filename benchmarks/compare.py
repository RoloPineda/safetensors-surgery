"""Compare safetensors-surgery against PEFT, PEFT low-memory, and mergekit.

Each tool measurement runs in an isolated subprocess so peak RSS readings
are independent. Timing includes both merge and save for every tool to
keep the comparison fair. Multiple runs are taken and the median is reported.
"""

# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "plotly",
#     "kaleido",
#     "torch",
#     "peft",
#     "transformers",
#     "huggingface-hub",
#     "mergekit",
#     "safetensors",
# ]
# ///

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from statistics import median

import plotly.graph_objects as go


PRESET_MODELS = {
    "opt-350m": {
        "base": "benchmarks/data/opt-350m/base",
        "adapter": "benchmarks/data/opt-350m/adapter",
        "label": "OPT-350M (662MB)",
    },
    "tinyllama-1.1b": {
        "base": "benchmarks/data/tinyllama/base",
        "adapter": "benchmarks/data/tinyllama/adapter",
        "label": "TinyLlama-1.1B (2.2GB)",
    },
    "mistral-7b": {
        "base": "benchmarks/data/mistral-7b/base",
        "adapter": "benchmarks/data/mistral-7b/adapter",
        "label": "Mistral-7B (14GB)",
    },
}

TOOLS = ["surgery", "peft", "peft_low_mem", "mergekit"]


def find_project_root():
    """Walks up from this file to find the directory containing Cargo.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "Cargo.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no Cargo.toml found)")


def find_cli_binary():
    """Returns the path to the release CLI binary."""
    root = find_project_root()
    cli_path = root / "target" / "release" / "cli"
    if not cli_path.exists():
        raise FileNotFoundError(
            f"Release binary not found at {cli_path}. "
            "Run 'cargo build --release' first."
        )
    return str(cli_path)


def find_safetensors_path(directory):
    """Finds the safetensors file or directory for sharded models.

    Args:
        directory: Path to a model directory.

    Returns:
        Path to the safetensors file, or the directory itself for sharded models.
    """
    directory = str(directory)
    index_path = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return directory

    single = os.path.join(directory, "model.safetensors")
    if os.path.exists(single):
        return single

    for f in os.listdir(directory):
        if f.endswith(".safetensors") and "adapter" not in f:
            return os.path.join(directory, f)

    raise FileNotFoundError(f"No safetensors file found in {directory}")


def parse_gnu_time_stderr(stderr):
    """Extracts peak RSS (MB) and wall-clock seconds from /usr/bin/time -v output.

    Args:
        stderr: The stderr string from a subprocess run with /usr/bin/time -v.

    Returns:
        Tuple of (peak_rss_mb, wall_seconds).
    """
    rss_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    wall_match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", stderr
    )

    peak_rss_mb = int(rss_match.group(1)) / 1024 if rss_match else 0.0

    wall_seconds = 0.0
    if wall_match:
        parts = wall_match.group(1).split(":")
        if len(parts) == 2:
            wall_seconds = float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            wall_seconds = (
                float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            )
        else:
            wall_seconds = float(parts[0])

    return peak_rss_mb, wall_seconds


def drop_caches():
    """Drops the OS page cache, dentries, and inodes.

    Requires root. Ensures each run starts with a cold disk cache
    for fair wall-clock comparison across tools.
    """
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Warning: could not drop caches: {e}", file=sys.stderr)


def measure_surgery(base_path, adapter_path, output_path):
    """Runs the Rust CLI under /usr/bin/time and returns (peak_rss_mb, wall_seconds).

    Args:
        base_path: Path to the base model safetensors file or directory.
        adapter_path: Path to the adapter directory.
        output_path: Path for the merged output file.

    Returns:
        Dict with peak_rss_mb and wall_seconds.
    """
    cli_path = find_cli_binary()
    cmd = [
        "/usr/bin/time", "-v",
        cli_path,
        "--base-model", base_path,
        "--adapter", adapter_path,
        "--output", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CLI stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"CLI exited with code {result.returncode}")

    peak_rss_mb, wall_seconds = parse_gnu_time_stderr(result.stderr)
    return {"peak_rss_mb": peak_rss_mb, "wall_seconds": wall_seconds}


def measure_peft_subprocess(base_dir, adapter_path, output_path, low_memory=False):
    """Runs PEFT merge in an isolated subprocess under /usr/bin/time.

    This ensures the peak RSS reading is not contaminated by prior runs
    in the same process.

    Args:
        base_dir: Path to the base model directory.
        adapter_path: Path to the adapter directory.
        output_path: Path for the merged output file.
        low_memory: If True, uses low_cpu_mem_usage=True for loading.

    Returns:
        Dict with peak_rss_mb and wall_seconds.
    """
    low_mem_flag = "1" if low_memory else "0"
    script = textwrap.dedent(f"""\
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        low_memory = {low_mem_flag} == "1"

        load_kwargs = {{"torch_dtype": "auto"}}
        if low_memory:
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(
            "{base_dir}", **load_kwargs
        )
        peft_model = PeftModel.from_pretrained(model, "{adapter_path}")
        merged = peft_model.merge_and_unload()
        merged.save_pretrained("{output_path}")
    """)

    cmd = [
        "/usr/bin/time", "-v",
        sys.executable, "-c", script,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  PEFT stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"PEFT subprocess exited with code {result.returncode}")

    peak_rss_mb, wall_seconds = parse_gnu_time_stderr(result.stderr)
    return {"peak_rss_mb": peak_rss_mb, "wall_seconds": wall_seconds}


def write_mergekit_config(base_dir, adapter_path, output_yaml_path):
    """Writes a mergekit YAML config for applying a single LoRA to a base model.

    Args:
        base_dir: Path to the base model directory.
        adapter_path: Path to the adapter directory.
        output_yaml_path: Where to write the YAML config.
    """
    config = textwrap.dedent(f"""\
        models:
          - model: {base_dir}
            lora: {adapter_path}
            parameters:
              weight: 1.0
        merge_method: passthrough
    """)
    with open(output_yaml_path, "w") as f:
        f.write(config)


def measure_mergekit_subprocess(base_dir, adapter_path, output_path):
    """Runs mergekit merge in an isolated subprocess under /usr/bin/time.

    Args:
        base_dir: Path to the base model directory.
        adapter_path: Path to the adapter directory.
        output_path: Path for the merged output directory.

    Returns:
        Dict with peak_rss_mb and wall_seconds.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name
        write_mergekit_config(base_dir, adapter_path, config_path)

    try:
        cmd = [
            "/usr/bin/time", "-v",
            sys.executable, "-m", "mergekit.scripts.run_yaml",
            config_path, output_path,
            "--copy-tokenizer",
            "--low-cpu-memory",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  mergekit stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(
                f"mergekit subprocess exited with code {result.returncode}"
            )

        peak_rss_mb, wall_seconds = parse_gnu_time_stderr(result.stderr)
        return {"peak_rss_mb": peak_rss_mb, "wall_seconds": wall_seconds}
    finally:
        os.unlink(config_path)


def load_tensors_from_path(path):
    """Loads all tensors from a safetensors file or directory into a dict.

    Args:
        path: A safetensors file or directory containing safetensors files.

    Returns:
        Dict mapping tensor name to (bytes, dtype_str, shape) tuples.
        Tensors are keyed by name and stored as raw bytes to avoid
        requiring torch as a dependency in the benchmark harness.
    """
    from safetensors import safe_open

    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.safetensors"))

    tensors = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for name in sf.keys():
                tensor = sf.get_tensor(name)
                tensors[name] = tensor
    return tensors


def verify_outputs(output_paths):
    """Compares all tool outputs tensor-by-tensor against the first tool.

    Loads each output with safetensors, then checks for missing/extra
    tensors, shape mismatches, and numerical differences. This is
    insensitive to tensor ordering within the file.

    Args:
        output_paths: Dict mapping tool name to output path (str).
    """
    import torch

    valid_paths = {
        name: p for name, p in output_paths.items()
        if p and os.path.exists(p)
    }

    if len(valid_paths) < 2:
        return

    tools = list(valid_paths.keys())
    ref_name = tools[0]
    ref_tensors = load_tensors_from_path(valid_paths[ref_name])
    all_match = True

    for other_name in tools[1:]:
        other_tensors = load_tensors_from_path(valid_paths[other_name])

        ref_keys = set(ref_tensors.keys())
        other_keys = set(other_tensors.keys())

        only_ref = ref_keys - other_keys
        only_other = other_keys - ref_keys

        if only_ref:
            all_match = False
            print(f"    Only in {ref_name}: {sorted(only_ref)}")
        if only_other:
            all_match = False
            print(f"    Only in {other_name}: {sorted(only_other)}")

        mismatched = []
        for tensor_name in sorted(ref_keys & other_keys):
            t_ref = ref_tensors[tensor_name]
            t_other = other_tensors[tensor_name]

            if t_ref.shape != t_other.shape:
                all_match = False
                mismatched.append(
                    f"      {tensor_name}: shape {list(t_ref.shape)} "
                    f"vs {list(t_other.shape)}"
                )
                continue

            if not torch.equal(t_ref, t_other):
                all_match = False
                diff = (t_ref.float() - t_other.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                mismatched.append(
                    f"      {tensor_name}: "
                    f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
                )

        if mismatched:
            print(f"    {ref_name} vs {other_name}: "
                  f"{len(mismatched)} tensor(s) differ")
            for line in mismatched:
                print(line)
        else:
            print(f"    {ref_name} vs {other_name}: ALL TENSORS MATCH")


def compute_f64_reference(base_dir, adapter_dir):
    """Computes f64 gold standard merge for LoRA target tensors.

    Loads base weights and LoRA A/B matrices as f64, computes the merge
    in f64, then downcasts to the base model's original dtype. This is
    as close to the mathematically correct result as possible.

    Args:
        base_dir: Path to the base model directory.
        adapter_dir: Path to the adapter directory.

    Returns:
        Tuple of (lora_reference, base_tensors, lora_target_names) where
        lora_reference maps base tensor names to gold standard tensors
        in the base model's original dtype,
        base_tensors maps all base tensor names to their original tensors,
        and lora_target_names is the set of base names that are LoRA targets.
    """
    import torch

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path) as f:
        config = json.load(f)

    alpha = config["lora_alpha"]
    r = config["r"]
    scaling = alpha / r
    fan_in_fan_out = config.get("fan_in_fan_out", False)

    base_path = find_safetensors_path(base_dir)
    base_tensors = load_tensors_from_path(base_path)
    adapter_tensors = load_tensors_from_path(adapter_dir)

    adapter_prefix = "base_model.model."
    lora_a_suffix = ".lora_A.weight"
    lora_b_suffix = ".lora_B.weight"

    lora_a_map = {}
    lora_b_map = {}

    for adapter_name in adapter_tensors:
        if not adapter_name.startswith(adapter_prefix):
            continue
        stripped = adapter_name[len(adapter_prefix):]

        if stripped.endswith(lora_a_suffix):
            base_part = stripped[: -len(lora_a_suffix)]
            base_name = f"{base_part}.weight"
            if base_name in base_tensors:
                lora_a_map[base_name] = adapter_name
        elif stripped.endswith(lora_b_suffix):
            base_part = stripped[: -len(lora_b_suffix)]
            base_name = f"{base_part}.weight"
            if base_name in base_tensors:
                lora_b_map[base_name] = adapter_name

    lora_reference = {}
    lora_target_names = set()

    for base_name in lora_a_map:
        if base_name not in lora_b_map:
            continue

        lora_target_names.add(base_name)

        base_w = base_tensors[base_name].to(torch.float64)
        lora_a = adapter_tensors[lora_a_map[base_name]].to(torch.float64)
        lora_b = adapter_tensors[lora_b_map[base_name]].to(torch.float64)

        if fan_in_fan_out:
            delta = lora_b @ lora_a.t()
        else:
            delta = lora_b @ lora_a

        merged_f64 = base_w + scaling * delta
        original_dtype = base_tensors[base_name].dtype
        lora_reference[base_name] = merged_f64.to(original_dtype)

    return lora_reference, base_tensors, lora_target_names


def measure_accuracy(tool_path, lora_reference, base_tensors, lora_target_names):
    """Measures a tool's output equivalence against the f64 gold standard.

    For LoRA target tensors, checks if the tool's output is byte-identical
    to the reference when both are in the output dtype. Also reports the
    intermediate f64 error and the output dtype's epsilon for context.

    For non-LoRA tensors, counts how many are bit-identical to the original
    base model.

    Args:
        tool_path: Path to the tool's output directory or file.
        lora_reference: Dict of gold standard tensors for LoRA targets
            (already downcast to base dtype).
        base_tensors: Dict of original base model tensors.
        lora_target_names: Set of tensor names that are LoRA targets.

    Returns:
        Dict with accuracy metrics: lora_max_error, lora_equivalent,
        lora_equivalent_total, output_dtype, output_epsilon,
        non_lora_identical, non_lora_total.
    """
    import torch

    dtype_epsilon = {
        torch.float16: 9.77e-4,
        torch.bfloat16: 7.81e-3,
        torch.float32: 1.19e-7,
    }

    tool_tensors = load_tensors_from_path(tool_path)

    lora_max_errors = []
    lora_equivalent_count = 0
    lora_equivalent_total = 0
    output_dtype = None

    for name in sorted(lora_target_names):
        if name not in tool_tensors or name not in lora_reference:
            continue

        ref_t = lora_reference[name]
        tool_t = tool_tensors[name]

        if output_dtype is None:
            output_dtype = ref_t.dtype

        lora_equivalent_total += 1
        if torch.equal(tool_t, ref_t):
            lora_equivalent_count += 1

        diff = (tool_t.float() - ref_t.float()).abs()
        lora_max_errors.append(diff.max().item())

    non_lora_total = 0
    non_lora_identical = 0

    for name in base_tensors:
        if name in lora_target_names:
            continue
        if name not in tool_tensors:
            continue

        non_lora_total += 1
        if torch.equal(base_tensors[name], tool_tensors[name]):
            non_lora_identical += 1

    max_err = max(lora_max_errors) if lora_max_errors else 0.0
    epsilon = dtype_epsilon.get(output_dtype, 0.0)

    return {
        "lora_max_error": max_err,
        "lora_equivalent": lora_equivalent_count,
        "lora_equivalent_total": lora_equivalent_total,
        "output_dtype": str(output_dtype),
        "output_epsilon": epsilon,
        "below_epsilon": max_err <= epsilon,
        "non_lora_identical": non_lora_identical,
        "non_lora_total": non_lora_total,
    }


def run_single_measurement(tool, base_dir, base_file, adapter_dir, tmp_root):
    """Runs a single measurement for one tool, returning stats.

    Args:
        tool: One of "surgery", "peft", "peft_low_mem", "mergekit".
        base_dir: Path to the base model directory.
        base_file: Path to the safetensors file (or dir for sharded).
        adapter_dir: Path to the adapter directory.
        tmp_root: Temporary directory for outputs.

    Returns:
        Tuple of (stats_dict, output_path).
    """
    output_path = os.path.join(tmp_root, f"{tool}_output")
    os.makedirs(output_path, exist_ok=True)

    if tool == "surgery":
        if os.path.isdir(base_file):
            stats = measure_surgery(base_file, adapter_dir, output_path)
        else:
            out_file = os.path.join(output_path, "merged.safetensors")
            stats = measure_surgery(base_file, adapter_dir, out_file)
        return stats, output_path

    if tool == "peft":
        stats = measure_peft_subprocess(base_dir, adapter_dir, output_path)
        return stats, output_path

    if tool == "peft_low_mem":
        stats = measure_peft_subprocess(
            base_dir, adapter_dir, output_path, low_memory=True
        )
        return stats, output_path

    if tool == "mergekit":
        stats = measure_mergekit_subprocess(base_dir, adapter_dir, output_path)
        return stats, output_path

    raise ValueError(f"Unknown tool: {tool}")


def run_benchmark(label, base_dir, adapter_dir, tools, num_runs=3,
                  drop_caches_enabled=False):
    """Runs all selected tools on a single model, with repeated measurements.

    First-run outputs are saved to a persistent temp directory so that
    per-tensor verification can compare them after all tools finish.

    Args:
        label: Human-readable model label for display.
        base_dir: Path to the base model directory.
        adapter_dir: Path to the adapter directory.
        tools: List of tool names to benchmark.
        num_runs: Number of runs per tool (median is reported).
        drop_caches_enabled: If True, drop OS page cache before each run.

    Returns:
        Dict with label and per-tool median results.
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {label}")
    print(f"{'=' * 60}")

    base_file = find_safetensors_path(base_dir)
    result = {"label": label}

    verify_dir = tempfile.mkdtemp(prefix="bench_verify_")
    first_run_paths = {}

    try:
        for tool in tools:
            tool_label = {
                "surgery": "safetensors-surgery",
                "peft": "PEFT merge_and_unload",
                "peft_low_mem": "PEFT (low memory)",
                "mergekit": "mergekit",
            }.get(tool, tool)

            print(f"  Running {tool_label} ({num_runs} runs)...")

            rss_values = []
            time_values = []

            for run_idx in range(num_runs):
                if drop_caches_enabled:
                    drop_caches()
                if run_idx == 0:
                    tool_out = os.path.join(verify_dir, tool)
                    os.makedirs(tool_out, exist_ok=True)
                    try:
                        stats, output_path = run_single_measurement(
                            tool, base_dir, base_file, adapter_dir, tool_out
                        )
                        rss_values.append(stats["peak_rss_mb"])
                        time_values.append(stats["wall_seconds"])
                        first_run_paths[tool] = output_path
                        print(
                            f"    Run {run_idx + 1}: "
                            f"RSS={stats['peak_rss_mb']:.0f}MB, "
                            f"Time={stats['wall_seconds']:.2f}s"
                        )
                    except Exception as e:
                        print(f"    Run {run_idx + 1} FAILED: {e}")
                else:
                    with tempfile.TemporaryDirectory() as tmp:
                        try:
                            stats, output_path = run_single_measurement(
                                tool, base_dir, base_file, adapter_dir, tmp
                            )
                            rss_values.append(stats["peak_rss_mb"])
                            time_values.append(stats["wall_seconds"])
                            print(
                                f"    Run {run_idx + 1}: "
                                f"RSS={stats['peak_rss_mb']:.0f}MB, "
                                f"Time={stats['wall_seconds']:.2f}s"
                            )
                        except Exception as e:
                            print(f"    Run {run_idx + 1} FAILED: {e}")

            if rss_values:
                median_rss = median(rss_values)
                median_time = median(time_values)
                result[tool] = {
                    "peak_rss_mb": median_rss,
                    "wall_seconds": median_time,
                    "all_rss": rss_values,
                    "all_times": time_values,
                }
                print(
                    f"    Median: RSS={median_rss:.0f}MB, Time={median_time:.2f}s"
                )
            else:
                print(f"    All runs failed for {tool_label}")
                result[tool] = None

        if len(first_run_paths) >= 2:
            print("\n  Output verification (per-tensor):")
            verify_outputs(first_run_paths)

        if first_run_paths:
            print("\n  Accuracy (vs f64 gold standard):")
            try:
                lora_ref, base_tensors, lora_targets = compute_f64_reference(
                    base_dir, adapter_dir
                )
                print(f"    LoRA targets: {len(lora_targets)}")
                tool_labels = {
                    "surgery": "safetensors-surgery",
                    "peft": "PEFT merge_and_unload",
                    "peft_low_mem": "PEFT (low memory)",
                    "mergekit": "mergekit",
                }

                for tool in tools:
                    if tool not in first_run_paths:
                        continue
                    acc = measure_accuracy(
                        first_run_paths[tool],
                        lora_ref,
                        base_tensors,
                        lora_targets,
                    )
                    label = tool_labels.get(tool, tool)
                    pct = (
                        100.0 * acc["non_lora_identical"] / acc["non_lora_total"]
                        if acc["non_lora_total"] > 0
                        else 0.0
                    )
                    eq_pct = (
                        100.0 * acc["lora_equivalent"] / acc["lora_equivalent_total"]
                        if acc["lora_equivalent_total"] > 0
                        else 0.0
                    )
                    status = "EQUIVALENT" if acc["below_epsilon"] else "DIVERGENT"
                    print(
                        f"    {label}:"
                        f"  max_intermediate_err={acc['lora_max_error']:.2e}"
                        f"  dtype_epsilon={acc['output_epsilon']:.2e}"
                        f"  status={status}"
                        f"  lora_bytewise={acc['lora_equivalent']}"
                        f"/{acc['lora_equivalent_total']}"
                        f" ({eq_pct:.0f}%)"
                        f"  passthrough="
                        f"{acc['non_lora_identical']}/{acc['non_lora_total']}"
                        f" ({pct:.0f}%)"
                    )

                    if result.get(tool) is not None:
                        result[tool]["accuracy"] = acc
            except Exception as e:
                print(f"    Accuracy measurement failed: {e}")

    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)

    return result


def print_summary(results, tools):
    """Prints a formatted summary table with performance and accuracy.

    Args:
        results: List of benchmark result dicts.
        tools: List of tool names that were benchmarked.
    """
    tool_labels = {
        "surgery": "safetensors-surgery",
        "peft": "PEFT merge_and_unload",
        "peft_low_mem": "PEFT (low memory)",
        "mergekit": "mergekit",
    }

    width = 120
    print(f"\n{'=' * width}")
    print("RESULTS (median of all runs)")
    print(f"{'=' * width}")
    print(
        f"{'Model':<22} {'Tool':<23} "
        f"{'RSS (MB)':>10} {'Time (s)':>10} "
        f"{'Max Err':>12} {'Status':>14} "
        f"{'Passthrough':>13}"
    )
    print("-" * width)

    for r in results:
        first_tool = True
        surgery_stats = r.get("surgery")

        for tool in tools:
            stats = r.get(tool)
            if stats is None:
                continue

            label_col = r["label"] if first_tool else ""
            tool_name = tool_labels.get(tool, tool)

            acc = stats.get("accuracy")
            if acc:
                pct = (
                    100.0 * acc["non_lora_identical"] / acc["non_lora_total"]
                    if acc["non_lora_total"] > 0
                    else 0.0
                )
                acc_max = f"{acc['lora_max_error']:.2e}"
                status = "Equivalent" if acc["below_epsilon"] else "Divergent"
                passthrough = f"{pct:.0f}%"
            else:
                acc_max = "n/a"
                status = "n/a"
                passthrough = "n/a"

            print(
                f"{label_col:<22} {tool_name:<23} "
                f"{stats['peak_rss_mb']:>10.0f} "
                f"{stats['wall_seconds']:>10.2f} "
                f"{acc_max:>12} {status:>14} "
                f"{passthrough:>13}"
            )
            first_tool = False

        if surgery_stats:
            print()
            for tool in tools:
                if tool == "surgery":
                    continue
                stats = r.get(tool)
                if stats is None:
                    continue
                tool_name = tool_labels.get(tool, tool)
                mem_ratio = stats["peak_rss_mb"] / max(
                    surgery_stats["peak_rss_mb"], 0.1
                )
                time_ratio = stats["wall_seconds"] / max(
                    surgery_stats["wall_seconds"], 0.01
                )
                print(
                    f"  vs {tool_name}: "
                    f"{mem_ratio:.1f}x memory, {time_ratio:.1f}x time"
                )

        print("-" * width)


def save_chart(results, tools, output_path):
    """Saves benchmark results as separate PNG charts and a JSON data file.

    Produces three individual charts (memory, time, accuracy) plus a
    combined HTML with all three. Each chart is styled for README embedding.

    Args:
        results: List of benchmark result dicts.
        tools: List of tool names that were benchmarked.
        output_path: Base output path (e.g., "benchmarks/results.png").
    """
    tool_labels = {
        "surgery": "safetensors-surgery",
        "peft": "PEFT merge_and_unload",
        "peft_low_mem": "PEFT (low memory)",
        "mergekit": "mergekit",
    }
    tool_colors = {
        "surgery": "#2563eb",
        "peft": "#dc2626",
        "peft_low_mem": "#ea580c",
        "mergekit": "#16a34a",
    }

    labels = [r["label"] for r in results]
    base_path = output_path.replace(".png", "")

    def make_chart(title, subtitle, y_values_per_tool, text_fmt, y_title,
                   filename, show_legend=True, log_y=False):
        """Creates and saves a single bar chart.

        Args:
            title: Chart title.
            subtitle: Explanation text shown below the chart.
            y_values_per_tool: Dict mapping tool name to list of y values.
            text_fmt: Callable that formats a value into a bar label string.
            y_title: Y-axis label.
            filename: Output filename (without extension).
            show_legend: Whether to show the legend.
            log_y: Whether to use log scale on y-axis.
        """
        fig = go.Figure()

        for tool in tools:
            if tool not in y_values_per_tool:
                continue
            vals = y_values_per_tool[tool]
            color = tool_colors.get(tool, "#6b7280")
            name = tool_labels.get(tool, tool)

            display_vals = vals
            if log_y:
                nonzero = [v for v in vals if v > 0]
                floor = min(nonzero) * 0.01 if nonzero else 1e-15
                display_vals = [v if v > 0 else floor for v in vals]

            fig.add_trace(go.Bar(
                name=name,
                x=labels,
                y=display_vals,
                marker_color=color,
                text=[text_fmt(v) for v in vals],
                textposition="outside",
                textfont=dict(size=12, color="#374151"),
            ))

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=22, color="#111827"),
                x=0.5,
                xanchor="center",
                y=0.97,
            ),
            barmode="group",
            bargap=0.25,
            bargroupgap=0.1,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, system-ui, sans-serif", size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.06,
                xanchor="center",
                x=0.5,
                font=dict(size=13),
                bgcolor="rgba(255,255,255,0.8)",
            ) if show_legend else dict(visible=False),
            height=550,
            width=720,
            margin=dict(t=130, b=100, l=80, r=40),
            yaxis=dict(
                title=dict(
                    text=y_title,
                    font=dict(size=13, color="#6b7280"),
                ),
                gridcolor="#e5e7eb",
                gridwidth=1,
                showgrid=True,
                zeroline=True,
                zerolinecolor="#d1d5db",
                zerolinewidth=1,
                type="log" if log_y else "linear",
            ),
            xaxis=dict(
                tickfont=dict(size=12, color="#374151"),
                tickangle=0,
            ),
            annotations=[
                dict(
                    text=subtitle,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=12, color="#6b7280"),
                    xanchor="center",
                ),
            ],
        )

        html_file = f"{filename}.html"
        fig.write_html(html_file, include_plotlyjs="cdn")

        try:
            fig.write_image(f"{filename}.png", scale=2)
            print(f"  Saved {filename}.png")
        except Exception as e:
            print(f"  Could not save {filename}.png (install kaleido): {e}")

    memory_vals = {}
    time_vals = {}

    for tool in tools:
        mem = []
        time_ = []
        for r in results:
            stats = r.get(tool)
            mem.append(stats["peak_rss_mb"] if stats else 0)
            time_.append(stats["wall_seconds"] if stats else 0)
        memory_vals[tool] = mem
        time_vals[tool] = time_

    def fmt_memory(v):
        """Formats memory values with MB suffix."""
        if v >= 1000:
            return f"{v / 1000:,.1f} GB"
        return f"{v:,.0f} MB"

    def fmt_time(v):
        """Formats time values with unit."""
        return f"{v:.1f}s"

    make_chart(
        title="Peak Memory Usage",
        subtitle="Maximum resident set size (RSS) of the merge process. Lower is better.",
        y_values_per_tool=memory_vals,
        text_fmt=fmt_memory,
        y_title="Peak RSS (MB)",
        filename=f"{base_path}_memory",
        show_legend=True,
    )

    make_chart(
        title="Merge Time",
        subtitle="Wall-clock time including merge computation and writing output to disk. Lower is better.",
        y_values_per_tool=time_vals,
        text_fmt=fmt_time,
        y_title="Seconds",
        filename=f"{base_path}_time",
        show_legend=True,
    )

    results_json = f"{base_path}.json"
    export = []
    for r in results:
        entry = {"model": r["label"]}
        for tool in tools:
            stats = r.get(tool)
            prefix = tool
            if stats:
                entry[f"{prefix}_peak_rss_mb"] = stats["peak_rss_mb"]
                entry[f"{prefix}_wall_seconds"] = stats["wall_seconds"]
                entry[f"{prefix}_all_rss"] = stats["all_rss"]
                entry[f"{prefix}_all_times"] = stats["all_times"]
                acc = stats.get("accuracy")
                if acc:
                    entry[f"{prefix}_lora_max_error"] = acc["lora_max_error"]
                    entry[f"{prefix}_lora_equivalent"] = acc["lora_equivalent"]
                    entry[f"{prefix}_lora_equivalent_total"] = acc["lora_equivalent_total"]
                    entry[f"{prefix}_below_epsilon"] = acc["below_epsilon"]
                    entry[f"{prefix}_output_dtype"] = acc["output_dtype"]
                    entry[f"{prefix}_non_lora_identical"] = acc["non_lora_identical"]
                    entry[f"{prefix}_non_lora_total"] = acc["non_lora_total"]
            else:
                entry[f"{prefix}_peak_rss_mb"] = None
                entry[f"{prefix}_wall_seconds"] = None
        export.append(entry)

    with open(results_json, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Raw data saved to {results_json}")


def check_tool_available(tool):
    """Checks whether a tool's dependencies are importable.

    Args:
        tool: Tool name string.

    Returns:
        True if the tool can be used, False otherwise.
    """
    if tool == "surgery":
        try:
            find_cli_binary()
            return True
        except FileNotFoundError:
            return False

    if tool in ("peft", "peft_low_mem"):
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    if tool == "mergekit":
        try:
            import mergekit  # noqa: F401
            return True
        except ImportError:
            return False

    return False


def main():
    """Runs benchmarks comparing safetensors-surgery, PEFT, and mergekit."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark safetensors-surgery against PEFT and mergekit"
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(PRESET_MODELS.keys()),
        default=["opt-350m"],
        help="Which preset models to benchmark",
    )
    parser.add_argument(
        "--base",
        help="Path to a custom base model directory (ignores --models)",
    )
    parser.add_argument(
        "--adapter",
        help="Path to a custom adapter directory (ignores --models)",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=TOOLS,
        default=None,
        help="Which tools to benchmark (default: all available)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per tool (reports median)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results.png",
        help="Path for the output chart",
    )
    parser.add_argument(
        "--drop-caches",
        action="store_true",
        help="Run 'sudo echo 3 > /proc/sys/vm/drop_caches' before each run (requires passwordless sudo)",
    )
    args = parser.parse_args()

    print("Building release binary...")
    subprocess.run(["cargo", "build", "--release"], check=True)

    if args.drop_caches:
        try:
            subprocess.run(
                ["sudo", "-n", "true"],
                check=True,
                capture_output=True,
            )
            print("Drop caches enabled (sudo verified)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                "Error: --drop-caches requires passwordless sudo.\n"
                "Run: echo \"$USER ALL=(ALL) NOPASSWD: ALL\" | sudo tee /etc/sudoers.d/$USER",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.tools:
        tools = args.tools
    else:
        tools = [t for t in TOOLS if check_tool_available(t)]
        print(f"Available tools: {', '.join(tools)}")
        unavailable = [t for t in TOOLS if t not in tools]
        if unavailable:
            print(f"Skipping (not installed): {', '.join(unavailable)}")

    if not tools:
        print("No tools available to benchmark.", file=sys.stderr)
        sys.exit(1)

    results = []

    if args.base and args.adapter:
        result = run_benchmark(
            "Custom", args.base, args.adapter, tools,
            num_runs=args.runs, drop_caches_enabled=args.drop_caches,
        )
        results.append(result)
    else:
        for model_key in args.models:
            preset = PRESET_MODELS[model_key]
            base_dir = preset["base"]
            adapter_dir = preset["adapter"]
            result = run_benchmark(
                preset["label"], base_dir, adapter_dir, tools,
                num_runs=args.runs, drop_caches_enabled=args.drop_caches,
            )
            results.append(result)

    print_summary(results, tools)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_chart(results, tools, args.output)


if __name__ == "__main__":
    main()