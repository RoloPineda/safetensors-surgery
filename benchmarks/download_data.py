"""Downloads benchmark models and adapters from HuggingFace Hub.

Run this once before benchmarking. Downloads to the standard HuggingFace
cache and creates symlinks in benchmarks/data/. If you already have the
models cached, this just creates the symlinks with no re-download.

Requires: pip install huggingface-hub
"""

# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "huggingface-hub",
# ]
# ///


import argparse
import os
import sys


MODELS = {
    "opt-350m": {
        "base": {
            "repo": "KoboldAI/OPT-350M-Erebus",
            "local": "benchmarks/data/opt-350m/base",
        },
        "adapter": {
            "repo": "opt-350m-vanilla_finetuning_with_lora-mnli-mm-d1_fs3",
            "local": "benchmarks/data/opt-350m/adapter",
        },
        "size": "~700MB",
    },
    "tinyllama-1.1b": {
        "base": {
            "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "local": "benchmarks/data/tinyllama/base",
        },
        "adapter": {
            "repo": "mo7amed-3bdalla7/tinyllama-python-lora",
            "local": "benchmarks/data/tinyllama/adapter",
        },
        "size": "~2.5GB",
    },
    "mistral-7b": {
        "base": {
            "repo": "mistralai/Mistral-7B-v0.1",
            "local": "benchmarks/data/mistral-7b/base",
        },
        "adapter": {
            "repo": "Reg1/marcus-aurelius-mistral7b-stoic-t1-support-lora",
            "local": "benchmarks/data/mistral-7b/adapter",
        },
        "size": "~15GB",
    },
}


def is_downloaded(local_path):
    """Checks if a model path exists and points to safetensors files.

    Handles both real directories and symlinks to the HF cache.

    Args:
        local_path: Path to check.

    Returns:
        True if the path has safetensors and config files.
    """
    if not os.path.exists(local_path):
        return False
    try:
        entries = os.listdir(local_path)
    except OSError:
        return False
    has_safetensors = any(f.endswith(".safetensors") for f in entries)
    has_config = any(
        f in ("config.json", "adapter_config.json") for f in entries
    )
    return has_safetensors and has_config


def download_model(repo_id, local_path):
    """Downloads a model to the HF cache and symlinks it locally.

    If the model is already in the HF cache, no download happens and
    only the symlink is created.

    Args:
        repo_id: HuggingFace repo ID (e.g., "facebook/opt-350m").
        local_path: Where to create the symlink.
    """
    from huggingface_hub import snapshot_download

    cache_path = snapshot_download(
        repo_id,
        ignore_patterns=["*.bin", "*.pt", "*.ot", "*.msgpack"],
    )

    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.islink(local_path):
        os.remove(local_path)
    elif os.path.isdir(local_path):
        os.rename(local_path, local_path + ".bak")

    os.symlink(os.path.abspath(cache_path), local_path)


def main():
    """Downloads benchmark models and adapters."""
    parser = argparse.ArgumentParser(
        description="Download models for benchmarking safetensors-surgery"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["opt-350m"],
        help="Which models to download (default: opt-350m)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already present",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available benchmark models:\n")
        for key, info in MODELS.items():
            status = ""
            base_ok = is_downloaded(info["base"]["local"])
            adapter_ok = is_downloaded(info["adapter"]["local"])
            if base_ok and adapter_ok:
                status = " [downloaded]"
            elif base_ok or adapter_ok:
                status = " [partial]"
            print(f"  {key:<20} {info['size']:<10}{status}")
            print(f"    base:    {info['base']['repo']}")
            print(f"    adapter: {info['adapter']['repo']}")
            print()
        return

    model_keys = list(MODELS.keys()) if "all" in args.models else args.models

    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print(
            "huggingface-hub is required. Install it with:\n"
            "  pip install huggingface-hub\n"
            "Or use uv:\n"
            "  uv pip install huggingface-hub",
            file=sys.stderr,
        )
        sys.exit(1)

    for key in model_keys:
        info = MODELS[key]
        print(f"\n{'=' * 50}")
        print(f"{key} ({info['size']})")
        print(f"{'=' * 50}")

        for part in ("base", "adapter"):
            repo = info[part]["repo"]
            local = info[part]["local"]

            if is_downloaded(local) and not args.force:
                print(f"  {part}: already present at {local}")
                continue

            print(f"  {part}: downloading {repo}...")
            try:
                download_model(repo, local)
                print(f"  {part}: done -> {local}")
            except Exception as e:
                print(f"  {part}: FAILED - {e}", file=sys.stderr)

    print("\nDone. Run benchmarks with:")
    model_list = " ".join(model_keys)
    print(f"  python3 benchmarks/compare.py --models {model_list}")


if __name__ == "__main__":
    main()