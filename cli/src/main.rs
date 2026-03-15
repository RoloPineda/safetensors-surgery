use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

/// Merge a PEFT LoRA adapter into a safetensors base model with bounded memory.
#[derive(Parser, Debug)]
#[command(name = "safetensors-surgery", version, about)]
struct Args {
    /// Path to the base model: a .safetensors file, or a directory containing
    /// model.safetensors or model.safetensors.index.json (sharded).
    #[arg(long)]
    base_model: PathBuf,

    /// Path to the adapter directory (containing adapter_config.json and adapter_model.safetensors).
    #[arg(long)]
    adapter: PathBuf,

    /// Output path: a .safetensors file for single-file models, or a directory for sharded models.
    #[arg(long)]
    output: PathBuf,

    /// Print what would be done without writing output.
    #[arg(long)]
    dry_run: bool,

    /// Use tiled merge to reduce peak memory at the cost of speed.
    #[arg(long)]
    low_memory: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.dry_run {
        let config_path = args.adapter.join("adapter_config.json");
        let config = safetensors_surgery::config::AdapterConfig::from_path(&config_path)
            .context("failed to read adapter config")?;
        println!("Adapter config:");
        println!("  rank: {}", config.rank());
        println!("  alpha: {}", config.alpha());
        println!("  scaling: {}", config.scaling());
        println!("  target_modules: {:?}", config.target_modules());
        println!("  fan_in_fan_out: {}", config.fan_in_fan_out());
        println!("  bias: {:?}", config.bias());

        let info = safetensors_surgery::dry_run_info(&args.base_model, &args.adapter)
            .context("failed to inspect model")?;
        println!("\nBase model:");
        if info.is_sharded {
            println!("  type: sharded ({} shards)", info.shard_count);
        } else {
            println!("  type: single file");
        }
        println!("  total tensors: {}", info.base_tensor_count);
        println!("\nMerge plan:");
        println!("  LoRA targets:   {}", info.lora_target_count);
        println!("  replacements:   {}", info.replacement_count);
        println!("  bias merges:    {}", info.bias_merge_count);
        println!("  pass-through:   {}", info.passthrough_count);
        println!(
            "  estimated size: {:.1} MB",
            info.estimated_output_bytes as f64 / (1024.0 * 1024.0)
        );
        println!("\nDry run complete. No output written.");
        return Ok(());
    }

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} tensors")
            .context("invalid progress bar template")?
            .progress_chars("##-"),
    );

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            anyhow::bail!(
                "output parent directory '{}' does not exist",
                parent.display()
            );
        }
    }

    let progress_callback = |current: usize, total: usize| {
        pb.set_length(total as u64);
        pb.set_position(current as u64);
    };

    let stats = safetensors_surgery::merge_adapter(
        &args.base_model,
        &args.adapter,
        &args.output,
        args.low_memory,
        Some(&progress_callback),
    )
    .context("merge failed")?;

    pb.finish_with_message("done");

    println!("\nMerge complete:");
    println!("  tensors copied:   {}", stats.tensors_copied);
    println!("  tensors merged:   {}", stats.tensors_merged);
    println!("  tensors replaced: {}", stats.tensors_replaced);
    println!("  biases merged:    {}", stats.biases_merged);
    println!("\nOutput written to: {}", args.output.display());

    Ok(())
}
