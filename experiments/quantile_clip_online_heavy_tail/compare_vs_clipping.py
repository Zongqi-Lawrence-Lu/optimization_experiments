"""Cross-experiment comparison: online quantile clipping vs fixed-threshold clipping.

Loads the sweep-best config for each condition, re-runs all five for a consistent
number of steps, then plots them together.

Run from the project root:
    python experiments/quantile_clip_online_heavy_tail/compare_vs_clipping.py [options]

Options:
    --output_dir DIR    Where to write final run logs
                        (default: outputs/quantile_vs_clipping/final)
    --device DEVICE     pytorch device (default: cpu)
    --steps INT         total_outer_steps for every condition (default: 2000)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from framework.configs import load_config
from framework.run import run as fw_run
from framework.plotting.comparison_curves import plot_comparison, save_comparison_summary

CONDITIONS = [
    # (condition_key, sweep_best_path, fallback_config_path, label)
    (
        "global_upper",
        "outputs/clipping_modes_heavy_tail/sweeps/global_upper/best_config.yaml",
        "experiments/clipping_modes_heavy_tail/configs/global_upper.yaml",
        "Fixed global L2",
    ),
    (
        "coord_upper",
        "outputs/clipping_modes_heavy_tail/sweeps/coord_upper/best_config.yaml",
        "experiments/clipping_modes_heavy_tail/configs/coord_upper.yaml",
        "Fixed coordinate",
    ),
    (
        "layerwise_upper",
        "outputs/clipping_modes_heavy_tail/sweeps/layerwise_upper/best_config.yaml",
        "experiments/clipping_modes_heavy_tail/configs/layerwise_upper.yaml",
        "Fixed layer-wise",
    ),
    (
        "global_quantile",
        "outputs/quantile_clip_online_heavy_tail/sweeps/global_quantile/best_config.yaml",
        "experiments/quantile_clip_online_heavy_tail/configs/global_quantile.yaml",
        "Online quantile global",
    ),
    (
        "coord_quantile",
        "outputs/quantile_clip_online_heavy_tail/sweeps/coord_quantile/best_config.yaml",
        "experiments/quantile_clip_online_heavy_tail/configs/coord_quantile.yaml",
        "Online quantile coord",
    ),
]


def _load_best_or_default(sweep_best: str, fallback: str) -> object:
    best_path = PROJECT_ROOT / sweep_best
    if best_path.exists():
        print(f"  using sweep best: {best_path}")
        return load_config(str(best_path))
    fb = str(PROJECT_ROOT / fallback)
    print(f"  sweep not found — using default config: {fb}")
    return load_config(fb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantile vs fixed-threshold clipping comparison")
    parser.add_argument("--output_dir", default="outputs/quantile_vs_clipping/final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    results_dir = "results/quantile_vs_clipping"
    plots_dir   = "plots/quantile_vs_clipping"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    runs = []
    for key, sweep_best, fallback, label in CONDITIONS:
        print(f"\n=== {key} ===")
        config = _load_best_or_default(sweep_best, fallback)
        run_name = f"cmp_{key}"
        config = replace(
            config,
            run_name=run_name,
            output_dir=args.output_dir,
            results_dir=results_dir,
            plots_dir=plots_dir,
            device=args.device,
            total_outer_steps=args.steps,
            eval_every=max(1, args.steps // 40),
            log_gradients_every=max(1, args.steps // 40),
            resume=False,
        )
        fw_run(config)
        runs.append((label, str(Path(args.output_dir) / run_name)))

    print("\n=== Generating comparison plots ===")
    plot_comparison(
        runs=runs,
        output_path=f"{plots_dir}/quantile_vs_clipping_val_loss.png",
        metrics=["val_loss"],
        smoothing=0.3,
        title="Online Quantile vs Fixed-Threshold Clipping — Heavy-tail Student-t(df=2)",
    )
    plot_comparison(
        runs=runs,
        output_path=f"{plots_dir}/quantile_vs_clipping_grad_stats.png",
        metrics=["val_loss", "grad_norm_raw", "clipped_fraction"],
        include_train_loss=False,
        smoothing=0.5,
        title="Gradient Statistics: Quantile vs Fixed Clipping",
    )
    save_comparison_summary(runs, f"{results_dir}/quantile_vs_clipping_comparison.json")
    print("Done.")


if __name__ == "__main__":
    main()
