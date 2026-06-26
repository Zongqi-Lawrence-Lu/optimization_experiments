"""Load best sweep config per quantile clipping scope, run final evaluations, and plot comparison.

Run from the project root:
    python experiments/quantile_clip_online_heavy_tail/compare.py [options]

Options:
    --sweeps_root DIR    Directory containing per-condition sweep outputs
                         (default: outputs/quantile_clip_online_heavy_tail/sweeps)
    --output_dir DIR     Where to write final run logs
                         (default: outputs/quantile_clip_online_heavy_tail/final)
    --device DEVICE      pytorch device (default: cpu)
    --steps INT          Override total_outer_steps for the final run (default: 2000)
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

CONDITIONS = ["global_quantile", "coord_quantile"]

DEFAULT_CONFIGS = {
    "global_quantile": "experiments/quantile_clip_online_heavy_tail/configs/global_quantile.yaml",
    "coord_quantile":  "experiments/quantile_clip_online_heavy_tail/configs/coord_quantile.yaml",
}

LABELS = {
    "global_quantile": "Global quantile (online)",
    "coord_quantile":  "Coordinate quantile (online)",
}


def _load_best_or_default(condition: str, sweeps_root: str) -> object:
    best_path = Path(sweeps_root) / condition / "best_config.yaml"
    if best_path.exists():
        print(f"  [{condition}] using sweep best: {best_path}")
        return load_config(str(best_path))
    fallback = str(PROJECT_ROOT / DEFAULT_CONFIGS[condition])
    print(f"  [{condition}] sweep not found — using default config: {fallback}")
    return load_config(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Online quantile clipping final comparison")
    parser.add_argument("--sweeps_root",
                        default="outputs/quantile_clip_online_heavy_tail/sweeps")
    parser.add_argument("--output_dir",
                        default="outputs/quantile_clip_online_heavy_tail/final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    results_dir = "results/quantile_clip_online_heavy_tail"
    plots_dir   = "plots/quantile_clip_online_heavy_tail"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    runs = []
    for condition in CONDITIONS:
        print(f"\n=== {condition} ===")
        config = _load_best_or_default(condition, args.sweeps_root)
        run_name = f"quantile_{condition}_final"
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
        runs.append((LABELS[condition], str(Path(args.output_dir) / run_name)))

    print("\n=== Generating comparison plots ===")
    plot_comparison(
        runs=runs,
        output_path=f"{plots_dir}/quantile_clip_val_loss.png",
        metrics=["val_loss"],
        smoothing=0.3,
        title="Online Quantile Clipping Comparison — Heavy-tail Student-t(df=2) Noise",
    )
    plot_comparison(
        runs=runs,
        output_path=f"{plots_dir}/quantile_clip_grad_stats.png",
        metrics=["val_loss", "grad_norm_raw", "clipped_fraction"],
        include_train_loss=False,
        smoothing=0.5,
        title="Gradient Statistics — Online Quantile Clipping",
    )
    save_comparison_summary(runs, f"{results_dir}/quantile_clip_comparison.json")
    print("Done.")


if __name__ == "__main__":
    main()
