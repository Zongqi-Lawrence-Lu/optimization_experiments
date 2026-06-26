"""Load the best config per clipping mode, run final evaluations, and plot comparison.

Run from the project root:
    python experiments/clipping_modes_heavy_tail/compare.py [options]

Options:
    --sweeps_root DIR    Directory containing per-condition sweep outputs
                         (default: outputs/clipping_modes_heavy_tail/sweeps)
    --output_dir DIR     Where to write final run logs
                         (default: outputs/clipping_modes_heavy_tail/final)
    --device DEVICE      pytorch device (default: cpu)
    --steps INT          Override total_outer_steps for the final run (default: 1000)

If a sweep has not been run yet, the script falls back to the default condition
config with its initial hyperparameters.
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

CONDITIONS = ["global_upper", "coord_upper", "layerwise_upper"]

DEFAULT_CONFIGS = {
    "global_upper":    "experiments/clipping_modes_heavy_tail/configs/global_upper.yaml",
    "coord_upper":     "experiments/clipping_modes_heavy_tail/configs/coord_upper.yaml",
    "layerwise_upper": "experiments/clipping_modes_heavy_tail/configs/layerwise_upper.yaml",
}

LABELS = {
    "global_upper":    "Global L2",
    "coord_upper":     "Coordinate-wise",
    "layerwise_upper": "Layer-wise",
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
    parser = argparse.ArgumentParser(description="Clipping modes final comparison")
    parser.add_argument("--sweeps_root",
                        default="outputs/clipping_modes_heavy_tail/sweeps")
    parser.add_argument("--output_dir",
                        default="outputs/clipping_modes_heavy_tail/final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    results_dir = "results/clipping_modes_heavy_tail"
    plots_dir   = "plots/clipping_modes_heavy_tail"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    runs = []
    for condition in CONDITIONS:
        print(f"\n=== {condition} ===")
        config = _load_best_or_default(condition, args.sweeps_root)
        run_name = f"clipping_{condition}_final"
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
        output_path=f"{plots_dir}/clipping_modes_val_loss.png",
        metrics=["val_loss"],
        smoothing=0.3,
        title="Clipping Mode Comparison — Heavy-tail Student-t(df=2) Noise",
    )
    plot_comparison(
        runs=runs,
        output_path=f"{plots_dir}/clipping_modes_grad_norms.png",
        metrics=["val_loss", "grad_norm_raw", "clipped_fraction"],
        include_train_loss=False,
        smoothing=0.5,
        title="Gradient Statistics by Clipping Mode",
    )
    save_comparison_summary(runs, f"{results_dir}/clipping_modes_comparison.json")
    print("Done.")


if __name__ == "__main__":
    main()
