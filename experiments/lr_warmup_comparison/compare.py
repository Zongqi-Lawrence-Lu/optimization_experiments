"""Load the best config per warmup condition, run final evaluations, and plot comparison.

Each condition (no_warmup vs. warmup, heavy-tail vs. Gaussian) is swept
independently — there is no shared learning rate between them. The comparison
therefore shows the best achievable performance for each condition.

Run from the project root:
    python experiments/lr_warmup_comparison/compare.py [options]

Options:
    --sweeps_root DIR    Directory containing per-condition sweep outputs
                         (default: outputs/lr_warmup_comparison/sweeps)
    --output_dir DIR     Where to write final run logs
                         (default: outputs/lr_warmup_comparison/final)
    --device DEVICE      pytorch device (default: cpu)
    --steps INT          Override total_outer_steps for the final run (default: 1000)

Produces:
    plots/lr_warmup_comparison/heavy_tail_comparison.png
    plots/lr_warmup_comparison/gaussian_comparison.png
    plots/lr_warmup_comparison/lr_schedule.png
    results/lr_warmup_comparison/warmup_comparison.json
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
    "no_warmup_heavy_tail",
    "warmup_heavy_tail",
    "no_warmup_gaussian",
    "warmup_gaussian",
]

DEFAULT_CONFIGS = {
    "no_warmup_heavy_tail": "experiments/lr_warmup_comparison/configs/no_warmup_heavy_tail.yaml",
    "warmup_heavy_tail":    "experiments/lr_warmup_comparison/configs/warmup_heavy_tail.yaml",
    "no_warmup_gaussian":   "experiments/lr_warmup_comparison/configs/no_warmup_gaussian.yaml",
    "warmup_gaussian":      "experiments/lr_warmup_comparison/configs/warmup_gaussian.yaml",
}

LABELS = {
    "no_warmup_heavy_tail": "No warmup (heavy-tail)",
    "warmup_heavy_tail":    "Warmup (heavy-tail)",
    "no_warmup_gaussian":   "No warmup (Gaussian)",
    "warmup_gaussian":      "Warmup (Gaussian)",
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
    parser = argparse.ArgumentParser(description="LR warmup final comparison")
    parser.add_argument("--sweeps_root",
                        default="outputs/lr_warmup_comparison/sweeps")
    parser.add_argument("--output_dir",
                        default="outputs/lr_warmup_comparison/final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    results_dir = "results/lr_warmup_comparison"
    plots_dir   = "plots/lr_warmup_comparison"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    log_dirs: dict[str, str] = {}
    for condition in CONDITIONS:
        print(f"\n=== {condition} ===")
        config = _load_best_or_default(condition, args.sweeps_root)
        run_name = f"warmup_{condition}_final"
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
        log_dirs[condition] = str(Path(args.output_dir) / run_name)

    print("\n=== Generating comparison plots ===")

    # Heavy-tail: no-warmup vs. warmup
    heavy_runs = [
        (LABELS["no_warmup_heavy_tail"], log_dirs["no_warmup_heavy_tail"]),
        (LABELS["warmup_heavy_tail"],    log_dirs["warmup_heavy_tail"]),
    ]
    plot_comparison(
        runs=heavy_runs,
        output_path=f"{plots_dir}/heavy_tail_warmup_vs_no_warmup.png",
        metrics=["val_loss"],
        smoothing=0.3,
        title="LR Warmup vs. No Warmup — Heavy-tail Student-t(df=2) Noise\n"
              "(each condition uses independently tuned best LR)",
    )
    plot_comparison(
        runs=heavy_runs,
        output_path=f"{plots_dir}/heavy_tail_lr_and_grad.png",
        metrics=["val_loss", "learning_rate", "grad_norm_raw"],
        include_train_loss=False,
        smoothing=0.5,
        title="LR Schedule and Gradient Norms — Heavy-tail Noise",
    )

    # Gaussian: no-warmup vs. warmup (control)
    gaussian_runs = [
        (LABELS["no_warmup_gaussian"], log_dirs["no_warmup_gaussian"]),
        (LABELS["warmup_gaussian"],    log_dirs["warmup_gaussian"]),
    ]
    plot_comparison(
        runs=gaussian_runs,
        output_path=f"{plots_dir}/gaussian_warmup_vs_no_warmup.png",
        metrics=["val_loss"],
        smoothing=0.3,
        title="LR Warmup vs. No Warmup — Gaussian (light-tail) Noise\n"
              "(each condition uses independently tuned best LR)",
    )

    # All four on one plot for cross-noise comparison
    all_runs = [
        (LABELS[c], log_dirs[c]) for c in CONDITIONS
    ]
    plot_comparison(
        runs=all_runs,
        output_path=f"{plots_dir}/all_conditions_val_loss.png",
        metrics=["val_loss"],
        smoothing=0.3,
        title="All Conditions: Warmup × Noise Type",
    )

    save_comparison_summary(all_runs, f"{results_dir}/warmup_comparison.json")
    print("Done.")


if __name__ == "__main__":
    main()
