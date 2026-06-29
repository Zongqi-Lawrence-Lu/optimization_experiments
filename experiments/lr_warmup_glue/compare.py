"""Load the best config per warmup condition, run final evaluations, and plot comparison.

Each condition (no_warmup vs. warmup on SST-2) is swept independently — no shared
learning rate between them. The comparison shows the best achievable performance.

Run from the project root:
    python experiments/lr_warmup_glue/compare.py [options]

Options:
    --sweeps_root DIR    Directory containing per-condition sweep outputs
                         (default: outputs/lr_warmup_glue/sweeps)
    --output_dir DIR     Where to write final run logs
                         (default: outputs/lr_warmup_glue/final)
    --device DEVICE      pytorch device (default: cuda)
    --steps INT          Override total_outer_steps for the final run (default: 3000)

Produces:
    plots/lr_warmup_glue/warmup_vs_no_warmup_loss.png
    plots/lr_warmup_glue/warmup_vs_no_warmup_accuracy.png
    plots/lr_warmup_glue/lr_schedule_and_grad.png
    results/lr_warmup_glue/warmup_comparison.json
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
    "no_warmup_sst2",
    "warmup_sst2",
]

DEFAULT_CONFIGS = {
    "no_warmup_sst2": "experiments/lr_warmup_glue/configs/no_warmup_sst2.yaml",
    "warmup_sst2":    "experiments/lr_warmup_glue/configs/warmup_sst2.yaml",
}

LABELS = {
    "no_warmup_sst2": "No warmup",
    "warmup_sst2":    "Warmup (linear)",
}


def _load_best_or_default(condition: str, sweeps_root: str) -> object:
    root = Path(sweeps_root) / condition
    # Two-stage sweep writes best_config.yaml inside stage2/ (finer sweep) or stage1/.
    for stage in ("stage2", "stage1"):
        best_path = root / stage / "best_config.yaml"
        if best_path.exists():
            print(f"  [{condition}] using sweep best: {best_path}")
            return load_config(str(best_path))
    fallback = str(PROJECT_ROOT / DEFAULT_CONFIGS[condition])
    print(f"  [{condition}] sweep not found — using default config: {fallback}")
    return load_config(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description="LR warmup final comparison — SST-2")
    parser.add_argument("--sweeps_root", default="outputs/lr_warmup_glue/sweeps")
    parser.add_argument("--output_dir",  default="outputs/lr_warmup_glue/final")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--steps",       type=int, default=3000)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    results_dir = "results/lr_warmup_glue"
    plots_dir   = "plots/lr_warmup_glue"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    log_dirs: dict[str, str] = {}
    for condition in CONDITIONS:
        print(f"\n=== {condition} ===")
        config = _load_best_or_default(condition, args.sweeps_root)
        run_name = f"warmup_glue_{condition}_final"
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
            checkpoint_every=max(1, args.steps // 6),
            checkpoint_interval_minutes=10.0,
            resume=False,
        )
        fw_run(config)
        log_dirs[condition] = str(Path(args.output_dir) / run_name)

    print("\n=== Generating comparison plots ===")

    all_runs = [(LABELS[c], log_dirs[c]) for c in CONDITIONS]

    plot_comparison(
        runs=all_runs,
        output_path=f"{plots_dir}/warmup_vs_no_warmup_loss.png",
        metrics=["val_loss"],
        smoothing=0.3,
        title="LR Warmup vs. No Warmup — RoBERTa-base on SST-2\n"
              "(each condition uses independently tuned best hyperparameters)",
    )

    plot_comparison(
        runs=all_runs,
        output_path=f"{plots_dir}/warmup_vs_no_warmup_accuracy.png",
        metrics=["val_accuracy"],
        smoothing=0.3,
        title="Validation Accuracy: LR Warmup vs. No Warmup — SST-2",
    )

    plot_comparison(
        runs=all_runs,
        output_path=f"{plots_dir}/lr_schedule_and_grad.png",
        metrics=["val_loss", "learning_rate", "grad_norm_raw"],
        include_train_loss=False,
        smoothing=0.5,
        title="LR Schedule and Gradient Norms — SST-2",
    )

    save_comparison_summary(all_runs, f"{results_dir}/warmup_comparison.json")
    print("Done.")


if __name__ == "__main__":
    main()
