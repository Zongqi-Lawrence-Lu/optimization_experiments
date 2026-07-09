"""Load the best config per condition, run final multi-seed evaluations, and
plot comparison — normalized-gradient counterpart of
lr_warmup_comparison/compare.py.

Each condition (no_warmup vs. warmup, heavy-tail vs. Gaussian) is swept
independently over lr (+ warmup_steps for the warmup conditions) only.
Clipping is fixed to clip_type=biclip, clip_scope=global, upper=lower=1.0 in
every condition's base config -- i.e. plain normalized-gradient descent,
update = -lr * g/||g|| -- so unlike lr_warmup_comparison there is no
clip-threshold axis to select alongside lr; the whole benefit-of-warmup
question collapses onto a single "max rate" axis.

Unlike the original lr_warmup_comparison/compare.py (single seed final run),
each condition's sweep winner is re-run here across multiple seeds (default
5, 42-46), and the comparison reports mean +/- std of best/final val loss and
test loss -- cheap at only 4 conditions and no df axis, and avoids resting
the headline warmup-benefit number on a single seed's variance (the project
hit exactly that problem once before; see
lr_warmup_comparison/README.md's "Revision history").

Run from the project root:
    python experiments/normalized_grad_warmup_benefit/compare.py [options]

Options:
    --sweeps_root DIR    Directory containing per-condition sweep outputs
                         (default: outputs/normalized_grad_warmup_benefit/sweeps)
    --output_dir DIR     Where to write final run logs
                         (default: outputs/normalized_grad_warmup_benefit/final)
    --device DEVICE      pytorch device (default: cpu)
    --steps INT          Override total_outer_steps for the final run (default: 1000)
    --seeds INT [INT ...] Seeds to average the final re-run over (default: 42 43 44 45 46)

Produces:
    plots/normalized_grad_warmup_benefit/heavy_tail_warmup_vs_no_warmup.png
    plots/normalized_grad_warmup_benefit/heavy_tail_lr_and_grad.png
    plots/normalized_grad_warmup_benefit/gaussian_warmup_vs_no_warmup.png
    plots/normalized_grad_warmup_benefit/all_conditions_test_loss.png
    plots/normalized_grad_warmup_benefit/best_val_loss_by_condition.png
    results/normalized_grad_warmup_benefit/warmup_comparison.json
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from framework.configs import load_config
from framework.run import run as fw_run
from framework.plotting.comparison_curves import plot_comparison

CONDITIONS = [
    "no_warmup_heavy_tail",
    "warmup_heavy_tail",
    "no_warmup_gaussian",
    "warmup_gaussian",
]

DEFAULT_CONFIGS = {
    "no_warmup_heavy_tail": "experiments/normalized_grad_warmup_benefit/configs/no_warmup_heavy_tail.yaml",
    "warmup_heavy_tail":    "experiments/normalized_grad_warmup_benefit/configs/warmup_heavy_tail.yaml",
    "no_warmup_gaussian":   "experiments/normalized_grad_warmup_benefit/configs/no_warmup_gaussian.yaml",
    "warmup_gaussian":      "experiments/normalized_grad_warmup_benefit/configs/warmup_gaussian.yaml",
}

LABELS = {
    "no_warmup_heavy_tail": "No warmup (heavy-tail, normalized grad)",
    "warmup_heavy_tail":    "Warmup (heavy-tail, normalized grad)",
    "no_warmup_gaussian":   "No warmup (Gaussian, normalized grad)",
    "warmup_gaussian":      "Warmup (Gaussian, normalized grad)",
}

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _load_best_or_default(condition: str, sweeps_root: str) -> object:
    best_path = Path(sweeps_root) / condition / "best_config.yaml"
    if best_path.exists():
        print(f"  [{condition}] using sweep best: {best_path}")
        return load_config(str(best_path))
    fallback = str(PROJECT_ROOT / DEFAULT_CONFIGS[condition])
    print(f"  [{condition}] sweep not found — using default config: {fallback}")
    return load_config(fallback)


def _final_run_metrics(config, run_name: str, seed: int, args) -> dict:
    config = replace(
        config,
        run_name=run_name,
        seed=seed,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        total_outer_steps=args.steps,
        eval_every=max(1, args.steps // 40),
        log_gradients_every=max(1, args.steps // 40),
        device=args.device,
        resume=False,
    )
    try:
        metrics = fw_run(config)
    except Exception as e:
        print(f"[diverged/failed] {run_name} (seed={seed}): {e}")
        return {
            "best_val_loss": None,
            "final_val_loss": None,
            "test_loss": None,
            "log_dir": str(Path(args.output_dir) / run_name),
            "diverged": True,
            "error": str(e),
        }
    best_val_loss = metrics.get("val_loss_best", metrics.get("best_val_loss"))
    final_val_loss = metrics.get("val_loss_final")
    test_loss = metrics.get("test_loss_final", metrics.get("test_loss_best"))
    diverged = any(
        v is not None and isinstance(v, float) and not math.isfinite(v)
        for v in (best_val_loss, final_val_loss, test_loss)
    )
    return {
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "test_loss": test_loss,
        "log_dir": str(Path(args.output_dir) / run_name),
        "diverged": diverged,
    }


def _final_run_metrics_multi_seed(config, run_name_prefix: str, seeds: list, args) -> dict:
    """Re-run `config` across `seeds`, returning per-seed metrics plus the
    mean/std of best_val_loss/final_val_loss/test_loss across seeds that did
    not diverge. Runs serially (one seed at a time)."""
    per_seed = []
    for seed in seeds:
        run_name = f"{run_name_prefix}_seed{seed}"
        m = _final_run_metrics(config, run_name, seed, args)
        m["seed"] = seed
        per_seed.append(m)

    agg: dict = {
        "per_seed": per_seed,
        "num_diverged": sum(1 for m in per_seed if m.get("diverged")),
    }
    for key in ("best_val_loss", "final_val_loss", "test_loss"):
        values = [m[key] for m in per_seed if isinstance(m[key], (int, float)) and math.isfinite(m[key])]
        if values:
            agg[f"{key}_mean"] = statistics.mean(values)
            agg[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None
    return agg


def _plot_bar_by_condition(comparison: dict, plots_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping bar plot")
        return

    labels = [LABELS[c] for c in CONDITIONS]
    means = [comparison[c]["best_val_loss_mean"] for c in CONDITIONS]
    stds = [comparison[c]["best_val_loss_std"] for c in CONDITIONS]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(CONDITIONS))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Best validation loss (mean ± std over seeds)")
    ax.set_title("Normalized-gradient warmup benefit — best val loss by condition")
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(plots_dir, "best_val_loss_by_condition.png")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalized-gradient LR warmup final comparison")
    parser.add_argument("--sweeps_root",
                        default="outputs/normalized_grad_warmup_benefit/sweeps")
    parser.add_argument("--output_dir",
                        default="outputs/normalized_grad_warmup_benefit/final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Seeds to average the final winner re-run over (per condition)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    args.results_dir = "results/normalized_grad_warmup_benefit"
    args.plots_dir = "plots/normalized_grad_warmup_benefit"
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    comparison: dict = {}
    log_dirs: dict[str, str] = {}  # seed-42 log dir per condition, for training-curve overlays
    for condition in CONDITIONS:
        print(f"\n=== {condition} ({len(args.seeds)} seeds) ===")
        config = _load_best_or_default(condition, args.sweeps_root)
        run_name_prefix = f"warmup_{condition}_final"
        agg = _final_run_metrics_multi_seed(config, run_name_prefix, args.seeds, args)
        comparison[condition] = agg
        log_dirs[condition] = agg["per_seed"][0]["log_dir"]  # seed==args.seeds[0], typically 42

    # Warmup-benefit deltas (positive = warmup variant wins), mirroring
    # lr_warmup_comparison/tail_index_compare.py's benefit computation.
    benefit = {}
    if comparison["no_warmup_heavy_tail"]["best_val_loss_mean"] is not None and \
       comparison["warmup_heavy_tail"]["best_val_loss_mean"] is not None:
        benefit["heavy_tail_val_loss_benefit"] = (
            comparison["no_warmup_heavy_tail"]["best_val_loss_mean"]
            - comparison["warmup_heavy_tail"]["best_val_loss_mean"]
        )
    if comparison["no_warmup_gaussian"]["best_val_loss_mean"] is not None and \
       comparison["warmup_gaussian"]["best_val_loss_mean"] is not None:
        benefit["gaussian_val_loss_benefit"] = (
            comparison["no_warmup_gaussian"]["best_val_loss_mean"]
            - comparison["warmup_gaussian"]["best_val_loss_mean"]
        )

    import json
    summary_path = f"{args.results_dir}/warmup_comparison.json"
    with open(summary_path, "w") as f:
        json.dump({"conditions": comparison, "warmup_benefit": benefit}, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    print("\n=== Generating comparison plots ===")

    heavy_runs = [
        (LABELS["no_warmup_heavy_tail"], log_dirs["no_warmup_heavy_tail"]),
        (LABELS["warmup_heavy_tail"],    log_dirs["warmup_heavy_tail"]),
    ]
    plot_comparison(
        runs=heavy_runs,
        output_path=f"{args.plots_dir}/heavy_tail_warmup_vs_no_warmup.png",
        metrics=["test_loss"],
        smoothing=0.3,
        title="Normalized-grad LR Warmup vs. No Warmup — Heavy-tail Student-t(df=2) Noise\n"
              "(each condition uses independently tuned best max rate; single representative seed shown)",
    )
    plot_comparison(
        runs=heavy_runs,
        output_path=f"{args.plots_dir}/heavy_tail_lr_and_grad.png",
        metrics=["val_loss", "learning_rate", "grad_norm_clipped"],
        include_train_loss=False,
        smoothing=0.5,
        title="LR Schedule and Post-normalization Grad Norm — Heavy-tail Noise\n"
              "(grad_norm_clipped should sit at ≈1.0 throughout — confirms normalization is active)",
    )

    gaussian_runs = [
        (LABELS["no_warmup_gaussian"], log_dirs["no_warmup_gaussian"]),
        (LABELS["warmup_gaussian"],    log_dirs["warmup_gaussian"]),
    ]
    plot_comparison(
        runs=gaussian_runs,
        output_path=f"{args.plots_dir}/gaussian_warmup_vs_no_warmup.png",
        metrics=["test_loss"],
        smoothing=0.3,
        title="Normalized-grad LR Warmup vs. No Warmup — Gaussian (light-tail) Noise\n"
              "(each condition uses independently tuned best max rate; single representative seed shown)",
    )

    all_runs = [(LABELS[c], log_dirs[c]) for c in CONDITIONS]
    plot_comparison(
        runs=all_runs,
        output_path=f"{args.plots_dir}/all_conditions_test_loss.png",
        metrics=["test_loss"],
        smoothing=0.3,
        title="All Conditions: Warmup × Noise Type (normalized gradient)",
    )

    _plot_bar_by_condition(comparison, args.plots_dir)

    print("Done.")


if __name__ == "__main__":
    main()
