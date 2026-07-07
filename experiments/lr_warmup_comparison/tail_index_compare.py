"""Aggregate the tail-index sweep results and plot the warmup/clip trend.

For each (condition, df) pair swept by tail_index_sweep.py (three conditions:
clip_only, warmup_clip, warmup_only), loads the sweep's winning config and
re-runs a longer final evaluation (same protocol as compare.py:
total_outer_steps=1000, eval/log every steps//40) across multiple seeds
(--seeds, default 42-46), recording the mean/std of best/final val loss and
test loss across seeds. A single fixed seed produced large, non-robust swings
at extreme tail indices in the earlier revision of this experiment (see
README) -- seed-averaging the final winner re-run (not every sweep trial,
which stays single-seed for cost reasons) is the fix.

Then, for each df, computes two benefit metrics (positive = the warmup
variant beats clip_only):
    warmup_clip_benefit(df) = clip_only_best_val_loss(df) - warmup_clip_best_val_loss(df)
    warmup_only_benefit(df) = clip_only_best_val_loss(df) - warmup_only_best_val_loss(df)
(mean best_val_loss across seeds in both terms). The trend shows whether
either benefit grows as the tail gets heavier (df decreasing from 2 towards 1.5).

Produces:
    results/lr_warmup_comparison/tail_index_trend.json
    plots/lr_warmup_comparison/tail_index_val_loss_trend.png
    plots/lr_warmup_comparison/tail_index_benefit_trend.png

Run from the project root (after tail_index_sweep.py has populated the index):
    python experiments/lr_warmup_comparison/tail_index_compare.py [options]
"""

from __future__ import annotations

import argparse
import glob
import json
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

CONDITIONS = ["clip_only", "warmup_clip", "warmup_only"]
LABELS = {
    "clip_only": "Clip only",
    "warmup_clip": "Warmup + clip",
    "warmup_only": "Warmup only (no clip)",
}
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _load_index(index_paths_pattern: str) -> dict:
    """Load and merge one or more sweep-index JSON files.

    tail_index_sweep.py writes a single shared index when run locally, but
    a SLURM array (one task per df value, see job_sweep_tail_index.sbatch)
    writes one shard per task (tail_index_sweep_index_task*.json) to avoid a
    concurrent-write race. Accept a glob pattern here and merge every match.
    """
    paths = sorted(glob.glob(index_paths_pattern))
    if not paths and os.path.exists(index_paths_pattern):
        paths = [index_paths_pattern]
    if not paths:
        raise FileNotFoundError(f"No index files matched '{index_paths_pattern}'")

    merged: dict = {}
    for path in paths:
        with open(path) as f:
            partial = json.load(f)
        for condition, entries in partial.items():
            merged.setdefault(condition, {}).update(entries)
    print(f"Loaded {len(paths)} index file(s): {paths}")
    return merged


def _final_run_metrics(config_path: str, run_name: str, seed: int, args) -> dict:
    config = load_config(config_path)
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
        # A no-clipping condition (warmup_only) can genuinely diverge (NaN
        # loss) under heavy-tailed noise over a long re-run -- that is a
        # legitimate outcome to record, not a reason to lose every other
        # already-computed (condition, df, seed) result in this process.
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
    # A run can complete without raising (the compute_gradient_stats fix
    # means non-finite gradients no longer crash) but still have genuinely
    # diverged -- NaN/Inf loss. Flag that explicitly rather than let a NaN
    # silently pass the "is not None" check in the seed-aggregation below.
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


def _final_run_metrics_multi_seed(config_path: str, run_name_prefix: str, seeds: list, args) -> dict:
    """Re-run the winning config across `seeds`, returning per-seed metrics
    plus the mean/std of best_val_loss/final_val_loss/test_loss across seeds
    that did not diverge. Runs serially (one seed at a time) -- see
    job_tail_index_compare.sbatch for the wall-time reasoning.
    """
    per_seed = []
    for seed in seeds:
        run_name = f"{run_name_prefix}_seed{seed}"
        m = _final_run_metrics(config_path, run_name, seed, args)
        m["seed"] = seed
        per_seed.append(m)

    agg: dict = {
        "per_seed": per_seed,
        "num_diverged": sum(1 for m in per_seed if m.get("diverged")),
    }
    for key in ("best_val_loss", "final_val_loss", "test_loss"):
        # Exclude None (failed run) and non-finite (diverged but did not
        # raise) values -- a raw NaN would otherwise silently poison
        # statistics.mean/stdev for the whole (condition, df) group.
        values = [m[key] for m in per_seed if isinstance(m[key], (int, float)) and math.isfinite(m[key])]
        if values:
            agg[f"{key}_mean"] = statistics.mean(values)
            agg[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail-index trend: warmup benefit vs. Student-t df")
    parser.add_argument("--index_path",
                        default="results/lr_warmup_comparison/tail_index_sweep_index*.json",
                        help="Path or glob pattern; merges every matching index/shard file")
    parser.add_argument("--output_dir", default="outputs/lr_warmup_comparison/tail_index/final")
    parser.add_argument("--results_dir", default="results/lr_warmup_comparison")
    parser.add_argument("--plots_dir", default="plots/lr_warmup_comparison")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Seeds to average the final winner re-run over (per condition, df)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    index = _load_index(args.index_path)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    trend_path = os.path.join(args.results_dir, "tail_index_trend.json")

    per_df: dict = {}  # noise_df -> {condition -> metrics}
    for condition in CONDITIONS:
        for tag, entry in index.get(condition, {}).items():
            df = entry["noise_df"]
            summary = entry.get("summary")
            if not summary or not summary.get("best_config_path"):
                print(f"[skip] {condition} df={df}: no valid sweep result")
                continue
            run_name_prefix = f"tail_df{tag}_{condition}_final"
            print(f"\n=== {condition} df={df} ({len(args.seeds)} seeds) ===")
            per_df.setdefault(df, {})[condition] = _final_run_metrics_multi_seed(
                summary["best_config_path"], run_name_prefix, args.seeds, args
            )
            # Write after every (condition, df) pair, not just at the very
            # end -- a later pair diverging (or any other crash) should not
            # discard every already-computed result in this process.
            with open(trend_path, "w") as f:
                json.dump(_build_trend(per_df), f, indent=2)

    trend = _build_trend(per_df)
    with open(trend_path, "w") as f:
        json.dump(trend, f, indent=2)
    print(f"\nTrend data written to {trend_path}")

    _plot_trend(trend, args.plots_dir)


def _build_trend(per_df: dict) -> list:
    trend = []
    for df in sorted(per_df.keys(), reverse=True):
        entry = per_df[df]
        row = {"noise_df": df}
        for condition, m in entry.items():
            for k, v in m.items():
                row[f"{condition}_{k}"] = v  # includes "{condition}_per_seed" (raw per-seed metrics)
        clip = entry.get("clip_only")
        if clip is not None and clip["best_val_loss_mean"] is not None:
            for warm_cond in ("warmup_clip", "warmup_only"):
                w = entry.get(warm_cond)
                if w is None or w["best_val_loss_mean"] is None:
                    continue
                row[f"{warm_cond}_val_loss_benefit"] = clip["best_val_loss_mean"] - w["best_val_loss_mean"]
                if clip["test_loss_mean"] is not None and w["test_loss_mean"] is not None:
                    row[f"{warm_cond}_test_loss_benefit"] = clip["test_loss_mean"] - w["test_loss_mean"]
        trend.append(row)
    return trend


def _plot_trend(trend: list, plots_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    if not trend:
        print("No trend data to plot.")
        return

    # Panel 1: best val loss (mean ± std across seeds) for all three
    # conditions, vs. tail index
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, marker, color in [
        ("clip_only", "o", "#1f77b4"),
        ("warmup_clip", "s", "#ff7f0e"),
        ("warmup_only", "^", "#2ca02c"),
    ]:
        pts = [
            (r["noise_df"], r.get(f"{condition}_best_val_loss_mean"), r.get(f"{condition}_best_val_loss_std"))
            for r in trend
        ]
        pts = [(x, y, s) for x, y, s in pts if y is not None]
        if pts:
            xs, ys, ss = zip(*pts)
            ax.errorbar(xs, ys, yerr=ss, marker=marker, color=color, label=LABELS[condition], capsize=3)
    ax.set_xlabel("Student-t degrees of freedom (tail index)")
    ax.set_ylabel("Best validation loss (mean ± std over seeds)")
    ax.set_title("Clip vs. warmup+clip vs. warmup-only — validation loss across the noise tail index")
    ax.invert_xaxis()  # heavier tail (lower df) plotted to the right
    ax.legend()
    ax.grid(True, alpha=0.3)
    path1 = os.path.join(plots_dir, "tail_index_val_loss_trend.png")
    plt.tight_layout()
    plt.savefig(path1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {path1}")

    # Panel 2: benefit of each warmup variant over clip_only, vs. tail index
    fig, ax = plt.subplots(figsize=(7, 5))
    any_pts = False
    for warm_cond, marker, color in [("warmup_clip", "s", "#ff7f0e"), ("warmup_only", "^", "#2ca02c")]:
        key = f"{warm_cond}_val_loss_benefit"
        pts = [(r["noise_df"], r[key]) for r in trend if key in r]
        if pts:
            any_pts = True
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=marker, color=color, label=LABELS[warm_cond])
    if any_pts:
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Student-t degrees of freedom (tail index)")
        ax.set_ylabel("Benefit over clip_only  (clip_only best val loss − variant best val loss)")
        ax.set_title("Does either warmup variant beat clipping alone, and does it grow with tail heaviness?")
        ax.invert_xaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        path2 = os.path.join(plots_dir, "tail_index_benefit_trend.png")
        plt.tight_layout()
        plt.savefig(path2, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved {path2}")


if __name__ == "__main__":
    main()
