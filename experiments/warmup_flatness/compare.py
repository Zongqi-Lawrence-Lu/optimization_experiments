"""Visualize warmup vs. no-warmup sharpness curves from warmup_flatness runs.

Reads the run index JSON produced by run_and_measure.py, loads each run's
eval_log.jsonl (which contains both val_loss and sharpness), and produces:

Per-setting:
  1. Sharpness-over-time panel grid — one column per LR, warmup vs. no-warmup
     overlaid on the same axes.  Curves are averaged over seeds with shaded ±1σ.
  2. Val-loss-over-time panel grid — same layout as above.
  3. Best-per-condition plot — single comparison using the best LR for each
     condition (warmup / no-warmup chosen independently).
  4. Final sharpness scatter — scatter of final sharpness vs. final val_loss,
     coloured by condition.

Usage (from project root):
    python experiments/warmup_flatness/compare.py [--settings gaussian heavy_tail sst2]

Options:
    --settings      which settings to process (default: gaussian heavy_tail sst2)
    --results_dir   directory with {setting}_runs.json (default: results/warmup_flatness)
    --plots_dir     where to save PNGs (default: plots/warmup_flatness)
    --smoothing     EMA smoothing coefficient for curves (default: 0.3)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        pass
    return records


def _extract_series(eval_records: List[dict], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for r in eval_records:
        if "outer_step" in r and key in r and r[key] is not None:
            xs.append(r["outer_step"])
            ys.append(float(r[key]))
    return xs, ys


def _ema_smooth(ys: List[float], alpha: float) -> List[float]:
    if not ys or alpha == 0:
        return list(ys)
    out = [ys[0]]
    for y in ys[1:]:
        out.append(alpha * out[-1] + (1 - alpha) * y)
    return out


def _interpolate_to_common(
    curves: List[Tuple[List[int], List[float]]]
) -> Tuple[List[int], List[List[float]]]:
    """Interpolate all (xs, ys) curves to a common step grid (union of all xs)."""
    if not curves:
        return [], []
    all_xs = sorted({x for xs, _ in curves for x in xs})
    interp_curves = []
    for xs, ys in curves:
        if not xs:
            interp_curves.append([float("nan")] * len(all_xs))
            continue
        # Linear interpolation for missing steps.
        arr = []
        for tx in all_xs:
            if tx in {x: y for x, y in zip(xs, ys)}:
                arr.append({x: y for x, y in zip(xs, ys)}[tx])
            else:
                # Find surrounding indices.
                lo, hi = None, None
                for i, x in enumerate(xs):
                    if x <= tx:
                        lo = i
                    if x >= tx and hi is None:
                        hi = i
                if lo is None:
                    arr.append(ys[0] if ys else float("nan"))
                elif hi is None:
                    arr.append(ys[-1])
                elif xs[lo] == xs[hi]:
                    arr.append(ys[lo])
                else:
                    t = (tx - xs[lo]) / (xs[hi] - xs[lo])
                    arr.append(ys[lo] + t * (ys[hi] - ys[lo]))
        interp_curves.append(arr)
    return all_xs, interp_curves


def _mean_std(
    values: List[float],
) -> Tuple[Optional[float], Optional[float]]:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None, None
    mean = sum(valid) / len(valid)
    var = sum((v - mean) ** 2 for v in valid) / max(1, len(valid) - 1)
    return mean, var ** 0.5


# ---------------------------------------------------------------------------
# Load and organise runs
# ---------------------------------------------------------------------------

def _load_setting(results_dir: str, setting: str) -> List[dict]:
    """Load run index (or merged partial indices) and attach sharpness + val_loss curves."""
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(results_dir, f"{setting}_runs*.json")))
    if not paths:
        print(f"[{setting}] run index not found in {results_dir}")
        return []

    run_index = []
    for p in paths:
        with open(p) as f:
            run_index.extend(json.load(f))
    if len(paths) > 1:
        print(f"  Merged {len(paths)} partial run index files for {setting}")

    for entry in run_index:
        log_dir = entry.get("log_dir", "")
        eval_records = _load_jsonl(os.path.join(log_dir, "eval_log.jsonl"))
        entry["sharpness_curve"] = _extract_series(eval_records, "sharpness")
        entry["val_loss_curve"]  = _extract_series(eval_records, "val_loss")
    return run_index


def _group_by_lr_and_condition(runs: List[dict]) -> Dict[float, Dict[bool, List[dict]]]:
    """Group runs by (lr, warmup_flag)."""
    groups: Dict[float, Dict[bool, List[dict]]] = defaultdict(lambda: {True: [], False: []})
    for r in runs:
        groups[r["lr"]][r["warmup"]].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel_grid(
    groups: Dict[float, Dict[bool, List[dict]]],
    metric: str,
    setting: str,
    plots_dir: str,
    smoothing: float = 0.3,
) -> None:
    """Multi-panel grid: columns = LR values, each panel has warmup vs no-warmup."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lr_values = sorted(groups.keys())
    n_cols = len(lr_values)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    COLORS = {"warmup": "#1f77b4", "no_warmup": "#ff7f0e"}
    LABELS = {"warmup": "Warmup", "no_warmup": "No warmup"}

    for ax, lr in zip(axes, lr_values):
        for warmup_flag, color_key in [(False, "no_warmup"), (True, "warmup")]:
            seed_runs = groups[lr][warmup_flag]
            curves = [r[f"{metric}_curve"] for r in seed_runs if r.get(f"{metric}_curve")]
            curves = [(xs, ys) for xs, ys in curves if xs]
            if not curves:
                continue

            common_xs, interp = _interpolate_to_common(curves)
            arr = np.array(interp)  # shape (n_seeds, n_steps)
            mean_curve = np.nanmean(arr, axis=0)
            std_curve  = np.nanstd(arr, axis=0)

            smooth_mean = _ema_smooth(mean_curve.tolist(), smoothing)

            ax.plot(common_xs, smooth_mean,
                    color=COLORS[color_key], label=LABELS[color_key], linewidth=1.8)
            ax.fill_between(
                common_xs,
                np.array(smooth_mean) - std_curve,
                np.array(smooth_mean) + std_curve,
                color=COLORS[color_key], alpha=0.18,
            )

        label = metric.replace("_", " ").title()
        ax.set_title(f"LR={lr:.2g}", fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{label} — {setting.replace('_', ' ').title()}  (τ fixed, mean±1σ over seeds)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, f"{setting}_{metric}_by_lr.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Panel grid saved to {out}")


def _plot_best_per_condition(
    groups: Dict[float, Dict[bool, List[dict]]],
    metric: str,
    setting: str,
    plots_dir: str,
    smoothing: float = 0.3,
) -> None:
    """Single plot: best-LR for each condition (warmup / no-warmup) independently."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def _best_lr_for_condition(warmup_flag: bool) -> Optional[float]:
        best_lr, best_loss = None, math.inf
        for lr, conds in groups.items():
            seed_runs = conds[warmup_flag]
            losses = [r["best_val_loss"] for r in seed_runs
                      if r.get("best_val_loss") is not None]
            if losses:
                mean_loss = sum(losses) / len(losses)
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_lr = lr
        return best_lr

    COLORS = {"warmup": "#1f77b4", "no_warmup": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(7, 4))

    for warmup_flag, color_key, label in [
        (False, "no_warmup", "No warmup (best LR)"),
        (True,  "warmup",    "Warmup (best LR)"),
    ]:
        best_lr = _best_lr_for_condition(warmup_flag)
        if best_lr is None:
            continue
        seed_runs = groups[best_lr][warmup_flag]
        curves = [(r[f"{metric}_curve"][0], r[f"{metric}_curve"][1])
                  for r in seed_runs if r.get(f"{metric}_curve") and r[f"{metric}_curve"][0]]
        if not curves:
            continue

        common_xs, interp = _interpolate_to_common(curves)
        arr = np.array(interp)
        mean_curve = np.nanmean(arr, axis=0)
        std_curve  = np.nanstd(arr, axis=0)
        smooth_mean = _ema_smooth(mean_curve.tolist(), smoothing)

        ax.plot(common_xs, smooth_mean, color=COLORS[color_key],
                label=f"{label} (lr={best_lr:.2g})", linewidth=2)
        ax.fill_between(
            common_xs,
            np.array(smooth_mean) - std_curve,
            np.array(smooth_mean) + std_curve,
            color=COLORS[color_key], alpha=0.18,
        )

    label_str = metric.replace("_", " ").title()
    ax.set_xlabel("Step")
    ax.set_ylabel(label_str)
    ax.set_title(
        f"{label_str}: best-per-condition — {setting.replace('_', ' ').title()}\n"
        f"(each condition uses its independently best LR)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(plots_dir, f"{setting}_{metric}_best_per_condition.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Best-per-condition plot saved to {out}")


def _plot_final_scatter(
    runs: List[dict],
    setting: str,
    plots_dir: str,
) -> None:
    """Scatter: final sharpness vs. final val_loss, coloured by warmup condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    COLORS = {False: "#ff7f0e", True: "#1f77b4"}
    LABELS = {False: "No warmup", True: "Warmup"}
    plotted = {False: False, True: False}

    for r in runs:
        xs, ys_sharp = r.get("sharpness_curve", ([], []))
        _, ys_val    = r.get("val_loss_curve", ([], []))
        if not xs or not ys_sharp or not ys_val:
            continue
        final_sharp = ys_sharp[-1]
        final_val   = ys_val[-1]
        w = r["warmup"]
        label = LABELS[w] if not plotted[w] else None
        ax.scatter(final_sharp, final_val, c=COLORS[w], alpha=0.6, s=40, label=label)
        plotted[w] = True

    ax.set_xlabel("Final sharpness (Δloss under perturbation)")
    ax.set_ylabel("Final val loss")
    ax.set_title(f"Sharpness vs. Val Loss — {setting.replace('_', ' ').title()}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(plots_dir, f"{setting}_sharpness_vs_val_loss.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Warmup flatness comparison plots")
    parser.add_argument("--settings", nargs="+", default=["gaussian", "heavy_tail", "sst2"])
    parser.add_argument("--results_dir", default="results/warmup_flatness")
    parser.add_argument("--plots_dir",   default="plots/warmup_flatness")
    parser.add_argument("--smoothing",   type=float, default=0.3)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    os.makedirs(args.plots_dir, exist_ok=True)

    all_summary = {}

    for setting in args.settings:
        print(f"\n=== {setting} ===")
        runs = _load_setting(args.results_dir, setting)
        if not runs:
            continue

        groups = _group_by_lr_and_condition(runs)

        # Panel grids per metric.
        _plot_panel_grid(groups, "sharpness", setting, args.plots_dir, args.smoothing)
        _plot_panel_grid(groups, "val_loss",  setting, args.plots_dir, args.smoothing)

        # Best-per-condition single comparison.
        _plot_best_per_condition(groups, "sharpness", setting, args.plots_dir, args.smoothing)
        _plot_best_per_condition(groups, "val_loss",  setting, args.plots_dir, args.smoothing)

        # Final sharpness scatter.
        _plot_final_scatter(runs, setting, args.plots_dir)

        # Numeric summary: at last sharpness step, warmup vs no-warmup.
        warmup_finals, nowarmup_finals = [], []
        for r in runs:
            xs, ys = r.get("sharpness_curve", ([], []))
            if not ys:
                continue
            if r["warmup"]:
                warmup_finals.append(ys[-1])
            else:
                nowarmup_finals.append(ys[-1])

        w_mean, w_std = _mean_std(warmup_finals)
        n_mean, n_std = _mean_std(nowarmup_finals)
        all_summary[setting] = {
            "warmup_final_sharpness_mean":     w_mean,
            "warmup_final_sharpness_std":      w_std,
            "no_warmup_final_sharpness_mean":  n_mean,
            "no_warmup_final_sharpness_std":   n_std,
            "warmup_sharper":                  (w_mean is not None and n_mean is not None
                                                and w_mean > n_mean),
        }
        if w_mean is not None:
            print(f"  Warmup final sharpness: {w_mean:.4g} ± {w_std:.4g}")
        else:
            print("  No warmup sharpness data")
        if n_mean is not None:
            print(f"  No-warmup final sharpness: {n_mean:.4g} ± {n_std:.4g}")

    if all_summary:
        out = os.path.join(args.results_dir, "flatness_summary.json")
        with open(out, "w") as f:
            json.dump(all_summary, f, indent=2)
        print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
