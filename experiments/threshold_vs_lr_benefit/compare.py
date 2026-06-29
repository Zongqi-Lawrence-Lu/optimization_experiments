"""Load grid results and produce heatmaps for threshold_vs_lr_benefit.

For each setting (gaussian, heavy_tail, sst2) that has a completed grid JSON,
produces:
  - A heatmap of mean best_val_loss over the (tau × lr) grid
  - A 1-D sensitivity plot showing the marginal effect of each axis
  - A summary JSON with the best cell and axis-improvement ratios

Run from project root after grid_sweep.py has completed:
    python experiments/threshold_vs_lr_benefit/compare.py [--settings gaussian heavy_tail sst2]

Options:
    --settings      which settings to plot (default: all three)
    --results_dir   directory with {setting}_grid.json files
                    (default: results/threshold_vs_lr_benefit)
    --plots_dir     where to write PNGs (default: plots/threshold_vs_lr_benefit)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_grid(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_and_merge_grids(results_dir: str, setting: str) -> Optional[dict]:
    """Load all partial grid files for a setting and merge their cells.

    Matches {setting}_grid.json and {setting}_grid_*.json so that both single-
    file runs and split sub-job runs are handled transparently.  Falls back to
    reconstructing the grid from individual trial summary.json files if no grid
    JSON exists (e.g., job timed out before writing the assembled result).
    """
    paths = sorted(glob.glob(os.path.join(results_dir, f"{setting}_grid*.json")))
    if paths:
        docs = [_load_grid(p) for p in paths]
        all_tau = sorted({t for d in docs for t in d["tau_values"]})
        cells_from_files = [c for d in docs for c in d["cells"]]

        # Supplement with any trial dirs whose tau isn't covered by the grid files
        # (happens when some sub-jobs timed out before writing their grid JSON).
        supplemental = _reconstruct_grid_from_trials(results_dir, setting)
        if supplemental:
            covered_tau = set(all_tau)
            extra_cells = [c for c in supplemental["cells"]
                           if c["tau"] not in covered_tau and c["mean_best_val_loss"] is not None]
            extra_tau   = sorted({c["tau"] for c in extra_cells})
            if extra_tau:
                print(f"  Supplementing grid files with {len(extra_cells)} cells from "
                      f"trial dirs for tau={extra_tau}")
                cells_from_files.extend(extra_cells)
                all_tau = sorted(set(all_tau) | set(extra_tau))

        merged = {
            "setting": setting,
            "warmup_fraction": docs[0]["warmup_fraction"],
            "n_seeds": docs[0]["n_seeds"],
            "tau_values": all_tau,
            "lr_values": docs[0]["lr_values"],
            "cells": cells_from_files,
        }
        if len(paths) > 1:
            print(f"  Merged {len(paths)} partial grid files for {setting}")
        return merged

    return _reconstruct_grid_from_trials(results_dir, setting)


def _reconstruct_grid_from_trials(results_dir: str, setting: str) -> Optional[dict]:
    """Reconstruct a grid doc from individual trial summary.json files.

    Used when the assembled grid JSON was never written (e.g., SLURM timeout).
    Each trial dir contains config.yaml (with the actual tau/lr values) and
    summary.json (with val_loss_best).
    """
    import re
    import yaml

    trial_pattern = re.compile(rf"thr_{re.escape(setting)}_t\d+_l\d+_s\d+$")
    trial_data = []

    for trial_dir in Path(results_dir).iterdir():
        if not trial_pattern.match(trial_dir.name):
            continue
        summary_path = trial_dir / "summary.json"
        config_path  = trial_dir / "config.yaml"
        if not summary_path.exists() or not config_path.exists():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            tau = cfg["inner_optimizer"]["clipping"]["upper"]
            lr  = cfg["inner_optimizer"]["lr"]
            val_loss = summary.get("val_loss_best", summary.get("best_val_loss"))
            trial_data.append({"tau": tau, "lr": lr, "val_loss_best": val_loss})
        except Exception:
            continue

    if not trial_data:
        return None

    tau_values = sorted({t["tau"] for t in trial_data})
    lr_values  = sorted({t["lr"]  for t in trial_data})

    cell_seeds: Dict = defaultdict(list)
    for t in trial_data:
        if t["val_loss_best"] is not None:
            cell_seeds[(t["tau"], t["lr"])].append(t["val_loss_best"])

    cells = []
    for tau in tau_values:
        for lr in lr_values:
            seed_vals = cell_seeds.get((tau, lr), [])
            if seed_vals:
                mean = sum(seed_vals) / len(seed_vals)
                var  = sum((v - mean) ** 2 for v in seed_vals) / max(1, len(seed_vals) - 1)
                std  = var ** 0.5
            else:
                mean = std = None
            cells.append({
                "tau": tau,
                "lr": lr,
                "mean_best_val_loss": mean,
                "std_best_val_loss": std,
                "n_valid_seeds": len(seed_vals),
                "seed_values": seed_vals,
            })

    print(f"  Reconstructed grid from {len(trial_data)} trial dirs for {setting} "
          f"(no grid JSON found — job may have timed out)")
    return {
        "setting": setting,
        "warmup_fraction": 0.1,
        "n_seeds": 5,
        "tau_values": tau_values,
        "lr_values": lr_values,
        "cells": cells,
    }


def _pivot(cells: List[dict], tau_values: List[float], lr_values: List[float]):
    """Return a 2-D array (n_tau × n_lr) of mean_best_val_loss values."""
    import numpy as np
    grid = np.full((len(tau_values), len(lr_values)), float("nan"))
    tau_idx = {t: i for i, t in enumerate(tau_values)}
    lr_idx  = {l: i for i, l in enumerate(lr_values)}
    for cell in cells:
        ti = tau_idx.get(cell["tau"])
        li = lr_idx.get(cell["lr"])
        if ti is not None and li is not None and cell["mean_best_val_loss"] is not None:
            grid[ti, li] = cell["mean_best_val_loss"]
    return grid


def plot_heatmap(doc: dict, plots_dir: str) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setting       = doc["setting"]
    tau_values    = doc["tau_values"]
    lr_values     = doc["lr_values"]
    warmup_frac   = doc["warmup_fraction"]
    cells         = doc["cells"]

    grid = _pivot(cells, tau_values, lr_values)

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(max(8, len(lr_values) * 1.4), max(6, len(tau_values) * 1.0)))

    vmin = float(np.nanmin(grid)) if not np.all(np.isnan(grid)) else 0.0
    vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Mean best val loss")

    ax.set_xticks(range(len(lr_values)))
    ax.set_xticklabels([f"{v:.1e}" for v in lr_values], rotation=45, ha="right")
    ax.set_yticks(range(len(tau_values)))
    ax.set_yticklabels([f"{v:.2g}" for v in tau_values])
    ax.set_xlabel("Terminal LR (η)")
    ax.set_ylabel("Clipping threshold (τ)")
    ax.set_title(
        f"Threshold vs. LR benefit — {setting.replace('_', ' ').title()}\n"
        f"(warmup fraction={warmup_frac}, mean over {doc['n_seeds']} seeds)"
    )

    for ti in range(len(tau_values)):
        for li in range(len(lr_values)):
            val = grid[ti, li]
            if not np.isnan(val):
                text_color = "white" if val < (vmin + vmax) / 2 else "black"
                ax.text(li, ti, f"{val:.3g}", ha="center", va="center",
                        fontsize=7, color=text_color)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, f"{setting}_heatmap.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {out}")


def plot_sensitivity(doc: dict, plots_dir: str) -> None:
    """1-D sensitivity: marginal effect of tau vs lr (averaging over the other axis)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setting    = doc["setting"]
    tau_values = doc["tau_values"]
    lr_values  = doc["lr_values"]
    cells      = doc["cells"]

    grid = _pivot(cells, tau_values, lr_values)

    tau_means = np.nanmean(grid, axis=1)   # mean over lr axis
    lr_means  = np.nanmean(grid, axis=0)   # mean over tau axis

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(tau_values, tau_means, marker="o", color="#1f77b4")
    axes[0].set_xlabel("Clipping threshold τ")
    axes[0].set_ylabel("Mean val loss (avg over LR axis)")
    axes[0].set_title("Effect of τ (marginal)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lr_values, lr_means, marker="s", color="#ff7f0e")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Terminal LR η")
    axes[1].set_ylabel("Mean val loss (avg over τ axis)")
    axes[1].set_title("Effect of η (marginal)")
    axes[1].grid(True, alpha=0.3)

    # Annotate with improvement range on each axis.
    tau_range = float(np.nanmax(tau_means) - np.nanmin(tau_means))
    lr_range  = float(np.nanmax(lr_means)  - np.nanmin(lr_means))
    fig.suptitle(
        f"{setting.replace('_', ' ').title()}:  "
        f"τ-axis improvement = {tau_range:.4f}  |  "
        f"η-axis improvement = {lr_range:.4f}",
        fontsize=11,
    )

    plt.tight_layout()
    out = os.path.join(plots_dir, f"{setting}_sensitivity.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Sensitivity plot saved to {out}")
    return tau_range, lr_range


def summarise(doc: dict, tau_range: float, lr_range: float) -> dict:
    """Return a dict with the best cell and axis-improvement ratios."""
    import numpy as np

    setting    = doc["setting"]
    tau_values = doc["tau_values"]
    lr_values  = doc["lr_values"]
    cells      = doc["cells"]

    grid = _pivot(cells, tau_values, lr_values)
    best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
    best_tau = tau_values[best_idx[0]]
    best_lr  = lr_values[best_idx[1]]
    best_val = float(grid[best_idx])

    return {
        "setting": setting,
        "best_tau": best_tau,
        "best_lr": best_lr,
        "best_mean_val_loss": best_val,
        "tau_axis_improvement": tau_range,
        "lr_axis_improvement": lr_range,
        "tau_benefit_ratio": tau_range / (lr_range + 1e-12),
        "interpretation": (
            "tau axis gives MORE benefit" if tau_range > lr_range
            else "lr axis gives MORE benefit"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Heatmap comparison for threshold_vs_lr_benefit")
    parser.add_argument("--settings", nargs="+", default=["gaussian", "heavy_tail", "sst2"])
    parser.add_argument("--results_dir", default="results/threshold_vs_lr_benefit")
    parser.add_argument("--plots_dir",   default="plots/threshold_vs_lr_benefit")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    os.makedirs(args.plots_dir, exist_ok=True)

    all_summaries = {}
    for setting in args.settings:
        doc = _load_and_merge_grids(args.results_dir, setting)
        if doc is None:
            print(f"[{setting}] no grid JSON found in {args.results_dir}, skipping.")
            continue

        print(f"\n=== {setting} ===")
        plot_heatmap(doc, args.plots_dir)
        tau_range, lr_range = plot_sensitivity(doc, args.plots_dir)
        summary = summarise(doc, tau_range, lr_range)
        all_summaries[setting] = summary
        print(f"  Best cell: tau={summary['best_tau']}, lr={summary['best_lr']:.3g}"
              f"  val={summary['best_mean_val_loss']:.4g}")
        print(f"  τ-axis improvement: {tau_range:.4f}  |  η-axis improvement: {lr_range:.4f}")
        print(f"  → {summary['interpretation']}")

    if all_summaries:
        out = os.path.join(args.results_dir, "comparison_summary.json")
        with open(out, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
