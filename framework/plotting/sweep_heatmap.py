"""Hyperparameter sweep visualization: heatmaps and sensitivity plots.

Usage (CLI):
    # 2-D heatmap of two hyperparameters vs a metric
    python -m framework.plotting.sweep_heatmap \\
        --csv outputs/sweeps/my_sweep/results_summary.csv \\
        --x inner_optimizer.lr \\
        --y inner_optimizer.clipping.upper \\
        --metric best_val_loss \\
        --output plots/sweep_heatmaps/lr_vs_clip.png

    # 1-D sensitivity plots for all swept axes
    python -m framework.plotting.sweep_heatmap \\
        --csv outputs/sweeps/my_sweep/results_summary.csv \\
        --metric best_val_loss \\
        --sensitivity \\
        --output plots/sweep_heatmaps/sensitivity.png
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np


def plot_sweep_heatmap(
    results_csv: str,
    x_axis: str,
    y_axis: str,
    metric: str = "best_val_loss",
    output_path: str = "heatmap.png",
    log_x: bool = False,
    log_y: bool = False,
    lower_is_better: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    annotate: bool = True,
) -> None:
    """Plot a 2-D heatmap of metric values over a grid of two hyperparameters.

    Parameters
    ----------
    results_csv     : path to results_summary.csv from a sweep run
    x_axis          : column name for the x-axis hyperparameter
    y_axis          : column name for the y-axis hyperparameter
    metric          : column name for the performance metric (lower-is-better by default)
    output_path     : where to save the PNG
    log_x, log_y    : use log scale for the axis tick labels
    lower_is_better : if False, reverse the colour map
    annotate        : write metric value in each cell
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        raise ImportError("Install matplotlib and pandas to use plotting utilities.")

    df = pd.read_csv(results_csv)

    if x_axis not in df.columns:
        raise ValueError(f"Column '{x_axis}' not found in {results_csv}")
    if y_axis not in df.columns:
        raise ValueError(f"Column '{y_axis}' not found in {results_csv}")
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in {results_csv}")

    pivot = df.pivot_table(index=y_axis, columns=x_axis, values=metric, aggfunc="mean")
    x_vals = pivot.columns.tolist()
    y_vals = pivot.index.tolist()
    data = pivot.values  # shape (n_y, n_x)

    cmap = "viridis_r" if lower_is_better else "viridis"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(
        [f"{v:.2e}" if isinstance(v, float) else str(v) for v in x_vals],
        rotation=45, ha="right",
    )
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(
        [f"{v:.2e}" if isinstance(v, float) else str(v) for v in y_vals]
    )
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{metric}\n({x_axis} vs {y_axis})")

    if annotate:
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3g}", ha="center", va="center",
                            fontsize=7, color="white" if val < np.nanmean(data) else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_path}")


def plot_sensitivity(
    results_csv: str,
    metric: str = "best_val_loss",
    output_path: str = "sensitivity.png",
    lower_is_better: bool = True,
    figsize: Tuple[int, int] = (14, 4),
) -> None:
    """Plot mean metric value vs each hyperparameter axis (1-D sensitivity).

    One subplot per swept hyperparameter.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        raise ImportError("Install matplotlib and pandas to use plotting utilities.")

    df = pd.read_csv(results_csv)
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in {results_csv}")

    metric_cols = {c for c in df.columns if any(m in c for m in ["loss", "step", "best", "final", "trial_id"])}
    hp_cols = [c for c in df.columns if c not in metric_cols and pd.api.types.is_numeric_dtype(df[c])]

    if not hp_cols:
        print("No numeric hyperparameter columns found.")
        return

    n = len(hp_cols)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]), squeeze=False)
    axes = axes[0]

    for ax, col in zip(axes, hp_cols):
        grouped = df.groupby(col)[metric].mean().reset_index()
        xs = grouped[col].tolist()
        ys = grouped[metric].tolist()
        ax.plot(xs, ys, marker="o")
        # Highlight the best value
        best_idx = int(np.argmin(ys)) if lower_is_better else int(np.argmax(ys))
        ax.axvline(xs[best_idx], color="red", linestyle="--", alpha=0.5, label=f"best={xs[best_idx]:.3g}")
        ax.set_xlabel(col)
        ax.set_ylabel(f"mean {metric}")
        ax.set_title(col)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sensitivity analysis — {metric}", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Sensitivity plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize hyperparameter sweep results")
    parser.add_argument("--csv", required=True, help="Path to results_summary.csv")
    parser.add_argument("--metric", default="best_val_loss")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--x", default=None, help="X-axis column for heatmap")
    parser.add_argument("--y", default=None, help="Y-axis column for heatmap")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Plot 1-D sensitivity instead of 2-D heatmap")
    args = parser.parse_args()

    if args.sensitivity or (args.x is None and args.y is None):
        plot_sensitivity(args.csv, args.metric, args.output)
    else:
        if args.x is None or args.y is None:
            parser.error("--x and --y are both required for heatmap mode")
        plot_sweep_heatmap(args.csv, args.x, args.y, args.metric, args.output)


if __name__ == "__main__":
    main()
