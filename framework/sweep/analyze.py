"""Standalone sweep result analysis.

Reads results_summary.csv and prints:
- Ranked trial table
- Per-axis sensitivity analysis
- Pair plots saved as PNG
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd


def analyze(
    results_csv: str,
    primary_metric: str = "best_val_loss",
    output_dir: Optional[str] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Load and analyze sweep results.

    Parameters
    ----------
    results_csv : path to results_summary.csv
    primary_metric : column name to rank by (lower is better assumed)
    output_dir : where to save PNG plots; defaults to same dir as csv
    top_n : number of top trials to display

    Returns
    -------
    DataFrame sorted by primary_metric.
    """
    df = pd.read_csv(results_csv)
    output_dir = output_dir or os.path.dirname(results_csv)
    os.makedirs(output_dir, exist_ok=True)

    if primary_metric not in df.columns:
        available = [c for c in df.columns if "loss" in c or "metric" in c]
        if available:
            primary_metric = available[0]
            print(f"[analyze] primary_metric not found, using '{primary_metric}'")
        else:
            print(f"[analyze] primary_metric '{primary_metric}' not in columns: {df.columns.tolist()}")
            return df

    df_sorted = df.sort_values(primary_metric, ascending=True).reset_index(drop=True)

    print(f"\n=== Top {top_n} trials by {primary_metric} ===")
    print(df_sorted.head(top_n).to_string(index=False))

    # Identify swept hyperparameter columns (everything that's not a metric or trial_id)
    metric_cols = {c for c in df.columns if any(m in c for m in ["loss", "step", "best", "final"])}
    hp_cols = [c for c in df.columns if c != "trial_id" and c not in metric_cols]

    # Per-axis sensitivity
    print("\n=== Per-axis sensitivity (mean primary metric per axis value) ===")
    sensitivity_rows = []
    for col in hp_cols:
        if df[col].nunique() <= 1:
            continue
        grouped = df.groupby(col)[primary_metric].mean().reset_index()
        grouped.columns = ["value", "mean_metric"]
        grouped.insert(0, "axis", col)
        print(f"\n  {col}:")
        print(grouped.to_string(index=False))
        sensitivity_rows.append(grouped)

    # Pair plots for top axis pairs
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from itertools import combinations

        numeric_hp_cols = [c for c in hp_cols if pd.api.types.is_numeric_dtype(df[c])]
        pairs = list(combinations(numeric_hp_cols[:5], 2))  # at most top-5 axes
        if pairs:
            n_pairs = len(pairs)
            fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4))
            if n_pairs == 1:
                axes = [axes]
            for ax, (col_x, col_y) in zip(axes, pairs):
                scatter = ax.scatter(
                    df[col_x],
                    df[col_y],
                    c=df[primary_metric],
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.colorbar(scatter, ax=ax, label=primary_metric)
                ax.set_xlabel(col_x)
                ax.set_ylabel(col_y)
                ax.set_title(f"{col_x} vs {col_y}")
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "pair_plots.png")
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"\nPair plots saved to {plot_path}")
    except ImportError:
        print("[analyze] matplotlib not available; skipping pair plots")

    return df_sorted


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("results_csv", help="Path to results_summary.csv")
    parser.add_argument("--primary_metric", default="best_val_loss")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()
    analyze(args.results_csv, args.primary_metric, args.output_dir, args.top_n)


if __name__ == "__main__":
    main()
