"""Overlay multiple training runs on shared axes for head-to-head comparison.

Usage:
    from framework.plotting.comparison_curves import plot_comparison, save_comparison_summary
    plot_comparison(
        runs=[("Global", "outputs/run_a"), ("Layerwise", "outputs/run_b")],
        output_path="plots/comparison.png",
        metrics=["val_loss", "grad_norm_raw"],
    )

Metrics can come from eval_log.jsonl (e.g. val_loss) or step_log.jsonl
(e.g. grad_norm_raw, clipped_fraction, learning_rate). The function tries
eval_log first and falls back to aggregating step_log.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
LINESTYLES = ["-", "--", "-.", ":"]

PANEL_LABELS = {
    "train_loss": "Train Loss",
    "learning_rate": "Learning Rate",
    "val_loss": "Val Loss",
    "grad_norm_raw": "Grad Norm (raw)",
    "grad_norm_clipped": "Grad Norm (clipped)",
    "clipped_fraction": "Clipped Fraction",
    "param_distance": "Param Distance ‖w*−ŵ‖",
}


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


def _extract_eval_series(records: List[dict], key: str) -> Tuple[List, List]:
    xs, ys = [], []
    for r in records:
        if "outer_step" in r and key in r and r[key] is not None:
            xs.append(r["outer_step"])
            ys.append(r[key])
    return xs, ys


def _extract_step_series(records: List[dict], key: str) -> Tuple[List, List]:
    """Aggregate a step_log key by outer_step (node 0, inner_step 0)."""
    agg: Dict[int, List[float]] = {}
    for r in records:
        if r.get("inner_step", 0) == 0 and r.get("node_id", 0) == 0:
            s = r.get("outer_step")
            if s is not None and key in r and r[key] is not None:
                agg.setdefault(s, []).append(r[key])
    xs = sorted(agg)
    ys = [float(np.mean(agg[x])) for x in xs]
    return xs, ys


def _smooth(ys: List[float], alpha: float) -> List[float]:
    if alpha == 0 or not ys:
        return ys
    out = [ys[0]]
    for y in ys[1:]:
        out.append(alpha * out[-1] + (1 - alpha) * y)
    return out


def _load_run(log_dir: str) -> Dict[str, List[dict]]:
    log_dir = Path(log_dir)
    return {
        "step": _load_jsonl(str(log_dir / "step_log.jsonl")),
        "eval": _load_jsonl(str(log_dir / "eval_log.jsonl")),
    }


def plot_comparison(
    runs: List[Tuple[str, str]],
    output_path: str,
    metrics: Optional[List[str]] = None,
    smoothing: float = 0.0,
    title: str = "",
    include_train_loss: bool = True,
    figsize_per_panel: Tuple[int, int] = (6, 4),
) -> None:
    """Overlay training curves from multiple runs on shared axes.

    Parameters
    ----------
    runs : list of (label, log_dir) pairs
    output_path : where to save the PNG
    metrics : metric keys to plot (can include both eval and step keys)
    smoothing : EMA smoothing for step-log series (0 = none)
    title : overall figure title
    include_train_loss : whether to add a train_loss panel
    figsize_per_panel : (width, height) per subplot in inches
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to use plotting utilities.")

    if metrics is None:
        metrics = ["val_loss"]

    panels = (["train_loss"] if include_train_loss else []) + list(metrics)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]))
    if n == 1:
        axes = [axes]

    for run_idx, (label, log_dir) in enumerate(runs):
        color = COLORS[run_idx % len(COLORS)]
        ls = LINESTYLES[run_idx % len(LINESTYLES)]
        data = _load_run(log_dir)
        step_records = data["step"]
        eval_records = data["eval"]

        for ax, panel in zip(axes, panels):
            if panel == "train_loss":
                xs, ys = _extract_step_series(step_records, "train_loss")
                ys_s = _smooth(ys, smoothing)
                ax.plot(xs, ys, alpha=0.15, color=color)
                ax.plot(xs, ys_s, color=color, linestyle=ls, label=label, linewidth=1.8)
            else:
                # Try eval_log first, fall back to step_log
                xs, ys = _extract_eval_series(eval_records, panel)
                if not xs:
                    xs, ys = _extract_step_series(step_records, panel)
                    ys = _smooth(ys, smoothing)
                    ax.plot(xs, ys, color=color, linestyle=ls, label=label, linewidth=1.8)
                else:
                    ax.plot(xs, ys, color=color, linestyle=ls, marker="o",
                            markersize=3, label=label, linewidth=1.8)

    for ax, panel in zip(axes, panels):
        ax.set_xlabel("Step")
        ax.set_ylabel(PANEL_LABELS.get(panel, panel))
        ax.set_title(PANEL_LABELS.get(panel, panel.replace("_", " ").title()))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {output_path}")


def save_comparison_summary(
    runs: List[Tuple[str, str]],
    output_path: str,
) -> None:
    """Write a JSON file with best/final val_loss per run."""
    summary = {}
    for label, log_dir in runs:
        data = _load_run(log_dir)
        eval_records = data["eval"]
        val_losses = [r["val_loss"] for r in eval_records if "val_loss" in r]
        summary[label] = {
            "best_val_loss": float(min(val_losses)) if val_losses else None,
            "final_val_loss": float(val_losses[-1]) if val_losses else None,
            "log_dir": str(log_dir),
        }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison summary saved to {output_path}")
