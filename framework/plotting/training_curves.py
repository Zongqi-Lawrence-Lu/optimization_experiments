"""Plot training and evaluation curves from run log files.

Reads step_log.jsonl and eval_log.jsonl produced by RunLogger and generates:
- Loss curve (train + val + test)
- Accuracy / custom metric curves
- Gradient norm curve

Usage (CLI):
    python -m framework.plotting.training_curves \\
        --log_dir outputs/my_run \\
        --output plots/training_curves/my_run.png \\
        --metrics val_loss accuracy param_distance
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_series(records: List[dict], x_key: str, y_key: str) -> Tuple[List, List]:
    xs, ys = [], []
    for r in records:
        if x_key in r and y_key in r and r[y_key] is not None:
            xs.append(r[x_key])
            ys.append(r[y_key])
    return xs, ys


def plot_training_curves(
    log_dir: str,
    output_path: str,
    metrics: Optional[List[str]] = None,
    smoothing: float = 0.0,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Plot training curves and save to output_path.

    Parameters
    ----------
    log_dir     : path to a run directory produced by RunLogger
    output_path : where to save the PNG
    metrics     : list of eval metric keys to plot (default: all found in eval_log)
    smoothing   : exponential smoothing factor in [0, 1) for step_log series (0 = no smoothing)
    figsize     : figure size in inches
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to use plotting utilities.")

    log_dir = Path(log_dir)
    step_log_path = log_dir / "step_log.jsonl"
    eval_log_path = log_dir / "eval_log.jsonl"

    step_records = _load_jsonl(str(step_log_path)) if step_log_path.exists() else []
    eval_records = _load_jsonl(str(eval_log_path)) if eval_log_path.exists() else []

    # Discover metric keys from eval_log
    eval_keys = set()
    for r in eval_records:
        eval_keys.update(k for k in r if k != "outer_step")
    if metrics is not None:
        plot_keys = [k for k in metrics if any(k in r for r in eval_records)]
    else:
        plot_keys = sorted(eval_keys)

    n_plots = 1 + len(plot_keys)  # train_loss + each eval metric
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0] * n_plots // max(1, n_plots), figsize[1]))
    if n_plots == 1:
        axes = [axes]

    def _smooth(ys: List[float], alpha: float) -> List[float]:
        if alpha == 0 or not ys:
            return ys
        out = [ys[0]]
        for y in ys[1:]:
            out.append(alpha * out[-1] + (1 - alpha) * y)
        return out

    # --- Train loss panel ---
    ax = axes[0]
    if step_records:
        # Average across inner_step == 0 and node_id == 0 to get one value per outer_step
        step_map: Dict[int, List[float]] = {}
        for r in step_records:
            if r.get("inner_step", 0) == 0 and r.get("node_id", 0) == 0:
                s = r["outer_step"]
                if "train_loss" in r:
                    step_map.setdefault(s, []).append(r["train_loss"])
        xs = sorted(step_map)
        ys = [np.mean(step_map[x]) for x in xs]
        ys_smooth = _smooth(ys, smoothing)
        ax.plot(xs, ys, alpha=0.3, color="steelblue", label="train_loss (raw)")
        ax.plot(xs, ys_smooth, color="steelblue", label="train_loss (smooth)")
    ax.set_xlabel("Outer step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Eval metric panels ---
    for ax, key in zip(axes[1:], plot_keys):
        xs, ys = _extract_series(eval_records, "outer_step", key)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=key)
        ax.set_xlabel("Outer step")
        ax.set_ylabel(key)
        ax.set_title(key.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(log_dir.name, fontsize=12, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_gradient_norms(
    log_dir: str,
    output_path: str,
    top_layers: int = 5,
) -> None:
    """Plot per-layer gradient norms over training steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to use plotting utilities.")

    log_dir = Path(log_dir)
    grad_log_path = log_dir / "gradient_stats.jsonl"
    if not grad_log_path.exists():
        print(f"No gradient_stats.jsonl found in {log_dir}")
        return

    records = _load_jsonl(str(grad_log_path))
    # Group by layer name
    layer_data: Dict[str, Tuple[List, List]] = {}
    for r in records:
        name = r.get("layer_name", "unknown")
        step = r.get("outer_step", 0)
        norm = r.get("grad_l2_norm", 0.0)
        if name not in layer_data:
            layer_data[name] = ([], [])
        layer_data[name][0].append(step)
        layer_data[name][1].append(norm)

    # Pick layers with highest mean norm
    mean_norms = {name: np.mean(norms) for name, (_, norms) in layer_data.items()}
    top = sorted(mean_norms, key=mean_norms.get, reverse=True)[:top_layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in top:
        steps, norms = layer_data[name]
        ax.plot(steps, norms, label=name[:40])
    ax.set_xlabel("Outer step")
    ax.set_ylabel("L2 gradient norm")
    ax.set_title("Per-layer Gradient Norms")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Gradient norm plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from a run log directory")
    parser.add_argument("--log_dir", required=True, help="Path to run directory")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--metrics", nargs="*", default=None, help="Eval metric keys to plot")
    parser.add_argument("--smoothing", type=float, default=0.8)
    args = parser.parse_args()
    plot_training_curves(args.log_dir, args.output, args.metrics, args.smoothing)


if __name__ == "__main__":
    main()
