"""Two-stage coarse-then-fine hyperparameter sweep.

Stage 1: Run a coarse grid/random search over a wide range of values.
Stage 2: Extract the best configuration from stage 1, then build a refined
         grid centred on the best value for each numeric axis and re-run.

Usage (CLI):
    python -m framework.sweep.two_stage configs/two_stage_sweep.yaml outputs/sweeps/ts_run

Two-stage sweep YAML format:
    base_config: configs/centralized_sgd.yaml
    parallel_jobs: 4

    stage1:
      search: grid          # "grid" | "random"
      max_trials: null      # limit for random search
      axes:
        - field: inner_optimizer.lr
          values: [0.001, 0.01, 0.1]
          log_scale: true
        - field: inner_optimizer.clipping.upper
          values: [0.1, 1.0, 5.0]

    stage2:
      search: grid
      n_values: 3           # number of values per axis in the fine grid
      zoom_factor: 0.5      # fraction of the coarse spacing to use for fine range
      axes:                 # optional: override which axes to refine (defaults to all numeric axes from stage1)
        - field: inner_optimizer.lr
          log_scale: true
        - field: inner_optimizer.clipping.upper
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from framework.configs import TrainingConfig, load_config, save_config, _dict_to_training_config
from framework.sweep.grid import expand_grid, write_grid_trials
from framework.sweep.random import write_random_trials
from framework.sweep.runner import _run_trial


# ---------------------------------------------------------------------------
# Stage-2 axis builder
# ---------------------------------------------------------------------------

def _build_fine_axes(
    coarse_axes: List[Dict[str, Any]],
    best_hp: Dict[str, Any],
    stage2_cfg: dict,
) -> List[Dict[str, Any]]:
    """Build stage-2 axes centred on best_hp values from stage 1.

    For numeric axes, generates n_values candidates symmetrically around the
    best stage-1 value.  Non-numeric axes keep their original value list.
    """
    n_values = stage2_cfg.get("n_values", 3)
    zoom_factor = stage2_cfg.get("zoom_factor", 0.5)
    # axes override: if provided, restrict refinement to listed fields
    axis_overrides = {ax["field"]: ax for ax in stage2_cfg.get("axes", [])}

    fine_axes = []
    for ax in coarse_axes:
        field = ax["field"]
        values = ax["values"]
        log_scale = ax.get("log_scale", False) or axis_overrides.get(field, {}).get("log_scale", False)

        if field not in best_hp or not all(isinstance(v, (int, float)) for v in values):
            # Keep original discrete list for non-numeric axes
            fine_axes.append(ax)
            continue

        best_val = best_hp[field]
        sorted_vals = sorted(float(v) for v in values)

        if log_scale and all(v > 0 for v in sorted_vals):
            log_vals = [math.log(v) for v in sorted_vals]
            best_log = math.log(max(best_val, 1e-30))
            # Spacing from adjacent values
            if len(log_vals) > 1:
                spacing = (max(log_vals) - min(log_vals)) / (len(log_vals) - 1)
            else:
                spacing = abs(best_log) * 0.5 or 1.0
            half = zoom_factor * spacing
            fine_log_vals = [best_log + half * (i - (n_values - 1) / 2) for i in range(n_values)]
            fine_values = [round(math.exp(v), 10) for v in fine_log_vals]
        else:
            if len(sorted_vals) > 1:
                spacing = (max(sorted_vals) - min(sorted_vals)) / (len(sorted_vals) - 1)
            else:
                spacing = abs(best_val) * 0.5 or 1.0
            half = zoom_factor * spacing
            fine_values = [best_val + half * (i - (n_values - 1) / 2) for i in range(n_values)]
            # Clamp to positive when all coarse values are positive (e.g. clipping thresholds).
            # Linear zoom around a small best value can otherwise produce negatives.
            if all(v > 0 for v in sorted_vals):
                eps = min(sorted_vals) * 0.1
                fine_values = [max(v, eps) for v in fine_values]

        # Remove duplicates and preserve order
        seen = set()
        deduped = []
        for v in fine_values:
            key = round(v, 12)
            if key not in seen:
                seen.add(key)
                deduped.append(v)

        fine_axes.append({
            "field": field,
            "values": deduped,
            "log_scale": log_scale,
        })

    return fine_axes


# ---------------------------------------------------------------------------
# Dispatch helper (mirrors runner.py)
# ---------------------------------------------------------------------------

def _dispatch_trials(
    trial_records: List[Tuple[str, str, Dict]],
    parallel_jobs: int,
) -> List[dict]:
    results = []
    with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        future_map = {
            executor.submit(_run_trial, config_path): (trial_id, hp)
            for trial_id, config_path, hp in trial_records
        }
        for future in as_completed(future_map):
            trial_id, hp = future_map[future]
            try:
                metrics = future.result()
            except Exception as exc:
                print(f"Trial {trial_id} failed: {exc}")
                metrics = {}
            results.append({"trial_id": trial_id, **hp, **metrics})
    return results


def _write_csv(results: List[dict], path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Main two-stage runner
# ---------------------------------------------------------------------------

def run_two_stage_sweep(sweep_config_path: str, sweep_dir: str) -> None:
    """Execute a two-stage coarse-then-fine hyperparameter sweep.

    Writes results to:
        <sweep_dir>/stage1/results_summary.csv
        <sweep_dir>/stage1/best_config.yaml
        <sweep_dir>/stage2/results_summary.csv
        <sweep_dir>/stage2/best_config.yaml
    """
    with open(sweep_config_path) as f:
        cfg = yaml.safe_load(f)

    base_config = load_config(cfg["base_config"])
    parallel_jobs = cfg.get("parallel_jobs", 1)
    primary_metric = cfg.get("primary_metric", "best_val_loss")
    lower_is_better = cfg.get("lower_is_better", True)

    stage1_cfg = cfg["stage1"]
    stage2_cfg = cfg["stage2"]

    # ---- Stage 1 ----
    stage1_dir = os.path.join(sweep_dir, "stage1")
    os.makedirs(stage1_dir, exist_ok=True)
    print(f"\n=== Stage 1: coarse sweep ({stage1_cfg.get('search', 'grid')}) ===")

    if stage1_cfg.get("search", "grid") == "random":
        n = stage1_cfg.get("max_trials", 20)
        s1_records = write_random_trials(base_config, stage1_cfg["axes"], stage1_dir, max_trials=n, seed=base_config.seed)
    else:
        s1_records = write_grid_trials(base_config, stage1_cfg["axes"], stage1_dir)
        if stage1_cfg.get("max_trials"):
            s1_records = s1_records[:stage1_cfg["max_trials"]]

    s1_results = _dispatch_trials(s1_records, parallel_jobs)
    s1_csv = os.path.join(stage1_dir, "results_summary.csv")
    _write_csv(s1_results, s1_csv)
    print(f"Stage 1 complete. Results → {s1_csv}")

    if not s1_results:
        print("Stage 1 produced no results; aborting stage 2.")
        return

    # Best stage-1 config
    if lower_is_better:
        best_s1 = min(s1_results, key=lambda r: r.get(primary_metric, float("inf")))
    else:
        best_s1 = max(s1_results, key=lambda r: r.get(primary_metric, float("-inf")))

    best_trial_id = best_s1["trial_id"]
    best_config_src = os.path.join(stage1_dir, best_trial_id, "config.yaml")
    best_config_dst = os.path.join(stage1_dir, "best_config.yaml")
    if os.path.exists(best_config_src):
        shutil.copy(best_config_src, best_config_dst)
    best_s1_config = load_config(best_config_dst) if os.path.exists(best_config_dst) else base_config

    # Extract the hp values for the best trial
    best_hp = {k: v for k, v in best_s1.items()
                if k not in {"trial_id"} and not any(m in k for m in ["loss", "step", "best", "final"])}
    print(f"Stage 1 best: {best_hp}  →  {primary_metric}={best_s1.get(primary_metric):.4f}")

    # ---- Stage 2 ----
    stage2_dir = os.path.join(sweep_dir, "stage2")
    os.makedirs(stage2_dir, exist_ok=True)
    print(f"\n=== Stage 2: fine sweep centred on stage-1 best ===")

    fine_axes = _build_fine_axes(stage1_cfg["axes"], best_hp, stage2_cfg)
    print(f"Fine axes: {[{a['field']: a['values']} for a in fine_axes]}")

    if stage2_cfg.get("search", "grid") == "random":
        n = stage2_cfg.get("max_trials", 20)
        s2_records = write_random_trials(best_s1_config, fine_axes, stage2_dir, max_trials=n, seed=base_config.seed + 1)
    else:
        s2_records = write_grid_trials(best_s1_config, fine_axes, stage2_dir)
        if stage2_cfg.get("max_trials"):
            s2_records = s2_records[:stage2_cfg["max_trials"]]

    s2_results = _dispatch_trials(s2_records, parallel_jobs)
    s2_csv = os.path.join(stage2_dir, "results_summary.csv")
    _write_csv(s2_results, s2_csv)
    print(f"Stage 2 complete. Results → {s2_csv}")

    if s2_results:
        if lower_is_better:
            best_s2 = min(s2_results, key=lambda r: r.get(primary_metric, float("inf")))
        else:
            best_s2 = max(s2_results, key=lambda r: r.get(primary_metric, float("-inf")))
        best_s2_src = os.path.join(stage2_dir, best_s2["trial_id"], "config.yaml")
        best_s2_dst = os.path.join(stage2_dir, "best_config.yaml")
        if os.path.exists(best_s2_src):
            shutil.copy(best_s2_src, best_s2_dst)
        print(f"Stage 2 best: {primary_metric}={best_s2.get(primary_metric):.4f}")
        print(f"Best config written to {best_s2_dst}")

        # Write an overall best to sweep_dir root
        overall_best_dst = os.path.join(sweep_dir, "best_config.yaml")
        if os.path.exists(best_s2_dst):
            shutil.copy(best_s2_dst, overall_best_dst)
            print(f"Overall best config → {overall_best_dst}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Two-stage hyperparameter sweep")
    parser.add_argument("sweep_config", help="Path to two-stage sweep YAML")
    parser.add_argument("sweep_dir", help="Output directory")
    args = parser.parse_args()
    run_two_stage_sweep(args.sweep_config, args.sweep_dir)


if __name__ == "__main__":
    main()
