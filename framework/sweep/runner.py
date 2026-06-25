"""Sweep runner: dispatches trials to a ProcessPoolExecutor and aggregates results."""

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import yaml

from framework.configs import TrainingConfig, load_config, save_config
from framework.sweep.grid import write_grid_trials
from framework.sweep.random import write_random_trials


# ---------------------------------------------------------------------------
# Single-trial entry point (must be importable at the module level for pickling)
# ---------------------------------------------------------------------------

def _run_trial(config_path: str) -> Dict[str, Any]:
    """Load a config and run one trial. Called in a subprocess."""
    from framework.run import run
    from framework.configs import load_config

    config = load_config(config_path)
    return run(config)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(sweep_config_path: str, sweep_dir: str) -> None:
    """Read a sweep YAML, dispatch all trials, and write results_summary.csv."""
    with open(sweep_config_path) as f:
        sweep_cfg = yaml.safe_load(f)

    base_config = load_config(sweep_cfg["base_config"])
    axes = sweep_cfg.get("axes", [])
    search = sweep_cfg.get("search", "grid")
    max_trials = sweep_cfg.get("max_trials", None)
    parallel_jobs = sweep_cfg.get("parallel_jobs", 1)

    os.makedirs(sweep_dir, exist_ok=True)

    # Generate trial configs on disk
    if search == "grid":
        trial_records = write_grid_trials(base_config, axes, sweep_dir)
    elif search == "random":
        n = max_trials or 20
        trial_records = write_random_trials(base_config, axes, sweep_dir, max_trials=n, seed=base_config.seed)
    else:
        raise ValueError(f"Unknown search type: '{search}'")

    if max_trials is not None and search == "grid":
        trial_records = trial_records[:max_trials]

    # Dispatch trials
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

    # Write results_summary.csv
    if results:
        fieldnames = list(results[0].keys())
        summary_path = os.path.join(sweep_dir, "results_summary.csv")
        with open(summary_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Sweep results written to {summary_path}")

        # Write best_config.yaml: trial with lowest best_val_loss
        best_row = min(
            results,
            key=lambda r: r.get("best_val_loss", float("inf")),
        )
        best_trial_id = best_row["trial_id"]
        best_config_src = os.path.join(sweep_dir, best_trial_id, "config.yaml")
        best_config_dst = os.path.join(sweep_dir, "best_config.yaml")
        if os.path.exists(best_config_src):
            import shutil
            shutil.copy(best_config_src, best_config_dst)
            print(f"Best config written to {best_config_dst}")
