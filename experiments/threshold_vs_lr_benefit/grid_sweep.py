"""2-D grid sweep over (tau, lr) to test whether increasing the clipping threshold
gives more benefit than increasing the terminal learning rate.

Each cell in the grid is a (tau, lr) pair run over n_seeds independent seeds.
Results are aggregated to mean/std of best_val_loss and saved as JSON.
The heatmap is produced by compare.py.

Usage (from project root):
    python experiments/threshold_vs_lr_benefit/grid_sweep.py --setting gaussian
    python experiments/threshold_vs_lr_benefit/grid_sweep.py --setting heavy_tail
    python experiments/threshold_vs_lr_benefit/grid_sweep.py --setting sst2

Options:
    --setting           gaussian | heavy_tail | sst2
    --tau_values        space-separated list of clipping thresholds (default varies by setting)
    --lr_values         space-separated list of terminal learning rates (default varies by setting)
    --n_seeds           number of seeds per cell (default: 5)
    --warmup_fraction   fraction of total_outer_steps used for warmup (default: 0.1)
    --total_steps       override total_outer_steps (default: from base config)
    --parallel_jobs     number of concurrent trials (default: 4)
    --output_dir        root for trial logs (default: outputs/threshold_vs_lr_benefit)
    --device            pytorch device (default: from base config)

Produces:
    results/threshold_vs_lr_benefit/{setting}_grid.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Default grids calibrated from prior sweep results
# (tau priors: heavy-tail ~6.9, gaussian ~1.1; lr priors: heavy-tail ~0.04, gaussian ~0.25)
# ---------------------------------------------------------------------------

_DEFAULT_TAU = {
    "gaussian":   [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    "heavy_tail": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    "sst2":       [0.1, 0.5, 1.0, 5.0, 10.0, 20.0],
}

_DEFAULT_LR = {
    "gaussian":   [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    "heavy_tail": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    "sst2":       [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
}

_BASE_CONFIG = {
    "gaussian":   "experiments/threshold_vs_lr_benefit/configs/base_gaussian.yaml",
    "heavy_tail": "experiments/threshold_vs_lr_benefit/configs/base_heavy_tail.yaml",
    "sst2":       "experiments/threshold_vs_lr_benefit/configs/base_sst2.yaml",
}

_BASE_SEED = 1000  # seeds per cell: base_seed, base_seed+1, ..., base_seed+n_seeds-1


# ---------------------------------------------------------------------------
# Trial runner (top-level for subprocess pickling)
# ---------------------------------------------------------------------------

def _run_trial(config_path: str) -> Dict[str, Any]:
    """Load a config yaml and run one training trial. Executed in a subprocess."""
    import sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parent.parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from framework.run import run
    from framework.configs import load_config
    from dataclasses import replace
    config = load_config(config_path)
    # Suppress all per-trial file logging (step logs, eval logs, checkpoints,
    # auto-plots). The run() return value carries all metrics we need in memory.
    config = replace(config, output_dir=None)
    return run(config)


# ---------------------------------------------------------------------------
# Trial builder
# ---------------------------------------------------------------------------

def _build_trial_config(
    base_config_path: str,
    tau: float,
    lr: float,
    seed: int,
    warmup_fraction: float,
    total_steps: Optional[int],
    output_dir: str,
    device: Optional[str],
    trial_name: str,
) -> str:
    """Build and save a trial config; return the path to the saved YAML."""
    from framework.configs import load_config, save_config

    config = load_config(base_config_path)

    # Compute warmup_steps as a fraction of total training steps.
    steps = total_steps if total_steps is not None else config.total_outer_steps
    warmup_steps = max(1, round(steps * warmup_fraction))

    clipping = replace(config.inner_optimizer.clipping, upper=tau)
    inner_opt = replace(
        config.inner_optimizer,
        lr=lr,
        clipping=clipping,
        warmup_steps=warmup_steps,
    )
    config = replace(
        config,
        run_name=trial_name,
        seed=seed,
        total_outer_steps=steps,
        inner_optimizer=inner_opt,
        output_dir=output_dir,
    )
    if device is not None:
        config = replace(config, device=device)

    trial_dir = Path(output_dir) / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = str(trial_dir / "config.yaml")
    save_config(config, config_path)
    return config_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="2-D (tau, lr) grid sweep")
    parser.add_argument("--setting", required=True, choices=["gaussian", "heavy_tail", "sst2"])
    parser.add_argument("--tau_values", nargs="+", type=float, default=None,
                        help="Clipping threshold values to sweep (default: preset per setting)")
    parser.add_argument("--lr_values", nargs="+", type=float, default=None,
                        help="Terminal learning rate values to sweep (default: preset per setting)")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of random seeds per (tau, lr) cell")
    parser.add_argument("--warmup_fraction", type=float, default=0.1,
                        help="Fraction of total steps used for linear LR warmup")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Override total_outer_steps from base config")
    parser.add_argument("--parallel_jobs", type=int, default=4)
    parser.add_argument("--output_dir", default="outputs/threshold_vs_lr_benefit")
    parser.add_argument("--device", default=None,
                        help="Override pytorch device (default: from base config)")
    parser.add_argument("--tau_offset", type=int, default=0,
                        help="Index offset for trial naming when running a tau subset (default: 0)")
    parser.add_argument("--result_tag", default="",
                        help="Suffix appended to the result filename, e.g. '_t0t1' writes "
                             "{setting}_grid_t0t1.json (default: '' = single file)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    setting = args.setting
    tau_values = args.tau_values or _DEFAULT_TAU[setting]
    lr_values  = args.lr_values  or _DEFAULT_LR[setting]
    n_seeds    = args.n_seeds
    output_dir = os.path.join(args.output_dir, setting)
    base_cfg   = str(PROJECT_ROOT / _BASE_CONFIG[setting])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/threshold_vs_lr_benefit", exist_ok=True)

    # Build all trial specs: (name, tau, lr, seed).
    trial_specs: List[Tuple[str, float, float, int]] = []
    for ti, tau in enumerate(tau_values):
        for li, lr in enumerate(lr_values):
            for s in range(n_seeds):
                seed = _BASE_SEED + s
                name = f"thr_{setting}_t{ti + args.tau_offset}_l{li}_s{s}"
                trial_specs.append((name, tau, lr, seed))

    total_trials = len(trial_specs)
    print(f"[{setting}] {len(tau_values)} tau × {len(lr_values)} lr × {n_seeds} seeds"
          f" = {total_trials} trials  (parallel_jobs={args.parallel_jobs})")

    # Write all trial configs to disk.
    config_paths: Dict[str, str] = {}
    for name, tau, lr, seed in trial_specs:
        config_paths[name] = _build_trial_config(
            base_config_path=base_cfg,
            tau=tau,
            lr=lr,
            seed=seed,
            warmup_fraction=args.warmup_fraction,
            total_steps=args.total_steps,
            output_dir=output_dir,
            device=args.device,
            trial_name=name,
        )

    # Dispatch via subprocess pool (spawn avoids CUDA fork issues).
    mp_ctx = multiprocessing.get_context("spawn")
    raw_results: Dict[str, Dict] = {}

    with ProcessPoolExecutor(max_workers=args.parallel_jobs, mp_context=mp_ctx) as exe:
        future_map = {
            exe.submit(_run_trial, config_paths[name]): (name, tau, lr, seed)
            for name, tau, lr, seed in trial_specs
        }
        done = 0
        for future in as_completed(future_map):
            name, tau, lr, seed = future_map[future]
            done += 1
            try:
                metrics = future.result()
            except Exception as exc:
                print(f"  [{done}/{total_trials}] {name} FAILED: {exc}")
                metrics = {}
            best_val = metrics.get("val_loss_best", metrics.get("best_val_loss", None))
            raw_results[name] = {
                "tau": tau, "lr": lr, "seed": seed,
                "best_val_loss": best_val,
            }
            bvl_str = f"{best_val:.4g}" if best_val is not None else "N/A"
            print(f"  [{done}/{total_trials}] {name}  tau={tau}  lr={lr:.3g}"
                  f"  seed={seed}  bvl={bvl_str}")

    # Aggregate: for each (tau, lr) cell compute mean and std over seeds.
    cells: List[Dict] = []
    for ti, tau in enumerate(tau_values):
        for li, lr in enumerate(lr_values):
            seed_vals = []
            for s in range(n_seeds):
                name = f"thr_{setting}_t{ti + args.tau_offset}_l{li}_s{s}"
                bvl = raw_results.get(name, {}).get("best_val_loss")
                if bvl is not None and not (isinstance(bvl, float) and math.isnan(bvl)):
                    seed_vals.append(bvl)
            if seed_vals:
                mean = sum(seed_vals) / len(seed_vals)
                variance = sum((v - mean) ** 2 for v in seed_vals) / max(1, len(seed_vals) - 1)
                std = variance ** 0.5
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

    result_doc = {
        "setting": setting,
        "warmup_fraction": args.warmup_fraction,
        "n_seeds": n_seeds,
        "tau_values": tau_values,
        "lr_values": lr_values,
        "cells": cells,
    }

    out_path = f"results/threshold_vs_lr_benefit/{setting}_grid{args.result_tag}.json"
    with open(out_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    print(f"\nGrid results saved to {out_path}")


if __name__ == "__main__":
    main()
