"""Sweep the Student-t tail index (degrees of freedom) for the LR-warmup comparison.

Three-way ablation: for each noise_df value, runs an independent two-stage
(coarse -> fine) hyperparameter sweep for each of three conditions:
    clip_only    - fixed-threshold global upper-clipping, no warmup
    warmup_clip  - warmup + clipping, both swept
    warmup_only  - warmup, clipping.clip_type="none" (no clip axis to sweep)

(Earlier revision only compared clip_only vs warmup_clip, and additionally
swept noise_df down to 1.1/1.25 -- those two points were dropped after seeing
a single fixed seed produce large, non-robust swings there; see
results/lr_warmup_comparison/tail_index_trend.json history / README for
that finding. This revision also re-runs each winner across multiple seeds
in tail_index_compare.py instead of a single fixed seed.)

Run from the project root:
    python experiments/lr_warmup_comparison/tail_index_sweep.py [options]

Use --dry_run first to see the planned trial counts before committing to a
full run (each (condition, df) pair is a full two-stage sweep).

Writes, per (condition, df):
    outputs/lr_warmup_comparison/tail_index/{condition}/df_{tag}/{stage1,stage2}/...
    outputs/lr_warmup_comparison/tail_index/{condition}/df_{tag}/best_config.yaml

And a running index of every sweep's outcome (updated after each df point, so
a partially-completed run is not lost):
    results/lr_warmup_comparison/tail_index_sweep_index.json

Follow up with tail_index_compare.py to turn the index into final-run metrics
and trend plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from framework.configs import load_config, TrainingConfig
from framework.sweep.two_stage import run_two_stage_sweep_for_config

CONDITIONS = ["clip_only", "warmup_clip", "warmup_only"]

BASE_CONFIG = {
    "clip_only":   "experiments/lr_warmup_comparison/configs/no_warmup_heavy_tail.yaml",
    "warmup_clip": "experiments/lr_warmup_comparison/configs/warmup_heavy_tail.yaml",
    "warmup_only": "experiments/lr_warmup_comparison/configs/warmup_only_heavy_tail.yaml",
}

# Same axes as sweeps/sweep_{no_warmup,warmup}_heavy.yaml (clip_only,
# warmup_clip) plus a new no-clip-axis grid for warmup_only — reused/added so
# every df point in the trend is tuned exactly as hard as the original df=2
# point.
SWEEP_SPEC = {
    "clip_only":   "experiments/lr_warmup_comparison/sweeps/sweep_no_warmup_heavy.yaml",
    "warmup_clip": "experiments/lr_warmup_comparison/sweeps/sweep_warmup_heavy.yaml",
    "warmup_only": "experiments/lr_warmup_comparison/sweeps/sweep_warmup_only_heavy.yaml",
}

# Kept to df in [1.5, 2.0]: a single fixed seed produced large, non-robust
# swings at df=1.1/1.25 (see README) that a heavier-tailed noise distribution
# makes hard to trust without seed-averaging on every sweep trial, which
# would be far more expensive. Multi-seed averaging is instead applied only
# to the final winner re-run in tail_index_compare.py.
DEFAULT_TAIL_INDICES = [2.0, 1.75, 1.5]


def _df_tag(df: float) -> str:
    return f"{df:.2f}".replace(".", "p")


def build_config(condition: str, df: float) -> TrainingConfig:
    config = load_config(str(PROJECT_ROOT / BASE_CONFIG[condition]))
    tag = _df_tag(df)
    return replace(
        config,
        run_name=f"tail_df{tag}_{condition}",
        data=replace(config.data, noise_df=df),
    )


def load_sweep_spec(condition: str) -> dict:
    path = PROJECT_ROOT / SWEEP_SPEC[condition]
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.pop("base_config", None)
    return cfg


def _planned_trial_count(spec: dict) -> int:
    s1_axes = spec["stage1"]["axes"]
    s1_n = 1
    for ax in s1_axes:
        s1_n *= len(ax["values"])
    s2_n = spec["stage2"].get("n_values", 3) ** len(s1_axes)
    return s1_n + s2_n


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail-index (Student-t df) sweep for LR warmup")
    parser.add_argument("--tail_indices", nargs="+", type=float, default=DEFAULT_TAIL_INDICES,
                        help="Student-t degrees-of-freedom values to sweep, e.g. 2.0 1.5 1.0")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--output_root", default="outputs/lr_warmup_comparison/tail_index")
    parser.add_argument("--index_path", default="results/lr_warmup_comparison/tail_index_sweep_index.json")
    parser.add_argument("--parallel_jobs", type=int, default=None,
                        help="Override parallel_jobs from the sweep YAML")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the planned (condition, df) grid and trial counts, run nothing")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.dry_run:
        total = 0
        for condition in args.conditions:
            spec = load_sweep_spec(condition)
            per_df = _planned_trial_count(spec)
            n = per_df * len(args.tail_indices)
            total += n
            print(f"[{condition}] ~{per_df} trials/df x {len(args.tail_indices)} df values "
                  f"= ~{n} trials  (parallel_jobs={args.parallel_jobs or spec.get('parallel_jobs', 1)})")
        print(f"Total planned trials across all conditions: ~{total}")
        return

    index: dict = {}
    if os.path.exists(args.index_path):
        with open(args.index_path) as f:
            index = json.load(f)

    for condition in args.conditions:
        spec = load_sweep_spec(condition)
        if args.parallel_jobs is not None:
            spec["parallel_jobs"] = args.parallel_jobs
        index.setdefault(condition, {})

        for df in args.tail_indices:
            tag = _df_tag(df)
            print(f"\n{'=' * 70}\ncondition={condition}  noise_df={df}\n{'=' * 70}")
            config = build_config(condition, df)
            sweep_dir = os.path.join(args.output_root, condition, f"df_{tag}")
            summary = run_two_stage_sweep_for_config(config, spec, sweep_dir)
            index[condition][tag] = {
                "noise_df": df,
                "sweep_dir": sweep_dir,
                "summary": summary,
            }
            os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
            with open(args.index_path, "w") as f:
                json.dump(index, f, indent=2)
            status = "ok" if summary else "FAILED (no valid trials)"
            print(f"[{condition} df={df}] {status}")

    print(f"\nDone. Index written to {args.index_path}")


if __name__ == "__main__":
    main()
