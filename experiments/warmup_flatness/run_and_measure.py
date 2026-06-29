"""Train with and without LR warmup while measuring loss-landscape sharpness
at periodic intervals throughout training.

Sharpness is measured via random-direction perturbation:
    sharpness(w) = mean_v [ L(w + ρv) - L(w) ]
where v is a random unit vector and ρ is the perturbation radius.

The sharpness values are logged alongside val_loss in each run's eval_log.jsonl
so that compare.py can use the standard plotting infrastructure.

Usage (from project root):
    python experiments/warmup_flatness/run_and_measure.py --setting gaussian
    python experiments/warmup_flatness/run_and_measure.py --setting heavy_tail
    python experiments/warmup_flatness/run_and_measure.py --setting sst2

Options:
    --setting               gaussian | heavy_tail | sst2
    --lr_values             space-separated LR grid (default varies by setting)
    --clip_upper            fixed clipping threshold τ for all runs (default: 1.0)
    --warmup_fraction       fraction of steps for warmup (default: 0.1)
    --total_steps           override total_outer_steps (default: from base config)
    --n_seeds               seeds per (lr, warmup) combination (default: 5)
    --rho                   perturbation radius for sharpness (default: 0.05)
    --n_directions          random directions per measurement (default: 30)
    --n_sharpness_batches   mini-batches used per sharpness estimate (default: 4)
    --sharpness_every       measure sharpness every N steps (default: eval_every)
    --parallel_jobs         concurrent processes (default: 4)
    --output_dir            root for trial logs (default: outputs/warmup_flatness)
    --device                pytorch device (default: from base config)

Produces:
    outputs/warmup_flatness/{setting}/{run_name}/eval_log.jsonl
        (contains "sharpness" field at each measurement step)
    results/warmup_flatness/{setting}_runs.json
        (index of all completed runs for compare.py)
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Default LR grids (calibrated from prior experiments)
# ---------------------------------------------------------------------------

_DEFAULT_LR = {
    "gaussian":   [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    "heavy_tail": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    "sst2":       [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
}

_BASE_CONFIG = {
    "gaussian":   "experiments/warmup_flatness/configs/base_gaussian.yaml",
    "heavy_tail": "experiments/warmup_flatness/configs/base_heavy_tail.yaml",
    "sst2":       "experiments/warmup_flatness/configs/base_sst2.yaml",
}

_BASE_SEED = 2000  # seeds: base_seed, base_seed+1, ...


# ---------------------------------------------------------------------------
# Single-trial runner (top-level for subprocess pickling)
# ---------------------------------------------------------------------------

def _run_flatness_trial(spec: dict) -> dict:
    """Execute one (lr, warmup, seed) trial with periodic sharpness measurement.

    Called inside a subprocess via ProcessPoolExecutor; all arguments are
    plain Python dicts/scalars for pickling compatibility.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parent.parent.parent)
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    import random
    import numpy as np
    import torch
    from framework.configs import load_config, save_config
    from framework.optimizers.registry import get_inner_optimizer
    from framework.models.wrappers import build_model
    from framework.tracking.logger import RunLogger
    from framework.sharpness.random_direction import compute_sharpness

    # ---- Config assembly ----
    config = load_config(spec["base_config_path"])
    steps          = spec.get("total_steps") or config.total_outer_steps
    warmup_steps   = max(1, round(steps * spec["warmup_fraction"])) if spec["warmup"] else 0
    eval_every     = config.eval_every
    sharpness_every = spec["sharpness_every"] if spec["sharpness_every"] > 0 else eval_every

    inner_opt_cfg  = replace(
        config.inner_optimizer,
        lr=spec["lr"],
        warmup_steps=warmup_steps,
    )
    clipping_cfg = replace(inner_opt_cfg.clipping, upper=spec["clip_upper"])
    inner_opt_cfg = replace(inner_opt_cfg, clipping=clipping_cfg)

    config = replace(
        config,
        run_name=spec["run_name"],
        seed=spec["seed"],
        total_outer_steps=steps,
        inner_optimizer=inner_opt_cfg,
        output_dir=spec["output_dir"],
    )
    if spec.get("device"):
        config = replace(config, device=spec["device"])

    device = config.device

    # ---- Seeding ----
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # ---- Data ----
    from framework.configs import SyntheticDataConfig, RealDataConfig
    true_weights = None
    if isinstance(config.data, SyntheticDataConfig):
        from framework.data.synthetic import build_synthetic_dataloaders
        train_loaders, val_loader, test_loader, dataset = build_synthetic_dataloaders(
            config.data,
            num_nodes=1,
            data_distribution="iid",
            batch_size=config.data.batch_size,
            seed=config.seed,
        )
        true_weights = dataset.true_weights
    elif isinstance(config.data, RealDataConfig) and config.data.glue_task is not None:
        from framework.data.glue_loader import build_glue_dataloaders
        from framework.data.glue_loader import _GLUE_NUM_LABELS
        if config.data.glue_task in _GLUE_NUM_LABELS:
            config.model.num_classes = _GLUE_NUM_LABELS[config.data.glue_task]
        train_loaders, val_loader, test_loader = build_glue_dataloaders(
            config.data, num_nodes=1, distribution="iid", seed=config.seed,
        )
    else:
        raise ValueError(f"Unsupported data config type: {type(config.data)}")

    train_loader = train_loaders[0]

    # ---- Model and optimizer ----
    model = build_model(config.model, true_weights=true_weights).to(device)
    model.train()
    inner_opt = get_inner_optimizer(config.inner_optimizer)

    # ---- Logger ----
    run_dir = Path(spec["output_dir"]) / spec["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(spec["run_name"], spec["output_dir"], verbose=False,
                       results_dir=config.results_dir)
    logger.save_config(asdict(config))

    # ---- Training loop with periodic sharpness measurement ----
    def _infinite(loader):
        while True:
            for b in loader:
                yield b

    def _move(batch, dev):
        if isinstance(batch, (list, tuple)):
            return (batch[0].to(dev), batch[1].to(dev))
        return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _evaluate(model, loader, metrics, prefix="val"):
        model.eval()
        total_loss, n = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = _move(batch, device)
                skip = False
                if isinstance(batch, dict):
                    lbl = batch.get("labels")
                    if lbl is not None and (lbl == -1).all():
                        skip = True
                if not skip:
                    try:
                        total_loss += model.compute_loss(batch).item()
                        n += 1
                    except Exception:
                        pass
                if isinstance(batch, dict):
                    try:
                        fwd = {k: v for k, v in batch.items() if k != "labels"}
                        out = model(fwd)
                        logits = out[1] if isinstance(out, tuple) else out
                        if logits is not None:
                            all_preds.append(logits.cpu())
                        lbl = batch.get("labels")
                        if lbl is not None and not (lbl == -1).all():
                            all_targets.append(lbl.cpu())
                    except Exception:
                        pass
                elif isinstance(batch, tuple):
                    try:
                        out = model(batch[0])
                        logits = out[1] if isinstance(out, tuple) else out
                        all_preds.append(logits.cpu())
                        all_targets.append(batch[1].cpu())
                    except Exception:
                        pass
        result = {f"{prefix}_loss": total_loss / max(1, n)}
        if "accuracy" in metrics and all_preds and all_targets:
            from framework.tracking import metrics as metric_module
            p = torch.cat(all_preds).numpy()
            t = torch.cat(all_targets).numpy()
            try:
                result[f"{prefix}_accuracy"] = metric_module.compute("accuracy", p, t)
            except Exception:
                pass
        model.train()
        return result

    data_iter = _infinite(train_loader)
    best_val_loss = math.inf

    for step in range(config.total_outer_steps):
        batch = _move(next(data_iter), device)
        inner_opt.step_accumulated(model, [batch])

        if step % eval_every == 0 or step == config.total_outer_steps - 1:
            eval_metrics = _evaluate(model, val_loader, config.metrics, prefix="val")

            # Sharpness measurement at the eval step (when aligned) or separately.
            # Also always measure at the final step so the scatter plot has a terminal point.
            if step % sharpness_every == 0 or step == config.total_outer_steps - 1:
                sharpness = compute_sharpness(
                    model, val_loader, device,
                    rho=spec["rho"],
                    n_directions=spec["n_directions"],
                    n_batches=spec["n_sharpness_batches"],
                    seed=spec["seed"] + step,   # per-step seed for reproducibility
                )
                eval_metrics["sharpness"] = sharpness

            logger.log_metrics(step, eval_metrics)
            val_loss = eval_metrics.get("val_loss", math.inf)
            best_val_loss = min(best_val_loss, val_loss)

    logger.close()

    return {
        "run_name":      spec["run_name"],
        "setting":       spec["setting"],
        "lr":            spec["lr"],
        "warmup":        spec["warmup"],
        "seed":          spec["seed"],
        "clip_upper":    spec["clip_upper"],
        "best_val_loss": best_val_loss,
        "log_dir":       str(run_dir),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Warmup flatness: train + measure sharpness")
    parser.add_argument("--setting", required=True, choices=["gaussian", "heavy_tail", "sst2"])
    parser.add_argument("--lr_values", nargs="+", type=float, default=None)
    parser.add_argument("--clip_upper", type=float, default=1.0,
                        help="Fixed clipping threshold τ (same for all runs)")
    parser.add_argument("--warmup_fraction", type=float, default=0.1,
                        help="Fraction of total_steps used for warmup when warmup=True")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--rho", type=float, default=0.05,
                        help="Perturbation radius for random-direction sharpness")
    parser.add_argument("--n_directions", type=int, default=30,
                        help="Number of random unit vectors per sharpness estimate")
    parser.add_argument("--n_sharpness_batches", type=int, default=4,
                        help="Mini-batches used per sharpness estimate")
    parser.add_argument("--sharpness_every", type=int, default=0,
                        help="Measure sharpness every N steps (0 = align with eval_every)")
    parser.add_argument("--parallel_jobs", type=int, default=4)
    parser.add_argument("--output_dir", default="outputs/warmup_flatness")
    parser.add_argument("--device", default=None)
    parser.add_argument("--lr_offset", type=int, default=0,
                        help="Index offset for run naming when running an lr subset (default: 0)")
    parser.add_argument("--result_tag", default="",
                        help="Suffix appended to the result filename, e.g. '_l0l1' writes "
                             "{setting}_runs_l0l1.json (default: '' = single file)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    setting    = args.setting
    lr_values  = args.lr_values or _DEFAULT_LR[setting]
    output_dir = os.path.join(args.output_dir, setting)
    base_cfg   = str(PROJECT_ROOT / _BASE_CONFIG[setting])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"results/warmup_flatness", exist_ok=True)

    # Build all trial specs: (lr, warmup={True,False}, seed).
    specs: List[dict] = []
    for li, lr in enumerate(lr_values):
        for warmup in (False, True):
            for s in range(args.n_seeds):
                seed = _BASE_SEED + s
                label = "w" if warmup else "n"
                run_name = f"flat_{setting}_l{li + args.lr_offset}_{label}_s{s}"
                specs.append({
                    "run_name":          run_name,
                    "setting":           setting,
                    "base_config_path":  base_cfg,
                    "lr":                lr,
                    "warmup":            warmup,
                    "warmup_fraction":   args.warmup_fraction,
                    "seed":              seed,
                    "clip_upper":        args.clip_upper,
                    "total_steps":       args.total_steps,
                    "rho":               args.rho,
                    "n_directions":      args.n_directions,
                    "n_sharpness_batches": args.n_sharpness_batches,
                    "sharpness_every":   args.sharpness_every,
                    "output_dir":        output_dir,
                    "device":            args.device,
                })

    total = len(specs)
    n_lr  = len(lr_values)
    print(f"[{setting}] {n_lr} lr × 2 conditions × {args.n_seeds} seeds"
          f" = {total} trials  (parallel_jobs={args.parallel_jobs})")

    mp_ctx = multiprocessing.get_context("spawn")
    run_index: List[dict] = []

    with ProcessPoolExecutor(max_workers=args.parallel_jobs, mp_context=mp_ctx) as exe:
        future_map = {exe.submit(_run_flatness_trial, spec): spec for spec in specs}
        done = 0
        for future in as_completed(future_map):
            spec = future_map[future]
            done += 1
            try:
                result = future.result()
            except Exception as exc:
                print(f"  [{done}/{total}] {spec['run_name']} FAILED: {exc}")
                result = {
                    "run_name":      spec["run_name"],
                    "setting":       setting,
                    "lr":            spec["lr"],
                    "warmup":        spec["warmup"],
                    "seed":          spec["seed"],
                    "clip_upper":    spec["clip_upper"],
                    "best_val_loss": None,
                    "log_dir":       str(Path(output_dir) / spec["run_name"]),
                }
            run_index.append(result)
            bvl = result.get("best_val_loss")
            bvl_str = f"{bvl:.4g}" if bvl is not None else "N/A"
            print(f"  [{done}/{total}] {spec['run_name']}"
                  f"  lr={spec['lr']:.3g}  warmup={spec['warmup']}"
                  f"  bvl={bvl_str}")

    # Save run index.
    index_path = f"results/warmup_flatness/{setting}_runs{args.result_tag}.json"
    with open(index_path, "w") as f:
        json.dump(run_index, f, indent=2)
    print(f"\nRun index saved to {index_path}")


if __name__ == "__main__":
    main()
