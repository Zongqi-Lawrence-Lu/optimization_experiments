"""Distributed (federated) training loop.

Implements the nested inner/outer optimization loop with partial node participation.
Supports:
- Time-based checkpointing (checkpoint_interval_minutes)
- Resume from latest checkpoint (config.resume = True)
- Optional test-set evaluation at the end of training
"""

from __future__ import annotations

import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from framework.configs import TrainingConfig
from framework.optimizers.clipping import compute_gradient_stats
from framework.optimizers.inner import InnerOptimizer
from framework.optimizers.registry import get_inner_optimizer, get_outer_optimizer
from framework.training.centralized import _evaluate, _find_latest_checkpoint, _move_batch, _seed_everything
from framework.tracking.logger import RunLogger


def _sample_active_nodes(
    num_nodes: int,
    participation_rate: float,
    rng: np.random.RandomState,
) -> List[int]:
    k = max(1, int(num_nodes * participation_rate))
    return sorted(rng.choice(num_nodes, size=k, replace=False).tolist())


def _node_local_update(
    node_id: int,
    global_model: nn.Module,
    inner_opt: InnerOptimizer,
    data_iter: Iterator,
    node_loader: DataLoader,
    local_steps: int,
    device: str,
) -> Dict[str, Any]:
    """Run local_steps of inner optimization on a copy of the global model.

    Returns:
        delta: dict mapping param name → displacement (x_i − x)
        diagnostics_list: per-inner-step diagnostic dicts
        data_iter: updated iterator (may have been reset on StopIteration)
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    inner_opt.reset_state()

    diagnostics_list = []

    for _ in range(local_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(node_loader)
            batch = next(data_iter)

        batch = _move_batch(batch, device)
        diag = inner_opt.step(local_model, batch)
        diagnostics_list.append(diag)

    delta: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for (name, local_p), (_, global_p) in zip(
            local_model.named_parameters(), global_model.named_parameters()
        ):
            delta[name] = (local_p.data - global_p.data).cpu()

    return {
        "node_id": node_id,
        "delta": delta,
        "diagnostics": diagnostics_list,
        "data_iter": data_iter,
    }


def run_distributed(
    config: TrainingConfig,
    model: nn.Module,
    train_loaders: List[DataLoader],
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    true_weights: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the distributed (federated) training loop. Returns the final metrics dict."""
    _seed_everything(config.seed, getattr(config, "deterministic", False))

    device = config.device
    model = model.to(device)
    model.train()

    num_nodes = config.distributed.num_nodes
    local_steps = config.distributed.local_steps
    participation_rate = config.distributed.participation_rate

    assert len(train_loaders) == num_nodes, (
        f"Expected {num_nodes} loaders, got {len(train_loaders)}"
    )

    inner_opts: List[InnerOptimizer] = [
        get_inner_optimizer(config.inner_optimizer) for _ in range(num_nodes)
    ]
    outer_opt = get_outer_optimizer(config.outer_optimizer)

    logger = RunLogger(config.run_name, config.output_dir, verbose=True,
                       results_dir=config.results_dir)
    logger.save_config(asdict(config))

    start_step = 0

    # --- Resume from checkpoint ---
    if config.resume and config.output_dir is not None:
        ckpt_dir = Path(config.output_dir) / config.run_name / "checkpoints"
        latest = _find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            state = torch.load(str(latest), map_location=device)
            model.load_state_dict(state["model_state_dict"])
            if "outer_optimizer_state" in state:
                outer_opt.load_state_dict(state["outer_optimizer_state"])
            if "inner_optimizer_states" in state:
                for i, inner_state in enumerate(state["inner_optimizer_states"]):
                    if i < len(inner_opts) and hasattr(inner_opts[i], "load_state_dict"):
                        inner_opts[i].load_state_dict(inner_state)
            start_step = state["outer_step"] + 1
            print(f"[resume] Loaded checkpoint from {latest}, continuing from step {start_step}")

    rng = np.random.RandomState(config.seed)
    best_val_loss = math.inf
    node_iters: List[Iterator] = [iter(loader) for loader in train_loaders]

    checkpoint_interval_sec = (
        (config.checkpoint_interval_minutes or 0) * 60
    )
    last_ckpt_time = time.time()

    # Advance RNG to match the state at start_step (so resumed runs are reproducible)
    for _ in range(start_step):
        _sample_active_nodes(num_nodes, participation_rate, rng)

    for outer_step in range(start_step, config.total_outer_steps):
        active_nodes = _sample_active_nodes(num_nodes, participation_rate, rng)

        node_results = []
        for i in active_nodes:
            result = _node_local_update(
                node_id=i,
                global_model=model,
                inner_opt=inner_opts[i],
                data_iter=node_iters[i],
                node_loader=train_loaders[i],
                local_steps=local_steps,
                device=device,
            )
            node_iters[i] = result["data_iter"]
            node_results.append(result)

        # Weighted mean of local displacements (uniform weights)
        n_active = len(active_nodes)
        pseudo_grad: Dict[str, torch.Tensor] = {}
        for result in node_results:
            for name, delta in result["delta"].items():
                if name not in pseudo_grad:
                    pseudo_grad[name] = torch.zeros_like(delta)
                pseudo_grad[name] += delta / n_active

        pseudo_grad = {k: v.to(device) for k, v in pseudo_grad.items()}

        outer_diag = outer_opt.step(model, pseudo_grad)

        for result in node_results:
            for inner_step, diag in enumerate(result["diagnostics"]):
                logger.log_step(
                    outer_step,
                    {**diag, **outer_diag},
                    inner_step=inner_step,
                    node_id=result["node_id"],
                )

        # Gradient statistics on pseudo_grad (raw == aggregated, no separate clipped version here)
        if outer_step % config.log_gradients_every == 0:
            grad_stats = []
            for name, delta in pseudo_grad.items():
                stats = compute_gradient_stats(
                    name, delta, delta,
                    config.outer_optimizer.clipping.upper,
                    config.outer_optimizer.clipping.lower,
                )
                grad_stats.append(stats)
            logger.log_gradient_stats(outer_step, grad_stats)

        # Validation
        if val_loader is not None and outer_step % config.eval_every == 0:
            eval_metrics = _evaluate(model, val_loader, config.metrics, device, true_weights,
                                     prefix="val", tokenizer=tokenizer)
            logger.log_metrics(outer_step, eval_metrics)
            val_loss = eval_metrics.get("val_loss", math.inf)
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
        else:
            is_best = False

        # Checkpointing (step-based + time-based)
        now = time.time()
        time_triggered = (
            checkpoint_interval_sec > 0
            and (now - last_ckpt_time) >= checkpoint_interval_sec
        )
        if (
            outer_step % config.checkpoint_every == 0
            or outer_step == config.total_outer_steps - 1
            or time_triggered
        ):
            inner_states = [
                opt.state_dict() if hasattr(opt, "state_dict") else {}
                for opt in inner_opts
            ]
            logger.save_checkpoint(
                outer_step, model, outer_opt.state_dict(), is_best=is_best,
                extra={"inner_optimizer_states": inner_states},
            )
            last_ckpt_time = now

    # Final test evaluation
    if test_loader is not None:
        test_metrics = _evaluate(model, test_loader, config.metrics, device, true_weights,
                                 prefix="test", tokenizer=tokenizer)
        logger.log_metrics(config.total_outer_steps - 1, test_metrics)

    logger.close()
    return logger._final_metrics
