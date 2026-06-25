"""Centralized (single-node) training loop.

In centralized mode, local_steps > 1 is treated as gradient accumulation.
Supports:
- Time-based checkpointing (checkpoint_interval_minutes)
- Resume from latest checkpoint (config.resume = True)
- Optional test-set evaluation at the end of training
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from framework.configs import TrainingConfig
from framework.optimizers.clipping import compute_gradient_stats
from framework.optimizers.registry import get_inner_optimizer
from framework.tracking.logger import RunLogger
from framework.tracking import metrics as metric_module


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _infinite_loader(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def _move_batch(batch, device: str):
    if isinstance(batch, (list, tuple)):
        return (batch[0].to(device), batch[1].to(device))
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    metric_names: List[str],
    device: str,
    true_weights: Optional[Any] = None,
    prefix: str = "val",
    tokenizer: Optional[Any] = None,
) -> Dict[str, float]:
    """Run evaluation and compute all requested metrics.

    Handles three families of models:
      * convex / classification: collect logits and compare against integer/float labels
      * generative seq2seq (T5): compute teacher-forcing loss and, for text metrics
        (bleu/meteor), decode ``model.generate`` output against the reference text.
        Logits are NOT accumulated here — for a vocab-sized output that would blow up
        memory on any non-trivial validation set.
    """
    model.eval()
    is_seq2seq = hasattr(model, "generate")
    text_metric_names = [m for m in metric_names if m in ("bleu", "meteor")]
    want_text = is_seq2seq and tokenizer is not None and len(text_metric_names) > 0

    all_preds, all_targets, total_loss, n_batches = [], [], 0.0, 0
    gen_hyps: List[str] = []
    gen_refs: List[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)

            # For GLUE test splits, labels may be -1 (unreleased); skip loss
            skip_loss = False
            if isinstance(batch, dict):
                labels = batch.get("labels")
                if labels is not None and (labels == -1).all():
                    skip_loss = True

            if not skip_loss:
                try:
                    loss = model.compute_loss(batch)
                    total_loss += loss.item()
                    n_batches += 1
                except Exception:
                    pass

            if is_seq2seq:
                # Decode generations only when a text metric is requested.
                if want_text and isinstance(batch, dict) and "input_ids" in batch:
                    try:
                        labels = batch["labels"]
                        gen_ids = model.generate(
                            batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            max_length=labels.shape[1],
                        )
                        hyps = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                        ref_ids = labels.clone()
                        ref_ids[ref_ids == -100] = tokenizer.pad_token_id
                        refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
                        gen_hyps.extend(hyps)
                        gen_refs.extend(refs)
                    except Exception:
                        pass
                continue  # never accumulate vocab-sized logits for seq2seq

            # Collect logits for metric computation (classification / regression)
            try:
                out = model(batch) if isinstance(batch, dict) else model(batch[0])
                if isinstance(out, tuple):
                    logits = out[1]   # (loss, logits)
                else:
                    logits = out
                if logits is not None:
                    all_preds.append(logits.cpu())
            except Exception:
                pass

            if isinstance(batch, tuple):
                all_targets.append(batch[1].cpu())
            elif isinstance(batch, dict) and "labels" in batch:
                lbl = batch["labels"]
                if not (lbl == -1).all():
                    all_targets.append(lbl.cpu())

    result: Dict[str, float] = {f"{prefix}_loss": total_loss / max(1, n_batches)}

    for name in metric_names:
        if name == "loss":
            result[f"{prefix}_loss"] = result[f"{prefix}_loss"]
        elif name == "param_distance" and true_weights is not None:
            for p in model.parameters():
                w_hat = p.data.cpu().flatten()
                break
            result["param_distance"] = metric_module.compute("param_distance", w_hat, true_weights)
        elif name in ("bleu", "meteor"):
            if gen_hyps and gen_refs:
                try:
                    result[f"{prefix}_{name}"] = metric_module.compute(name, gen_hyps, gen_refs)
                except Exception as e:
                    print(f"[eval] {name} computation failed: {e}")
        elif all_preds and all_targets:
            preds_cat = torch.cat(all_preds, dim=0).numpy()
            tgts_cat = torch.cat(all_targets, dim=0).numpy()
            try:
                result[f"{prefix}_{name}"] = metric_module.compute(name, preds_cat, tgts_cat)
            except Exception:
                pass

    model.train()
    return result


def _find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return the path to the checkpoint with the highest step number, or None."""
    if not checkpoint_dir.exists():
        return None
    ckpts = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return ckpts[-1] if ckpts else None


def run_centralized(
    config: TrainingConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    true_weights: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the centralized training loop. Returns the final metrics dict."""
    _seed_everything(config.seed, getattr(config, "deterministic", False))

    device = config.device
    model = model.to(device)
    model.train()

    inner_opt = get_inner_optimizer(config.inner_optimizer)
    local_steps = config.distributed.local_steps

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
            if "inner_optimizer_state" in state and hasattr(inner_opt, "load_state_dict"):
                inner_opt.load_state_dict(state["inner_optimizer_state"])
            start_step = state["outer_step"] + 1
            print(f"[resume] Loaded checkpoint from {latest}, continuing from step {start_step}")

    data_iter = _infinite_loader(train_loader)
    best_val_loss = math.inf
    checkpoint_interval_sec = (
        (config.checkpoint_interval_minutes or 0) * 60
    )
    last_ckpt_time = time.time()

    for outer_step in range(start_step, config.total_outer_steps):
        batches = []
        for _ in range(local_steps):
            batch = next(data_iter)
            batches.append(_move_batch(batch, device))

        diagnostics = inner_opt.step_accumulated(model, batches)
        logger.log_step(outer_step, diagnostics)

        # Gradient statistics
        if outer_step % config.log_gradients_every == 0:
            grad_stats = []
            for (name, raw_g), (_, clipped_g) in zip(
                inner_opt._last_raw_grads, inner_opt._last_clipped_grads
            ):
                stats = compute_gradient_stats(
                    name, raw_g, clipped_g,
                    config.inner_optimizer.clipping.upper,
                    config.inner_optimizer.clipping.lower,
                )
                grad_stats.append(stats)
            if grad_stats:
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

        # Step-based checkpointing
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
            inner_state = inner_opt.state_dict() if hasattr(inner_opt, "state_dict") else {}
            logger.save_checkpoint(
                outer_step, model, {}, is_best=is_best,
                extra={"inner_optimizer_state": inner_state},
            )
            last_ckpt_time = now

    # Final test evaluation
    if test_loader is not None:
        test_metrics = _evaluate(model, test_loader, config.metrics, device, true_weights,
                                 prefix="test", tokenizer=tokenizer)
        logger.log_metrics(config.total_outer_steps - 1, test_metrics)

    logger.close()
    return logger._final_metrics
