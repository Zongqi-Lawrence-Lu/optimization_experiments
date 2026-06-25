"""Top-level entry point: builds everything from a config and dispatches to the
correct training loop based on config.distributed.mode.

Usage (Python API):
    from framework.run import run
    from framework.configs import TrainingConfig, ...
    metrics = run(config)

Usage (CLI):
    python -m framework.run path/to/config.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from framework.configs import RealDataConfig, SyntheticDataConfig, TrainingConfig, load_config
from framework.models.wrappers import build_model
from framework.training.centralized import run_centralized
from framework.training.distributed import run_distributed


def _auto_plot(config: TrainingConfig) -> None:
    """Generate training curve plots after a run completes."""
    if config.output_dir is None:
        return
    try:
        from framework.plotting.training_curves import plot_training_curves, plot_gradient_norms
    except ImportError:
        return

    log_dir = Path(config.output_dir) / config.run_name
    if not log_dir.exists():
        return

    plots_base = Path(config.plots_dir or "plots") / "training_curves"
    plots_base.mkdir(parents=True, exist_ok=True)

    curve_path = str(plots_base / f"{config.run_name}.png")
    grad_path = str(plots_base / f"{config.run_name}_grad_norms.png")

    try:
        plot_training_curves(str(log_dir), curve_path, metrics=config.metrics)
    except Exception as e:
        print(f"[auto-plot] training curves failed: {e}")

    try:
        plot_gradient_norms(str(log_dir), grad_path)
    except Exception as e:
        print(f"[auto-plot] gradient norm plot failed: {e}")


def run(config: TrainingConfig) -> Dict[str, Any]:
    """Build data, model, and run the training loop selected by config.distributed.mode."""
    true_weights = None
    test_loader = None

    if isinstance(config.data, SyntheticDataConfig):
        from framework.data.synthetic import build_synthetic_dataloaders

        train_loaders, val_loader, dataset = build_synthetic_dataloaders(
            config.data,
            num_nodes=config.distributed.num_nodes,
            data_distribution=config.distributed.data_distribution,
            batch_size=config.data.batch_size,
            seed=config.seed,
            partition_file=config.distributed.partition_file,
        )
        true_weights = dataset.true_weights

    elif isinstance(config.data, RealDataConfig):
        # Auto-wire the number of labels for GLUE so switching glue_task does not
        # silently train with the wrong head size.
        if config.data.glue_task is not None and config.model.kind in ("hf_seq_cls", "hf_pretrained"):
            from framework.data.glue_loader import _GLUE_NUM_LABELS
            if config.data.glue_task in _GLUE_NUM_LABELS:
                config.model.num_classes = _GLUE_NUM_LABELS[config.data.glue_task]

        # Route to specialised loaders based on task type or explicit fields
        if config.data.glue_task is not None:
            from framework.data.glue_loader import build_glue_dataloaders
            train_loaders, val_loader, test_loader = build_glue_dataloaders(
                config.data,
                num_nodes=config.distributed.num_nodes,
                distribution=config.distributed.data_distribution,
                seed=config.seed,
            )
        elif config.data.src_lang is not None and config.data.tgt_lang is not None:
            from framework.data.wmt_loader import build_wmt_dataloaders
            train_loaders, val_loader, test_loader = build_wmt_dataloaders(
                config.data,
                num_nodes=config.distributed.num_nodes,
                distribution=config.distributed.data_distribution,
                seed=config.seed,
            )
        else:
            from framework.data.loaders import build_dataloaders
            train_loaders, val_loader, test_loader = build_dataloaders(
                config.data,
                num_nodes=config.distributed.num_nodes,
                distribution=config.distributed.data_distribution,
                seed=config.seed,
            )
    else:
        raise ValueError(f"Unknown data config type: {type(config.data)}")

    model = build_model(config.model, true_weights=true_weights)

    # For generative (seq2seq) models, text metrics like BLEU/METEOR require a
    # tokenizer to decode generated ids back to strings during evaluation.
    tokenizer = None
    if (
        isinstance(config.data, RealDataConfig)
        and config.data.tokenizer_name
        and hasattr(model, "generate")
        and any(m in ("bleu", "meteor") for m in config.metrics)
    ):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
        except Exception as e:
            print(f"[run] could not load tokenizer for text metrics: {e}")

    mode = config.distributed.mode
    if mode == "centralized":
        final_metrics = run_centralized(
            config, model,
            train_loader=train_loaders[0],
            val_loader=val_loader,
            test_loader=test_loader,
            true_weights=true_weights,
            tokenizer=tokenizer,
        )
    elif mode == "distributed":
        final_metrics = run_distributed(
            config, model,
            train_loaders=train_loaders,
            val_loader=val_loader,
            test_loader=test_loader,
            true_weights=true_weights,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(
            f"Unknown distributed.mode '{mode}'. Choose 'centralized' or 'distributed'."
        )

    _auto_plot(config)
    return final_metrics


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m framework.run <config.yaml>")
        sys.exit(1)
    config = load_config(sys.argv[1])
    final_metrics = run(config)
    print("\nFinal metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
