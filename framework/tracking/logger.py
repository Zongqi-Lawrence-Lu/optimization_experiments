"""Run-level logger.

Writes structured records in JSON Lines format to disk and optionally to stdout.
All log methods are no-ops when output_dir is None, so training loops never
need conditional logging guards.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class RunLogger:
    """Manages all log files for a single training run."""

    def __init__(
        self,
        run_name: str,
        output_dir: Optional[str],
        verbose: bool = True,
        results_dir: Optional[str] = None,
    ):
        self.run_name = run_name
        self.output_dir = output_dir
        self.verbose = verbose
        self._run_dir: Optional[Path] = None
        self._results_run_dir: Optional[Path] = None

        if output_dir is not None:
            self._run_dir = Path(output_dir) / run_name
            self._run_dir.mkdir(parents=True, exist_ok=True)
            self._checkpoint_dir = self._run_dir / "checkpoints"
            self._checkpoint_dir.mkdir(exist_ok=True)

            self._step_log = open(self._run_dir / "step_log.jsonl", "a")
            self._eval_log = open(self._run_dir / "eval_log.jsonl", "a")
            self._grad_log = open(self._run_dir / "gradient_stats.jsonl", "a")
            self._checkpoint_index: Dict[int, str] = {}
        else:
            self._step_log = None
            self._eval_log = None
            self._grad_log = None
            self._checkpoint_index = {}

        # Separate results directory for organized JSON summaries
        _rdir = results_dir or ("results" if output_dir is not None else None)
        if _rdir is not None:
            self._results_run_dir = Path(_rdir) / run_name
            self._results_run_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_loss: float = float("inf")
        self._best_step: int = -1
        self._final_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Step log
    # ------------------------------------------------------------------

    def log_step(self, outer_step: int, record: Dict[str, Any], inner_step: int = 0, node_id: int = 0) -> None:
        entry = {
            "outer_step": outer_step,
            "inner_step": inner_step,
            "node_id": node_id,
            **record,
        }
        if self._step_log is not None:
            self._step_log.write(json.dumps(entry) + "\n")
            self._step_log.flush()
        if self.verbose and outer_step % 10 == 0 and inner_step == 0 and node_id == 0:
            loss_str = f"loss={record.get('train_loss', 'N/A'):.4f}" if isinstance(record.get("train_loss"), float) else ""
            print(f"[step {outer_step}] {loss_str}", file=sys.stdout)

    # Metrics where higher values are better (all others default to lower-is-better)
    _HIGHER_IS_BETTER = frozenset({
        "accuracy", "f1_binary", "f1_macro", "mcc", "spearman", "pearson", "bleu", "meteor",
    })

    @staticmethod
    def _is_higher_better(metric_name: str) -> bool:
        base = metric_name.split("_", 1)[-1] if metric_name.startswith(("val_", "test_")) else metric_name
        return any(h in base for h in RunLogger._HIGHER_IS_BETTER)

    # ------------------------------------------------------------------
    # Eval log
    # ------------------------------------------------------------------

    def log_metrics(self, outer_step: int, metrics: Dict[str, float]) -> None:
        entry = {"outer_step": outer_step, **metrics}
        if self._eval_log is not None:
            self._eval_log.write(json.dumps(entry) + "\n")
            self._eval_log.flush()

        val_loss = metrics.get("val_loss", metrics.get("loss", float("inf")))
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_step = outer_step

        for name, value in metrics.items():
            key = f"{name}_best"
            higher_better = self._is_higher_better(name)
            if key not in self._final_metrics:
                self._final_metrics[key] = value
                self._final_metrics[f"{name}_best_step"] = outer_step
            elif higher_better and value > self._final_metrics[key]:
                self._final_metrics[key] = value
                self._final_metrics[f"{name}_best_step"] = outer_step
            elif not higher_better and value < self._final_metrics[key]:
                self._final_metrics[key] = value
                self._final_metrics[f"{name}_best_step"] = outer_step
            self._final_metrics[f"{name}_final"] = value
            self._final_metrics[f"{name}_final_step"] = outer_step

        if self.verbose:
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
            print(f"[eval step {outer_step}] {metric_str}", file=sys.stdout)

    # ------------------------------------------------------------------
    # Gradient stats log
    # ------------------------------------------------------------------

    def log_gradient_stats(self, outer_step: int, stats_list: List[Dict[str, Any]]) -> None:
        for stats in stats_list:
            entry = {"outer_step": outer_step, **stats}
            if self._grad_log is not None:
                self._grad_log.write(json.dumps(entry) + "\n")
                self._grad_log.flush()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        outer_step: int,
        model: nn.Module,
        outer_optimizer_state: dict,
        is_best: bool = False,
        extra: Optional[dict] = None,
    ) -> None:
        if self._run_dir is None:
            return

        state = {
            "outer_step": outer_step,
            "model_state_dict": model.state_dict(),
            "outer_optimizer_state": outer_optimizer_state,
        }
        if extra:
            state.update(extra)
        path = self._checkpoint_dir / f"step_{outer_step}.pt"
        torch.save(state, str(path))
        self._checkpoint_index[outer_step] = str(path)
        self._save_checkpoint_index()

        if is_best:
            best_path = self._checkpoint_dir / "best.pt"
            torch.save(state, str(best_path))

    def _save_checkpoint_index(self) -> None:
        if self._run_dir is None:
            return
        index_path = self._run_dir / "checkpoints.json"
        with open(index_path, "w") as f:
            json.dump(self._checkpoint_index, f, indent=2)

    # ------------------------------------------------------------------
    # Config snapshot and final metrics
    # ------------------------------------------------------------------

    def save_config(self, config_dict: dict) -> None:
        import yaml
        if self._run_dir is not None:
            config_path = self._run_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        if self._results_run_dir is not None:
            config_path = self._results_run_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def save_final_metrics(self) -> None:
        self._final_metrics["best_val_loss"] = self._best_val_loss
        self._final_metrics["best_step"] = self._best_step
        if self._run_dir is not None:
            metrics_path = self._run_dir / "final_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self._final_metrics, f, indent=2)
        if self._results_run_dir is not None:
            metrics_path = self._results_run_dir / "summary.json"
            with open(metrics_path, "w") as f:
                json.dump(self._final_metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.save_final_metrics()
        for fh in [self._step_log, self._eval_log, self._grad_log]:
            if fh is not None:
                fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
