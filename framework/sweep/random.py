"""Random search: sample one value per axis uniformly (or log-uniformly)."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from framework.configs import TrainingConfig, save_config, _dict_to_training_config
from framework.sweep.grid import _set_nested


def _sample_value(
    values: List[Any],
    log_scale: bool,
    rng: np.random.RandomState,
) -> Any:
    """Sample a value uniformly at random, optionally on a log scale."""
    if log_scale and all(isinstance(v, (int, float)) for v in values):
        log_vals = [math.log(v) for v in values]
        log_sampled = rng.uniform(min(log_vals), max(log_vals))
        return math.exp(log_sampled)
    return values[rng.randint(len(values))]


def expand_random(
    base_config: TrainingConfig,
    axes: List[Dict[str, Any]],
    max_trials: int,
    seed: int = 0,
) -> List[Tuple[str, TrainingConfig, Dict[str, Any]]]:
    """Generate max_trials random configurations.

    Each axis dict may optionally include a 'log_scale' boolean key.
    """
    rng = np.random.RandomState(seed)
    trials = []
    for trial_num in range(max_trials):
        trial_id = f"trial_{trial_num:04d}"
        base_dict = asdict(base_config)
        hp = {}
        for ax in axes:
            field = ax["field"]
            values = ax["values"]
            log_scale = ax.get("log_scale", False)
            value = _sample_value(values, log_scale, rng)
            _set_nested(base_dict, field, value)
            hp[field] = value
        trial_config = _dict_to_training_config(base_dict)
        trials.append((trial_id, trial_config, hp))
    return trials


def write_random_trials(
    base_config: TrainingConfig,
    axes: List[Dict[str, Any]],
    sweep_dir: str,
    max_trials: int,
    seed: int = 0,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Write each random trial config to disk.

    Returns list of (trial_id, config_path, hyperparams_dict).
    """
    trials = expand_random(base_config, axes, max_trials, seed)
    os.makedirs(sweep_dir, exist_ok=True)
    result = []
    for trial_id, trial_config, hp in trials:
        trial_dir = os.path.join(sweep_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        trial_config = replace(
            trial_config,
            run_name=trial_id,
            output_dir=sweep_dir,
        )
        config_path = os.path.join(trial_dir, "config.yaml")
        save_config(trial_config, config_path)
        result.append((trial_id, config_path, hp))
    return result
