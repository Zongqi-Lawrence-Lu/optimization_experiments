"""Grid search: Cartesian product over all axis value lists."""

from __future__ import annotations

import copy
import itertools
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import yaml

from framework.configs import TrainingConfig, save_config, _dict_to_training_config


def _set_nested(d: dict, dotpath: str, value: Any) -> dict:
    """Set a value in a nested dict using a dot-path key like 'inner_optimizer.lr'."""
    keys = dotpath.split(".")
    node = d
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value
    return d


def _get_nested(d: dict, dotpath: str) -> Any:
    keys = dotpath.split(".")
    node = d
    for key in keys:
        node = node[key]
    return node


def expand_grid(
    base_config: TrainingConfig,
    axes: List[Dict[str, Any]],
) -> List[Tuple[str, TrainingConfig, Dict[str, Any]]]:
    """Generate all (trial_id, config, hyperparams) combinations.

    Parameters
    ----------
    base_config : TrainingConfig
        The base configuration to modify.
    axes : list of dicts with keys 'field' and 'values'
        Each axis specifies a dot-path field and a list of candidate values.

    Returns
    -------
    List of (trial_id, resolved_config, hyperparams_dict)
    """
    fields = [ax["field"] for ax in axes]
    value_lists = [ax["values"] for ax in axes]

    trials = []
    for trial_num, combo in enumerate(itertools.product(*value_lists)):
        trial_id = f"trial_{trial_num:04d}"
        base_dict = asdict(base_config)
        hp = {}
        for field, value in zip(fields, combo):
            _set_nested(base_dict, field, value)
            hp[field] = value
        trial_config = _dict_to_training_config(base_dict)
        trials.append((trial_id, trial_config, hp))

    return trials


def write_grid_trials(
    base_config: TrainingConfig,
    axes: List[Dict[str, Any]],
    sweep_dir: str,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Write each grid trial config to disk.

    Returns list of (trial_id, config_path, hyperparams_dict).
    """
    trials = expand_grid(base_config, axes)
    os.makedirs(sweep_dir, exist_ok=True)
    result = []
    for trial_id, trial_config, hp in trials:
        trial_dir = os.path.join(sweep_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        # Override run_name and output_dir so logs go to the trial subdir
        from dataclasses import replace
        trial_config = replace(
            trial_config,
            run_name=trial_id,
            output_dir=sweep_dir,
        )
        config_path = os.path.join(trial_dir, "config.yaml")
        save_config(trial_config, config_path)
        result.append((trial_id, config_path, hp))
    return result
