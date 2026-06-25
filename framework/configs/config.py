"""Structured config dataclasses for the optimization framework.

All hyperparameters flow through these configs; no magic strings or hardcoded
constants should appear in training loops.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ClippingConfig:
    # Operation type: "none" | "upper" | "biclip"
    clip_type: str = "none"
    # Scope: "global" | "layerwise" | "coordinate"
    clip_scope: str = "global"
    upper: float = 1.0           # upper clipping threshold
    lower: float = 0.0           # lower threshold (biclip modes only)
    layer_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Dynamic threshold: derive upper (and lower) from gradient statistics each step
    dynamic: bool = False
    dynamic_percentile: float = 0.9   # for coordinate scope: percentile of |g| used as upper
    dynamic_ema_decay: float = 0.99   # for global/layerwise scope: EMA decay for norm history

    def __post_init__(self):
        valid_types = {"none", "upper", "biclip"}
        valid_scopes = {"global", "layerwise", "coordinate"}
        if self.clip_type not in valid_types:
            raise ValueError(f"ClippingConfig.clip_type must be one of {valid_types}, got '{self.clip_type}'")
        if self.clip_scope not in valid_scopes:
            raise ValueError(f"ClippingConfig.clip_scope must be one of {valid_scopes}, got '{self.clip_scope}'")
        if self.clip_type != "none":
            if self.upper <= 0:
                raise ValueError(f"ClippingConfig.upper must be positive, got {self.upper}")
            if self.lower < 0:
                raise ValueError(f"ClippingConfig.lower must be non-negative, got {self.lower}")
            if self.lower > self.upper:
                raise ValueError(f"ClippingConfig.lower ({self.lower}) must be <= upper ({self.upper})")

    # Legacy convenience: synthesise a mode string for old code paths
    @property
    def mode(self) -> str:
        if self.clip_type == "none":
            return "none"
        if self.clip_type == "upper":
            if self.clip_scope == "global":
                return "l2"
            if self.clip_scope == "layerwise":
                return "layerwise"
            return "upper_coord"   # coordinate upper
        # biclip
        if self.clip_scope == "global":
            return "biclip_global"
        if self.clip_scope == "layerwise":
            return "biclip_layerwise"
        return "bidirectional_coord"


def _clipping_from_legacy(d: dict) -> ClippingConfig:
    """Reconstruct a ClippingConfig from a dict that may use the legacy 'mode' key."""
    d = dict(d)
    if "mode" in d:
        mode = d.pop("mode")
        _mode_to_type_scope = {
            "none":              ("none",  "global"),
            "l2":                ("upper", "global"),
            "bidirectional_coord": ("biclip", "coordinate"),
            "layerwise":         ("upper", "layerwise"),
            "upper_coord":       ("upper", "coordinate"),
            "biclip_global":     ("biclip", "global"),
            "biclip_layerwise":  ("biclip", "layerwise"),
        }
        if mode not in _mode_to_type_scope:
            raise ValueError(f"Unknown legacy clipping mode: '{mode}'")
        clip_type, clip_scope = _mode_to_type_scope[mode]
        # Special case: layerwise with lower > 0 → biclip_layerwise
        if mode == "layerwise" and d.get("lower", 0.0) > 0:
            clip_type = "biclip"
        d.setdefault("clip_type", clip_type)
        d.setdefault("clip_scope", clip_scope)
    return ClippingConfig(**d)


@dataclass
class InnerOptimizerConfig:
    name: str = "sgd"           # "sgd" | "adam" | "adagrad" | "adagrad_norm" | "rmsprop" | "adamw" | "custom"
    lr: float = 0.01
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    custom_factory: Optional[str] = None
    # Learning rate warmup
    warmup_steps: int = 0               # 0 = no warmup
    warmup_max_lr: Optional[float] = None  # peak LR; defaults to lr if None
    warmup_schedule: str = "linear"     # "linear" | "cosine"

    def __post_init__(self):
        valid_names = {"sgd", "sgd_l2clip", "sgd_bidir", "adam", "adagrad", "adagrad_norm",
                       "rmsprop", "adamw", "custom"}
        if self.name not in valid_names:
            raise ValueError(f"InnerOptimizerConfig.name must be one of {valid_names}, got '{self.name}'")
        if self.name == "custom" and self.custom_factory is None:
            raise ValueError("custom_factory must be set when name='custom'")
        if not isinstance(self.clipping, ClippingConfig):
            self.clipping = _clipping_from_legacy(self.clipping)
        valid_warmup = {"linear", "cosine"}
        if self.warmup_schedule not in valid_warmup:
            raise ValueError(f"warmup_schedule must be one of {valid_warmup}, got '{self.warmup_schedule}'")


@dataclass
class OuterOptimizerConfig:
    name: str = "average"       # "average" | "sgd" | "adagrad" | "adagrad_norm" | "rmsprop" | "adam" | "adamw" | "clipped" | "custom"
    lr: float = 1.0
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    custom_factory: Optional[str] = None
    # Learning rate warmup (applied per outer step)
    warmup_steps: int = 0
    warmup_max_lr: Optional[float] = None
    warmup_schedule: str = "linear"

    def __post_init__(self):
        valid_names = {"average", "sgd", "adagrad", "adagrad_norm", "rmsprop", "adam", "adamw", "clipped", "custom"}
        if self.name not in valid_names:
            raise ValueError(f"OuterOptimizerConfig.name must be one of {valid_names}, got '{self.name}'")
        if self.name == "custom" and self.custom_factory is None:
            raise ValueError("custom_factory must be set when name='custom'")
        if not isinstance(self.clipping, ClippingConfig):
            self.clipping = _clipping_from_legacy(self.clipping)
        valid_warmup = {"linear", "cosine"}
        if self.warmup_schedule not in valid_warmup:
            raise ValueError(f"warmup_schedule must be one of {valid_warmup}, got '{self.warmup_schedule}'")


@dataclass
class DistributedConfig:
    mode: str = "centralized"   # "centralized" | "distributed"
    num_nodes: int = 1
    local_steps: int = 1
    data_distribution: str = "iid"  # "iid" | "noniid"
    participation_rate: float = 1.0
    partition_file: Optional[str] = None

    def __post_init__(self):
        valid_modes = {"centralized", "distributed"}
        if self.mode not in valid_modes:
            raise ValueError(f"DistributedConfig.mode must be one of {valid_modes}")
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        if self.local_steps < 1:
            raise ValueError("local_steps must be >= 1")
        if not (0.0 < self.participation_rate <= 1.0):
            raise ValueError("participation_rate must be in (0, 1]")


@dataclass
class SyntheticDataConfig:
    task: str = "regression"
    num_samples: int = 1000
    num_features: int = 10
    noise_distribution: str = "gaussian"   # "gaussian" | "student_t" | "cauchy" | "uniform" | "none"
    noise_scale: float = 0.1
    noise_df: float = 3.0                  # degrees of freedom for student_t
    feature_distribution: str = "gaussian" # "gaussian" | "bernoulli_mixed"
    common_feature_fraction: float = 0.5
    common_prob: float = 0.9
    rare_prob: float = 0.1
    random_seed: int = 42
    batch_size: int = 32
    dirichlet_alpha: float = 0.5           # concentration for non-IID Dirichlet partition


@dataclass
class RealDataConfig:
    dataset_name: str = ""
    task_type: str = "classification"  # "classification" | "seq2seq" | "token_classification" | "regression"
    split_train: str = "train"
    split_val: str = "validation"
    split_test: str = "test"
    tokenizer_name: Optional[str] = None
    max_seq_len: Optional[int] = None
    batch_size: int = 32
    num_workers: int = 0
    # GLUE-specific
    glue_task: Optional[str] = None        # e.g. "sst2", "mnli", "qqp", ...
    # WMT-specific
    src_lang: Optional[str] = None         # e.g. "en"
    tgt_lang: Optional[str] = None         # e.g. "de"


@dataclass
class ModelConfig:
    kind: str = "linear"           # "linear" | "hf_seq_cls" | "hf_seq2seq" | "torchvision" | "custom"
    in_features: int = 10          # for "linear"
    pretrained_name: str = ""      # for HuggingFace models
    arch_name: str = ""            # for "torchvision"
    pretrained: bool = False
    num_classes: int = 10
    custom_factory: Optional[str] = None

    def __post_init__(self):
        if self.kind == "custom" and self.custom_factory is None:
            raise ValueError("custom_factory must be set when kind='custom'")
        # Legacy alias
        if self.kind == "hf_pretrained":
            self.kind = "hf_seq_cls"


@dataclass
class TrainingConfig:
    run_name: str = "run"
    seed: int = 42
    total_outer_steps: int = 100
    eval_every: int = 10
    log_gradients_every: int = 10
    checkpoint_every: int = 50
    checkpoint_interval_minutes: Optional[float] = 10.0  # also checkpoint every N wall-clock minutes
    resume: bool = False           # if True, resume from latest checkpoint in output_dir/run_name
    output_dir: Optional[str] = None   # checkpoints and raw logs go here
    results_dir: Optional[str] = None  # organized JSON summaries go here (defaults to "results/")
    plots_dir: Optional[str] = None    # auto-generated plots go here (defaults to "plots/")
    deterministic: bool = False
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: Union[SyntheticDataConfig, RealDataConfig] = field(default_factory=SyntheticDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inner_optimizer: InnerOptimizerConfig = field(default_factory=InnerOptimizerConfig)
    outer_optimizer: OuterOptimizerConfig = field(default_factory=OuterOptimizerConfig)
    metrics: List[str] = field(default_factory=lambda: ["loss"])
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.distributed, DistributedConfig):
            self.distributed = DistributedConfig(**self.distributed)
        if not isinstance(self.inner_optimizer, InnerOptimizerConfig):
            cfg = dict(self.inner_optimizer)
            if isinstance(cfg.get("clipping"), dict):
                cfg["clipping"] = _clipping_from_legacy(cfg["clipping"])
            self.inner_optimizer = InnerOptimizerConfig(**cfg)
        if not isinstance(self.outer_optimizer, OuterOptimizerConfig):
            cfg = dict(self.outer_optimizer)
            if isinstance(cfg.get("clipping"), dict):
                cfg["clipping"] = _clipping_from_legacy(cfg["clipping"])
            self.outer_optimizer = OuterOptimizerConfig(**cfg)
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(**self.model)
        if not isinstance(self.data, (SyntheticDataConfig, RealDataConfig)):
            d = self.data
            if "dataset_name" in d:
                self.data = RealDataConfig(**d)
            else:
                self.data = SyntheticDataConfig(**d)


def save_config(config: TrainingConfig, path: str) -> None:
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    data = asdict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(path: str) -> TrainingConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return _dict_to_training_config(data)


def _dict_to_training_config(d: dict) -> TrainingConfig:
    """Recursively reconstruct a TrainingConfig from a plain dict."""
    d = dict(d)

    if "distributed" in d and isinstance(d["distributed"], dict):
        d["distributed"] = DistributedConfig(**d["distributed"])

    if "inner_optimizer" in d and isinstance(d["inner_optimizer"], dict):
        io = dict(d["inner_optimizer"])
        if "clipping" in io and isinstance(io["clipping"], dict):
            io["clipping"] = _clipping_from_legacy(io["clipping"])
        d["inner_optimizer"] = InnerOptimizerConfig(**io)

    if "outer_optimizer" in d and isinstance(d["outer_optimizer"], dict):
        oo = dict(d["outer_optimizer"])
        if "clipping" in oo and isinstance(oo["clipping"], dict):
            oo["clipping"] = _clipping_from_legacy(oo["clipping"])
        d["outer_optimizer"] = OuterOptimizerConfig(**oo)

    if "model" in d and isinstance(d["model"], dict):
        d["model"] = ModelConfig(**d["model"])

    if "data" in d and isinstance(d["data"], dict):
        data_dict = d["data"]
        if "dataset_name" in data_dict:
            d["data"] = RealDataConfig(**data_dict)
        else:
            d["data"] = SyntheticDataConfig(**data_dict)

    return TrainingConfig(**d)
