"""Outer optimizer implementations.

Outer optimizers receive the aggregated pseudo-gradient Δ (average of local model-weight
displacements) and update the global model. They maintain their own moment buffers,
which are never transmitted to nodes.
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from framework.configs import OuterOptimizerConfig
from framework.optimizers.clipping import build_clipping_operator


class OuterOptimizer(ABC):
    """Abstract base class for outer optimizers."""

    def _init_warmup(self, config: OuterOptimizerConfig) -> None:
        self._warmup_steps = config.warmup_steps
        self._base_lr = config.lr
        self._warmup_max_lr = config.warmup_max_lr if config.warmup_max_lr is not None else config.lr
        self._warmup_schedule = config.warmup_schedule
        self._warmup_t: int = 0

    def _get_lr(self) -> float:
        self._warmup_t += 1
        t = self._warmup_t
        if self._warmup_steps <= 0 or t > self._warmup_steps:
            return self._base_lr
        frac = t / self._warmup_steps
        if self._warmup_schedule == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * frac))
        return self._warmup_max_lr * frac

    @abstractmethod
    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        """Update global_model in-place using pseudo_grad; return diagnostics."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class AverageOuter(OuterOptimizer):
    """Simple averaging: x ← x + Δ (FedAvg outer step with lr=1)."""

    def __init__(self, config: OuterOptimizerConfig):
        self._init_warmup(config)

    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        lr = self._get_lr()
        delta_norm = 0.0
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name in pseudo_grad:
                    delta = pseudo_grad[name]
                    p.data += lr * delta
                    delta_norm += delta.norm(2).item() ** 2
        return {"delta_norm": delta_norm ** 0.5, "outer_lr": lr}


class SGDOuter(OuterOptimizer):
    """SGD outer step with optional momentum and clipping on Δ."""

    def __init__(self, config: OuterOptimizerConfig):
        self.beta1 = config.beta1
        self.clip = build_clipping_operator(config.clipping)
        self._momentum: Dict[str, Tensor] = {}
        self._init_warmup(config)

    def state_dict(self):
        return {"momentum": {k: v.cpu() for k, v in self._momentum.items()}, "warmup_t": self._warmup_t}

    def load_state_dict(self, state):
        self._momentum = {k: v for k, v in state.get("momentum", {}).items()}
        self._warmup_t = state.get("warmup_t", self._warmup_t)

    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        lr = self._get_lr()
        named_deltas = list(pseudo_grad.items())
        clipped = dict(self.clip.clip_grad_list(named_deltas))

        delta_norm = 0.0
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name not in clipped:
                    continue
                g = clipped[name]
                if self.beta1 > 0:
                    if name not in self._momentum:
                        self._momentum[name] = torch.zeros_like(g)
                    self._momentum[name] = self.beta1 * self._momentum[name] + g
                    g = self._momentum[name]
                p.data += lr * g
                delta_norm += g.norm(2).item() ** 2

        return {"delta_norm": delta_norm ** 0.5, "outer_lr": lr}


class AdagradOuter(OuterOptimizer):
    """Adagrad outer step: accumulates squared pseudo-gradients."""

    def __init__(self, config: OuterOptimizerConfig):
        self.eps = config.eps
        self.clip = build_clipping_operator(config.clipping)
        self._G: Dict[str, Tensor] = {}
        self._init_warmup(config)

    def state_dict(self):
        return {"G": {k: v.cpu() for k, v in self._G.items()}, "warmup_t": self._warmup_t}

    def load_state_dict(self, state):
        self._G = {k: v for k, v in state.get("G", {}).items()}
        self._warmup_t = state.get("warmup_t", self._warmup_t)

    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        lr = self._get_lr()
        named_deltas = list(pseudo_grad.items())
        clipped = dict(self.clip.clip_grad_list(named_deltas))

        delta_norm = 0.0
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name not in clipped:
                    continue
                g = clipped[name]
                if name not in self._G:
                    self._G[name] = torch.zeros_like(g)
                self._G[name] += g ** 2
                update = lr * g / (self._G[name].sqrt() + self.eps)
                p.data += update
                delta_norm += update.norm(2).item() ** 2

        return {"delta_norm": delta_norm ** 0.5, "outer_lr": lr}


class AdagradNormOuter(OuterOptimizer):
    """Adagrad-norm outer step: uses a single global accumulated squared norm as preconditioner.

    update = lr * Δ / (sqrt(G_scalar) + eps)  where G_scalar = sum_t ||Δ_t||^2
    """

    def __init__(self, config: OuterOptimizerConfig):
        self.eps = config.eps
        self.clip = build_clipping_operator(config.clipping)
        self._G_scalar: float = 0.0
        self._t = 0
        self._init_warmup(config)

    def state_dict(self):
        return {"G_scalar": self._G_scalar, "t": self._t, "warmup_t": self._warmup_t}

    def load_state_dict(self, state):
        self._G_scalar = state.get("G_scalar", 0.0)
        self._t = state.get("t", 0)
        self._warmup_t = state.get("warmup_t", self._warmup_t)

    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        self._t += 1
        lr = self._get_lr()
        named_deltas = list(pseudo_grad.items())
        clipped = dict(self.clip.clip_grad_list(named_deltas))

        sq_norm = sum(g.norm(2).item() ** 2 for g in clipped.values())
        self._G_scalar += sq_norm
        scale = lr / (self._G_scalar ** 0.5 + self.eps)

        delta_norm = 0.0
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name not in clipped:
                    continue
                g = clipped[name]
                update = scale * g
                p.data += update
                delta_norm += update.norm(2).item() ** 2

        return {"delta_norm": delta_norm ** 0.5, "outer_lr": lr, "adagrad_norm_scale": scale}


class RMSPropOuter(OuterOptimizer):
    """RMSProp outer step: EMA of squared pseudo-gradients."""

    def __init__(self, config: OuterOptimizerConfig):
        self.beta2 = config.beta2
        self.eps = config.eps
        self.clip = build_clipping_operator(config.clipping)
        self._v: Dict[str, Tensor] = {}
        self._init_warmup(config)

    def state_dict(self):
        return {"v": {k: v.cpu() for k, v in self._v.items()}, "warmup_t": self._warmup_t}

    def load_state_dict(self, state):
        self._v = {k: v for k, v in state.get("v", {}).items()}
        self._warmup_t = state.get("warmup_t", self._warmup_t)

    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        lr = self._get_lr()
        named_deltas = list(pseudo_grad.items())
        clipped = dict(self.clip.clip_grad_list(named_deltas))

        delta_norm = 0.0
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name not in clipped:
                    continue
                g = clipped[name]
                if name not in self._v:
                    self._v[name] = torch.zeros_like(g)
                self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g ** 2
                update = lr * g / (self._v[name].sqrt() + self.eps)
                p.data += update
                delta_norm += update.norm(2).item() ** 2

        return {"delta_norm": delta_norm ** 0.5, "outer_lr": lr}


class AdamOuter(OuterOptimizer):
    """Adam outer step on pseudo-gradients."""

    def __init__(self, config: OuterOptimizerConfig):
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps
        self.clip = build_clipping_operator(config.clipping)
        self._m: Dict[str, Tensor] = {}
        self._v: Dict[str, Tensor] = {}
        self._t = 0
        self._init_warmup(config)

    def state_dict(self):
        return {
            "m": {k: v.cpu() for k, v in self._m.items()},
            "v": {k: v.cpu() for k, v in self._v.items()},
            "t": self._t,
            "warmup_t": self._warmup_t,
        }

    def load_state_dict(self, state):
        self._m = {k: v for k, v in state.get("m", {}).items()}
        self._v = {k: v for k, v in state.get("v", {}).items()}
        self._t = state.get("t", 0)
        self._warmup_t = state.get("warmup_t", self._warmup_t)

    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        self._t += 1
        lr = self._get_lr()
        named_deltas = list(pseudo_grad.items())
        clipped = dict(self.clip.clip_grad_list(named_deltas))

        bias_c1 = 1.0 - self.beta1 ** self._t
        bias_c2 = 1.0 - self.beta2 ** self._t

        delta_norm = 0.0
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name not in clipped:
                    continue
                g = clipped[name]
                if name not in self._m:
                    self._m[name] = torch.zeros_like(g)
                    self._v[name] = torch.zeros_like(g)
                self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * g
                self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g ** 2
                m_hat = self._m[name] / bias_c1
                v_hat = self._v[name] / bias_c2
                update = lr * m_hat / (v_hat.sqrt() + self.eps)
                p.data += update
                delta_norm += update.norm(2).item() ** 2

        return {"delta_norm": delta_norm ** 0.5, "outer_step": self._t, "outer_lr": lr}


class AdamWOuter(OuterOptimizer):
    """Adam with decoupled weight decay on the outer step."""

    def __init__(self, config: OuterOptimizerConfig):
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps
        self.weight_decay = config.weight_decay
        self.clip = build_clipping_operator(config.clipping)
        self._m: Dict[str, Tensor] = {}
        self._v: Dict[str, Tensor] = {}
        self._t = 0
        self._init_warmup(config)

    def state_dict(self):
        return {
            "m": {k: v.cpu() for k, v in self._m.items()},
            "v": {k: v.cpu() for k, v in self._v.items()},
            "t": self._t,
            "warmup_t": self._warmup_t,
        }

    def load_state_dict(self, state):
        self._m = {k: v for k, v in state.get("m", {}).items()}
        self._v = {k: v for k, v in state.get("v", {}).items()}
        self._t = state.get("t", 0)
        self._warmup_t = state.get("warmup_t", self._warmup_t)

    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        self._t += 1
        lr = self._get_lr()
        named_deltas = list(pseudo_grad.items())
        clipped = dict(self.clip.clip_grad_list(named_deltas))

        bias_c1 = 1.0 - self.beta1 ** self._t
        bias_c2 = 1.0 - self.beta2 ** self._t

        delta_norm = 0.0
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name not in clipped:
                    continue
                g = clipped[name]
                if name not in self._m:
                    self._m[name] = torch.zeros_like(g)
                    self._v[name] = torch.zeros_like(g)
                self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * g
                self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g ** 2
                m_hat = self._m[name] / bias_c1
                v_hat = self._v[name] / bias_c2
                p.data -= self.weight_decay * lr * p.data
                update = lr * m_hat / (v_hat.sqrt() + self.eps)
                p.data += update
                delta_norm += update.norm(2).item() ** 2

        return {"delta_norm": delta_norm ** 0.5, "outer_lr": lr}


class ClippedOuter(OuterOptimizer):
    """Clips Δ then applies a plain SGD step."""

    def __init__(self, config: OuterOptimizerConfig):
        self.clip = build_clipping_operator(config.clipping)
        self._init_warmup(config)

    def state_dict(self):
        return {"warmup_t": self._warmup_t}

    def load_state_dict(self, state):
        self._warmup_t = state.get("warmup_t", self._warmup_t)

    def step(self, global_model: nn.Module, pseudo_grad: Dict[str, Tensor]) -> dict:
        lr = self._get_lr()
        named_deltas = list(pseudo_grad.items())
        clipped = dict(self.clip.clip_grad_list(named_deltas))

        delta_norm = 0.0
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name not in clipped:
                    continue
                g = clipped[name]
                p.data += lr * g
                delta_norm += g.norm(2).item() ** 2

        return {"delta_norm": delta_norm ** 0.5, "outer_lr": lr}


def _load_custom_outer(config: OuterOptimizerConfig) -> OuterOptimizer:
    module_path, fn_name = config.custom_factory.rsplit(".", 1)
    module = importlib.import_module(module_path)
    factory = getattr(module, fn_name)
    return factory(config)


def build_outer_optimizer(config: OuterOptimizerConfig) -> OuterOptimizer:
    dispatch = {
        "average": AverageOuter,
        "sgd": SGDOuter,
        "adagrad": AdagradOuter,
        "adagrad_norm": AdagradNormOuter,
        "rmsprop": RMSPropOuter,
        "adam": AdamOuter,
        "adamw": AdamWOuter,
        "clipped": ClippedOuter,
    }
    if config.name == "custom":
        return _load_custom_outer(config)
    cls = dispatch.get(config.name)
    if cls is None:
        raise ValueError(f"Unknown outer optimizer: '{config.name}'")
    return cls(config)
