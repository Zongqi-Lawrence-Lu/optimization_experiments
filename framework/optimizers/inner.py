"""Inner optimizer implementations.

Each inner optimizer performs one (or more) gradient update steps on a local model copy
and returns scalar diagnostics. Clipping is applied to the raw stochastic gradient
*before* it is fed into the adaptive update rule.

All optimizers expose:
  step(model, batch)              -> diagnostics dict
  step_accumulated(model, batches) -> diagnostics dict  (mean gradient, ONE update)
  _last_raw_grads                  list[(name, Tensor)] raw gradients from the last update
  _last_clipped_grads              list[(name, Tensor)] clipped gradients from the last update
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from framework.configs import InnerOptimizerConfig
from framework.optimizers.clipping import build_clipping_operator, compute_gradient_stats


class InnerOptimizer(ABC):
    def __init__(self):
        self._last_raw_grads: List[Tuple[str, Tensor]] = []
        self._last_clipped_grads: List[Tuple[str, Tensor]] = []
        # Warmup step counter — NOT reset by reset_state() so warmup spans distributed rounds
        self._warmup_t: int = 0
        self._warmup_steps: int = 0
        self._warmup_max_lr: float = 0.0
        self._warmup_schedule: str = "linear"
        self._base_lr: float = 0.0

    def _init_warmup(self, config: InnerOptimizerConfig) -> None:
        self._warmup_steps = config.warmup_steps
        self._base_lr = config.lr
        self._warmup_max_lr = config.warmup_max_lr if config.warmup_max_lr is not None else config.lr
        self._warmup_schedule = config.warmup_schedule

    def _get_lr(self) -> float:
        """Return effective lr at the current warmup step, then increment counter."""
        self._warmup_t += 1
        t = self._warmup_t
        if self._warmup_steps <= 0 or t > self._warmup_steps:
            return self._base_lr
        frac = t / self._warmup_steps
        if self._warmup_schedule == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * frac))
        # Interpolate from 0 (or a tiny value) to warmup_max_lr
        effective = self._warmup_max_lr * frac
        return effective

    @abstractmethod
    def step(self, model: nn.Module, batch: Any) -> dict:
        """Single mini-batch step: forward, backward, update. Returns diagnostics."""

    def _do_update(self, model: nn.Module, named_grads: List[Tuple[str, Tensor]], loss_val: float) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _do_update."
        )

    def step_accumulated(self, model: nn.Module, batches: List[Any]) -> dict:
        """Accumulate gradients over n batches, then do ONE parameter update."""
        n = len(batches)
        if n == 1:
            return self.step(model, batches[0])

        model.zero_grad(set_to_none=False)
        total_loss = 0.0
        for batch in batches:
            loss = model.compute_loss(batch)
            (loss / n).backward()
            total_loss += loss.item()

        named_grads = [
            (nm, p.grad.detach().clone())
            for nm, p in model.named_parameters()
            if p.grad is not None
        ]
        return self._do_update(model, named_grads, total_loss / n)

    def reset_state(self) -> None:
        """Reset stateful accumulators. Called before each node's local run.
        Note: _warmup_t is intentionally NOT reset here so warmup spans all rounds."""


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class SGDInner(InnerOptimizer):
    def __init__(self, config: InnerOptimizerConfig):
        super().__init__()
        self._base_lr = config.lr
        self.clip = build_clipping_operator(config.clipping)
        self.config = config
        self._init_warmup(config)

    def _do_update(self, model: nn.Module, named_grads: List[Tuple[str, Tensor]], loss_val: float) -> dict:
        lr = self._get_lr()
        clipped_grads = self.clip.clip_grad_list(named_grads)
        clipped_map = dict(clipped_grads)
        self._last_raw_grads = named_grads
        self._last_clipped_grads = clipped_grads

        grad_norm_raw = sum(g.norm(2).item() ** 2 for _, g in named_grads) ** 0.5
        grad_norm_clipped = sum(g.norm(2).item() ** 2 for _, g in clipped_grads) ** 0.5
        total_coords = sum(g.numel() for _, g in named_grads)
        clipped_coords = sum(
            (raw != clipped).sum().item()
            for (_, raw), (_, clipped) in zip(named_grads, clipped_grads)
        )

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None:
                    g = clipped_map.get(name, p.grad)
                    p.data -= lr * g
                    p.grad.zero_()

        return {
            "train_loss": loss_val,
            "grad_norm_raw": grad_norm_raw,
            "grad_norm_clipped": grad_norm_clipped,
            "clipped_fraction": clipped_coords / total_coords if total_coords > 0 else 0.0,
            "learning_rate": lr,
        }

    def step(self, model: nn.Module, batch: Any) -> dict:
        loss = model.compute_loss(batch)
        loss.backward()
        named_grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        return self._do_update(model, named_grads, loss.item())


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class AdamInner(InnerOptimizer):
    def __init__(self, config: InnerOptimizerConfig):
        super().__init__()
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps
        self.clip = build_clipping_operator(config.clipping)
        self._m: Dict[str, Tensor] = {}
        self._v: Dict[str, Tensor] = {}
        self._t = 0
        self._init_warmup(config)

    def reset_state(self):
        self._m.clear()
        self._v.clear()
        self._t = 0
        # _warmup_t is NOT reset intentionally

    def _do_update(self, model: nn.Module, named_grads: List[Tuple[str, Tensor]], loss_val: float) -> dict:
        self._t += 1
        lr = self._get_lr()
        clipped_grads = self.clip.clip_grad_list(named_grads)
        clipped_map = dict(clipped_grads)
        self._last_raw_grads = named_grads
        self._last_clipped_grads = clipped_grads

        bias_c1 = 1.0 - self.beta1 ** self._t
        bias_c2 = 1.0 - self.beta2 ** self._t

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = clipped_map[name]
                if name not in self._m:
                    self._m[name] = torch.zeros_like(g)
                    self._v[name] = torch.zeros_like(g)
                self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * g
                self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g ** 2
                m_hat = self._m[name] / bias_c1
                v_hat = self._v[name] / bias_c2
                p.data -= lr * m_hat / (v_hat.sqrt() + self.eps)
                p.grad.zero_()

        grad_norm = sum(g.norm(2).item() ** 2 for _, g in named_grads) ** 0.5
        return {"train_loss": loss_val, "grad_norm_raw": grad_norm, "learning_rate": lr}

    def step(self, model: nn.Module, batch: Any) -> dict:
        loss = model.compute_loss(batch)
        loss.backward()
        named_grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        return self._do_update(model, named_grads, loss.item())

    def state_dict(self) -> dict:
        return {"m": self._m, "v": self._v, "t": self._t, "warmup_t": self._warmup_t}

    def load_state_dict(self, state: dict) -> None:
        self._m = state.get("m", {})
        self._v = state.get("v", {})
        self._t = state.get("t", 0)
        self._warmup_t = state.get("warmup_t", self._warmup_t)


# ---------------------------------------------------------------------------
# Adagrad (coordinate-wise accumulated squared gradient)
# ---------------------------------------------------------------------------

class AdagradInner(InnerOptimizer):
    def __init__(self, config: InnerOptimizerConfig):
        super().__init__()
        self.eps = config.eps
        self.clip = build_clipping_operator(config.clipping)
        self._G: Dict[str, Tensor] = {}
        self._t = 0
        self._init_warmup(config)

    def reset_state(self):
        self._G.clear()
        self._t = 0

    def _do_update(self, model: nn.Module, named_grads: List[Tuple[str, Tensor]], loss_val: float) -> dict:
        self._t += 1
        lr = self._get_lr()
        clipped_grads = self.clip.clip_grad_list(named_grads)
        clipped_map = dict(clipped_grads)
        self._last_raw_grads = named_grads
        self._last_clipped_grads = clipped_grads

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = clipped_map[name]
                if name not in self._G:
                    self._G[name] = torch.zeros_like(g)
                self._G[name] += g ** 2
                p.data -= lr * g / (self._G[name].sqrt() + self.eps)
                p.grad.zero_()

        grad_norm = sum(g.norm(2).item() ** 2 for _, g in named_grads) ** 0.5
        return {"train_loss": loss_val, "grad_norm_raw": grad_norm, "learning_rate": lr}

    def step(self, model: nn.Module, batch: Any) -> dict:
        loss = model.compute_loss(batch)
        loss.backward()
        named_grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        return self._do_update(model, named_grads, loss.item())

    def state_dict(self) -> dict:
        return {"G": self._G, "t": self._t, "warmup_t": self._warmup_t}

    def load_state_dict(self, state: dict) -> None:
        self._G = state.get("G", {})
        self._t = state.get("t", 0)
        self._warmup_t = state.get("warmup_t", self._warmup_t)


# ---------------------------------------------------------------------------
# AdagradNorm (global accumulated squared gradient norm as scalar preconditioner)
# ---------------------------------------------------------------------------

class AdagradNormInner(InnerOptimizer):
    """Adagrad with a single global accumulated squared norm as preconditioner.

    The update is: p -= lr * g / (sqrt(G_scalar) + eps)
    where G_scalar = sum over t of ||g_t||^2.

    Unlike coordinate-wise Adagrad, this uses the same scalar step-size for
    all coordinates, scaled by the global gradient norm history. This is more
    appropriate when gradients are sparse in a structured way (e.g. token features).
    """

    def __init__(self, config: InnerOptimizerConfig):
        super().__init__()
        self.eps = config.eps
        self.clip = build_clipping_operator(config.clipping)
        self._G_scalar: float = 0.0  # accumulated sum of squared gradient norms
        self._t = 0
        self._init_warmup(config)

    def reset_state(self):
        self._G_scalar = 0.0
        self._t = 0

    def _do_update(self, model: nn.Module, named_grads: List[Tuple[str, Tensor]], loss_val: float) -> dict:
        self._t += 1
        lr = self._get_lr()
        clipped_grads = self.clip.clip_grad_list(named_grads)
        clipped_map = dict(clipped_grads)
        self._last_raw_grads = named_grads
        self._last_clipped_grads = clipped_grads

        # Accumulate global squared norm
        sq_norm = sum(g.norm(2).item() ** 2 for _, g in clipped_grads)
        self._G_scalar += sq_norm
        scale = lr / (self._G_scalar ** 0.5 + self.eps)

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = clipped_map[name]
                p.data -= scale * g
                p.grad.zero_()

        grad_norm = sum(g.norm(2).item() ** 2 for _, g in named_grads) ** 0.5
        return {
            "train_loss": loss_val,
            "grad_norm_raw": grad_norm,
            "learning_rate": lr,
            "adagrad_norm_scale": scale,
        }

    def step(self, model: nn.Module, batch: Any) -> dict:
        loss = model.compute_loss(batch)
        loss.backward()
        named_grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        return self._do_update(model, named_grads, loss.item())

    def state_dict(self) -> dict:
        return {"G_scalar": self._G_scalar, "t": self._t, "warmup_t": self._warmup_t}

    def load_state_dict(self, state: dict) -> None:
        self._G_scalar = state.get("G_scalar", 0.0)
        self._t = state.get("t", 0)
        self._warmup_t = state.get("warmup_t", self._warmup_t)


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------

class RMSPropInner(InnerOptimizer):
    def __init__(self, config: InnerOptimizerConfig):
        super().__init__()
        self.beta2 = config.beta2
        self.eps = config.eps
        self.clip = build_clipping_operator(config.clipping)
        self._v: Dict[str, Tensor] = {}
        self._init_warmup(config)

    def reset_state(self):
        self._v.clear()

    def _do_update(self, model: nn.Module, named_grads: List[Tuple[str, Tensor]], loss_val: float) -> dict:
        lr = self._get_lr()
        clipped_grads = self.clip.clip_grad_list(named_grads)
        clipped_map = dict(clipped_grads)
        self._last_raw_grads = named_grads
        self._last_clipped_grads = clipped_grads

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = clipped_map[name]
                if name not in self._v:
                    self._v[name] = torch.zeros_like(g)
                self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g ** 2
                p.data -= lr * g / (self._v[name].sqrt() + self.eps)
                p.grad.zero_()

        grad_norm = sum(g.norm(2).item() ** 2 for _, g in named_grads) ** 0.5
        return {"train_loss": loss_val, "grad_norm_raw": grad_norm, "learning_rate": lr}

    def step(self, model: nn.Module, batch: Any) -> dict:
        loss = model.compute_loss(batch)
        loss.backward()
        named_grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        return self._do_update(model, named_grads, loss.item())

    def state_dict(self) -> dict:
        return {"v": self._v, "warmup_t": self._warmup_t}

    def load_state_dict(self, state: dict) -> None:
        self._v = state.get("v", {})
        self._warmup_t = state.get("warmup_t", self._warmup_t)


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------

class AdamWInner(InnerOptimizer):
    def __init__(self, config: InnerOptimizerConfig):
        super().__init__()
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps
        self.weight_decay = config.weight_decay
        self.clip = build_clipping_operator(config.clipping)
        self._m: Dict[str, Tensor] = {}
        self._v: Dict[str, Tensor] = {}
        self._t = 0
        self._init_warmup(config)

    def reset_state(self):
        self._m.clear()
        self._v.clear()
        self._t = 0

    def _do_update(self, model: nn.Module, named_grads: List[Tuple[str, Tensor]], loss_val: float) -> dict:
        self._t += 1
        lr = self._get_lr()
        clipped_grads = self.clip.clip_grad_list(named_grads)
        clipped_map = dict(clipped_grads)
        self._last_raw_grads = named_grads
        self._last_clipped_grads = clipped_grads

        bias_c1 = 1.0 - self.beta1 ** self._t
        bias_c2 = 1.0 - self.beta2 ** self._t

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = clipped_map[name]
                if name not in self._m:
                    self._m[name] = torch.zeros_like(g)
                    self._v[name] = torch.zeros_like(g)
                self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * g
                self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g ** 2
                m_hat = self._m[name] / bias_c1
                v_hat = self._v[name] / bias_c2
                p.data -= self.weight_decay * lr * p.data
                p.data -= lr * m_hat / (v_hat.sqrt() + self.eps)
                p.grad.zero_()

        grad_norm = sum(g.norm(2).item() ** 2 for _, g in named_grads) ** 0.5
        return {"train_loss": loss_val, "grad_norm_raw": grad_norm, "learning_rate": lr}

    def step(self, model: nn.Module, batch: Any) -> dict:
        loss = model.compute_loss(batch)
        loss.backward()
        named_grads = [
            (name, p.grad.detach().clone())
            for name, p in model.named_parameters()
            if p.grad is not None
        ]
        return self._do_update(model, named_grads, loss.item())

    def state_dict(self) -> dict:
        return {"m": self._m, "v": self._v, "t": self._t, "warmup_t": self._warmup_t}

    def load_state_dict(self, state: dict) -> None:
        self._m = state.get("m", {})
        self._v = state.get("v", {})
        self._t = state.get("t", 0)
        self._warmup_t = state.get("warmup_t", self._warmup_t)


# ---------------------------------------------------------------------------
# Custom loader
# ---------------------------------------------------------------------------

def _load_custom_inner(config: InnerOptimizerConfig) -> InnerOptimizer:
    module_path, fn_name = config.custom_factory.rsplit(".", 1)
    module = importlib.import_module(module_path)
    factory = getattr(module, fn_name)
    return factory(config)


def build_inner_optimizer(config: InnerOptimizerConfig) -> InnerOptimizer:
    dispatch = {
        "sgd": SGDInner,
        "sgd_l2clip": SGDInner,   # legacy aliases; clipping is set via config.clipping
        "sgd_bidir": SGDInner,
        "adam": AdamInner,
        "adagrad": AdagradInner,
        "adagrad_norm": AdagradNormInner,
        "rmsprop": RMSPropInner,
        "adamw": AdamWInner,
    }
    if config.name == "custom":
        return _load_custom_inner(config)
    cls = dispatch.get(config.name)
    if cls is None:
        raise ValueError(f"Unknown inner optimizer: '{config.name}'")
    return cls(config)
