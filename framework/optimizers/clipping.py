"""Clipping operators: global L2, coordinate-wise, layer-wise, biclip variants, and dynamic.

All operators share a unified interface:
    clip = build_clipping_operator(config)
    clipped = clip(tensor)
    clipped_list = clip.clip_grad_list([(name, grad), ...])

Dynamic operators compute thresholds from the current gradient each call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from framework.configs import ClippingConfig


class ClippingOperator:
    """Base class; subclasses must implement __call__ and clip_grad_list."""

    def __init__(self, config: ClippingConfig):
        self.config = config

    def __call__(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        return [(name, self(g)) for name, g in grads]

    def diagnostics(self, raw: Tensor, clipped: Tensor) -> dict:
        total = raw.numel()
        clipped_count = (raw != clipped).sum().item()
        return {
            "grad_l2_norm_raw": raw.norm(2).item(),
            "grad_l2_norm_clipped": clipped.norm(2).item(),
            "grad_max_abs": raw.abs().max().item(),
            "grad_min_abs": raw.abs().min().item(),
            "clipped_fraction": clipped_count / total if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# No-op
# ---------------------------------------------------------------------------

class NoClipping(ClippingOperator):
    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor

    def clip_grad_list(self, grads):
        return grads


# ---------------------------------------------------------------------------
# Upper clipping — global (L2 norm rescaling)
# ---------------------------------------------------------------------------

class GlobalUpperClippingOperator(ClippingOperator):
    """Rescale the full gradient vector so its L2 norm ≤ upper.

    clip(g, C) = g * min(1, C / ||g||_2)
    """

    def __call__(self, tensor: Tensor) -> Tensor:
        norm = tensor.norm(2).item()
        if norm == 0:
            return tensor
        scale = min(1.0, self.config.upper / norm)
        return tensor * scale

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        flat = torch.cat([g.flatten() for _, g in grads])
        norm = flat.norm(2).item()
        if norm == 0:
            return grads
        scale = min(1.0, self.config.upper / norm)
        return [(name, g * scale) for name, g in grads]


# Alias for backward compatibility
L2ClippingOperator = GlobalUpperClippingOperator


# ---------------------------------------------------------------------------
# Upper clipping — coordinate-wise
# ---------------------------------------------------------------------------

class CoordUpperClippingOperator(ClippingOperator):
    """Clip each element to [−upper, +upper]."""

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.clamp(-self.config.upper, self.config.upper)


# ---------------------------------------------------------------------------
# Biclip — coordinate-wise (amplify small, clip large)
# ---------------------------------------------------------------------------

class BidirectionalCoordClippingOperator(ClippingOperator):
    """Element-wise bidirectional clipping.

    sign(v) * lower    if |v| < lower   (amplify small signals)
    v                  if lower ≤ |v| ≤ upper
    sign(v) * upper    if |v| > upper
    """

    def __call__(self, tensor: Tensor) -> Tensor:
        upper = self.config.upper
        lower = self.config.lower
        abs_t = tensor.abs()
        sign_t = tensor.sign()

        result = tensor.clone()
        if lower > 0:
            below_mask = abs_t < lower
            result = torch.where(below_mask, sign_t * lower, result)
        above_mask = abs_t > upper
        result = torch.where(above_mask, sign_t * upper, result)
        return result


# ---------------------------------------------------------------------------
# Biclip — global (scale global norm into [lower, upper])
# ---------------------------------------------------------------------------

class GlobalBiclipOperator(ClippingOperator):
    """Scale the full gradient vector so its L2 norm ∈ [lower, upper].

    If ||g|| < lower: g ← g * (lower / ||g||)   (amplify)
    If ||g|| > upper: g ← g * (upper / ||g||)   (clip)
    """

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._scale(tensor, self.config.upper, self.config.lower)

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        flat = torch.cat([g.flatten() for _, g in grads])
        norm = flat.norm(2).item()
        if norm == 0:
            return grads
        upper, lower = self.config.upper, self.config.lower
        if norm > upper:
            scale = upper / norm
        elif lower > 0 and norm < lower:
            scale = lower / norm
        else:
            return grads
        return [(name, g * scale) for name, g in grads]

    @staticmethod
    def _scale(tensor: Tensor, upper: float, lower: float) -> Tensor:
        norm = tensor.norm(2).item()
        if norm == 0:
            return tensor
        if norm > upper:
            return tensor * (upper / norm)
        if lower > 0 and norm < lower:
            return tensor * (lower / norm)
        return tensor


# ---------------------------------------------------------------------------
# Layer-wise clipping (upper or biclip per parameter tensor)
# ---------------------------------------------------------------------------

class LayerwiseClippingOperator(ClippingOperator):
    """Apply per-layer clipping. layer_overrides gives per-prefix upper/lower overrides."""

    def __init__(self, config: ClippingConfig):
        super().__init__(config)
        # Determine the per-tensor operation from clip_type
        self.is_biclip = (config.clip_type == "biclip")

    def _get_thresholds(self, param_name: str) -> Tuple[float, float]:
        for prefix, overrides in self.config.layer_overrides.items():
            if param_name.startswith(prefix):
                return overrides.get("upper", self.config.upper), overrides.get("lower", self.config.lower)
        return self.config.upper, self.config.lower

    def _clip_tensor(self, tensor: Tensor, upper: float, lower: float) -> Tensor:
        norm = tensor.norm(2).item()
        if norm == 0:
            return tensor
        if norm > upper:
            tensor = tensor * (upper / norm)
        elif self.is_biclip and lower > 0 and norm < lower:
            tensor = tensor * (lower / norm)
        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._clip_tensor(tensor, self.config.upper, self.config.lower)

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        result = []
        for name, g in grads:
            upper, lower = self._get_thresholds(name)
            result.append((name, self._clip_tensor(g, upper, lower)))
        return result


# ---------------------------------------------------------------------------
# Dynamic clipping wrappers
# ---------------------------------------------------------------------------

class DynamicCoordUpperClipping(ClippingOperator):
    """Coordinate-wise upper clipping where the threshold is the p-th percentile of |g|."""

    def __call__(self, tensor: Tensor) -> Tensor:
        threshold = torch.quantile(tensor.abs().float(), self.config.dynamic_percentile).item()
        threshold = max(threshold, 1e-8)
        return tensor.clamp(-threshold, threshold)

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        flat = torch.cat([g.abs().flatten().float() for _, g in grads])
        threshold = torch.quantile(flat, self.config.dynamic_percentile).item()
        threshold = max(threshold, 1e-8)
        return [(name, g.clamp(-threshold, threshold)) for name, g in grads]


class DynamicGlobalUpperClipping(ClippingOperator):
    """Global L2 upper clipping where the threshold is an EMA of recent gradient norms."""

    def __init__(self, config: ClippingConfig):
        super().__init__(config)
        self._ema_norm: Optional[float] = None

    def _update_ema(self, norm: float) -> float:
        decay = self.config.dynamic_ema_decay
        if self._ema_norm is None:
            self._ema_norm = norm
        else:
            self._ema_norm = decay * self._ema_norm + (1 - decay) * norm
        return self._ema_norm

    def __call__(self, tensor: Tensor) -> Tensor:
        norm = tensor.norm(2).item()
        threshold = self._update_ema(norm)
        threshold = max(threshold, 1e-8)
        if norm > threshold:
            return tensor * (threshold / norm)
        return tensor

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        flat = torch.cat([g.flatten() for _, g in grads])
        norm = flat.norm(2).item()
        threshold = self._update_ema(norm)
        threshold = max(threshold, 1e-8)
        if norm > threshold:
            scale = threshold / norm
            return [(name, g * scale) for name, g in grads]
        return grads


class DynamicLayerwiseUpperClipping(ClippingOperator):
    """Layerwise L2 upper clipping where each layer threshold is an EMA of its own norm."""

    def __init__(self, config: ClippingConfig):
        super().__init__(config)
        self._ema_norms: Dict[str, float] = {}

    def _update_ema(self, name: str, norm: float) -> float:
        decay = self.config.dynamic_ema_decay
        if name not in self._ema_norms:
            self._ema_norms[name] = norm
        else:
            self._ema_norms[name] = decay * self._ema_norms[name] + (1 - decay) * norm
        return max(self._ema_norms[name], 1e-8)

    def __call__(self, tensor: Tensor) -> Tensor:
        norm = tensor.norm(2).item()
        threshold = self._update_ema("_default", norm)
        if norm > threshold:
            return tensor * (threshold / norm)
        return tensor

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        result = []
        for name, g in grads:
            norm = g.norm(2).item()
            threshold = self._update_ema(name, norm)
            if norm > threshold:
                result.append((name, g * (threshold / norm)))
            else:
                result.append((name, g))
        return result


# ---------------------------------------------------------------------------
# Online quantile clipping — Robbins-Monro stochastic approximation
#
# The τ-quantile estimate q is updated each step as:
#   q ← q + η_q * (τ − 1(x ≤ q))
# At equilibrium, E[1(x ≤ q)] = τ, so q converges to the τ-quantile of x.
# ---------------------------------------------------------------------------

class OnlineQuantileGlobalClipping(ClippingOperator):
    """Global L2 clipping with threshold = online estimate of τ-quantile of gradient norms.

    Each step: q ← q + η_q * (τ − 1(‖g‖₂ ≤ q)), then clip ‖g‖₂ to q.
    """

    def __init__(self, config: ClippingConfig):
        super().__init__(config)
        self._q: float = config.upper  # initialise from upper; adapts online

    def _update(self, norm: float) -> float:
        tau = self.config.quantile_level
        eta = self.config.quantile_lr
        indicator = 1.0 if norm <= self._q else 0.0
        self._q = max(self._q + eta * (tau - indicator), 1e-8)
        return self._q

    def __call__(self, tensor: Tensor) -> Tensor:
        norm = tensor.norm(2).item()
        threshold = self._update(norm)
        if norm > threshold:
            return tensor * (threshold / norm)
        return tensor

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        flat = torch.cat([g.flatten() for _, g in grads])
        norm = flat.norm(2).item()
        threshold = self._update(norm)
        if norm > threshold:
            scale = threshold / norm
            return [(name, g * scale) for name, g in grads]
        return grads


class OnlineQuantileCoordClipping(ClippingOperator):
    """Coordinate-wise clipping with threshold = online estimate of τ-quantile of |g_i|.

    Uses the batch-mean indicator to update the shared scalar estimate:
      q ← q + η_q * (τ − mean_i 1(|g_i| ≤ q)), then clip each |g_i| to q.
    """

    def __init__(self, config: ClippingConfig):
        super().__init__(config)
        self._q: float = config.upper

    def _update(self, abs_flat: Tensor) -> float:
        tau = self.config.quantile_level
        eta = self.config.quantile_lr
        frac_below = (abs_flat <= self._q).float().mean().item()
        self._q = max(self._q + eta * (tau - frac_below), 1e-8)
        return self._q

    def __call__(self, tensor: Tensor) -> Tensor:
        threshold = self._update(tensor.abs().flatten())
        return tensor.clamp(-threshold, threshold)

    def clip_grad_list(self, grads: List[Tuple[str, Tensor]]) -> List[Tuple[str, Tensor]]:
        flat = torch.cat([g.abs().flatten() for _, g in grads])
        threshold = self._update(flat)
        return [(name, g.clamp(-threshold, threshold)) for name, g in grads]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_clipping_operator(config: ClippingConfig) -> ClippingOperator:
    """Construct the appropriate ClippingOperator from a config."""
    if config.clip_type == "none":
        return NoClipping(config)

    if config.clip_type == "quantile":
        if config.clip_scope == "coordinate":
            return OnlineQuantileCoordClipping(config)
        else:  # global (layerwise falls back to global for simplicity)
            return OnlineQuantileGlobalClipping(config)

    if config.dynamic:
        # Dynamic thresholds
        if config.clip_scope == "coordinate":
            return DynamicCoordUpperClipping(config)
        elif config.clip_scope == "layerwise":
            return DynamicLayerwiseUpperClipping(config)
        else:  # global
            return DynamicGlobalUpperClipping(config)

    # Static thresholds
    if config.clip_type == "upper":
        if config.clip_scope == "global":
            return GlobalUpperClippingOperator(config)
        elif config.clip_scope == "layerwise":
            return LayerwiseClippingOperator(config)
        else:  # coordinate
            return CoordUpperClippingOperator(config)

    # biclip
    if config.clip_scope == "global":
        return GlobalBiclipOperator(config)
    elif config.clip_scope == "layerwise":
        return LayerwiseClippingOperator(config)
    else:  # coordinate
        return BidirectionalCoordClippingOperator(config)


def compute_gradient_stats(
    param_name: str,
    raw_grad: Tensor,
    clipped_grad: Tensor,
    upper: float,
    lower: float,
    num_bins: int = 20,
) -> dict:
    """Compute per-layer gradient statistics for logging."""
    raw_flat = raw_grad.detach().flatten()
    clipped_flat = clipped_grad.detach().flatten()
    abs_raw = raw_flat.abs()

    hist_counts, hist_edges = torch.histogram(raw_flat.float().cpu(), bins=num_bins)

    return {
        "layer_name": param_name,
        "grad_l2_norm": raw_flat.norm(2).item(),
        "grad_max_abs": abs_raw.max().item() if abs_raw.numel() > 0 else 0.0,
        "grad_min_abs": abs_raw.min().item() if abs_raw.numel() > 0 else 0.0,
        "fraction_above_upper_clip": (abs_raw > upper).float().mean().item() if upper > 0 else 0.0,
        "fraction_below_lower_clip": (abs_raw < lower).float().mean().item() if lower > 0 else 0.0,
        "histogram_edges": hist_edges.tolist(),
        "histogram_counts": hist_counts.tolist(),
    }
