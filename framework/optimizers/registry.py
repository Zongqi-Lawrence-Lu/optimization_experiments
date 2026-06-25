"""Registry for inner and outer optimizers.

Resolves string keys to concrete optimizer instances. Custom optimizers are loaded
dynamically from fully-qualified Python import paths via the config's custom_factory.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

from framework.configs import InnerOptimizerConfig, OuterOptimizerConfig
from framework.optimizers.inner import InnerOptimizer, build_inner_optimizer
from framework.optimizers.outer import OuterOptimizer, build_outer_optimizer


# ---------------------------------------------------------------------------
# Inner optimizer registry
# ---------------------------------------------------------------------------

_INNER_REGISTRY: Dict[str, Callable[[InnerOptimizerConfig], InnerOptimizer]] = {}


def register_inner(name: str, factory: Callable[[InnerOptimizerConfig], InnerOptimizer]) -> None:
    """Register a custom inner optimizer factory under a string key."""
    _INNER_REGISTRY[name] = factory


def get_inner_optimizer(config: InnerOptimizerConfig) -> InnerOptimizer:
    """Resolve and instantiate an inner optimizer from config."""
    if config.name in _INNER_REGISTRY:
        return _INNER_REGISTRY[config.name](config)
    return build_inner_optimizer(config)


# ---------------------------------------------------------------------------
# Outer optimizer registry
# ---------------------------------------------------------------------------

_OUTER_REGISTRY: Dict[str, Callable[[OuterOptimizerConfig], OuterOptimizer]] = {}


def register_outer(name: str, factory: Callable[[OuterOptimizerConfig], OuterOptimizer]) -> None:
    """Register a custom outer optimizer factory under a string key."""
    _OUTER_REGISTRY[name] = factory


def get_outer_optimizer(config: OuterOptimizerConfig) -> OuterOptimizer:
    """Resolve and instantiate an outer optimizer from config."""
    if config.name in _OUTER_REGISTRY:
        return _OUTER_REGISTRY[config.name](config)
    return build_outer_optimizer(config)
