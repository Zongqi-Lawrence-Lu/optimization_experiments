"""Two-layer MLP regression model.

Parameter names fc1.* and fc2.* allow per-layer clipping thresholds via
ClippingConfig.layer_overrides = {"fc1": {"upper": C1}, "fc2": {"upper": C2}}.
"""

from __future__ import annotations
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TwoLayerMLP(nn.Module):
    """Two-layer MLP for scalar regression (MSE loss).

    Parameters
    ----------
    in_features : input dimension
    hidden_size : width of the hidden layer
    """

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, X: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(X))).squeeze(-1)

    def compute_loss(self, batch: Any) -> Tensor:
        X, y = batch
        X = X.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).float()
        return nn.functional.mse_loss(self.forward(X), y)
