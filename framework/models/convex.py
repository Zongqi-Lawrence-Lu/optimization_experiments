"""Convex model: linear regression with MSE loss.

The model is a single nn.Linear(in_features, 1, bias=False), which is
µ-strongly convex. During evaluation the distance to the true parameter
vector w* is tracked via log ||w* - ŵ||_2.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LinearRegressionModel(nn.Module):
    """Single-layer linear regression model.

    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    true_weights : Tensor, optional
        Ground-truth weight vector w*; stored as a buffer for evaluation.
    """

    def __init__(self, in_features: int, true_weights: Optional[Tensor] = None):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=False)
        if true_weights is not None:
            self.register_buffer("true_weights", true_weights.float())
        else:
            self.register_buffer("true_weights", None)

    def forward(self, X: Tensor) -> Tensor:
        return self.linear(X).squeeze(-1)

    def compute_loss(self, batch: Any) -> Tensor:
        X, y = batch
        X = X.to(self.linear.weight.device)
        y = y.to(self.linear.weight.device).float()
        pred = self.forward(X)
        return nn.functional.mse_loss(pred, y)

    @torch.no_grad()
    def param_distance(self) -> float:
        """Return ||w* - ŵ||_2."""
        if self.true_weights is None:
            raise ValueError("true_weights not set; cannot compute parameter distance.")
        w_hat = self.linear.weight.data.squeeze()
        return (self.true_weights - w_hat).norm(2).item()

    def param_vector(self) -> Tensor:
        """Return the current parameter as a flat vector."""
        return self.linear.weight.data.squeeze().clone()
