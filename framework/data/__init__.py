from .synthetic import (
    TensorDataset,
    SyntheticRegressionDataset,
    build_synthetic_dataloaders,
)
from .loaders import build_dataloaders, dirichlet_partition

__all__ = [
    "TensorDataset",
    "SyntheticRegressionDataset",
    "build_synthetic_dataloaders",
    "build_dataloaders",
    "dirichlet_partition",
]
