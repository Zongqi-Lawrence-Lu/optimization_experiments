"""Configurable synthetic data generator for regression experiments.

Generates a feature matrix X, true weight vector w*, and noisy labels y = Xw* + noise.
Supports IID and non-IID partitioning across N nodes.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from framework.configs import SyntheticDataConfig


class TensorDataset(Dataset):
    """Simple dataset wrapping feature and label tensors."""

    def __init__(self, X: Tensor, y: Tensor):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


class SyntheticRegressionDataset:
    """Generates a synthetic linear regression dataset.

    Attributes
    ----------
    true_weights : Tensor, shape (num_features,)
        The ground-truth parameter vector w* used to generate labels.
    X : Tensor, shape (num_samples, num_features)
    y : Tensor, shape (num_samples,)
    """

    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        rng = np.random.RandomState(config.random_seed)

        X = self._generate_features(rng)
        w_star = rng.randn(config.num_features).astype(np.float32)
        noise = self._generate_noise(rng, config.num_samples)
        y = X @ w_star + noise

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.true_weights = torch.from_numpy(w_star)

    def _generate_features(self, rng: np.random.RandomState) -> np.ndarray:
        cfg = self.config
        n, d = cfg.num_samples, cfg.num_features

        if cfg.feature_distribution == "gaussian":
            return rng.randn(n, d).astype(np.float32)

        elif cfg.feature_distribution == "bernoulli_mixed":
            num_common = max(1, int(d * cfg.common_feature_fraction))
            num_rare = d - num_common
            X = np.zeros((n, d), dtype=np.float32)
            X[:, :num_common] = rng.binomial(1, cfg.common_prob, size=(n, num_common)).astype(np.float32)
            if num_rare > 0:
                X[:, num_common:] = rng.binomial(1, cfg.rare_prob, size=(n, num_rare)).astype(np.float32)
            return X

        else:
            raise ValueError(f"Unknown feature_distribution: '{cfg.feature_distribution}'")

    def _generate_noise(self, rng: np.random.RandomState, n: int) -> np.ndarray:
        cfg = self.config
        dist = cfg.noise_distribution

        if dist == "none":
            return np.zeros(n, dtype=np.float32)
        elif dist == "gaussian":
            return (rng.randn(n) * cfg.noise_scale).astype(np.float32)
        elif dist == "uniform":
            return (rng.uniform(-cfg.noise_scale, cfg.noise_scale, n)).astype(np.float32)
        elif dist == "student_t":
            samples = rng.standard_t(df=cfg.noise_df, size=n)
            return (samples * cfg.noise_scale).astype(np.float32)
        elif dist == "cauchy":
            # Cauchy: ratio of two standard normals
            samples = rng.standard_cauchy(size=n)
            return (samples * cfg.noise_scale).astype(np.float32)
        else:
            raise ValueError(f"Unknown noise_distribution: '{dist}'")

    def as_torch_dataset(self) -> TensorDataset:
        return TensorDataset(self.X, self.y)

    def partition_iid(
        self,
        num_nodes: int,
        batch_size: int,
        rng_seed: int = 0,
    ) -> List[DataLoader]:
        """Split dataset uniformly at random across N nodes."""
        n = len(self.X)
        rng = np.random.RandomState(rng_seed)
        indices = rng.permutation(n)
        splits = np.array_split(indices, num_nodes)
        dataset = self.as_torch_dataset()
        loaders = []
        for split_idx in splits:
            subset = Subset(dataset, split_idx.tolist())
            loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
        return loaders

    def partition_noniid(
        self,
        num_nodes: int,
        batch_size: int,
        rng_seed: int = 0,
        partition_file: Optional[str] = None,
        dirichlet_alpha: float = 0.5,
    ) -> List[DataLoader]:
        """Non-IID partition using Dirichlet allocation over quantized label bins.

        Regression labels are quantized into bins (pseudo-classes) via quantiles, then
        a Dirichlet(alpha) distribution assigns each bin's samples across nodes.
        Lower alpha → more heterogeneous; higher alpha → closer to IID.

        If partition_file is given, load pre-computed index arrays from that file instead.
        """
        n = len(self.X)
        dataset = self.as_torch_dataset()

        if partition_file is not None:
            import json
            with open(partition_file) as f:
                node_indices = json.load(f)
            loaders = []
            for idx_list in node_indices:
                subset = Subset(dataset, idx_list)
                loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
            return loaders

        # Quantize continuous labels into pseudo-class bins for Dirichlet partitioning.
        from framework.data.loaders import dirichlet_partition
        num_bins = min(10, max(2, num_nodes))
        y_np = self.y.numpy()
        bin_edges = np.quantile(y_np, np.linspace(0, 1, num_bins + 1))
        bin_edges[-1] += 1e-8  # ensure max value falls in last bin
        pseudo_labels = np.searchsorted(bin_edges[1:-1], y_np).astype(int)

        rng = np.random.RandomState(rng_seed)
        node_indices = dirichlet_partition(pseudo_labels, num_nodes, dirichlet_alpha, rng)

        loaders = []
        for idx_list in node_indices:
            if len(idx_list) == 0:
                idx_list = [rng.randint(n)]  # give empty node at least one sample
            subset = Subset(dataset, idx_list)
            loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
        return loaders


def build_synthetic_dataloaders(
    config: SyntheticDataConfig,
    num_nodes: int,
    data_distribution: str,
    batch_size: Optional[int],
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    partition_file: Optional[str] = None,
) -> Tuple[List[DataLoader], DataLoader, DataLoader, "SyntheticRegressionDataset"]:
    """Build per-node training loaders plus shared validation and test loaders.

    The split is: train = 1 - val_fraction - test_fraction, val = val_fraction,
    test = test_fraction.  Val is used for hyperparameter selection; test is
    held out entirely and used only for final evaluation.

    Returns
    -------
    (train_loaders, val_loader, test_loader, dataset)
    """
    dataset = SyntheticRegressionDataset(config)
    bs = batch_size if batch_size is not None else config.batch_size

    n = len(dataset.X)
    rng = np.random.RandomState(seed)
    all_idx = rng.permutation(n)
    val_size  = max(1, int(n * val_fraction))
    test_size = max(1, int(n * test_fraction))
    val_idx   = all_idx[:val_size].tolist()
    test_idx  = all_idx[val_size:val_size + test_size].tolist()
    train_idx = all_idx[val_size + test_size:].tolist()

    full_ds = dataset.as_torch_dataset()
    val_loader  = DataLoader(Subset(full_ds, val_idx),  batch_size=bs, shuffle=False)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=bs, shuffle=False)

    train_X = dataset.X[train_idx]
    train_y = dataset.y[train_idx]

    train_shell = copy.copy(dataset)
    train_shell.X = train_X
    train_shell.y = train_y

    if data_distribution == "iid":
        train_loaders = train_shell.partition_iid(num_nodes, bs, rng_seed=seed)
    elif data_distribution == "noniid":
        train_loaders = train_shell.partition_noniid(
            num_nodes, bs, rng_seed=seed,
            partition_file=partition_file,
            dirichlet_alpha=config.dirichlet_alpha,
        )
    else:
        raise ValueError(f"Unknown data_distribution: '{data_distribution}'")

    return train_loaders, val_loader, test_loader, dataset
