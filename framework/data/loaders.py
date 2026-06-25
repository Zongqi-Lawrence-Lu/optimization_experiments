"""Real-world dataset loaders.

Supports HuggingFace datasets and torchvision datasets, with IID and non-IID
(Dirichlet allocation) partitioning across N nodes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from framework.configs import RealDataConfig


# ---------------------------------------------------------------------------
# Dirichlet non-IID partitioning
# ---------------------------------------------------------------------------

def dirichlet_partition(
    labels: np.ndarray,
    num_nodes: int,
    alpha: float,
    rng: np.random.RandomState,
) -> List[List[int]]:
    """Partition sample indices across nodes using a Dirichlet distribution over classes.

    Higher alpha → more IID; lower alpha → more heterogeneous.
    """
    classes = np.unique(labels)
    node_indices: List[List[int]] = [[] for _ in range(num_nodes)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * num_nodes)
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)
        splits = np.split(cls_idx, proportions[:-1])
        for node_id, split in enumerate(splits):
            node_indices[node_id].extend(split.tolist())

    return node_indices


# ---------------------------------------------------------------------------
# HuggingFace dataset loader
# ---------------------------------------------------------------------------

class HFDatasetWrapper(Dataset):
    """Wraps a HuggingFace dataset split as a PyTorch Dataset."""

    def __init__(self, hf_dataset, feature_keys: List[str], label_key: str):
        self.dataset = hf_dataset
        self.feature_keys = feature_keys
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        features = {k: row[k] for k in self.feature_keys}
        label = row[self.label_key]
        return features, label


class TokenizedDataset(Dataset):
    """Wraps a tokenized HuggingFace dataset for sequence classification / LM tasks."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.dataset[idx].items()}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: RealDataConfig,
    num_nodes: int,
    distribution: str,
    seed: int = 42,
    dirichlet_alpha: float = 0.5,
    label_key: str = "label",
) -> Tuple[List[DataLoader], DataLoader, Optional[DataLoader]]:
    """Load a dataset and partition it across N nodes.

    Returns
    -------
    (train_loaders, val_loader, test_loader)
    test_loader may be None if no test split exists.
    """
    try:
        import datasets as hf_datasets
    except ImportError:
        raise ImportError("Install 'datasets' to use real-world data loaders.")

    rng = np.random.RandomState(seed)

    ds = hf_datasets.load_dataset(config.dataset_name)

    train_split = ds[config.split_train]
    val_split = ds.get(config.split_val)
    test_split = ds.get(config.split_test)

    # Tokenize if needed
    if config.tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        max_len = config.max_seq_len or 128

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"] if "text" in batch else batch["sentence"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )

        train_split = train_split.map(tokenize_fn, batched=True)
        if val_split is not None:
            val_split = val_split.map(tokenize_fn, batched=True)
        if test_split is not None:
            test_split = test_split.map(tokenize_fn, batched=True)
        train_split.set_format("torch")
        if val_split is not None:
            val_split.set_format("torch")
        if test_split is not None:
            test_split.set_format("torch")

        train_ds = TokenizedDataset(train_split)
        val_ds = TokenizedDataset(val_split) if val_split is not None else None
        test_ds = TokenizedDataset(test_split) if test_split is not None else None
    else:
        train_ds = TokenizedDataset(train_split)
        val_ds = TokenizedDataset(val_split) if val_split is not None else None
        test_ds = TokenizedDataset(test_split) if test_split is not None else None

    # Partition train set
    n = len(train_ds)
    if distribution == "iid":
        all_idx = rng.permutation(n)
        splits = np.array_split(all_idx, num_nodes)
        node_indices = [s.tolist() for s in splits]
    elif distribution == "noniid":
        try:
            labels = np.array(train_split[label_key])
        except Exception:
            # Fall back to IID if labels cannot be extracted
            all_idx = rng.permutation(n)
            splits = np.array_split(all_idx, num_nodes)
            node_indices = [s.tolist() for s in splits]
        else:
            node_indices = dirichlet_partition(labels, num_nodes, dirichlet_alpha, rng)
    else:
        raise ValueError(f"Unknown distribution: '{distribution}'")

    def _collate(batch):
        if isinstance(batch[0], dict):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
        return torch.utils.data.dataloader.default_collate(batch)

    train_loaders = [
        DataLoader(
            Subset(train_ds, idx_list),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=_collate,
        )
        for idx_list in node_indices
    ]

    val_loader = (
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                   num_workers=config.num_workers, collate_fn=_collate)
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                   num_workers=config.num_workers, collate_fn=_collate)
        if test_ds is not None
        else None
    )

    return train_loaders, val_loader, test_loader
