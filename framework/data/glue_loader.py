"""GLUE benchmark dataset loader for RoBERTa fine-tuning.

Supports all 9 GLUE tasks: cola, sst2, mrpc, qqp, stsb, mnli, qnli, rte, wnli.
For MNLI, the validation split is 'validation_matched'.

Returns per-node DataLoaders using IID or Dirichlet non-IID partitioning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np

from framework.configs import RealDataConfig
from framework.data.loaders import dirichlet_partition


# GLUE task metadata: (sentence1_key, sentence2_key, is_regression)
_GLUE_META: Dict[str, Tuple[str, Optional[str], bool]] = {
    "cola":  ("sentence",  None,        False),
    "sst2":  ("sentence",  None,        False),
    "mrpc":  ("sentence1", "sentence2", False),
    "qqp":   ("question1", "question2", False),
    "stsb":  ("sentence1", "sentence2", True),   # regression task
    "mnli":  ("premise",   "hypothesis", False),
    "qnli":  ("question",  "sentence",  False),
    "rte":   ("sentence1", "sentence2", False),
    "wnli":  ("sentence1", "sentence2", False),
}

_GLUE_NUM_LABELS: Dict[str, int] = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "stsb": 1, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2,
}

_GLUE_VALIDATION_SPLIT: Dict[str, str] = {
    "mnli": "validation_matched",
}


class TokenizedGlueDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, sent1_key: str,
                 sent2_key: Optional[str], max_len: int, label_key: str = "label"):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.sent1_key = sent1_key
        self.sent2_key = sent2_key
        self.max_len = max_len
        self.label_key = label_key
        # Pre-tokenize for efficiency
        self._encodings = self._tokenize_all()
        self._labels = [item[label_key] for item in hf_dataset]

    def _tokenize_all(self):
        texts1 = [item[self.sent1_key] for item in self.data]
        texts2 = [item[self.sent2_key] for item in self.data] if self.sent2_key else None
        return self.tokenizer(
            texts1, texts2,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self._encodings.items()}
        label = self._labels[idx]
        # STS-B labels are floats; all others are integers
        item["labels"] = torch.tensor(label, dtype=torch.float if isinstance(label, float) else torch.long)
        return item


def build_glue_dataloaders(
    config: RealDataConfig,
    num_nodes: int,
    distribution: str = "iid",
    seed: int = 42,
) -> Tuple[List[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build GLUE DataLoaders for RoBERTa fine-tuning.

    Returns (train_loaders, val_loader, test_loader).
    Note: GLUE test labels are not public; test_loader uses the validation split as a proxy.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install 'datasets' and 'transformers' to use GLUE loaders.")

    task = config.glue_task or config.dataset_name
    if task not in _GLUE_META:
        raise ValueError(f"Unknown GLUE task '{task}'. Valid tasks: {list(_GLUE_META.keys())}")

    sent1_key, sent2_key, is_regression = _GLUE_META[task]
    tokenizer_name = config.tokenizer_name or "roberta-base"
    max_len = config.max_seq_len or 128

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load dataset; GLUE is under "glue" namespace
    ds = load_dataset("glue", task)
    val_split_name = _GLUE_VALIDATION_SPLIT.get(task, "validation")

    train_hf = ds["train"]
    val_hf = ds.get(val_split_name) or ds.get("validation")
    # Use val as test proxy since test labels are unreleased
    test_hf = ds.get("test")

    def _make_ds(hf_split):
        if hf_split is None:
            return None
        return TokenizedGlueDataset(
            hf_split, tokenizer, sent1_key, sent2_key, max_len,
            label_key="label",
        )

    train_ds = _make_ds(train_hf)
    val_ds = _make_ds(val_hf)
    test_ds = _make_ds(test_hf)

    # Partition training set
    rng = np.random.RandomState(seed)
    n = len(train_ds)

    if distribution == "iid":
        all_idx = rng.permutation(n)
        splits = np.array_split(all_idx, num_nodes)
        node_indices = [s.tolist() for s in splits]
    elif distribution == "noniid":
        labels = np.array([train_hf[i]["label"] for i in range(n)])
        if is_regression:
            # Quantize regression labels for Dirichlet partitioning
            n_bins = min(10, max(2, num_nodes))
            bin_edges = np.quantile(labels, np.linspace(0, 1, n_bins + 1))
            bin_edges[-1] += 1e-8
            labels = np.searchsorted(bin_edges[1:-1], labels)
        node_indices = dirichlet_partition(labels, num_nodes, alpha=0.5, rng=rng)
    else:
        raise ValueError(f"Unknown distribution: '{distribution}'")

    def _collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

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
        if val_ds is not None else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                   num_workers=config.num_workers, collate_fn=_collate)
        if test_ds is not None else None
    )

    return train_loaders, val_loader, test_loader
