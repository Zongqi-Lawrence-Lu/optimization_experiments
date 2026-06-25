"""WMT / IWSLT machine translation dataset loader for T5 fine-tuning.

Supported dataset names:
  "iwslt2017"        — TED Talks (small, ~200k pairs; readily downloadable)
  "opus100"          — OPUS-100 multilingual corpus (~100k per language pair)
  "news_commentary"  — WMT News Commentary (~300k pairs)
  "wmt14"            — WMT 2014 (large, ~4.5M En-De pairs)
  "wmt16"            — WMT 2016
  Any HuggingFace dataset id with a "translation" feature

Languages: English → German ("de") and English → French ("fr").

Usage:
    config = RealDataConfig(
        dataset_name="iwslt2017",
        src_lang="en",
        tgt_lang="de",
        tokenizer_name="t5-small",
        max_seq_len=128,
        batch_size=16,
    )
    train_loaders, val_loader, test_loader = build_wmt_dataloaders(config, num_nodes=1)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from framework.configs import RealDataConfig
from framework.data.loaders import dirichlet_partition


# ---------------------------------------------------------------------------
# HuggingFace dataset / config name resolution
# ---------------------------------------------------------------------------

# Maps user-facing alias → (hf_dataset_id, hf_config_template)
# hf_config_template uses {src}-{tgt} substitution; both orderings are tried.
_DATASET_SPECS: Dict[str, Tuple[str, Optional[str]]] = {
    # TED Talks (~200k pairs per language pair, readily downloadable)
    "iwslt2017":         ("iwslt2017", "iwslt2017-{src}-{tgt}"),
    # Small, readily downloadable (~1M pairs per language pair, has train/val/test)
    "opus100":           ("Helsinki-NLP/opus-100", "{src}-{tgt}"),
    # WMT corpora (larger, cached after first download)
    "wmt14":             ("wmt/wmt14",    "{src}-{tgt}"),
    "wmt16":             ("wmt/wmt16",    "{src}-{tgt}"),
    # News Commentary (~300k pairs)
    "news_commentary":   ("Helsinki-NLP/news_commentary", "{src}-{tgt}"),
}

_LANG_NAMES: Dict[str, str] = {
    "en": "English", "de": "German", "fr": "French",
    "ro": "Romanian", "cs": "Czech", "fi": "Finnish",
}


def _load_hf_translation(dataset_name: str, src_lang: str, tgt_lang: str):
    """Load a HuggingFace translation dataset, trying both language-pair orderings.

    Returns (hf_dataset_dict, src_lang, tgt_lang) where src_lang/tgt_lang are
    always the *user-specified* values — the dataset may store the pair in either
    order in its config name, but the translation dict always contains both keys.
    """
    from datasets import load_dataset

    spec = _DATASET_SPECS.get(dataset_name)
    if spec is not None:
        hf_id, config_template = spec
        for s, t in [(src_lang, tgt_lang), (tgt_lang, src_lang)]:
            config_name = config_template.format(src=s, tgt=t) if config_template else None
            try:
                ds = load_dataset(hf_id, config_name)
                # Always return the user's intended src/tgt direction
                return ds, src_lang, tgt_lang
            except Exception:
                continue
        raise ValueError(
            f"Could not load '{hf_id}' with language pair "
            f"'{src_lang}-{tgt_lang}' or '{tgt_lang}-{src_lang}'."
        )
    else:
        # Generic: try loading dataset_name directly with both orderings
        for s, t in [(src_lang, tgt_lang), (tgt_lang, src_lang)]:
            try:
                ds = load_dataset(dataset_name, f"{s}-{t}")
                return ds, src_lang, tgt_lang
            except Exception:
                continue
        raise ValueError(
            f"Could not load dataset '{dataset_name}' with language pair "
            f"'{src_lang}-{tgt_lang}'."
        )


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """Pre-tokenized seq2seq dataset for T5.

    Stores all tokenized inputs in memory for small-to-medium datasets.
    For very large datasets (WMT14 full), pass max_samples to limit size.
    """

    def __init__(
        self,
        hf_split,
        tokenizer,
        src_lang: str,
        tgt_lang: str,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        task_prefix: str = "",
        max_samples: Optional[int] = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = tokenizer
        self.task_prefix = task_prefix

        if max_samples is not None:
            hf_split = hf_split.select(range(min(max_samples, len(hf_split))))

        self._src_texts = [
            task_prefix + item["translation"][src_lang] for item in hf_split
        ]
        self._tgt_texts = [item["translation"][tgt_lang] for item in hf_split]

        # Tokenize source
        self._src_enc = tokenizer(
            self._src_texts,
            max_length=max_src_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Tokenize target (using text_target for modern transformers)
        self._tgt_enc = tokenizer(
            text_target=self._tgt_texts,
            max_length=max_tgt_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Replace pad token id in labels with -100 (ignored in loss)
        pad_id = tokenizer.pad_token_id
        self._label_ids = self._tgt_enc["input_ids"].clone()
        self._label_ids[self._label_ids == pad_id] = -100

    def __len__(self) -> int:
        return self._label_ids.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self._src_enc["input_ids"][idx],
            "attention_mask": self._src_enc["attention_mask"][idx],
            "labels":         self._label_ids[idx],
        }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_wmt_dataloaders(
    config: RealDataConfig,
    num_nodes: int,
    distribution: str = "iid",
    seed: int = 42,
    max_train_samples: Optional[int] = None,
) -> Tuple[List[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build translation DataLoaders for T5 fine-tuning.

    Parameters
    ----------
    config            : RealDataConfig with dataset_name, src_lang, tgt_lang, tokenizer_name
    num_nodes         : number of partitions for federated training
    distribution      : "iid" or "noniid"
    seed              : random seed for partitioning
    max_train_samples : limit training examples (useful for large WMT corpora)

    Returns
    -------
    (train_loaders, val_loader, test_loader)
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install 'transformers' to use translation loaders.")

    src_lang = config.src_lang or "en"
    tgt_lang = config.tgt_lang or "de"
    tokenizer_name = config.tokenizer_name or "t5-small"
    max_len = config.max_seq_len or 128

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ds, actual_src, actual_tgt = _load_hf_translation(config.dataset_name, src_lang, tgt_lang)

    # Build T5 task prefix
    src_name = _LANG_NAMES.get(actual_src, actual_src)
    tgt_name = _LANG_NAMES.get(actual_tgt, actual_tgt)
    task_prefix = f"translate {src_name} to {tgt_name}: "

    def _make_ds(split_key: str, limit: Optional[int] = None) -> Optional[TranslationDataset]:
        split = ds.get(split_key)
        # Some datasets use different split names for validation
        if split is None and split_key == "validation":
            split = ds.get("valid") or ds.get("dev")
        if split is None:
            return None
        return TranslationDataset(
            split, tokenizer, actual_src, actual_tgt,
            max_src_len=max_len, max_tgt_len=max_len,
            task_prefix=task_prefix,
            max_samples=limit,
        )

    train_ds = _make_ds(config.split_train, limit=max_train_samples)
    val_ds = _make_ds(config.split_val)
    test_ds = _make_ds(config.split_test)

    # Fallback: some datasets (e.g. news_commentary) only have a train split
    if train_ds is None:
        raise ValueError(
            f"Training split '{config.split_train}' not found in dataset '{config.dataset_name}'. "
            f"Available splits: {list(ds.keys())}"
        )

    # Partition training set
    rng = np.random.RandomState(seed)
    n = len(train_ds)

    if distribution == "iid":
        all_idx = rng.permutation(n)
        splits = np.array_split(all_idx, num_nodes)
        node_indices = [s.tolist() for s in splits]
    elif distribution == "noniid":
        # Use source token-count as a proxy for non-IID partitioning
        src_lengths = (train_ds._src_enc["attention_mask"].sum(dim=1)).numpy()
        n_bins = min(10, max(2, num_nodes))
        bin_edges = np.quantile(src_lengths, np.linspace(0, 1, n_bins + 1))
        bin_edges[-1] += 1
        pseudo_labels = np.searchsorted(bin_edges[1:-1], src_lengths).astype(int)
        node_indices = dirichlet_partition(pseudo_labels, num_nodes, alpha=0.5, rng=rng)
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
