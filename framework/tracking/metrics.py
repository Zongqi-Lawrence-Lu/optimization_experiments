"""Task-specific metric computation.

All metric functions share the signature:
    fn(predictions, targets, **kwargs) -> float

A global registry allows users to add custom metrics:
    metrics.register("my_metric", fn)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable] = {}


def register(name: str, fn: Callable) -> None:
    _REGISTRY[name] = fn


def compute(name: str, predictions: Any, targets: Any, **kwargs) -> float:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown metric: '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](predictions, targets, **kwargs)


def available() -> List[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------

def _loss(predictions: Any, targets: Any, **kwargs) -> float:
    preds = torch.as_tensor(predictions).float()
    tgts = torch.as_tensor(targets).float()
    if preds.shape == tgts.shape:
        return torch.nn.functional.mse_loss(preds, tgts).item()
    return torch.nn.functional.cross_entropy(preds, tgts.long()).item()


def _accuracy(predictions: Any, targets: Any, **kwargs) -> float:
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    return float((preds == tgts).mean())


def _f1_macro(predictions: Any, targets: Any, **kwargs) -> float:
    from sklearn.metrics import f1_score
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    return float(f1_score(tgts, preds, average="macro", zero_division=0))


def _f1_binary(predictions: Any, targets: Any, **kwargs) -> float:
    from sklearn.metrics import f1_score
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    if preds.ndim > 1:
        preds = (preds[:, 1] >= 0.5).astype(int)
    return float(f1_score(tgts, preds, average="binary", zero_division=0))


def _mcc(predictions: Any, targets: Any, **kwargs) -> float:
    from sklearn.metrics import matthews_corrcoef
    preds = _to_numpy(predictions)
    tgts = _to_numpy(targets)
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    return float(matthews_corrcoef(tgts, preds))


def _spearman(predictions: Any, targets: Any, **kwargs) -> float:
    from scipy.stats import spearmanr
    preds = _to_numpy(predictions).flatten()
    tgts = _to_numpy(targets).flatten()
    corr, _ = spearmanr(preds, tgts)
    return float(corr)


def _pearson(predictions: Any, targets: Any, **kwargs) -> float:
    from scipy.stats import pearsonr
    preds = _to_numpy(predictions).flatten()
    tgts = _to_numpy(targets).flatten()
    corr, _ = pearsonr(preds, tgts)
    return float(corr)


def _bleu(predictions: Any, targets: Any, **kwargs) -> float:
    """BLEU score. predictions and targets are lists of strings."""
    import sacrebleu
    if isinstance(predictions[0], str):
        refs = [[t] for t in targets]
        result = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
        return float(result.score)
    return 0.0


def _meteor(predictions: Any, targets: Any, **kwargs) -> float:
    """METEOR score. predictions and targets are lists of strings."""
    import nltk
    try:
        nltk.data.find("wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    scores = [
        nltk.translate.meteor_score.single_meteor_score(ref.split(), hyp.split())
        for hyp, ref in zip(predictions, targets)
    ]
    return float(np.mean(scores)) if scores else 0.0


def _perplexity(predictions: Any, targets: Any, **kwargs) -> float:
    """Perplexity from per-token log-probabilities."""
    log_probs = _to_numpy(predictions).flatten()
    return float(np.exp(-log_probs.mean()))


def _param_distance(predictions: Any, targets: Any, **kwargs) -> float:
    """||w* - ŵ||_2 for convex models.

    predictions: current model weights (Tensor or array)
    targets: true weights w* (Tensor or array)
    """
    w_hat = _to_numpy(predictions).flatten()
    w_star = _to_numpy(targets).flatten()
    return float(np.linalg.norm(w_star - w_hat))


# Register all built-in metrics
register("loss", _loss)
register("accuracy", _accuracy)
register("f1_macro", _f1_macro)
register("f1_binary", _f1_binary)
register("mcc", _mcc)
register("spearman", _spearman)
register("pearson", _pearson)
register("bleu", _bleu)
register("meteor", _meteor)
register("perplexity", _perplexity)
register("param_distance", _param_distance)
