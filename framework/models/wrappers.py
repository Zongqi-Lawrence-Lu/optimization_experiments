"""Thin wrappers around arbitrary nn.Module models.

ModelWrapper provides a unified interface:
    model.compute_loss(batch) -> Tensor   (for training)
    model.forward(batch)     -> (loss, logits)   (for evaluation)

Supports HuggingFace sequence classification, sequence-to-sequence (T5),
torchvision models, and user-supplied nn.Module.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from framework.configs import ModelConfig


class ModelWrapper(nn.Module):
    """Wraps any nn.Module with a unified loss/forward interface."""

    def __init__(self, backbone: nn.Module, loss_fn: Optional[Callable] = None):
        super().__init__()
        self.backbone = backbone
        self._loss_fn = loss_fn or nn.CrossEntropyLoss()

    def forward(self, batch: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """Returns (loss, logits). logits may be None for some model types."""
        if isinstance(batch, dict):
            outputs = self.backbone(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            logits = outputs.logits if hasattr(outputs, "logits") else None
            if loss is None and logits is not None and "labels" in batch:
                loss = self._loss_fn(logits, batch["labels"])
            return loss, logits

        X, y = batch
        logits = self.backbone(X)
        loss = self._loss_fn(logits, y)
        return loss, logits

    def compute_loss(self, batch: Any) -> Tensor:
        loss, _ = self.forward(batch)
        return loss


class Seq2SeqModelWrapper(nn.Module):
    """Wrapper for encoder-decoder (seq2seq) models such as T5.

    Handles dict batches with input_ids, attention_mask, labels.
    Returns (loss, generated_ids) from forward; loss is from the language
    model head (teacher-forcing cross-entropy).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch: Any) -> Tuple[Tensor, Optional[Tensor]]:
        if not isinstance(batch, dict):
            raise ValueError("Seq2SeqModelWrapper expects a dict batch with input_ids/labels")
        outputs = self.backbone(**batch)
        loss = outputs.loss
        logits = outputs.logits if hasattr(outputs, "logits") else None
        return loss, logits

    def compute_loss(self, batch: Any) -> Tensor:
        loss, _ = self.forward(batch)
        return loss

    def generate(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.backbone.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config: ModelConfig, true_weights=None) -> nn.Module:
    """Instantiate a model from config."""
    if config.kind == "linear":
        from framework.models.convex import LinearRegressionModel
        return LinearRegressionModel(config.in_features, true_weights=true_weights)

    elif config.kind == "two_layer_mlp":
        from framework.models.mlp import TwoLayerMLP
        return TwoLayerMLP(config.in_features, config.hidden_size)

    elif config.kind in ("hf_seq_cls", "hf_pretrained"):
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            raise ImportError("Install 'transformers' to use HuggingFace models.")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.pretrained_name,
            num_labels=config.num_classes,
        )
        return ModelWrapper(model)

    elif config.kind == "hf_seq2seq":
        try:
            from transformers import AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError("Install 'transformers' to use HuggingFace seq2seq models.")
        model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_name)
        return Seq2SeqModelWrapper(model)

    elif config.kind == "torchvision":
        try:
            import torchvision.models as tv_models
        except ImportError:
            raise ImportError("Install 'torchvision' to use torchvision models.")
        factory = getattr(tv_models, config.arch_name)
        backbone = factory(pretrained=config.pretrained)

        if hasattr(backbone, "fc"):
            in_f = backbone.fc.in_features
            backbone.fc = nn.Linear(in_f, config.num_classes)
        elif hasattr(backbone, "classifier"):
            if isinstance(backbone.classifier, nn.Sequential):
                in_f = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Linear(in_f, config.num_classes)
            else:
                in_f = backbone.classifier.in_features
                backbone.classifier = nn.Linear(in_f, config.num_classes)

        return ModelWrapper(backbone)

    elif config.kind == "custom":
        module_path, fn_name = config.custom_factory.rsplit(".", 1)
        module = importlib.import_module(module_path)
        factory = getattr(module, fn_name)
        return factory(config)

    else:
        raise ValueError(f"Unknown model kind: '{config.kind}'")
