"""Pooled backbone and independent classification heads; no SSR components."""
from __future__ import annotations

import torch
from torch import nn

from shared.backbone import build_encoder
from shared.metrics import class_weights
from .runtime import isolated_model_rng, model_forward_seed


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(config, num_classes: int, device, *, initialize_pretrained: bool = True) -> Classifier:
    pretrained = config.pretrained and initialize_pretrained
    encoder, dimensions = build_encoder(config.name, pretrained=pretrained,
                                       pretrained_weights=config.pretrained_weights if pretrained else None,
                                       drop_path_rate=config.drop_path_rate)
    return Classifier(encoder, dimensions, num_classes).to(device)


def build_teacher_models(config, num_classes: int, device, *, initialize_pretrained: bool = True):
    """Build independent modules, optionally isolating each initialization RNG.

    A shared pretrained checkpoint still supplies identical encoder weights.
    Seeds control randomly initialized components and stochastic training, not
    the contents of that checkpoint.
    """
    seeds = config.correction.model_seeds
    if seeds is None:
        return [build_model(config.model, num_classes, device,
                            initialize_pretrained=initialize_pretrained) for _ in range(2)]
    models = []
    for seed in seeds:
        with isolated_model_rng(seed, device):
            models.append(build_model(config.model, num_classes, device,
                                      initialize_pretrained=initialize_pretrained))
    return models


def teacher_forward(models, images, *, model_seeds, device, epoch: int, step: int):
    """Use the same image tensor with separate deterministic stochastic streams."""
    if model_seeds is None:
        return [model(images) for model in models]
    outputs = []
    for model, seed in zip(models, model_seeds, strict=True):
        with isolated_model_rng(model_forward_seed(seed, epoch, step), device):
            outputs.append(model(images))
    return outputs


def build_optimizer(model, training):
    groups = [{"params": model.encoder.parameters(), "lr": training.encoder_learning_rate},
              {"params": model.classifier.parameters(), "lr": training.learning_rate}]
    if training.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=training.weight_decay)
    return torch.optim.SGD(groups, momentum=training.momentum, weight_decay=training.weight_decay)


def teacher_model_weights(original_labels, classes: int, strategy: str):
    """Use original prepared Train labels throughout all teacher phases."""
    if strategy not in {"inverse_sqrt", "none"}:
        raise ValueError("correction.model_class_weighting must be inverse_sqrt or none.")
    weights, counts = class_weights(original_labels, classes)
    if strategy == "none":
        weights = torch.ones_like(weights)
    return weights, counts
