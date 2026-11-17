"""One pooled timm encoder factory for training and fixed-feature metrics."""
from __future__ import annotations

import os
from pathlib import Path

from torch import nn


def build_encoder(
    name: str,
    *,
    pretrained: bool,
    drop_path_rate: float = 0.0,
    pretrained_weights: Path | None = None,
) -> tuple[nn.Module, int]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from timm import create_model

    options = {}
    if pretrained and pretrained_weights is not None:
        weights = Path(pretrained_weights)
        if not weights.is_file():
            raise FileNotFoundError(weights)
        options["pretrained_cfg_overlay"] = {"file": str(weights)}
    if name == "convnextv2_tiny":
        name = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    encoder = create_model(
        name, pretrained=pretrained, num_classes=0, global_pool="avg",
        drop_path_rate=drop_path_rate, **options,
    )
    dimensions = int(encoder.num_features)
    if dimensions < 1:
        raise ValueError(f"Backbone {name} has no pooled feature dimension.")
    return encoder, dimensions
