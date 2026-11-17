"""Image transforms for the saved students and correction inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image
from torch import Tensor
from torchvision import transforms

from .config import AugmentationConfig, DataConfig

ImageTransform = Callable[[Image.Image], Tensor]


@dataclass(frozen=True, slots=True)
class ImageTransforms:
    evaluation: ImageTransform
    strong: ImageTransform


def build_image_transforms(data: DataConfig, config: AugmentationConfig) -> ImageTransforms:
    """Apply augmentation after ROI cropping and resizing."""
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(data.mean, data.std)
    return ImageTransforms(
        evaluation=transforms.Compose([to_tensor, normalize]),
        strong=transforms.Compose([
            transforms.RandomHorizontalFlip(config.horizontal_flip),
            transforms.RandomVerticalFlip(config.vertical_flip),
            transforms.RandomRotation(config.strong_rotation),
            transforms.ColorJitter(*config.color_jitter),
            to_tensor,
            normalize,
        ]),
    )
