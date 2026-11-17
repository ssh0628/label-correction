"""Original split rows and the image views used by Teacher and Student."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.augmentation import build_image_transforms
from shared.bbox import BBox
from shared.config import AugmentationConfig, DataConfig
from shared.data import IndexedImageDataset, SplitSource, _file_hash, _resolve_array, prepare_dataset
from shared.roi import LETTERBOX_FILL, ROICrop
from shared.views import FixedViewsDataset


def source_hashes(config: DataConfig, split: str) -> dict[str, str]:
    """Hash the original arrays, including their original order and offsets."""
    description = {"train": config.train, "val": config.validation, "test": config.test}.get(split)
    if description is None:
        raise ValueError(f"Dataset split is not configured: {split!r}.")
    return {key: _file_hash(_resolve_array(config.root, getattr(description, key)))
            for key in ("paths", "labels")}


def input_config(config: DataConfig) -> dict:
    """Record preprocessing alongside every exported label artifact."""
    return {
        "image_size": config.image_size,
        "label_offset": config.label_offset,
        "crop_bbox": config.crop_bbox,
        "missing_bbox": config.missing_bbox,
        "crop_method": config.crop_method,
        "multi_roi": config.multi_roi,
        "roi_scales": list(config.roi_scales),
        "roi_shift_ratio": config.roi_shift_ratio,
        "mean": list(config.mean),
        "std": list(config.std),
        "letterbox_fill": list(LETTERBOX_FILL),
        "roi_padding_fill": 0,
    }


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Original annotations and indexed image views; targets are never overwritten."""

    sources: dict[str, SplitSource]
    boxes: dict[str, tuple[BBox | None, ...]]
    train: IndexedImageDataset
    evaluation_train: IndexedImageDataset
    validation: IndexedImageDataset | None
    test: IndexedImageDataset | None


def build_data(config, *, report_path: Path | None = None,
               train_multi_roi: bool | None = None) -> PreparedData:
    """Prepare one training view and fixed-center holdouts using stable row indices.

    Both correcting networks receive the same augmented tensor. Their independent
    parameters and stochastic network layers provide the two prediction streams.
    Alternate student targets are loaded separately, leaving these sources intact.
    """
    data = config.data
    sources, boxes = prepare_dataset(data, report_path=report_path)
    transforms = build_image_transforms(data, config.augmentation)
    random_roi = data.multi_roi if train_multi_roi is None else train_multi_roi
    if random_roi and not data.crop_bbox:
        raise ValueError("Random Multi-ROI requires data.crop_bbox=True.")

    def dataset(name: str, *, training: bool = False):
        source = sources.get(name)
        if source is None:
            return None
        return IndexedImageDataset(
            source,
            transforms.strong if training else transforms.evaluation,
            allow_truncated=data.allow_truncated_images,
            bboxes=boxes.get(name),
            roi=ROICrop(size=data.image_size, method=data.crop_method,
                        random_view=training and random_roi,
                        scales=data.roi_scales, shift_ratio=data.roi_shift_ratio),
        )

    return PreparedData(sources=sources, boxes=boxes, train=dataset("train", training=True),
                        evaluation_train=dataset("train"), validation=dataset("val"),
                        test=dataset("test"))


def build_fixed_views_dataset(source: SplitSource, boxes: tuple[BBox | None, ...] | None,
                              data: DataConfig, augmentation: AugmentationConfig) -> FixedViewsDataset:
    return FixedViewsDataset(source, boxes, data, augmentation)
