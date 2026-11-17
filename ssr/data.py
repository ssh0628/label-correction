"""SSR/LSL views and aligned original or corrected datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from shared.augmentation import build_image_transforms as shared_transforms
from shared.config import AugmentationConfig, DataConfig
from shared.data import IndexedImageDataset, _corrected_targets, prepare_dataset
from shared.roi import ROICrop

ImageTransform = Callable[[Image.Image], Tensor]


@dataclass(frozen=True, slots=True)
class ImageTransforms:
    evaluation: ImageTransform
    weak: ImageTransform
    strong: ImageTransform


class TwoStrongViews:
    """Two independent strong views for SSR's selected-sample mixup CE."""

    def __init__(self, strong_transform: ImageTransform) -> None:
        self.strong_transform = strong_transform

    def __call__(self, image: Image.Image) -> list[Tensor]:
        return [self.strong_transform(image), self.strong_transform(image)]


class AllSampleViews:
    """Weak/strong feature-consistency views, plus LSL's independent strong view."""

    def __init__(
        self,
        weak_transform: ImageTransform,
        strong_transform: ImageTransform,
        *,
        include_structural_view: bool,
    ) -> None:
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.include_structural_view = include_structural_view

    def __call__(self, image: Image.Image) -> list[Tensor]:
        views = [self.weak_transform(image), self.strong_transform(image)]
        if self.include_structural_view:
            views.append(self.strong_transform(image))
        return views


def build_image_transforms(data: DataConfig, config: AugmentationConfig) -> ImageTransforms:
    """Augment images already cropped and resized once by the dataset."""
    common = shared_transforms(data, config)
    weak = transforms.Compose([
        transforms.RandomHorizontalFlip(config.horizontal_flip),
        transforms.RandomVerticalFlip(config.vertical_flip),
        transforms.RandomRotation(config.weak_rotation),
        transforms.ToTensor(),
        transforms.Normalize(data.mean, data.std),
    ])
    return ImageTransforms(evaluation=common.evaluation, weak=weak, strong=common.strong)


@dataclass(frozen=True, slots=True)
class ExperimentData:
    calibration_train: Dataset
    selected_train: Dataset
    evaluation_train: Dataset
    all_train: Dataset
    validation: Dataset | None
    test: Dataset | None
    noisy_labels: Tensor
    num_classes: int
    class_names: tuple[str, ...]
    training_targets: dict | None = None
    validation_correction: Tensor | None = None
    test_correction: Tensor | None = None
    validation_targets: dict | None = None
    test_targets: dict | None = None



def build_experiment_data(
    config: DataConfig,
    *,
    structural_labels_enabled: bool,
    augmentation: AugmentationConfig,
    training_method: str = "ssr",
    corrected_labels_path: Path | None = None,
    corrected_validation_labels_path: Path | None = None,
    corrected_test_labels_path: Path | None = None,
    report_path: Path | None = None,
) -> ExperimentData:
    """Bind fixed corrections to each split; train transforms run in seeded workers."""
    sources, boxes = prepare_dataset(config, report_path=report_path)
    correction_paths = {"train": corrected_labels_path, "val": corrected_validation_labels_path,
                        "test": corrected_test_labels_path}
    if training_method not in {"ssr", "ce"} or (any(path is not None for path in correction_paths.values()) and training_method != "ce"):
        raise ValueError("Corrected labels require plain CE retraining.")
    if any(path is not None for path in correction_paths.values()) and not all(correction_paths.values()):
        raise ValueError("Corrected retraining requires Train, Validation and Test correction files.")
    corrections = {}
    for name, path in correction_paths.items():
        if path is not None:
            if name not in sources:
                raise ValueError(f"Correction requires the {name} split.")
            corrections[name] = _corrected_targets(config, sources[name], path)
    training_labels, training_targets = corrections.get("train", (sources["train"].labels, None))
    if corrections:
        identities = {tuple(provenance["metadata"].get(key) for key in
                            ("checkpoint_sha256", "correction_threshold"))
                      for _, provenance in corrections.values()}
        if len(identities) != 1:
            raise ValueError("Train, Validation and Test corrections must share a teacher and correction threshold.")
    transforms = build_image_transforms(config, augmentation)

    def dataset(
        name: str, transform: Callable, *, labeled: bool = False, training: bool = False
    ) -> IndexedImageDataset | None:
        source = sources.get(name)
        if source is None:
            return None
        return IndexedImageDataset(
            source, transform,
            allow_truncated=config.allow_truncated_images, labeled=labeled,
            bboxes=boxes.get(name),
            roi=ROICrop(
                size=config.image_size, method=config.crop_method,
                random_view=training and config.multi_roi,
                scales=config.roi_scales, shift_ratio=config.roi_shift_ratio,
            ),
        )

    fixed_train = dataset("train", transforms.evaluation)
    return ExperimentData(
        calibration_train=fixed_train,
        selected_train=dataset("train", TwoStrongViews(transforms.strong), training=True),
        evaluation_train=fixed_train,
        all_train=dataset(
            "train",
            transforms.strong if training_method == "ce" else AllSampleViews(
                transforms.weak, transforms.strong,
                include_structural_view=structural_labels_enabled,
            ),
            training=True,
        ),
        validation=dataset("val", transforms.evaluation, labeled=True),
        test=dataset("test", transforms.evaluation, labeled=True),
        noisy_labels=training_labels,
        num_classes=config.num_classes,
        class_names=tuple(config.class_names),
        training_targets=training_targets,
        validation_correction=corrections["val"][0] if "val" in corrections else None,
        test_correction=corrections["test"][0] if "test" in corrections else None,
        validation_targets=corrections["val"][1] if "val" in corrections else None,
        test_targets=corrections["test"][1] if "test" in corrections else None,
    )
