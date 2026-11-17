"""Settings shared by the two historical input formats."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, get_args

DATASET_ROOT = Path("/root/project/dataset/npy_path/consensus/sqrt/modify_same_as_orig")

MissingBBoxPolicy = Literal["drop", "error", "full"]
ROICropMethod = Literal["roi_resize", "aspect_letterbox"]


def validate_missing_bbox(policy: str) -> None:
    if policy not in get_args(MissingBBoxPolicy):
        raise ValueError(f"data.missing_bbox must be one of {get_args(MissingBBoxPolicy)}.")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """root 기준 NPY 파일명. 절대 경로도 사용 가능."""

    paths: str  # - 이미지 경로 NPY
    labels: str  # - 정수 class index NPY
    bboxes: str | None = None  # - None: bbox cache 자동 탐색·생성

    def validate(self) -> None:
        if not self.paths.strip() or not self.labels.strip():
            raise ValueError("Split paths/labels filenames must not be empty.")
        if self.bboxes is not None and not self.bboxes.strip():
            raise ValueError("Split bbox filename must not be empty.")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """이미지·라벨·bbox 입력."""

    root: Path = field(default_factory=lambda: DATASET_ROOT)  # - NPY 루트
    image_root: Path | None = None  # - 상대 이미지 경로 기준; None: root
    annotation_root: Path | None = None  # - JSON 루트; None: 이미지 옆
    name: str = "a1-a7-multi-roi-sqrt"  # - 실험 이름
    class_names: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")  # - 라벨 순서
    train: SplitConfig = field(
        default_factory=lambda: SplitConfig("train_path.npy", "train_labels.npy")
    )
    validation: SplitConfig | None = field(
        default_factory=lambda: SplitConfig("val_path.npy", "val_labels.npy")
    )  # - None: validation 생략
    test: SplitConfig | None = field(
        default_factory=lambda: SplitConfig("test_path.npy", "test_labels.npy")
    )  # - None: test 생략
    label_offset: int = 0  # - 라벨 시작 번호
    crop_bbox: bool = True  # - 전체 이미지에서 bbox crop
    multi_roi: bool = True  # - 학습: 27개 scale·방향 중 1개
    crop_method: ROICropMethod = "aspect_letterbox"  # - 비율 유지·패딩 / roi_resize: 정사각 변형
    roi_scales: tuple[float, ...] = (0.8, 1.0, 1.2)  # - bbox 크기 배율
    roi_shift_ratio: float = 0.15  # - 방향별 이동 상한; crop 크기 기준
    missing_bbox: MissingBBoxPolicy = "drop"  # - 누락: 제외 / 중단 / 전체 이미지
    image_size: int = 224  # - crop 후 resize
    bbox_workers: int = 16  # - bbox JSON·이미지 크기 확인 worker
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # - 정규화 평균
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # - 정규화 표준편차
    allow_truncated_images: bool = True  # - 잘린 이미지 재시도
    verify_images: bool = False  # - 사전 decode 검사
    image_check_workers: int = 8  # - 검사 worker

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def splits(self) -> dict[str, SplitConfig]:
        return {name: split for name, split in (
            ("train", self.train), ("val", self.validation), ("test", self.test)
        ) if split is not None}

    def validate(self) -> None:
        if not self.root.is_absolute():
            raise ValueError("data.root must be an absolute path.")
        if not isinstance(self.train, SplitConfig):
            raise ValueError("data.train must specify paths and labels files.")
        if self.image_root is not None and not self.image_root.is_absolute():
            raise ValueError("data.image_root must be an absolute path.")
        if self.annotation_root is not None and not self.annotation_root.is_absolute():
            raise ValueError("data.annotation_root must be an absolute path.")
        validate_missing_bbox(self.missing_bbox)
        if self.multi_roi and not self.crop_bbox:
            raise ValueError("data.multi_roi requires data.crop_bbox=True.")
        if self.crop_method not in get_args(ROICropMethod):
            raise ValueError(f"data.crop_method must be one of {get_args(ROICropMethod)}.")
        if not self.roi_scales or any(not isfinite(v) or v <= 0 for v in self.roi_scales):
            raise ValueError("data.roi_scales must contain positive finite scales.")
        if not isfinite(self.roi_shift_ratio) or not 0 <= self.roi_shift_ratio <= 1:
            raise ValueError("data.roi_shift_ratio must be in [0, 1].")
        if (
            not self.name
            or not self.name.isascii()
            or not self.name.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("data.name must contain only letters, digits, '-' or '_'.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("data.class_names must contain at least two unique names.")
        if any(not isinstance(name, str) or not name.strip() for name in self.class_names):
            raise ValueError("data.class_names must be non-empty strings.")
        for split in (self.train, self.validation, self.test):
            if split is not None:
                split.validate()
        if self.image_size < 1:
            raise ValueError("data.image_size must be positive.")
        if not isinstance(self.label_offset, int):
            raise ValueError("data.label_offset must be an integer.")
        if (
            len(self.mean) != 3
            or len(self.std) != 3
            or not all(isfinite(v) for v in (*self.mean, *self.std))
            or any(v <= 0 for v in self.std)
        ):
            raise ValueError("data.mean/std must have three finite channels and positive std.")
        if self.image_check_workers < 1:
            raise ValueError("data.image_check_workers must be positive.")
        if self.bbox_workers < 1:
            raise ValueError("data.bbox_workers must be positive.")


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """학습 전용 증강."""

    horizontal_flip: float = 0.5  # - 좌우 반전 확률
    vertical_flip: float = 0.5  # - 상하 반전 확률
    weak_rotation: float = 10.0  # - weak 회전 ±도
    strong_rotation: float = 15.0  # - strong 회전 ±도
    color_jitter: tuple[float, float, float, float] = (0.1, 0.1, 0.05, 0.02)  # - 밝기/대비/채도/색상

    def validate(self) -> None:
        if not 0 <= self.horizontal_flip <= 1 or not 0 <= self.vertical_flip <= 1:
            raise ValueError("augmentation flip probabilities must be in [0, 1].")
        if any(not isfinite(v) or v < 0 for v in (self.weak_rotation, self.strong_rotation)):
            raise ValueError("augmentation rotation must be finite and non-negative.")
        if (
            len(self.color_jitter) != 4
            or any(not isfinite(v) or v < 0 for v in self.color_jitter)
            or self.color_jitter[3] > 0.5
        ):
            raise ValueError("Invalid color_jitter; hue must be in [0, 0.5].")


def data_config_from_dict(values: dict, *, base_dir: Path | None = None,
                          legacy_roi_defaults: bool = False) -> DataConfig:
    """Restore saved input geometry without changing historical ROI defaults."""
    values = dict(values)
    for key in ("root", "image_root", "annotation_root"):
        if values.get(key) is not None:
            value = Path(values[key])
            if base_dir is not None:
                value = value.expanduser()
                value = (value if value.is_absolute() else base_dir / value).resolve()
            values[key] = value
    for key in ("train", "validation", "test"):
        if values.get(key) is not None:
            values[key] = SplitConfig(**values[key])
    for key in ("class_names", "mean", "std", "roi_scales"):
        if key in values:
            values[key] = tuple(values[key])
    if legacy_roi_defaults:
        values.setdefault("crop_method", "roi_resize")
        values.setdefault("multi_roi", False)
    config = DataConfig(**values)
    config.validate()
    return config
