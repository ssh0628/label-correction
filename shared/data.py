"""Read and validate the saved dataset, bbox rows and fixed correction targets."""

from __future__ import annotations

import json
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from .bbox import BBox, prepare_bboxes, validate_crop_boxes
from .artifacts import JsonlWriter, file_sha256 as _file_hash
from .config import DataConfig, SplitConfig
from .image_io import load_rgb_image, verify_image_files
from .roi import ROICrop


@dataclass(frozen=True, slots=True)
class SplitSource:
    name: str
    paths: tuple[str, ...]
    labels: Tensor
    raw_paths: tuple[str, ...] = ()


def _record_dataset_metadata(config: DataConfig, report_path: Path | None) -> None:
    """Record optional NPY provenance, without sampling an already sampled dataset."""
    metadata = {}
    for filename in ("classes.json", "split_config.json"):
        path = config.root / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        classes = payload.get("classes")
        if classes is not None and classes != list(config.class_names):
            raise ValueError(f"{path}: classes order does not match data.class_names.")
        metadata[filename] = payload
    with JsonlWriter(report_path, append=True, flush=False) as report:
        report.write({"status": "dataset_metadata", "root": str(config.root),
                      "sampling": "as_provided", "metadata": metadata})


def _resolve_array(root: Path, filename: str) -> Path:
    path = Path(filename)
    path = path if path.is_absolute() else root / path
    if not path.is_file() and path.name in {
        f"{split}_{suffix}.npy"
        for split in ("train", "val", "test")
        for suffix in ("path", "paths")
    }:
        old, new = (
            ("_paths.npy", "_path.npy")
            if path.name.endswith("_paths.npy") else ("_path.npy", "_paths.npy")
        )
        alias = path.with_name(path.name.replace(old, new))
        if alias.is_file():
            return alias
    if not path.is_file():
        raise FileNotFoundError(f"Dataset array not found: {path}")
    return path


def _load_split(config: DataConfig, split: SplitConfig, name: str) -> SplitSource:
    paths_file = _resolve_array(config.root, split.paths)
    labels_file = _resolve_array(config.root, split.labels)
    paths_array = np.load(paths_file, allow_pickle=False)
    labels_array = np.load(labels_file, allow_pickle=False)
    if (
        paths_array.ndim != 1 or labels_array.ndim != 1
        or len(paths_array) != len(labels_array)
    ):
        raise ValueError(f"{name}: paths and labels must be aligned rank-1 arrays.")
    if not len(paths_array):
        raise ValueError(f"{name}: dataset is empty.")
    if paths_array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name}: paths must be a string array saved without pickle.")
    if labels_array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name}: labels must be numeric class indices.")
    if labels_array.dtype.kind == "f" and (
        not np.isfinite(labels_array).all()
        or np.any(labels_array != np.floor(labels_array))
    ):
        raise ValueError(f"{name}: labels must be finite integer class indices.")
    minimum, maximum = config.label_offset, config.label_offset + config.num_classes
    if np.any(labels_array < minimum) or np.any(labels_array >= maximum):
        raise ValueError(
            f"{name}: labels must be in [{minimum}, {maximum}); "
            "check class_names and label_offset."
        )
    labels = labels_array.astype(np.int64, copy=False)
    if config.label_offset:
        labels = labels - config.label_offset
    image_root = config.image_root or config.root
    paths: list[str] = []
    for raw_path in tqdm(paths_array, desc=f"Load {name}", unit="img", mininterval=5):
        value = raw_path.decode("utf-8") if isinstance(raw_path, bytes) else str(raw_path)
        if not value.strip():
            raise ValueError(f"{name}: empty image path at index {len(paths)}.")
        path = Path(value).expanduser()
        paths.append(str((path if path.is_absolute() else image_root / path).resolve()))
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name}: duplicate image paths are not allowed.")
    raw_paths = tuple(str(value.item()) if isinstance(value, np.generic) else str(value) for value in paths_array)
    return SplitSource(name, tuple(paths), torch.from_numpy(labels), raw_paths)


def dataset_bboxes(
    config: DataConfig, sources: dict[str, SplitSource], *,
    report_path: Path | None = None,
) -> dict[str, tuple[BBox | None, ...]]:
    """Resolve coordinates once, before image preprocessing or model creation."""
    if not config.crop_bbox:
        return {}
    return {
        name: prepare_bboxes(
            config, config.splits[name], name, source.paths,
            raw_paths=source.raw_paths or None, report_path=report_path,
        )
        for name, source in sources.items()
    }


def filter_bbox_sources(
    config: DataConfig,
    sources: dict[str, SplitSource],
    boxes: dict[str, tuple[BBox | None, ...]],
    *, report_path: Path | None = None,
) -> tuple[dict[str, SplitSource], dict[str, tuple[BBox | None, ...]]]:
    """Apply one stable mask to paths, labels and boxes before training."""
    if not config.crop_bbox or config.missing_bbox != "drop":
        return sources, boxes
    filtered_sources: dict[str, SplitSource] = {}
    filtered_boxes: dict[str, tuple[BBox | None, ...]] = {}
    for name, source in sources.items():
        values = boxes[name]
        if len(values) != len(source.paths):
            raise ValueError(f"{name}: bbox count does not match image count.")
        excluded = [index for index, box in enumerate(values) if box is None]
        input_count = len(values)
        if excluded:
            kept = [index for index, box in enumerate(values) if box is not None]
            labels = source.labels[torch.tensor(kept, dtype=torch.int64)]
        else:
            labels = source.labels
        counts = torch.bincount(labels, minlength=config.num_classes).tolist()
        class_counts = dict(zip(config.class_names, counts))
        with JsonlWriter(report_path, append=True, flush=False) as report:
            for index in excluded:
                label = int(source.labels[index])
                report.write({"status": "bbox_excluded", "split": name, "index": index,
                              "path": source.paths[index], "label": label,
                              "class_name": config.class_names[label], "reason": "missing_bbox"})
            report.write({"status": "bbox_filter_summary", "split": name,
                          "input_count": input_count, "excluded_count": len(excluded),
                          "retained_count": len(labels), "class_counts": class_counts})
        if not len(labels):
            raise ValueError(f"{name}: no samples remain after excluding missing bboxes.")
        if excluded:
            source = replace(
                source, paths=tuple(source.paths[index] for index in kept), labels=labels,
                raw_paths=tuple(source.raw_paths[index] for index in kept) if source.raw_paths else (),
            )
            values = tuple(values[index] for index in kept)
        filtered_sources[name] = source
        filtered_boxes[name] = values
    return filtered_sources, filtered_boxes


def inspect_dataset(
    config: DataConfig, *, report_path: Path | None = None, split: str | None = None,
) -> dict[str, SplitSource]:
    """Validate configured arrays, split separation, and optional full image decode."""
    config.validate()
    if split is not None and split not in config.splits:
        raise ValueError(f"Dataset split is not configured: {split!r}.")
    sources = {name: _load_split(config, part, name) for name, part in config.splits.items()
               if split is None or name == split}
    _record_dataset_metadata(config, report_path)
    seen: set[str] = set()
    for name, source in sources.items():
        overlap = seen.intersection(source.paths)
        if overlap:
            raise ValueError(
                f"train/val/test paths must be disjoint; "
                f"{name} overlaps at {min(overlap)}."
            )
        seen.update(source.paths)
    if config.verify_images:
        _verify_sources(config, sources, report_path)
    return sources


def _verify_sources(
    config: DataConfig, sources: dict[str, SplitSource], report_path: Path | None,
) -> None:
    tasks = (
        (name, index, path, config.allow_truncated_images)
        for name, source in sources.items()
        for index, path in enumerate(source.paths)
    )
    verify_image_files(
        tasks, count=sum(len(source.paths) for source in sources.values()),
        workers=config.image_check_workers, report_path=report_path,
    )


class IndexedImageDataset(Dataset):
    def __init__(
        self,
        source: SplitSource,
        transform: Callable[[Image.Image], object],
        *,
        allow_truncated: bool,
        roi: ROICrop,
        labeled: bool = False,
        bboxes: tuple[BBox | None, ...] | None = None,
    ) -> None:
        self.source = source
        self.transform = transform
        self.allow_truncated = allow_truncated
        self.labeled = labeled
        self.roi = roi
        self.bboxes = bboxes

    def __len__(self) -> int:
        return len(self.source.paths)

    def __getitem__(self, index: int) -> tuple[object, int | Tensor]:
        path = self.source.paths[index]
        box = self.bboxes[index] if self.bboxes is not None else None
        try:
            image, _ = load_rgb_image(path, allow_truncated=self.allow_truncated)
            with closing(image):
                cropped = self.roi(image, box)
            # Full-resolution RGB is released before multi-view transforms.
            with closing(cropped):
                transformed = self.transform(cropped)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise RuntimeError(f"Failed to load {self.source.name}[{index}]: {path}") from error
        return transformed, self.source.labels[index] if self.labeled else index


def correction_checkpoint(metadata: dict) -> dict:
    """Read the same Teacher identity from native SSR or Co correction metadata."""
    checkpoint = metadata.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        raise ValueError("Correction checkpoint must be a JSON object.")
    return {"sha256": metadata.get("checkpoint_sha256", checkpoint.get("sha256")),
            "completed_epochs": metadata.get("teacher_completed_epochs", checkpoint.get("completed_epochs")),
            "selection": metadata.get("teacher_selection", checkpoint.get("selection", {})) or {}}


def _corrected_targets(config: DataConfig, source: SplitSource, path: Path) -> tuple[Tensor, dict]:
    """Bind alternate targets to unchanged, bbox-filtered rows of one split."""
    split = config.splits.get(source.name)
    if split is None:
        raise ValueError(f"Correction split {source.name!r} is not configured.")
    path = Path(path)
    digest = _file_hash(path)
    with np.load(path, allow_pickle=False) as artifact:
        if not {"paths", "original_labels", "corrected_labels", "metadata"} <= set(artifact.files):
            raise ValueError("Correction NPZ requires paths/original_labels/corrected_labels/metadata.")
        encoded = artifact["metadata"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("Correction metadata must be scalar Unicode JSON.")
        metadata = json.loads(encoded.item())
        if (metadata.get("split") != source.name or metadata.get("class_names") != list(config.class_names)
            or metadata.get("label_offset") != 0 or metadata.get("original_label_offset") != config.label_offset):
            raise ValueError(f"Correction split/class_names/label_offset do not match {source.name}.")
        if not np.array_equal(artifact["paths"], np.asarray(source.paths)):
            raise ValueError(f"Correction paths must match every original {source.name} row in order.")
        original, labels = artifact["original_labels"], artifact["corrected_labels"]
        if (original.dtype.kind not in {"i", "u"} or labels.dtype.kind not in {"i", "u"}
            or original.shape != (len(source.paths),) or labels.shape != original.shape
            or not np.array_equal(original, source.labels.numpy())
            or np.any(labels < 0) or np.any(labels >= config.num_classes)):
            raise ValueError(f"Correction original_labels/targets are incompatible with {source.name}.")
        targets = torch.from_numpy(labels.astype(np.int64, copy=True))
    expected_hashes = {
        "paths": _file_hash(_resolve_array(config.root, split.paths)),
        "labels": _file_hash(_resolve_array(config.root, split.labels)),
    }
    geometry = {
        "image_size": config.image_size, "crop_bbox": config.crop_bbox,
        "crop_method": config.crop_method, "mean": list(config.mean), "std": list(config.std),
    }
    if (metadata.get("source_hashes") != expected_hashes
        or any(metadata.get("input_config", {}).get(key) != value for key, value in geometry.items())
        or _file_hash(path) != digest):
        raise ValueError("Correction provenance/input_config differs from the original dataset.")
    return targets, {"source_path": str(path), "artifact_sha256": digest, "metadata": metadata,
                     "sample_count": len(targets), "changed_count": int(targets.ne(source.labels).sum())}


def prepare_dataset(
    config: DataConfig, *, report_path: Path | None = None, split: str | None = None,
) -> tuple[dict[str, SplitSource], dict[str, tuple[BBox | None, ...]]]:
    """Shared run/audit preparation: arrays → bbox exclusion → image validation."""
    sources = inspect_dataset(
        replace(config, verify_images=False), report_path=report_path, split=split,
    )
    boxes = dataset_bboxes(config, sources, report_path=report_path)
    sources, boxes = filter_bbox_sources(config, sources, boxes, report_path=report_path)
    if config.verify_images:
        _verify_sources(config, sources, report_path)
    boxes = {
        name: validate_crop_boxes(
            sources[name].paths, values, name, workers=config.bbox_workers,
            missing=config.missing_bbox, report_path=report_path,
            cache_path=config.root / f"{name}_bbox_dimensions_v1.npz",
        )
        for name, values in boxes.items()
    }
    return sources, boxes
