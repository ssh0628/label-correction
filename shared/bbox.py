"""Ordered bbox metadata compatible with multi_roi's version-1 caches."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from zipfile import BadZipFile

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import validate_missing_bbox
from .artifacts import JsonlWriter, atomic_save_npz
from .image_io import bounded_map

if TYPE_CHECKING:
    from .config import DataConfig, SplitConfig

BBox = tuple[int, int, int, int]
_MISSING = (-1, -1, -1, -1)
_VERSION = 1


def _fingerprint(paths: Iterable[str]) -> str:
    digest = hashlib.sha256(b"bbox-cache-v1\n")
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_path(config: DataConfig, split: SplitConfig, name: str) -> Path:
    if split.bboxes is not None:
        path = Path(split.bboxes)
        path = path if path.is_absolute() else config.root / path
        if path.is_file():
            return path
        aliases = (f"{name}_bbox_cache_v1.npz", f"{name}_bboxes_cache_v1.npz")
        if path.name in aliases:
            alternate = path.with_name(aliases[1 - aliases.index(path.name)])
            if alternate.is_file():
                return alternate
        return path
    candidates = [
        config.root / f"{name}_{suffix}_cache_v1.npz"
        for suffix in ("bbox", "bboxes")
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(
            f"{name}: both bbox cache aliases exist; set the split's bboxes filename explicitly."
        )
    return existing[0] if existing else candidates[1]


def _read_cache(path: Path, paths: tuple[str, ...]) -> tuple[BBox | None, ...]:
    try:
        with np.load(path, allow_pickle=False) as cached:
            if cached["version"].item() != _VERSION:
                raise ValueError("unsupported bbox cache version")
            if cached["paths_sha256"].item() != _fingerprint(paths):
                raise ValueError("ordered image paths do not match the bbox cache")
            array = cached["bboxes"]
            if array.shape != (len(paths), 4) or array.dtype.kind not in "iuf":
                raise ValueError("bboxes must be a numeric (number of images, 4) array")
            if not np.isfinite(array).all() or np.any(array != np.floor(array)):
                raise ValueError("bbox coordinates must be finite integers")
            limits = np.iinfo(np.int32)
            if np.any(array < limits.min) or np.any(array > limits.max):
                raise ValueError("bbox coordinates exceed int32 range")
            boxes: list[BBox | None] = []
            for index, row in enumerate(array):
                box = tuple(int(value) for value in row)
                if box == _MISSING:
                    boxes.append(None)
                elif box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"empty or inverted bbox at index {index}")
                else:
                    boxes.append(box)
            return tuple(boxes)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, BadZipFile) as error:
        raise RuntimeError(f"Invalid bbox cache {path}: {error}") from error


def _annotation_path(config: DataConfig, image_path: str) -> Path | None:
    image = Path(image_path)
    if config.annotation_root is not None:
        try:
            relative = image.expanduser().resolve().relative_to(
                (config.image_root or config.root).expanduser().resolve()
            )
        except ValueError as error:
            raise ValueError(
                f"Cannot map {image} into annotation_root. "
                "Set data.image_root to the common image directory; "
                "annotation_root must mirror its subdirectories."
            ) from error
        image = config.annotation_root / relative
    candidates = (
        image.with_suffix(".json"), image.with_suffix(".JSON"),
        image.with_name(image.name + ".json"),
    )
    return next((path for path in candidates if path.is_file()), None)


def _coordinate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("bbox coordinates must be numeric integers")
    if not np.isfinite(value) or value != int(value):
        raise ValueError("bbox coordinates must be finite integers")
    return int(value)


def _read_annotation(task: tuple[DataConfig, str]) -> tuple[BBox | None, str, str | None]:
    config, image_path = task
    annotation = None
    try:
        annotation = _annotation_path(config, image_path)
        if annotation is None:
            return None, "missing", "annotation JSON not found"
        document = json.loads(annotation.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("labelingInfo"), list):
            raise ValueError("annotation must contain a labelingInfo list")
        for item in document["labelingInfo"]:
            if not isinstance(item, dict):
                raise ValueError("labelingInfo items must be objects")
            box = item.get("box")
            if not box:
                continue
            if not isinstance(box, dict):
                raise ValueError("box must be an object")
            locations = box.get("location") or []
            if not isinstance(locations, list):
                raise ValueError("box.location must be a list")
            if not locations:
                continue
            location = locations[0]
            if not isinstance(location, dict):
                raise ValueError("box.location items must be objects")
            x, y, width, height = (
                _coordinate(location.get(key)) for key in ("x", "y", "width", "height")
            )
            if width <= 0 or height <= 0:
                continue
            coordinates = (x, y, x + width, y + height)
            limits = np.iinfo(np.int32)
            if any(value < limits.min or value > limits.max for value in coordinates):
                raise ValueError("bbox coordinates exceed int32 range")
            return coordinates, "ok", None
        return None, "missing", f"no positive-size bbox in {annotation}"
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, "unreadable_json", f"{annotation or image_path}: {error}"
    except (OSError, ValueError, TypeError, OverflowError) as error:
        return None, "invalid", f"{annotation or image_path}: {error}"


def _audit_missing(
    boxes: tuple[BBox | None, ...], paths: tuple[str, ...], name: str,
    policy: str, report_path: Path | None,
) -> None:
    missing = sum(box is None for box in boxes)
    with JsonlWriter(report_path, append=True, flush=False) as report:
        for index, box in enumerate(boxes):
            if box is None:
                report.write({"status": "bbox_missing", "split": name,
                              "index": index, "path": paths[index]})
        report.write({"status": "bbox_summary", "split": name, "checked": len(boxes),
                      "with_bbox": len(boxes) - missing, "missing": missing, "policy": policy})
    if missing and policy == "error":
        examples = list(islice((paths[i] for i, box in enumerate(boxes) if box is None), 5))
        raise RuntimeError(
            f"{name}: {missing} missing bbox(es); crop requires valid annotations.\n"
            + "\n".join(examples)
            + (f"\nFull report: {report_path}" if report_path else "")
        )
    if missing and policy == "full":
        print(f"WARNING [{name}]: using full images for {missing} missing bbox(es) (missing_bbox='full').")


def prepare_bboxes(
    config: DataConfig, split: SplitConfig, name: str, paths: tuple[str, ...], *,
    raw_paths: tuple[str, ...] | None = None, report_path: Path | None = None,
) -> tuple[BBox | None, ...]:
    """Build metadata or repair only missing entries; preserve original row alignment."""
    if not config.crop_bbox:
        return (None,) * len(paths)
    validate_missing_bbox(config.missing_bbox)
    originals = raw_paths if raw_paths is not None else paths
    if len(originals) != len(paths):
        raise ValueError("Raw and resolved bbox paths must have the same length.")
    path = _cache_path(config, split, name)
    cached_boxes = None
    if path.is_file():
        try:
            cached_boxes = _read_cache(path, originals)
        except RuntimeError:
            pass
    boxes_list = list(cached_boxes) if cached_boxes is not None else [None] * len(paths)
    indices = [index for index, box in enumerate(boxes_list) if box is None]
    if not indices:
        boxes = tuple(boxes_list)
        _audit_missing(boxes, paths, name, config.missing_bbox, report_path)
        return boxes
    workers = max(1, min(config.bbox_workers, len(indices)))
    recovered = []
    invalid, examples = 0, []
    with JsonlWriter(report_path, append=True, flush=False) as report, ThreadPoolExecutor(max_workers=workers) as executor:
        action = "Repair" if cached_boxes is not None else "Cache"
        tasks = ((config, paths[index]) for index in indices)
        results = bounded_map(_read_annotation, tasks, executor, batch_size=workers * 32)
        for index, (box, status, error) in zip(indices, tqdm(
                results, total=len(indices), desc=f"{action} {name} bboxes", unit="img", mininterval=5)):
            boxes_list[index] = box
            if cached_boxes is not None and box is not None:
                recovered.append(index)
            if status in {"invalid", "unreadable_json"}:
                exclude = status == "unreadable_json" and config.missing_bbox == "drop"
                if not exclude:
                    invalid += 1
                    if len(examples) < 5:
                        examples.append(f"{paths[index]}: {error}")
                report.write({"status": "bbox_invalid", "split": name, "index": index,
                              "path": paths[index], "error": error,
                              "action": "exclude" if exclude else "error",
                              "reason": "invalid_json" if status == "unreadable_json" else "invalid_bbox"})
    boxes = tuple(boxes_list)
    if invalid:
        raise RuntimeError(
            f"{name}: {invalid} invalid bbox annotation(s); cache was not changed.\n"
            + "\n".join(examples)
            + (f"\nFull report: {report_path}" if report_path else "")
        )
    if cached_boxes is None:
        _audit_missing(boxes, paths, name, config.missing_bbox, report_path)
        _save_cache(path, originals, boxes)
    else:
        if recovered:
            _save_cache(path, originals, boxes)
        with JsonlWriter(report_path, append=True, flush=False) as report:
            for index in recovered:
                report.write({"status": "bbox_recovered", "split": name,
                              "index": index, "path": paths[index], "bbox": boxes[index]})
            report.write({"status": "bbox_repair_summary", "split": name,
                          "checked": len(indices), "recovered": len(recovered),
                          "missing": len(indices) - len(recovered)})
        _audit_missing(boxes, paths, name, config.missing_bbox, report_path)
    return boxes


def _save_cache(path: Path, originals: tuple[str, ...], boxes: tuple[BBox | None, ...]) -> None:
    """Publish a complete ordered cache only after all annotation reads succeed."""
    _save_npz(
        path, bboxes=np.asarray([box or _MISSING for box in boxes], dtype=np.int32).reshape(-1, 4),
        paths_sha256=np.asarray(_fingerprint(originals)), version=np.asarray(_VERSION, dtype=np.int32),
    )


def _save_npz(path: Path, **values: np.ndarray) -> None:
    atomic_save_npz(path, compressed=False, overwrite=True, **values)


def clamp_bbox(bbox: BBox | None, image_width: int, image_height: int) -> BBox | None:
    """Clip against the original pixels; None means no usable intersection."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    clipped = max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2)
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


_NO_DIMENSIONS = (-1,) * 7


def _read_dimensions(path: Path, paths: tuple[str, ...]) -> np.ndarray | None:
    """An absent/stale metadata sidecar falls back to reading image headers."""
    try:
        with np.load(path, allow_pickle=False) as cached:
            if (
                cached["version"].item() != 1
                or cached["paths_sha256"].item() != _fingerprint(paths)
                or cached["max_image_pixels"].item() != str(Image.MAX_IMAGE_PIXELS)
            ):
                return None
            values = cached["image_metadata"]
            if values.shape != (len(paths), 7) or values.dtype != np.int64:
                return None
            if np.any(~((values[:, -2:] > 0).all(axis=1) | (values == -1).all(axis=1))):
                return None
            return values
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, BadZipFile):
        return None


def _image_signature(path: str) -> tuple[int, ...]:
    info = os.stat(path)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _clip_to_image(
    task: tuple[str, BBox | None, tuple[int, ...] | None],
) -> tuple[BBox | None, tuple[int, ...]]:
    path, box, previous = task
    if box is None:
        return None, _NO_DIMENSIONS
    try:
        signature = _image_signature(path) if previous is not None else None
        if signature is not None and signature == previous[:5]:
            return clamp_bbox(box, *previous[-2:]), previous
        with Image.open(path) as image:
            size = image.size
        # Do not reuse a header if the file changed while it was being read.
        metadata = _NO_DIMENSIONS
        if signature is not None and signature == _image_signature(path):
            metadata = (*signature, *size)
        return clamp_bbox(box, *size), metadata
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise RuntimeError(f"Failed to read image dimensions: {path}") from error


def validate_crop_boxes(
    paths: tuple[str, ...], boxes: tuple[BBox | None, ...], name: str,
    *, workers: int, missing: str, report_path: Path | None, cache_path: Path | None = None,
) -> tuple[BBox | None, ...]:
    """Recheck file states; reuse native dimensions, never pixels or crop decisions."""
    if len(boxes) != len(paths):
        raise ValueError(f"{name}: bbox count does not match image count.")
    workers = max(1, min(workers, len(paths)))
    previous = _read_dimensions(cache_path, paths) if cache_path is not None else None
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    valid: list[BBox | None] = []
    metadata: list[tuple[int, ...]] = []
    failures: list[str] = []
    with JsonlWriter(report_path, append=True, flush=False) as report, pool as executor:
        saved = (tuple(int(value) for value in row) for row in previous) if previous is not None else (
            (_NO_DIMENSIONS if cache_path is not None else None) for _ in paths
        )
        results = bounded_map(_clip_to_image, zip(paths, boxes, saved), executor, batch_size=workers * 32)
        progress = tqdm(results, total=len(paths), desc=f"Check {name} bbox bounds", unit="img", mininterval=5)
        for index, (path, box, (clipped, dimensions)) in enumerate(zip(paths, boxes, progress)):
            if box is not None and (clipped is None or clipped != box):
                report.write({"split": name, "index": index, "path": path,
                              "status": "bbox_clipped" if clipped else "bbox_outside_image",
                              "bbox": box, "clipped_bbox": clipped})
                if clipped is None and missing != "full" and len(failures) < 10:
                    failures.append(f"{name}[{index}]: {path} bbox={box}")
            valid.append(clipped)
            metadata.append(dimensions)
    if failures:
        raise ValueError("Bboxes outside source images:\n" + "\n".join(failures))
    if cache_path is not None:
        values = np.asarray(metadata, dtype=np.int64).reshape(-1, 7)
        if previous is None or not np.array_equal(previous, values):
            try:
                _save_npz(
                    cache_path, image_metadata=values, paths_sha256=np.asarray(_fingerprint(paths)),
                    version=np.asarray(1, dtype=np.int32), max_image_pixels=np.asarray(str(Image.MAX_IMAGE_PIXELS)),
                )
            except OSError:
                pass
    return tuple(valid)
