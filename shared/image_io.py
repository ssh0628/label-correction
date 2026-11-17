"""Image decoding shared by preflight checks and DataLoader workers."""

from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile
from tqdm import tqdm

from .artifacts import JsonlWriter


_DECODER_LOCK = threading.RLock()


def load_rgb_image(path: str, *, allow_truncated: bool) -> tuple[Image.Image, bool]:
    """Strictly decode first; retry a truncated stream only when allowed.

    Pillow's decoder flag is process-global. Restore it after each read and
    serialize callers in the same process; the preflight uses separate processes.
    """
    with _DECODER_LOCK:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            try:
                with Image.open(path) as source:
                    return source.convert("RGB"), False
            except OSError as error:
                if not allow_truncated or "truncated" not in str(error).lower():
                    raise
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as source:
                return source.convert("RGB"), True
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


@dataclass(frozen=True, slots=True)
class ImageCheck:
    split: str
    index: int
    path: str
    status: str
    error: str | None = None


def _check_image(task: tuple[str, int, str, bool]) -> ImageCheck:
    split, index, path, allow_truncated = task
    try:
        image, recovered = load_rgb_image(path, allow_truncated=allow_truncated)
        image.close()
        return ImageCheck(split, index, path, "recovered" if recovered else "ok")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return ImageCheck(
            split, index, path, "failed", f"{type(error).__name__}: {error}"
        )


def bounded_map(function, tasks, executor=None, *, batch_size=32):
    """Bound eagerly submitted executor work; preserve input order."""
    if executor is None:
        yield from map(function, tasks)
        return
    pending = iter(tasks)
    while batch := list(islice(pending, batch_size)):
        yield from executor.map(function, batch, chunksize=16)


def verify_image_files(
    tasks: Iterable[tuple[str, int, str, bool]],
    *,
    count: int,
    workers: int,
    report_path: Path | None,
) -> dict[str, int]:
    """Fully decode every image, report exceptional paths, and keep all indices.

    Invalid images stop the run after the scan so one report lists all failures.
    No placeholder images are created and no samples are silently removed.
    """
    summary = {"checked": 0, "recovered": 0, "failed": 0}
    failure_examples: list[str] = []
    executor_context = (
        ProcessPoolExecutor(
            max_workers=min(workers, count),
            mp_context=multiprocessing.get_context("spawn"),
        )
        if workers > 1 and count > 1
        else nullcontext(None)
    )
    with JsonlWriter(report_path, append=True, flush=False) as report, executor_context as executor:
        results = bounded_map(_check_image, tasks, executor, batch_size=min(workers, count) * 32)
        for result in tqdm(results, total=count, desc="Check images", unit="img", mininterval=5):
            summary["checked"] += 1
            if result.status == "ok":
                continue
            summary[result.status] += 1
            report.write(asdict(result))
            if result.status == "failed" and len(failure_examples) < 5:
                failure_examples.append(
                    f"{result.split}[{result.index}] {result.path}: {result.error}"
                )
        report.write({"status": "summary", **summary})
    if summary["failed"]:
        raise RuntimeError(
            f"{summary['failed']} unreadable image(s); no samples were removed.\n"
            + "\n".join(failure_examples)
            + (f"\nFull report: {report_path}" if report_path else "")
        )
    return summary
