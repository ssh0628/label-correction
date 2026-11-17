"""Training-only density curriculum with bounded-memory pairwise distances.

Implements Co-Correcting's PCA, percentile distance cutoff, density and
distance-to-higher-density equations. The curriculum is built separately
inside each ORIGINAL training class; neither holdout labels nor predictions
are used to establish the class partitions. The local copy of upstream's
trainer calls an unavailable curriculum utility, so the precise ranking
rule here is explicit: densest center first, then increasing distance to a
strictly denser neighbor (ties: higher density, then original row order),
split into approximately equal easy/medium/hard groups.

Distance percentiles are exact, not subsampled. One class's distance matrix
is temporarily memory-mapped to disk, partitioned in-place for its quantile,
then discarded. Density and difficulty passes recompute distances in chunks.
No whole-dataset N x N matrix is allocated in RAM, and no sklearn dependency
or code outside this folder is required.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.spatial.distance import cdist


def _as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def pca_features(features, dimensions: int) -> np.ndarray:
    """Deterministic covariance PCA of already concatenated network features.

The covariance matrix is feature_dim squared; pairwise sample distances are
never constructed here. Inputs are copied to avoid changing caller arrays.
"""
    values = _as_numpy(features)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("Curriculum features require nonempty [samples, features] shape.")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Curriculum features must be finite numeric values.")
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("pca_dimensions must be a positive integer.")
    values = np.asarray(values, dtype=np.float32).copy()
    values -= values.mean(axis=0, keepdims=True)
    keep = min(dimensions, values.shape[1], max(1, values.shape[0] - 1))
    if len(values) == 1:
        return np.zeros((1, keep), dtype=np.float32)
    if keep == values.shape[1]:
        # A full orthonormal projection preserves distances exactly; centering
        # alone avoids an unnecessary eigensolve when no dimensions are cut.
        return values
    covariance = values.T @ values
    covariance /= len(values) - 1
    _, vectors = np.linalg.eigh(covariance)
    result = values @ vectors[:, -keep:]
    if not np.isfinite(result).all():
        raise FloatingPointError("PCA produced non-finite features.")
    return np.asarray(result, dtype=np.float32)


def _exact_distance_percentile(
    values: np.ndarray, percentile: float, chunk_size: int, directory: Path,
) -> float:
    """Exact linear percentile of all directed pairs, including self pairs."""
    count = len(values)
    matrix_path = directory / "distances.bin"
    matrix = np.memmap(matrix_path, dtype=np.float64, mode="w+", shape=(count, count))
    try:
        for start in range(0, count, chunk_size):
            matrix[start:start + chunk_size] = cdist(values[start:start + chunk_size], values)
        flat = matrix.reshape(-1)
        position = percentile / 100.0 * (len(flat) - 1)
        lower, upper = int(np.floor(position)), int(np.ceil(position))
        # ndarray.partition mutates the memory map rather than copying all
        # distances into RAM. Row ordering is no longer needed after this.
        flat.partition((lower, upper))
        cutoff = float(flat[lower] + (position - lower) * (flat[upper] - flat[lower]))
        del flat
        return cutoff
    finally:
        matrix.flush()
        # Explicit close allows cleanup on platforms that lock mapped files.
        matrix._mmap.close()
        del matrix
        matrix_path.unlink(missing_ok=True)


def density_difficulty(
    features: np.ndarray, *, percentile: float = 60.0, chunk_size: int = 512,
    work_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return density rho and distance delta from the paper's equations.

For a point without a strictly denser neighbor, delta is its maximum
distance to any point in the class. Equal-density ties remain ties rather
than being turned into a fictitious higher-density relation.
"""
    values = _as_numpy(features)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Density features must be a finite nonempty matrix.")
    if not isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError("density_percentile must be between 0 and 100.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("curriculum_chunk_size must be positive.")
    values = np.asarray(values, dtype=np.float64)
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="curriculum-", dir=work_dir) as temporary:
        cutoff = _exact_distance_percentile(values, percentile, chunk_size, Path(temporary))
    densities = np.empty(len(values), dtype=np.int64)
    for start in range(0, len(values), chunk_size):
        distances = cdist(values[start:start + chunk_size], values)
        densities[start:start + chunk_size] = (distances < cutoff).sum(axis=1)
    del distances
    delta = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        distances = cdist(values[start:start + chunk_size], values)
        rows = np.arange(start, min(start + chunk_size, len(values)))
        higher = densities[np.newaxis, :] > densities[rows, np.newaxis]
        candidate = np.where(higher, distances, np.inf).min(axis=1)
        delta[rows] = np.where(higher.any(axis=1), candidate, distances.max(axis=1))
    return densities, delta


def build_curriculum(
    features,
    original_labels,
    config,
    *,
    seed: int = 0,
    work_dir: Path | None = None,
) -> np.ndarray:
    """Assign every training row to a group 0..groups-1, preserving row order.

``features`` is the concatenation of fixed-view A/B features, collected
after warm-up. ``config`` is ExperimentConfig.correction. ``seed`` is kept
in the API for provenance; exact PCA/distances and stable ranks use no RNG.
"""
    labels = _as_numpy(original_labels)
    values = _as_numpy(features)
    groups = config.curriculum_groups
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
        raise ValueError("curriculum_groups must be a positive integer.")
    if (labels.shape != (len(values),) or labels.dtype.kind not in {"i", "u"}
            or np.any(labels < 0)):
        raise ValueError("Original training class ids must align with features.")
    projected = pca_features(values, config.pca_dimensions)
    assignments = np.empty(len(labels), dtype=np.int64)
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        density, difficulty = density_difficulty(
            projected[indices], percentile=config.density_percentile,
            chunk_size=config.curriculum_chunk_size, work_dir=work_dir,
        )
        center = int(np.argmax(density))
        rest = np.delete(np.arange(len(indices)), center)
        rank = rest[np.lexsort((indices[rest], -density[rest], difficulty[rest]))]
        order = np.concatenate(([center], rank))
        # Small classes fill the earliest groups; no singleton class is held
        # back until the last correction stage merely because it is rare.
        for group, positions in enumerate(np.array_split(order, groups)):
            assignments[indices[positions]] = group
    return assignments


def active_group_at(epoch: int, config) -> int:
    """Return last enabled group, or -1 outside the label-correction phase.

Every group receives at least one epoch when the configured correction
duration is >= the number of groups. Stage IV freezes the learned labels.
"""
    start, end, groups = config.warmup_epochs, config.correction_end_epoch, config.curriculum_groups
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (epoch, start, end, groups)):
        raise ValueError("Curriculum epochs and group count must be integers.")
    if epoch < 0 or start < 0 or groups < 1 or end - start < groups:
        raise ValueError("Correction duration must provide at least one epoch per curriculum group.")
    if epoch < start or epoch >= end:
        return -1
    return min(groups - 1, (epoch - start) * groups // (end - start))


def eligible_mask(groups, epoch: int, config) -> torch.Tensor:
    """Boolean mask on the same device as tensor group ids, CPU otherwise."""
    assignments = torch.as_tensor(groups)
    if (assignments.ndim != 1 or assignments.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64,
                                                         torch.uint8}
            or bool(((assignments < 0) | (assignments >= config.curriculum_groups)).any())):
        raise ValueError("Curriculum assignments contain invalid group ids.")
    active = active_group_at(epoch, config)
    return (assignments <= active) if active >= 0 else torch.zeros_like(assignments, dtype=torch.bool)
