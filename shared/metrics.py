"""Classification scores from confusion counts and mean cross-entropy loss.

Scores are fractions in [0, 1], not percentages. Confusion matrices use truth
rows / prediction columns. Balanced accuracy excludes classes with no true
support; macro scores include every configured class with undefined class
scores set to zero. Cross-entropy loss uses natural logs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .precision import full_precision


_LABEL_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def class_weights(labels: Tensor, num_classes: int) -> tuple[Tensor, Tensor]:
    """Inverse square-root Train frequencies, normalized to mean one."""
    labels = torch.as_tensor(labels).detach().cpu()
    if (type(num_classes) is not int or num_classes < 1 or labels.ndim != 1 or not labels.numel()
            or labels.dtype not in _LABEL_DTYPES or ((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("WCE requires nonempty integer Train labels within the configured class range.")
    counts = torch.bincount(labels.long(), minlength=num_classes)
    missing = counts.eq(0).nonzero().flatten().tolist()
    if missing:
        raise ValueError(f"WCE requires every class in Train; missing classes={missing}.")
    weights = counts.float().rsqrt()
    return weights / weights.mean(), counts


def metrics_from_confusion_matrix(
    confusion_matrix: Tensor | list[list[int]],
    *,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute JSON-compatible scores; undefined class scores are zero."""
    matrix = torch.as_tensor(confusion_matrix).detach().to("cpu", torch.float64)
    if matrix.ndim != 2 or matrix.size(0) == 0 or matrix.size(0) != matrix.size(1):
        raise ValueError("Confusion matrix must be a nonempty square matrix.")
    if not torch.isfinite(matrix).all() or (matrix < 0).any() or (matrix != matrix.round()).any():
        raise ValueError("Confusion matrix must contain finite nonnegative integer counts.")
    count = int(matrix.sum().item())
    if count == 0:
        raise ValueError("Confusion matrix must contain at least one sample.")
    classes = matrix.size(0)
    names = tuple(str(index) for index in range(classes)) if class_names is None else tuple(class_names)
    if len(names) != classes or any(not isinstance(name, str) for name in names):
        raise ValueError("class_names must contain one string per output class.")

    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    correct = matrix.diagonal()
    precision = correct / predicted.clamp_min(1)
    recall = correct / support.clamp_min(1)
    f1 = 2 * correct / (support + predicted).clamp_min(1)
    accuracy = float(correct.sum().item() / count)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(recall[support > 0].mean().item()),
        "macro_f1": float(f1.mean().item()),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "sample_count": count,
        "confusion_matrix": matrix.to(torch.int64).tolist(),
        "per_class": [
            {
                "class_id": index,
                "class_name": names[index],
                "support": int(support[index]),
                "predicted_count": int(predicted[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(classes)
        ],
    }


def _labels_for_batch(labels: Tensor, batch_size: int, classes: int, device: torch.device) -> Tensor:
    # Normal DataLoader labels are still on CPU: reject malformed labels before
    # invoking CUDA gather / bincount kernels, without a device synchronization.
    labels = torch.as_tensor(labels)
    if labels.ndim != 1 or labels.numel() != batch_size or labels.dtype not in _LABEL_DTYPES:
        raise ValueError("Evaluation labels must be a one-dimensional integer tensor matching the batch.")
    if ((labels < 0) | (labels >= classes)).any().item():
        raise ValueError(f"Evaluation labels must lie in [0, {classes - 1}].")
    return labels.to(device, dtype=torch.long, non_blocking=True)


def _validate_logits(logits: Tensor, batch_size: int | None = None) -> None:
    if (
        not isinstance(logits, Tensor) or logits.ndim != 2 or logits.size(1) == 0
        or (batch_size is not None and logits.size(0) != batch_size)
    ):
        raise ValueError("Classifier logits must have shape (batch_size, num_classes).")
    if logits.size(0) == 0:
        raise ValueError("Evaluation batches must not be empty.")
    if not logits.is_floating_point():
        raise ValueError("Classifier logits must be floating point.")


def _metrics_from_batches(
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    class_names: tuple[str, ...] | None,
) -> dict[str, Any]:
    """Accumulate validated logits with the same definitions for every evaluator."""
    # MPS does not support float64; model inference itself remains FP32.
    accumulation_dtype = torch.float32 if device.type == "mps" else torch.float64
    loss_sum = torch.zeros((), dtype=accumulation_dtype, device=device)
    matrix: Tensor | None = None
    sample_count = 0

    with full_precision(device):
        for logits, labels in batches:
            classes = logits.size(1)
            if matrix is None:
                if class_names is not None and (
                    len(class_names) != classes or any(not isinstance(name, str) for name in class_names)
                ):
                    raise ValueError("class_names must contain one string per output class.")
                matrix = torch.zeros((classes, classes), dtype=torch.int64, device=device)
            elif matrix.size(0) != classes:
                raise ValueError("Classifier output class count changed between batches.")
            labels = _labels_for_batch(labels, logits.size(0), classes, device)
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            predictions = logits.argmax(dim=1)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").to(accumulation_dtype)
            matrix += torch.bincount(
                labels * classes + predictions, minlength=classes * classes
            ).reshape(classes, classes)
            sample_count += logits.size(0)

    if matrix is None or sample_count == 0:
        raise RuntimeError("Classification evaluation loader is empty.")
    if not torch.isfinite(loss_sum).item():
        raise FloatingPointError("Classification evaluation produced non-finite loss.")
    result = metrics_from_confusion_matrix(matrix, class_names=class_names)
    result["loss"] = loss_sum.item() / sample_count
    return result


@torch.no_grad()
def metrics_from_logits(
    logits: Tensor,
    labels: Tensor,
    *,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Score final [N, C] logits, e.g. logits averaged over multiple ROI views."""
    _validate_logits(logits)
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("Classification evaluation received non-finite logits.")
    return _metrics_from_batches(((logits, labels),), logits.device, class_names)
