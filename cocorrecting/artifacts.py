"""Atomic experiment artifacts with immutable original labels and provenance.

Train corrections are learned label distributions, not a final model's argmax.
Teacher-inferred Val/Test labels remain separate from the original annotations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .data import input_config, source_hashes as _source_hashes
from shared.artifacts import (_json_default, atomic_save_npz as _atomic_npz, atomic_write_json,
                              file_sha256)
from shared.data import _corrected_targets
from .model import teacher_model_weights


def _data(config):
    return getattr(config, "data", config)


def source_hashes(config, split: str) -> dict[str, str]:
    return _source_hashes(_data(config), split)


def dataset_fingerprint(sources: dict, config) -> str:
    """Bind a checkpoint to ordered, bbox-filtered samples and input geometry."""
    data = _data(config)
    payload = {"class_names": list(data.class_names), "input_config": input_config(data),
               "splits": {name: {"paths": list(source.paths),
                                  "labels": torch.as_tensor(source.labels).cpu().tolist(),
                                  "source_hashes": source_hashes(data, name)}
                          for name, source in sorted(sources.items())}}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_label_logits(source, label_logits, classes: int) -> Tensor:
    values = torch.as_tensor(label_logits).detach().cpu()
    if (values.ndim != 2 or values.shape != (len(source.paths), classes)
            or not values.is_floating_point() or not torch.isfinite(values).all()):
        raise ValueError("Label logits must be finite floating point [source_count, num_classes].")
    original = torch.as_tensor(source.labels).detach().cpu()
    if (original.shape != (len(source.paths),) or original.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    } or ((original < 0) | (original >= classes)).any()):
        raise ValueError("Source labels must contain one in-range integer for every path.")
    return values.float()


def _summary(original: Tensor, corrected: Tensor, data) -> dict:
    classes = len(data.class_names)
    transitions = torch.bincount(original * classes + corrected,
                                minlength=classes * classes).reshape(classes, classes)
    changed = corrected.ne(original)
    original_counts = torch.bincount(original, minlength=classes)
    changed_counts = torch.bincount(original[changed], minlength=classes)
    return {"sample_count": len(original), "changed_count": int(changed.sum()),
            "changed_fraction": float(changed.float().mean()),
            "original_class_counts": original_counts.tolist(),
            "corrected_class_counts": torch.bincount(corrected, minlength=classes).tolist(),
            "changed_count_by_original_class": changed_counts.tolist(),
            "changed_fraction_by_original_class": (changed_counts / original_counts.clamp_min(1)).tolist(),
            "label_transitions": transitions.tolist(), "class_names": list(data.class_names)}


def _metadata(config, source, origin: str, checkpoint_info: dict | None) -> dict:
    data = _data(config)
    return {"schema_version": 1, "method": "co_correcting", "split": source.name,
            "class_names": list(data.class_names), "label_offset": 0,
            "original_label_offset": data.label_offset,
            "source_hashes": source_hashes(data, source.name),
            "input_config": input_config(data), "label_origin": origin,
            "ground_truth_verified": False, "checkpoint": checkpoint_info or {}}


def write_train_correction(
    config, source, label_logits, output_path: Path | str, *,
    checkpoint_info: dict | None = None,
) -> dict:
    """Export the checkpoint's optimized train label logits, without inference."""
    if source.name != "train":
        raise ValueError("Learned label logits can only be exported for the training split.")
    data, output_path = _data(config), Path(output_path)
    values = _validated_label_logits(source, label_logits, len(data.class_names))
    probabilities = values.softmax(1)
    corrected = values.argmax(1)
    original = torch.as_tensor(source.labels).detach().cpu().long()
    metadata = _metadata(config, source, "optimized_train_label_logits", checkpoint_info)
    _atomic_npz(output_path, overwrite=False, paths=np.asarray(source.paths), original_labels=original.numpy(),
                corrected_labels=corrected.numpy(), label_logits=values.numpy(),
                soft_labels=probabilities.numpy(), changed_mask=corrected.ne(original).numpy(),
                confidences=probabilities.max(1).values.numpy(),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, default=_json_default, allow_nan=False)))
    return {**_summary(original, corrected, data), "artifact": str(output_path.resolve()),
            "artifact_sha256": file_sha256(output_path), "metadata": metadata}


def write_holdout_reference(
    config, source, prediction_logits, output_path: Path | str, *,
    threshold: float = 0.7, checkpoint_info: dict | None = None,
) -> dict:
    """Export teacher-inferred Val/Test labels without replacing original annotations."""
    if source.name not in {"val", "test"}:
        raise ValueError("Inferred holdout references are only defined for val/test.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Holdout threshold must be in [0,1].")
    data, output_path = _data(config), Path(output_path)
    values = _validated_label_logits(source, prediction_logits, len(data.class_names))
    probabilities = values.softmax(1)
    confidence, predictions = probabilities.max(1)
    original = torch.as_tensor(source.labels).detach().cpu().long()
    corrected = torch.where(confidence > threshold, predictions, original)
    metadata = _metadata(config, source, "teacher_prediction_reference", checkpoint_info)
    metadata.update({"correction_threshold": threshold,
                     "interpretation": "teacher_agreement_only_not_verified_ground_truth"})
    _atomic_npz(output_path, overwrite=False, paths=np.asarray(source.paths), original_labels=original.numpy(),
                corrected_labels=corrected.numpy(), predictions=predictions.numpy(),
                prediction_logits=values.numpy(), confidences=confidence.numpy(),
                changed_mask=corrected.ne(original).numpy(),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, default=_json_default, allow_nan=False)))
    return {**_summary(original, corrected, data), "artifact": str(output_path.resolve()),
            "artifact_sha256": file_sha256(output_path), "metadata": metadata}


def load_train_correction(config, source, path: Path | str) -> tuple[Tensor, dict]:
    """Validate provenance and learned distributions before using fixed targets."""
    if source.name != "train":
        raise ValueError("Only training corrections may supply retraining targets.")
    data, path = _data(config), Path(path)
    before_hash = file_sha256(path)
    required = {"paths", "original_labels", "corrected_labels", "label_logits", "soft_labels",
                "changed_mask", "metadata"}
    with np.load(path, allow_pickle=False) as artifact:
        if not required <= set(artifact.files):
            raise ValueError("Train correction is missing learned distributions or provenance.")
        encoded = artifact["metadata"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("Correction metadata must be scalar Unicode JSON.")
        metadata = json.loads(encoded.item())
        if not isinstance(metadata, dict):
            raise ValueError("Correction metadata must be a JSON object.")
        expected = _metadata(config, source, "optimized_train_label_logits", None)
        for key in ("schema_version", "method", "split", "class_names", "label_offset",
                    "original_label_offset", "source_hashes", "input_config", "label_origin"):
            if metadata.get(key) != expected[key]:
                raise ValueError(f"Correction provenance mismatch: {key}.")
        if not np.array_equal(artifact["paths"], np.asarray(source.paths)):
            raise ValueError("Correction paths differ from the ordered, filtered training split.")
        original = torch.as_tensor(source.labels).detach().cpu().long()
        stored_original, corrected = artifact["original_labels"], artifact["corrected_labels"]
        if (stored_original.dtype.kind not in {"i", "u"} or corrected.dtype.kind not in {"i", "u"}
                or stored_original.shape != (len(original),) or corrected.shape != (len(original),)
                or not np.array_equal(stored_original, original.numpy())):
            raise ValueError("Correction original labels differ from the training source.")
        values = _validated_label_logits(source, artifact["label_logits"], len(data.class_names))
        labels = torch.from_numpy(corrected.astype(np.int64, copy=True))
        if not torch.equal(labels, values.argmax(1)):
            raise ValueError("Correction hard targets do not match the learned label logits.")
        soft = torch.as_tensor(artifact["soft_labels"])
        if (soft.shape != values.shape or not soft.is_floating_point() or not torch.isfinite(soft).all()
                or not torch.allclose(soft.float(), values.softmax(1), atol=1e-6, rtol=1e-5)):
            raise ValueError("Correction probabilities do not match the learned label logits.")
        changed = artifact["changed_mask"]
        if changed.dtype.kind != "b" or not np.array_equal(changed, labels.ne(original).numpy()):
            raise ValueError("Correction changed_mask does not match labels.")
    if file_sha256(path) != before_hash:
        raise ValueError("Correction artifact changed while being read.")
    return labels, {"source_path": str(path.resolve()), "artifact_sha256": before_hash,
                    "metadata": metadata, **_summary(original, labels, data)}


def load_holdout_correction(config, source, path: Path | str) -> tuple[Tensor, dict]:
    """Load fixed teacher labels without changing the original Val/Test arrays."""
    if source.name not in {"val", "test"}:
        raise ValueError("Holdout corrections require a val or test split.")
    data, path = _data(config), Path(path)
    labels, report = _corrected_targets(data, source, path)
    metadata = report["metadata"]
    expected = _metadata(config, source, "teacher_prediction_reference", None)
    for key in ("schema_version", "method", "split", "class_names", "label_offset",
                "original_label_offset", "source_hashes", "input_config", "label_origin",
                "ground_truth_verified"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Holdout correction provenance mismatch: {key}.")
    threshold = metadata.get("correction_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Holdout correction threshold must be in [0,1].")
    original = torch.as_tensor(source.labels).detach().cpu().long()
    with np.load(path, allow_pickle=False) as artifact:
        if not {"prediction_logits", "predictions", "confidences", "changed_mask"} <= set(artifact.files):
            raise ValueError("Holdout correction is missing predictions or confidence provenance.")
        logits = _validated_label_logits(source, artifact["prediction_logits"], data.num_classes)
        confidences, predictions = logits.softmax(1).max(1)
        stored_predictions, stored_confidences = artifact["predictions"], artifact["confidences"]
        if (stored_predictions.dtype.kind not in {"i", "u"}
                or stored_predictions.shape != (len(original),)
                or not np.array_equal(stored_predictions, predictions.numpy())):
            raise ValueError("Holdout correction predictions do not match teacher logits.")
        if (stored_confidences.dtype.kind != "f" or stored_confidences.shape != (len(original),)
                or not np.isfinite(stored_confidences).all()
                or not torch.allclose(torch.as_tensor(stored_confidences).float(), confidences,
                                      atol=1e-6, rtol=1e-5)):
            raise ValueError("Holdout correction confidences do not match teacher logits.")
        if not torch.equal(labels, torch.where(confidences > threshold, predictions, original)):
            raise ValueError("Holdout correction labels violate the teacher confidence threshold.")
        changed = artifact["changed_mask"]
        if changed.dtype.kind != "b" or not np.array_equal(changed, labels.ne(original).numpy()):
            raise ValueError("Holdout correction changed_mask does not match labels.")
    if file_sha256(path) != report["artifact_sha256"]:
        raise ValueError("Holdout correction artifact changed while being read.")
    return labels, {**report, "source_path": str(path.resolve()), **_summary(original, labels, data)}


def validate_correction_pair(train: dict, validation: dict) -> str:
    """Require corrected Train and Val targets exported from the same teacher."""
    train_checkpoint = train["metadata"].get("checkpoint", {})
    val_checkpoint = validation["metadata"].get("checkpoint", {})
    if not isinstance(train_checkpoint, dict) or not isinstance(val_checkpoint, dict):
        raise ValueError("Corrected Train and Validation must identify the same teacher checkpoint.")
    digest = train_checkpoint.get("sha256")
    completed = train_checkpoint.get("completed_epochs")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(completed) is not int or completed < 1
            or val_checkpoint.get("sha256") != digest
            or val_checkpoint.get("completed_epochs") != completed):
        raise ValueError("Corrected Train and Validation must come from the same teacher checkpoint.")
    return digest


def validate_teacher_weighting(state, original_labels, classes: int, strategy: str):
    """Never silently resume or export an unweighted teacher as a weighted run."""
    weights, counts = teacher_model_weights(original_labels, classes, strategy)
    if (state.get("kind") != "teacher"
            or state.get("config", {}).get("correction", {}).get("model_class_weighting") != strategy
            or state.get("model_class_weighting") != strategy
            or state.get("class_weight_label_reference") != "original_train"
            or state.get("class_weight_reduction") != "selected_weight_sum"
            or state.get("class_counts") != counts.tolist()):
        raise ValueError("Teacher class weighting provenance mismatch. Use a new runtime.name "
                         "for the weighted experiment; historical unweighted checkpoints remain evaluable.")
    try:
        saved_weights = torch.as_tensor(state.get("class_weights", []), dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Teacher class weighting has invalid saved weights.") from error
    if (saved_weights.shape != weights.shape or not bool(torch.isfinite(saved_weights).all())
            or not torch.equal(saved_weights, weights)):
        raise ValueError("Teacher class weighting does not match fixed original-Train weights.")
    return weights, counts
