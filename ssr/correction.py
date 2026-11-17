"""Export aligned Train, Validation and Test corrections from one Teacher."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json

from torch.utils.data import DataLoader
import numpy as np
import torch

from .config import ExperimentConfig
from .engine import configure_execution, seed_everything
from .features import (
    _evaluation_images,
    _extract_features_and_predictions,
    _record_indices,
    _validate_model_outputs,
)
from .model import build_ce_networks
from .neighbors import relabel_from_predictions, select_samples
from shared.augmentation import build_image_transforms
from shared.data import IndexedImageDataset, _file_hash, _resolve_array, prepare_dataset
from shared.precision import full_precision
from shared.roi import LETTERBOX_FILL, ROICrop

def checkpoint_identity(path: Path) -> tuple[dict, str]:
    """Keep one CPU checkpoint load; hash the exact file used for correction."""
    fingerprint = _file_hash(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    completed = state.get("completed_epochs")
    if not isinstance(completed, int) or completed < 0:
        raise ValueError(f"{path}: missing valid completed_epochs.")
    return state, fingerprint


def _validate_checkpoint(state: dict, config: ExperimentConfig) -> None:
    expected = {
        "model_name": config.model.name, "num_classes": config.data.num_classes,
        "class_names": list(config.data.class_names),
    }
    inputs = {
        key: value for key, value in asdict(config.data).items()
        if key in {"image_size", "label_offset", "crop_bbox", "crop_method", "missing_bbox",
                   "multi_roi", "roi_shift_ratio"}
    }
    inputs.update(mean=list(config.data.mean), std=list(config.data.std),
                  roi_scales=list(config.data.roi_scales), letterbox_fill=list(LETTERBOX_FILL),
                  roi_padding_fill=0)
    for actual, reference in ((state, expected), (state.get("input_config", {}), inputs)):
        for key, value in reference.items():
            if actual.get(key) != value:
                raise ValueError(f"Checkpoint/config mismatch for {key}.")
    if state.get("source_hashes") != _source_hashes(config, "train"):
        raise ValueError("Checkpoint source_hashes do not match the original train arrays.")
    if state.get("ssr_config") != asdict(config.ssr):
        raise ValueError("Checkpoint SSR correction settings do not match the configuration.")


def _source_hashes(config: ExperimentConfig, name: str) -> dict[str, str]:
    split = {"train": config.data.train, "val": config.data.validation, "test": config.data.test}[name]
    return {key: _file_hash(_resolve_array(config.data.root, getattr(split, key)))
            for key in ("paths", "labels")}


@torch.inference_mode()
def _probabilities(loader, networks, device, *, num_classes: int, channels_last: bool):
    """Holdouts need probabilities only, not a dataset-sized feature matrix."""
    networks.eval()
    values = torch.empty((len(loader.dataset), num_classes), dtype=torch.float32)
    seen = torch.zeros(len(loader.dataset), dtype=torch.bool)
    with full_precision(device):
        for images, indices in loader:
            indices = _record_indices(indices, images.size(0), seen)
            images = _evaluation_images(images, device, channels_last)
            features = networks.encoder(images)
            logits = networks.classifier(features)
            _validate_model_outputs(features, logits)
            if logits.shape != (len(indices), num_classes):
                raise ValueError("Classifier output does not match configured classes.")
            values[indices] = logits.float().softmax(1).cpu()
    if not seen.all():
        raise RuntimeError("Correction loader did not cover every dataset row.")
    return values


def _split_summary(original: np.ndarray, corrected: np.ndarray, classes: tuple[str, ...]):
    count = len(classes)
    transitions = np.bincount(original * count + corrected, minlength=count * count)
    return {
        "sample_count": len(original), "changed_count": int(np.count_nonzero(original != corrected)),
        "original_class_counts": dict(zip(classes, np.bincount(original, minlength=count).tolist())),
        "corrected_class_counts": dict(zip(classes, np.bincount(corrected, minlength=count).tolist())),
        "label_transitions": transitions.reshape(count, count).tolist(),
        "ground_truth_available": False,
    }


def _validate_snapshot(snapshot: dict, source, config: ExperimentConfig) -> dict[str, torch.Tensor]:
    """Validate saved pre-update corrections from historical teacher checkpoints."""
    if tuple(snapshot.get("paths", ())) != source.paths:
        raise ValueError("Checkpoint correction paths do not match the prepared train dataset.")
    size, classes = len(source.paths), config.data.num_classes
    values = {}
    for key in ("original_labels", "corrected_labels", "predictions", "confidences",
                "selected_mask", "consistency"):
        value = snapshot.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (size,):
            raise ValueError(f"Invalid checkpoint correction snapshot: {key}.")
        values[key] = value.detach().cpu()
    for key in ("original_labels", "corrected_labels", "predictions"):
        value = values[key]
        if value.dtype != torch.int64 or bool(((value < 0) | (value >= classes)).any()):
            raise ValueError(f"Invalid checkpoint class indices: {key}.")
    if not torch.equal(values["original_labels"], source.labels):
        raise ValueError("Checkpoint original labels do not match the prepared train dataset.")
    confidence, consistency = values["confidences"], values["consistency"]
    if (values["selected_mask"].dtype != torch.bool
        or not confidence.is_floating_point() or not consistency.is_floating_point()
        or not bool(torch.isfinite(confidence).all() & torch.isfinite(consistency).all())
        or bool(((confidence < 0) | (confidence > 1) | (consistency < 0) | (consistency > 1)).any())):
        raise ValueError("Invalid checkpoint confidence/consistency/selection snapshot.")
    expected = torch.where(confidence > config.ssr.relabel_threshold,
                           values["predictions"], values["original_labels"])
    if not torch.equal(expected, values["corrected_labels"]):
        raise ValueError("Checkpoint correction snapshot violates the relabel threshold.")
    if not torch.equal(values["selected_mask"], consistency >= config.ssr.selection_threshold):
        raise ValueError("Checkpoint correction snapshot violates the selection threshold.")
    return values


def export_corrections(
    config: ExperimentConfig, checkpoint_path: Path, output_dir: Path, *, criterion: str = "last",
) -> Path:
    """Train: SSR relabel+selection; holdouts: confidence rule without train k-NN."""
    config.validate()
    if config.data.validation is None or config.data.test is None:
        raise ValueError("Label correction requires Train, Validation and Test splits.")
    state, fingerprint = checkpoint_identity(checkpoint_path)
    _validate_checkpoint(state, config)
    if state.get("training_method", "ssr") != "ssr":
        raise ValueError("Correction requires a teacher SSR checkpoint.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources, boxes = prepare_dataset(config.data, report_path=output_dir / "data_audit.jsonl")
    transform = build_image_transforms(config.data, config.augmentation).evaluation
    device = config.resolve_device()
    seed_everything(config.seed)
    networks = build_ce_networks(replace(config.model, pretrained=False), device, config.data.num_classes)
    networks.encoder.load_state_dict(state.pop("encoder"), strict=True)
    networks.classifier.load_state_dict(state.pop("classifier"), strict=True)
    inputs, completed, selection = state["input_config"], state["completed_epochs"], state.get("selection", {})
    snapshot = state.get("train_correction")
    del state  # Release optimizer/projector tensors before feature extraction.
    configure_execution(config, networks, device)
    channels_last = config.training.channels_last and device.type == "cuda"
    if channels_last:
        networks.encoder.to(memory_format=torch.channels_last)
    report = {
        "criterion": criterion, "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": fingerprint, "teacher_completed_epochs": completed,
        "teacher_selection": selection, "correction_threshold": config.ssr.relabel_threshold,
        "corrected_splits": ["train", "val", "test"],
        "student_selection": "corrected_validation_macro_f1",
        "student_evaluation": "corrected_test",
        "corrected_ground_truth_verified": False, "splits": {},
    }
    workers = config.training.eval_num_workers
    options = dict(batch_size=config.training.eval_batch_size, shuffle=False,
                   num_workers=workers, pin_memory=device.type == "cuda")
    if workers:
        options.update(prefetch_factor=config.training.prefetch_factor,
                       persistent_workers=False, multiprocessing_context="spawn")
    for name, source in sources.items():
        dataset = IndexedImageDataset(
            source, transform, allow_truncated=config.data.allow_truncated_images,
            bboxes=boxes.get(name), roi=ROICrop(size=config.data.image_size,
                method=config.data.crop_method, random_view=False,
                scales=config.data.roi_scales, shift_ratio=config.data.roi_shift_ratio),
        )
        loader = DataLoader(dataset, **options)
        extra = {}
        if name == "train" and snapshot is not None:
            paired = _validate_snapshot(snapshot, source, config)
            corrected, confidence, predictions = (paired[key] for key in (
                "corrected_labels", "confidences", "predictions"))
            extra = {key: paired[key].numpy() for key in ("selected_mask", "consistency")}
        elif name == "train":
            features, probabilities = _extract_features_and_predictions(
                loader, networks, device, channels_last=channels_last,
            )
            result = select_samples(
                features, source.labels.to(device), probabilities,
                relabel_threshold=config.ssr.relabel_threshold,
                selection_threshold=config.ssr.selection_threshold,
                neighbors=config.ssr.neighbors, chunks=config.ssr.knn_chunks,
                num_classes=config.data.num_classes,
            )
            corrected, confidence = result.modified_labels.cpu(), result.confidences.cpu()
            selected = torch.zeros(len(source.paths), dtype=torch.bool)
            selected[result.selected_indices.cpu()] = True
            extra = {"selected_mask": selected.numpy(), "consistency": result.consistency.cpu().numpy()}
            predictions = probabilities.argmax(1).cpu()
            del features, probabilities, result
        else:
            probabilities = _probabilities(loader, networks, device,
                num_classes=config.data.num_classes, channels_last=channels_last)
            corrected, _, confidence = relabel_from_predictions(
                source.labels, probabilities, config.ssr.relabel_threshold,
            )
            predictions = probabilities.argmax(1)
        original, corrected = source.labels.numpy(), corrected.numpy()
        metadata = {
            "split": name, "class_names": list(config.data.class_names), "label_offset": 0,
            "original_label_offset": config.data.label_offset,
            "checkpoint_sha256": fingerprint, "teacher_completed_epochs": completed,
            "teacher_selection": selection, "correction_threshold": config.ssr.relabel_threshold,
            "input_config": inputs,
            "source_hashes": _source_hashes(config, name),
            "train_label_origin": "paired_snapshot" if name == "train" and snapshot is not None else "checkpoint_inference",
        }
        target = output_dir / f"{name}_correction.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, paths=np.asarray(source.paths), original_labels=original,
                corrected_labels=corrected, predictions=predictions.numpy(),
                confidences=confidence.numpy(), changed_mask=original != corrected,
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)), **extra)
        summary = _split_summary(original, corrected, config.data.class_names)
        summary["artifact"] = str(target.resolve())
        if extra:
            summary["selected_count"] = int(extra["selected_mask"].sum())
        report["splits"][name] = summary
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path
