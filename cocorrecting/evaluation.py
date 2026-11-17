"""FP32 inference, warm-up features and selected-checkpoint Test evaluation.

ROI views and dual networks are averaged as raw logits. Corrected-Val Students
are evaluated against the matching Teacher-corrected Test labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import numpy as np
from contextlib import contextmanager

import torch
from tqdm import tqdm
from torch import Tensor, nn

from shared.metrics import metrics_from_logits
from shared.precision import full_precision
from .artifacts import (_atomic_npz, atomic_write_json, dataset_fingerprint, file_sha256,
                        load_holdout_correction, validate_teacher_weighting)
from .config import ModelConfig
from .data import build_data, build_fixed_views_dataset
from .model import build_model
from .runtime import (configure_runtime, load_checkpoint, managed_loader, resolve_device,
                      validate_teacher_seed_metadata)


@contextmanager
def _evaluation_mode(models: Sequence[nn.Module]):
    flags = [(module, module.training) for model in models for module in model.modules()]
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for module, was_training in flags:
            module.training = was_training


def _models(models: Sequence[nn.Module] | nn.Module) -> tuple[nn.Module, ...]:
    result = (models,) if isinstance(models, nn.Module) else tuple(models)
    if not result or any(not isinstance(model, nn.Module) for model in result):
        raise ValueError("Evaluation requires one or more torch models.")
    return result


def _indices(indices, count: int, total: int, seen: Tensor) -> Tensor:
    indices = torch.as_tensor(indices).detach().cpu()
    if indices.ndim != 1 or len(indices) != count or indices.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    }:
        raise ValueError("Evaluation requires integer sample indices matching each image batch.")
    indices = indices.long()
    if ((indices < 0) | (indices >= total)).any():
        raise ValueError("Evaluation index is outside the dataset.")
    if len(indices.unique()) != count or seen[indices].any():
        raise ValueError("Evaluation loader repeated a sample index.")
    seen[indices] = True
    return indices


def _to_device(images: Tensor, device: torch.device, channels_last: bool) -> Tensor:
    if not images.is_floating_point():
        raise ValueError("Evaluation images must be floating point tensors.")
    return images.to(device=device, dtype=torch.float32, non_blocking=True,
                     memory_format=torch.channels_last if channels_last else torch.contiguous_format)


@torch.no_grad()
def predict_logits(
    models: Sequence[nn.Module] | nn.Module, loader, device: torch.device | str, *,
    channels_last: bool = False, view_chunk_size: int = 128, progress_label: str | None = None,
) -> Tensor:
    """Return CPU [N,C] logits in original dataset order; reject partial loaders.

    A batch is ``(images, indices)`` with images [B,C,H,W] or [B,V,C,H,W].
    Shuffled complete loaders are allowed because indices are checked and used.
    """
    models, device = _models(models), torch.device(device)
    if type(view_chunk_size) is not int or view_chunk_size < 1:
        raise ValueError("view_chunk_size must be a positive integer.")
    total = len(loader.dataset)
    if total < 1:
        raise ValueError("Evaluation dataset is empty.")
    seen, results = torch.zeros(total, dtype=torch.bool), None
    with _evaluation_mode(models), full_precision(device):
        batches = tqdm(loader, desc=progress_label, leave=False, disable=progress_label is None, mininterval=5)
        for images, raw_indices in batches:
            if not isinstance(images, Tensor) or images.ndim not in (4, 5) or not len(images):
                raise ValueError("Expected nonempty images [B,C,H,W] or [B,V,C,H,W].")
            batch = len(images)
            views = images.shape[1] if images.ndim == 5 else 1
            if views < 1:
                raise ValueError("A sample must have at least one view.")
            indices = _indices(raw_indices, batch, total, seen)
            flat = images.reshape(batch * views, *images.shape[-3:])
            view_logits = None
            for start in range(0, len(flat), view_chunk_size):
                inputs = _to_device(flat[start:start + view_chunk_size], device, channels_last)
                summed = None
                for model in models:
                    logits = model(inputs)
                    if (not isinstance(logits, Tensor) or logits.ndim != 2
                            or logits.shape[0] != len(inputs) or logits.shape[1] < 2
                            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
                        raise ValueError("Models must return finite floating point [batch, classes] logits.")
                    logits = logits.float()
                    if summed is not None and summed.shape != logits.shape:
                        raise ValueError("Teacher networks have different output class counts.")
                    summed = logits if summed is None else summed + logits
                averaged = (summed / len(models)).cpu()
                if view_logits is None:
                    view_logits = torch.empty((len(flat), averaged.shape[1]), dtype=torch.float32)
                elif view_logits.shape[1] != averaged.shape[1]:
                    raise ValueError("Model output class count changed across chunks.")
                view_logits[start:start + len(inputs)] = averaged
            sample_logits = view_logits.reshape(batch, views, -1).mean(1)
            if results is None:
                results = torch.empty((total, sample_logits.shape[1]), dtype=torch.float32)
            elif results.shape[1] != sample_logits.shape[1]:
                raise ValueError("Model output class count changed between batches.")
            results[indices] = sample_logits
    if results is None or not seen.all():
        raise ValueError("Evaluation loader did not cover every sample exactly once.")
    return results


@torch.no_grad()
def extract_features(
    models: Sequence[nn.Module] | nn.Module, loader, device: torch.device | str, *,
    channels_last: bool = False,
) -> Tensor:
    """Concatenate fixed-center features from the two warm-up networks on CPU."""
    models, device = _models(models), torch.device(device)
    total = len(loader.dataset)
    if total < 1:
        raise ValueError("Feature dataset is empty.")
    seen, results = torch.zeros(total, dtype=torch.bool), None
    with _evaluation_mode(models), full_precision(device):
        for images, raw_indices in loader:
            if not isinstance(images, Tensor) or images.ndim != 4 or not len(images):
                raise ValueError("Curriculum features require fixed-center [B,C,H,W] images.")
            indices = _indices(raw_indices, len(images), total, seen)
            inputs = _to_device(images, device, channels_last)
            values = []
            for model in models:
                features = model.features(inputs)
                if (not isinstance(features, Tensor) or features.ndim != 2
                        or len(features) != len(images) or features.shape[1] < 1
                        or not torch.isfinite(features).all()):
                    raise ValueError("Models must expose finite pooled features [B,D].")
                values.append(features.float().cpu())
            features = torch.cat(values, dim=1)
            if results is None:
                results = torch.empty((total, features.shape[1]), dtype=torch.float32)
            elif results.shape[1] != features.shape[1]:
                raise ValueError("Feature dimensions changed between batches.")
            results[indices] = features
    if results is None or not seen.all():
        raise ValueError("Feature loader did not cover every sample exactly once.")
    return results


def evaluate_logits(
    logits: Tensor, labels: Tensor, *, class_names: tuple[str, ...] | None = None,
) -> dict:
    """Score one explicit label reference; callers keep reference results separate."""
    return metrics_from_logits(torch.as_tensor(logits).detach().cpu(),
                               torch.as_tensor(labels).cpu(), class_names=class_names)


def _model_config(saved: dict, current) -> ModelConfig:
    if not isinstance(saved, dict) or saved.get("name") != current.name:
        raise ValueError("Checkpoint model architecture does not match the evaluation config.")
    known = {field.name for field in fields(ModelConfig)}
    payload = {key: value for key, value in saved.items() if key in known}
    # No pretrained download or original weight file is needed for evaluation.
    payload.update(pretrained=False, pretrained_weights=None)
    return ModelConfig(**payload)


def _load_model(state: dict, config, prepared, device):
    kind = state.get("kind")
    if state.get("schema_version") != 1 or kind not in {"student", "teacher"}:
        raise ValueError("Unsupported checkpoint kind/schema.")
    if state.get("dataset_fingerprint") != dataset_fingerprint(prepared.sources, config):
        raise ValueError("Checkpoint dataset fingerprint does not match this evaluation data.")
    saved = state.get("config", {})
    if saved.get("data", {}).get("class_names") != list(config.data.class_names):
        raise ValueError("Checkpoint class names do not match the current class order.")
    model_config = _model_config(saved.get("model", {}), config.model)
    states = state.get("models", [])
    expected_networks = 2 if kind == "teacher" else 1
    if not isinstance(states, (list, tuple)) or len(states) != expected_networks:
        raise ValueError("Checkpoint has an invalid number of networks.")
    if kind == "teacher":
        originals, logits = state.get("original_labels"), state.get("label_logits")
        reference = prepared.sources["train"].labels
        if (not torch.is_tensor(originals) or originals.dtype != reference.dtype
                or not torch.equal(originals, reference)
                or not torch.is_tensor(logits) or not logits.is_floating_point()
                or logits.shape != (len(reference), config.data.num_classes)
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("Teacher Train label bank does not match the prepared source rows.")
        validate_teacher_weighting(state, reference, config.data.num_classes,
                                   config.correction.model_class_weighting)
        validate_teacher_seed_metadata(state, config.correction.model_seeds)
    models = []
    for weights in states:
        model = build_model(model_config, config.data.num_classes, device, initialize_pretrained=False)
        model.load_state_dict(weights, strict=True)
        models.append(model)
    return (tuple(models) if kind == "teacher" else models[0]), kind


def _primary_test_reference(state, reference_teacher_hashes):
    """Choose the matching corrected Test reference for a corrected-Val Student."""
    selection_reference = state.get("selection_label_reference",
                                    state.get("selection", {}).get("label_reference", "original"))
    if state.get("kind") != "student" or selection_reference != "teacher_corrected":
        return "original", "original_test"
    teacher_hash = state.get("correction_checkpoint_sha256")
    matching = [name for name, digest in reference_teacher_hashes.items()
                if teacher_hash is not None and digest == teacher_hash]
    if not teacher_hash or len(matching) != 1:
        raise ValueError("Corrected-Val Student requires exactly one corrected Test reference "
                         "from the same Teacher checkpoint as its Train/Val labels.")
    return "teacher_corrected", matching[0]


def evaluate_checkpoint(
    config, checkpoint_path: Path | str, output_dir: Path | str, *,
    correction_paths: dict[str, Path] | None = None,
) -> dict:
    """Evaluate an explicitly supplied checkpoint; persist logits for paired analyses.

    ``correction_paths`` maps reference names to Test correction NPZ files.
    Teacher, baseline and student may share a reference without repeating inference.
    ``logit_avg`` is None when 27-view evaluation is disabled in the config.
    """
    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if config.data.test is None:
        raise ValueError("Test split is required for final checkpoint evaluation.")
    if (output_dir / "evaluation.json").exists() or (output_dir / "predictions.npz").exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = build_data(config, report_path=output_dir / "data_audit.jsonl")
    if prepared.test is None or "test" not in prepared.sources:
        raise ValueError("No Test samples were prepared.")
    source, references = prepared.sources["test"], {}
    for name, path in (correction_paths or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("Correction reference names must be nonempty strings.")
        labels, provenance = load_holdout_correction(config, source, Path(path))
        references[name] = (labels, provenance)
    state = load_checkpoint(checkpoint_path)
    primary_label_reference, primary_reference = _primary_test_reference(
        state, {name: provenance.get("metadata", {}).get("checkpoint", {}).get("sha256")
                for name, (_, provenance) in references.items()})
    device = resolve_device(config.runtime.device)
    configure_runtime(config, device)
    model, kind = _load_model(state, config, prepared, device)
    report = {"checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": file_sha256(checkpoint_path),
              "kind": kind, "completed_epochs": state.get("completed_epochs"),
              "selection": state.get("selection", {}),
              "selection_label_reference": state.get("selection_label_reference",
                  state.get("selection", {}).get("label_reference", "original")),
              "corrected_validation_labels_sha256": state.get("corrected_validation_labels_sha256"),
              "primary_label_reference": primary_label_reference, "primary_reference": primary_reference,
              "dataset_fingerprint": dataset_fingerprint(prepared.sources, config),
              "evaluation_label_reference": "original", "corrected_ground_truth_verified": False,
              "metric_units": "rates in [0,1]; loss in nats",
              "aggregation": "27_view_logit_avg" if config.pipeline.evaluate_27_views else "single_center"}
    if kind == "teacher":
        report.update(network_count=2, network_aggregation="raw_logit_mean",
                      teacher_model_seeds=state.get("teacher_model_seeds"),
                      model_seed_policy=state.get("model_seed_policy", "legacy_shared_rng"),
                      teacher_phase=state.get("metrics", {}).get("phase"),
                      correction_end_epoch=config.correction.correction_end_epoch,
                      teacher_label_updates_completed=(type(state.get("completed_epochs")) is int
                          and state["completed_epochs"] >= config.correction.correction_end_epoch))
    del state
    channels_last = config.training.channels_last and device.type == "cuda"
    if channels_last:
        for network in (model if isinstance(model, tuple) else (model,)):
            network.to(memory_format=torch.channels_last)
    with managed_loader(prepared.test, config, device) as center_loader:
        center = predict_logits(model, center_loader, device, channels_last=channels_last,
                                view_chunk_size=config.pipeline.evaluation_view_chunk_size,
                                progress_label="Co eval center")
    average = None
    if config.pipeline.evaluate_27_views:
        views = build_fixed_views_dataset(source, prepared.boxes.get("test"), config.data, config.augmentation)
        with managed_loader(views, config, device, batch_size=config.pipeline.evaluation_image_batch_size) as views_loader:
            average = predict_logits(model, views_loader, device, channels_last=channels_last,
                                     view_chunk_size=config.pipeline.evaluation_view_chunk_size,
                                     progress_label="Co eval 27-view")
    report["single_center"] = evaluate_logits(center, source.labels, class_names=config.data.class_names)
    report["logit_avg"] = (evaluate_logits(average, source.labels, class_names=config.data.class_names)
                           if average is not None else None)
    report["corrected_references"] = {}
    for name, (labels, provenance) in references.items():
        report["corrected_references"][name] = {
            "single_center": evaluate_logits(center, labels, class_names=config.data.class_names),
            "logit_avg": (evaluate_logits(average, labels, class_names=config.data.class_names)
                          if average is not None else None),
            "ground_truth_verified": False, "interpretation": "agreement_with_teacher_reference",
            "labels_path": provenance["source_path"], "artifact_sha256": provenance["artifact_sha256"],
            "teacher_checkpoint_sha256": provenance.get("metadata", {}).get("checkpoint", {}).get("sha256"),
        }
    primary = (report["corrected_references"][primary_reference]
               if primary_label_reference == "teacher_corrected" else report)
    key = "logit_avg" if config.pipeline.evaluate_27_views else "single_center"
    report["primary_metrics"] = primary[key]
    arrays = {"paths": np.asarray(source.paths), "indices": np.arange(len(source.paths)),
              "original_labels": source.labels.cpu().numpy(), "single_center_logits": center.numpy()}
    if average is not None:
        arrays["logit_avg_logits"] = average.numpy()
    _atomic_npz(output_dir / "predictions.npz", overwrite=False, **arrays)
    report["predictions_path"] = str((output_dir / "predictions.npz").resolve())
    report["predictions_sha256"] = file_sha256(output_dir / "predictions.npz")
    atomic_write_json(output_dir / "evaluation.json", report)
    print(f"[Co eval] primary={primary_label_reference} ({primary_reference}) "
          f"Macro F1={report['primary_metrics']['macro_f1']:.4f}", flush=True)
    return report
