"""Fresh Multi-ROI WCE training with matching original or corrected Train/Val labels.

The teacher supplies immutable target artifacts, never student model parameters.
"""
from __future__ import annotations

import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .artifacts import (atomic_write_json, dataset_fingerprint, file_sha256,
                        load_holdout_correction, load_train_correction,
                        validate_correction_pair)
from .evaluation import evaluate_logits, predict_logits
from .model import build_model, build_optimizer, class_weights
from .runtime import (atomic_torch_save, comparable_config, configure_runtime,
                      managed_loader, reconcile_metrics, resolve_device, start_run,
                      seed_everything, set_loader_epoch, validate_checkpoint_history)
from .config import to_dict
from .data import build_data
from shared.precision import PrecisionPolicy
from shared.artifacts import JsonlWriter


def train_student(config, run_dir, *, corrected_labels_path=None,
                  corrected_validation_labels_path=None, resume=None,
                  stop_after_epochs=None) -> Path:
    """Train 300 epochs by default, or resume only this run's atomic last.pt.

    ``stop_after_epochs`` is an absolute completed-epoch boundary for deliberate
    interruption/testing; it never changes the saved total training schedule.
    """
    config.validate(student=True)
    if (corrected_labels_path is None) != (corrected_validation_labels_path is None):
        raise ValueError("Corrected student training requires both Train and Validation correction files; "
                         "omit both for the original-label baseline.")
    selection_reference = "teacher_corrected" if corrected_labels_path is not None else "original"
    validation_hash = (file_sha256(corrected_validation_labels_path)
                       if corrected_validation_labels_path is not None else None)
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be a positive integer.")
    run_dir, state = start_run(run_dir, resume)
    if state is not None:
        saved_reference = state.get("selection_label_reference",
                                    state.get("metrics", {}).get("validation_label_reference", "original"))
        if (saved_reference != selection_reference
                or state.get("corrected_validation_labels_sha256") != validation_hash):
            raise ValueError("Student Validation labels/selection policy changed. Use a new output directory; "
                             "an original-Validation run cannot resume as corrected-Validation training.")
        validate_checkpoint_history(run_dir, state)
    device = resolve_device(config.runtime.device)
    configure_runtime(config, device)
    prepared = build_data(config, report_path=run_dir / "data_audit.jsonl")
    fingerprint = dataset_fingerprint(prepared.sources, config)
    originals = prepared.sources["train"].labels.clone()
    labels, correction = (load_train_correction(config, prepared.sources["train"], corrected_labels_path)
                          if corrected_labels_path is not None else (originals.clone(), None))
    correction_hash = correction["artifact_sha256"] if correction is not None else None
    validation_labels = prepared.sources["val"].labels.clone()
    validation_correction = None
    correction_checkpoint_hash = None
    if corrected_validation_labels_path is not None:
        validation_labels, validation_correction = load_holdout_correction(
            config, prepared.sources["val"], corrected_validation_labels_path)
        if validation_correction["artifact_sha256"] != validation_hash:
            raise ValueError("The fixed Validation correction changed before student training.")
        correction_checkpoint_hash = validate_correction_pair(correction, validation_correction)
    weights, counts = class_weights(labels, config.data.num_classes)
    saved_config = to_dict(config)
    start_epoch = 0
    best = {"macro_f1": None}
    best_epochs = {}
    if state is not None:
        if state.get("schema_version") != 1 or state.get("kind") != "student":
            raise ValueError("Resume requires a student checkpoint of schema version 1.")
        if comparable_config(state.get("config", {})) != comparable_config(saved_config):
            raise ValueError("Student resume configuration differs from the saved run.")
        if (state.get("dataset_fingerprint") != fingerprint
                or state.get("corrected_labels_sha256") != correction_hash
                or state.get("correction_checkpoint_sha256") != correction_checkpoint_hash
                or state.get("class_weights") != weights.tolist()
                or state.get("class_counts") != counts.tolist()):
            raise ValueError("Student resume dataset, fixed correction or WCE weights differ.")
        if validation_correction is not None:
            targets_path = run_dir / "validation_targets.json"
            if not targets_path.is_file():
                raise ValueError("Resume requires this run's fixed validation_targets.json.")
            saved_targets = json.loads(targets_path.read_text(encoding="utf-8"))
            if (saved_targets.get("artifact_sha256") != validation_hash
                    or saved_targets.get("metadata") != validation_correction["metadata"]):
                raise ValueError("Resume Validation target provenance differs from this run.")
        start_epoch = state.get("completed_epochs")
        if type(start_epoch) is not int or not 0 <= start_epoch <= config.training.epochs:
            raise ValueError("Student resume epoch is invalid.")
        best = {"macro_f1": state["best_scores"].get("macro_f1")}
        best_epochs = {key: value for key, value in state.get("best_epochs", {}).items()
                       if key == "macro_f1"}
        reconcile_metrics(run_dir / "metrics.jsonl", start_epoch)
    else:
        atomic_write_json(run_dir / "config.json", saved_config)
        atomic_write_json(run_dir / "class_weights.json", {
            "rule": "inverse_sqrt_train_count_normalized_mean_one",
            "class_names": list(config.data.class_names), "class_counts": counts.tolist(),
            "class_weights": weights.tolist(), "label_source": "corrected_train" if correction else "original_train",
            "corrected_labels_sha256": correction_hash,
        })
        if correction is not None:
            atomic_write_json(run_dir / "training_targets.json", correction)
            atomic_write_json(run_dir / "validation_targets.json", validation_correction)

    model = build_model(config.model, config.data.num_classes, device,
                        initialize_pretrained=state is None)
    precision = PrecisionPolicy.from_config(config.training, device)
    if precision.channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, config.training)
    if state is not None:
        if len(state.get("models", [])) != 1 or len(state.get("optimizers", [])) != 1:
            raise ValueError("Student resume must contain exactly one model and optimizer.")
        model.load_state_dict(state["models"][0])
        optimizer.load_state_dict(state["optimizers"][0])
    labels_device, weights_device = labels.to(device), weights.to(device)
    completed = start_epoch
    last_metrics = state.get("metrics") if state is not None else None

    def checkpoint(selection, metrics):
        return {
            "schema_version": 1, "kind": "student", "config": saved_config,
            "dataset_fingerprint": fingerprint, "models": [model.state_dict()],
            "optimizers": [optimizer.state_dict()], "completed_epochs": completed,
            "selection": selection, "class_weights": weights.tolist(),
            "class_counts": counts.tolist(), "corrected_labels_sha256": correction_hash,
            "corrected_validation_labels_sha256": validation_hash,
            "correction_checkpoint_sha256": correction_checkpoint_hash,
            "selection_label_reference": selection_reference,
            "best_scores": dict(best), "best_epochs": dict(best_epochs),
            "metrics": metrics,
            "training_method": "wce", "supervised_loss_kind": "wce",
        }

    with ExitStack() as stack:
        train_loader = stack.enter_context(managed_loader(prepared.train, config, device, training=True))
        validation_loader = stack.enter_context(managed_loader(prepared.validation, config, device))
        metrics_log = stack.enter_context(JsonlWriter(run_dir / "metrics.jsonl", append=True))
        target = min(config.training.epochs, stop_after_epochs or config.training.epochs)
        epochs = tqdm(range(start_epoch, target), desc="Co Student", unit="epoch", mininterval=5)
        for epoch in epochs:
            set_loader_epoch(train_loader, config, epoch)
            seed_everything(config.seed + epoch * 100003 + 307)
            factor = config.training.student_min_lr_ratio + (
                1 - config.training.student_min_lr_ratio
            ) * (1 + math.cos(math.pi * epoch / config.training.epochs)) / 2
            for group, base_lr in zip(optimizer.param_groups,
                                      (config.training.encoder_learning_rate, config.training.learning_rate)):
                group["lr"] = base_lr * factor
            model.train()
            losses, observed, correct = 0.0, 0, 0
            batches = tqdm(train_loader, desc=f"Student {epoch + 1}/{config.training.epochs}", leave=False, mininterval=5)
            for images, indices in batches:
                optimizer.zero_grad(set_to_none=True)
                inputs = precision.to_device(images)
                targets = labels_device[indices.to(device)]
                with precision.autocast():
                    logits = model(inputs)
                loss = F.cross_entropy(logits.float(), targets, weight=weights_device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Student WCE became non-finite.")
                loss.backward()
                optimizer.step()
                losses += float(loss.detach()) * len(indices)
                observed += len(indices)
                correct += int(logits.detach().argmax(1).eq(targets).sum())
                batches.set_postfix(loss=f"{losses / observed:.4f}", accuracy=f"{correct / observed:.3f}")
            if observed != len(labels):
                raise RuntimeError("Student loader must cover every fixed training target once per epoch.")
            logits = predict_logits(model, validation_loader, device, channels_last=precision.channels_last)
            validation = evaluate_logits(logits, validation_labels,
                                         class_names=config.data.class_names)
            completed = epoch + 1
            metrics = {
                "epoch": epoch, "completed_epochs": completed, "method": "wce",
                "supervised_loss": losses / observed, "supervised_loss_kind": "wce",
                "train_accuracy": correct / observed,
                "validation": validation, "validation_macro_f1": validation["macro_f1"],
                "validation_label_reference": selection_reference,
                "corrected_validation_labels_sha256": validation_hash,
            }
            score = float(validation["macro_f1"])
            if not math.isfinite(score):
                raise FloatingPointError("Student validation macro_f1 is non-finite.")
            if best["macro_f1"] is None or score > best["macro_f1"]:
                best["macro_f1"], best_epochs["macro_f1"] = score, completed
                selection = {"method": "best", "split": "validation", "criterion": "macro_f1",
                             "label_reference": selection_reference, "score": score}
                atomic_torch_save(checkpoint(selection, metrics), run_dir / "best_macro_f1.pt")
            atomic_torch_save(checkpoint({"method": "last", "criterion": "last_epoch"}, metrics),
                              run_dir / "last.pt")
            metrics_log.write(metrics)
            last_metrics = metrics
            epochs.set_postfix(loss=f"{losses / observed:.4f}", accuracy=f"{validation['accuracy']:.3f}",
                               f1=f"{validation['macro_f1']:.3f}")
    # Detect a target artifact changed by another process while training.
    if corrected_labels_path is not None and file_sha256(corrected_labels_path) != correction_hash:
        raise ValueError("The fixed training correction changed during student training.")
    if (corrected_validation_labels_path is not None
            and file_sha256(corrected_validation_labels_path) != validation_hash):
        raise ValueError("The fixed Validation correction changed during student training.")
    atomic_write_json(run_dir / "run_summary.json", {
        "status": "completed" if completed == config.training.epochs else "interrupted",
        "kind": "student", "completed_epochs": completed, "training_method": "wce",
        "primary_checkpoint": "best_macro_f1.pt", "selection_label_reference": selection_reference,
        "best_scores": best, "best_epochs": best_epochs,
        "corrected_labels_sha256": correction_hash,
        "corrected_validation_labels_sha256": validation_hash,
        "correction_checkpoint_sha256": correction_checkpoint_hash,
        "checkpoints": [name for name in ("best_macro_f1.pt", "last.pt") if (run_dir / name).is_file()],
        "last_metrics": last_metrics,
    }, overwrite=True)
    return run_dir
