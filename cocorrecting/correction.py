"""Export fixed Train, Validation and Test labels from the correction-end Teacher."""
from __future__ import annotations

from pathlib import Path

import torch

from shared.artifacts import atomic_write_json, file_sha256
from .artifacts import (dataset_fingerprint,
                        holdout_settings,
                        load_holdout_correction, load_train_correction,
                        write_holdout_correction, write_train_correction, validate_teacher_weighting)
from .algorithm import optimize_holdout_labels
from .evaluation import predict_logits
from .model import build_model
from .runtime import (configure_runtime, load_checkpoint, managed_loader, resolve_device,
                      validate_teacher_seed_metadata)
from .data import build_data


def export_corrections(config, checkpoint_path, output_dir, *, resume=False):
    """Export Train's learned labels and optimize Val/Test labels with fixed Teachers.

    Reused artifacts must match the Teacher hash, epoch and prepared source rows.
    Moving an intact Teacher directory does not invalidate its checkpoint hash.
    """
    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    state = load_checkpoint(checkpoint_path)
    if (
            state.get("completed_epochs") != config.correction.correction_end_epoch
            or state.get("selection", {}).get("criterion") != "end_of_label_updates"):
        raise ValueError("Expected the saved end-of-label-updates checkpoint.")
    output_dir.mkdir(parents=True, exist_ok=resume)
    data = build_data(config, report_path=output_dir / "data_audit.jsonl")
    fingerprint = dataset_fingerprint(data.sources, config)
    if (state.get("kind") != "teacher" or state.get("dataset_fingerprint") != fingerprint
            or not torch.equal(state.get("original_labels", torch.empty(0)), data.sources["train"].labels)):
        raise ValueError("Teacher checkpoint does not match these prepared source rows.")
    weights, counts = validate_teacher_weighting(
        state, data.sources["train"].labels, config.data.num_classes,
        config.correction.model_class_weighting)
    seed_metadata = validate_teacher_seed_metadata(state, config.correction.model_seeds)
    if state["completed_epochs"] <= config.correction.warmup_epochs:
        raise ValueError("The checkpoint predates label correction.")
    info = {"path": str(checkpoint_path.resolve()), "sha256": file_sha256(checkpoint_path),
            "completed_epochs": state["completed_epochs"], "selection": state["selection"],
            "model_class_weighting": state["model_class_weighting"],
            "class_weights": weights.tolist(), "class_counts": counts.tolist(),
            "class_weight_label_reference": state["class_weight_label_reference"],
            "class_weight_reduction": state["class_weight_reduction"]}
    if "teacher_model_seeds" in state:
        info.update(seed_metadata)

    def verify_checkpoint(metadata):
        actual = metadata.get("checkpoint", {})
        if ({key: value for key, value in actual.items() if key != "path"}
                != {key: value for key, value in info.items() if key != "path"}):
            raise ValueError("Existing correction artifact belongs to a different Teacher checkpoint.")

    reports, paths = {}, {}
    train_path = output_dir / "train_correction.npz"
    if train_path.exists() and resume:
        labels, provenance = load_train_correction(config, data.sources["train"], train_path)
        verify_checkpoint(provenance["metadata"])
        if not torch.equal(labels, state["label_logits"].argmax(1)):
            raise ValueError("Existing Train correction differs from the Teacher label bank.")
        reports["train"] = {"path": str(train_path), "reused": True,
                            "changed_count": int(labels.ne(data.sources["train"].labels).sum())}
    else:
        reports["train"] = write_train_correction(config, data.sources["train"], state["label_logits"],
                                                 train_path, checkpoint_info=info)
    paths["train"] = str(train_path.resolve())
    models = None
    for split, dataset in (("val", data.validation), ("test", data.test)):
        if dataset is None:
            continue
        path = output_dir / f"{split}_correction.npz"
        if path.exists() and resume:
            labels, provenance = load_holdout_correction(config, data.sources[split], path)
            verify_checkpoint(provenance["metadata"])
            reports[split] = {"path": str(path), "reused": True,
                             "changed_count": int(labels.ne(data.sources[split].labels).sum()),
                             "ground_truth_verified": False}
        else:
            if models is None:
                device = resolve_device(config.runtime.device)
                configure_runtime(config, device)
                if len(state.get("models", [])) != 2:
                    raise ValueError("Teacher correction requires exactly two networks.")
                models = [build_model(config.model, config.data.num_classes, device, initialize_pretrained=False)
                          for _ in state["models"]]
                for model, parameters in zip(models, state["models"], strict=True):
                    model.load_state_dict(parameters)
                    model.requires_grad_(False)
            with managed_loader(dataset, config, device) as loader:
                logits = predict_logits(models, loader, device, channels_last=config.training.channels_last,
                                        progress_label=f"Co {split} predictions", per_model=True)
            labels, history = optimize_holdout_labels(
                logits.to(device), data.sources[split].labels.to(device), config.correction,
                epochs=config.pipeline.holdout_correction_epochs,
                batch_size=config.correction.label_lr_reference_batch_size,
                progress_label=f"Co {split} label updates",
            )
            reports[split] = write_holdout_correction(
                config, data.sources[split], labels, path, checkpoint_info=info,
                teacher_logits=logits, history=history,
            )
        paths[split] = str(path.resolve())
    report = {"checkpoint": info, "paths": paths, "splits": reports,
              "dataset_fingerprint": fingerprint,
              "training_labels": "optimized_train_label_logits_argmax",
              "holdout_labels": "frozen_teacher_optimized_label_logits",
              "holdout_settings": holdout_settings(config),
              "ground_truth_verified": False, "label_condition": "corrected_train_val_test"}
    atomic_write_json(output_dir / "report.json", report, overwrite=resume)
    return report
