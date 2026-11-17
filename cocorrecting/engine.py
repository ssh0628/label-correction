"""Dual-network training, curriculum construction and frozen-label fine-tuning."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .algorithm import (forget_rate_at, per_sample_loss,
                        select_small_loss_indices, update_label_logits)
from .artifacts import (atomic_write_json, dataset_fingerprint, source_hashes,
                        validate_teacher_weighting)
from .curriculum import build_curriculum, eligible_mask
from .evaluation import evaluate_logits, extract_features, predict_logits
from .model import build_optimizer, build_teacher_models, teacher_forward, teacher_model_weights
from .runtime import (atomic_torch_save, comparable_config, configure_runtime,
                      managed_loader, reconcile_metrics, resolve_device, start_run,
                      seed_everything, set_loader_epoch, teacher_seed_metadata,
                      validate_checkpoint_history, validate_teacher_seed_metadata)
from .config import to_dict
from .data import build_data
from shared.precision import PrecisionPolicy
from shared.artifacts import JsonlWriter


def phase_at(epoch, correction):
    if epoch < correction.warmup_epochs:
        return "warmup"
    if epoch < correction.correction_end_epoch:
        return "correction"
    return "finetune"


def set_teacher_lr(optimizers, config, epoch):
    t, c = config.training, config.correction
    ratio = 1.0
    if epoch >= c.correction_end_epoch:
        third = max(1, (t.epochs - c.correction_end_epoch) // 3)
        decay = min(2, (epoch - c.correction_end_epoch) // third)
        ratio = t.finetune_lr_ratio / 10**decay
    for optimizer in optimizers:
        for group, initial in zip(optimizer.param_groups, (t.encoder_learning_rate, t.learning_rate)):
            group["lr"] = initial * ratio


def train_epoch(models, optimizers, loader, original_labels, label_logits, groups,
                config, device, epoch, *, model_class_weights=None):
    """Weight only model optimization; selection and label gradients stay raw."""
    for model in models:
        model.train()
    policy = PrecisionPolicy.from_config(config.training, device)
    correction = config.correction
    classes = label_logits.shape[1]
    if model_class_weights is None:
        model_class_weights, _ = teacher_model_weights(
            original_labels, classes, correction.model_class_weighting)
    model_class_weights = torch.as_tensor(model_class_weights, dtype=torch.float32, device=device).detach()
    if (model_class_weights.shape != (classes,) or not bool(torch.isfinite(model_class_weights).all())
            or not bool((model_class_weights > 0).all())):
        raise ValueError("Teacher model class weights must be finite, positive and aligned with classes.")
    phase = phase_at(epoch, correction)
    forget = forget_rate_at(epoch, correction.forget_rate, correction.num_gradual)
    eligible = (eligible_mask(groups, epoch, correction) if groups is not None
                else torch.zeros(len(original_labels), dtype=torch.bool))
    seen = torch.zeros(len(original_labels), dtype=torch.bool)
    total = selected = correct = fallback_batches = 0
    loss_sums, weight_sums = [0.0, 0.0], [0.0, 0.0]
    progress = tqdm(loader, desc=f"Co {phase} {epoch + 1}", leave=False, mininterval=5)
    for step, (images, indices) in enumerate(progress):
        indices = torch.as_tensor(indices, dtype=torch.long)
        if (indices.ndim != 1 or len(indices) != len(images) or indices.unique().numel() != len(indices)
                or bool(((indices < 0) | (indices >= len(seen))).any()) or bool(seen[indices].any())):
            raise ValueError("Training loader must visit every stable index exactly once.")
        seen[indices] = True
        images = policy.to_device(images)
        originals = original_labels[indices].to(device)
        labels = label_logits[indices].to(device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with policy.autocast():
            outputs = teacher_forward(models, images, model_seeds=correction.model_seeds,
                                      device=device, epoch=epoch, step=step)
        outputs = [output.float() for output in outputs]
        if phase == "warmup":
            losses = [F.cross_entropy(output, originals, reduction="none") for output in outputs]
        else:
            alpha, beta = (correction.alpha, correction.beta) if phase == "correction" else (0.0, 0.0)
            losses = [per_sample_loss(output, labels, originals, alpha, beta,
                                      correction.loss_variant) for output in outputs]
        chosen = select_small_loss_indices(*outputs, *losses, forget)
        if phase == "warmup" and not len(chosen[0]):
            # Independently initialized heads need full-batch CE to bootstrap
            # when they initially agree on no examples.
            all_indices = torch.arange(len(originals), device=device)
            chosen = (all_indices, all_indices)
            fallback_batches += 1
        detached = [output.detach() for output in outputs]
        sample_weights = model_class_weights[originals]
        for model_index, (loss, keep, optimizer) in enumerate(zip(losses, chosen, optimizers)):
            if len(keep):
                numerator = (loss[keep] * sample_weights[keep]).sum()
                denominator = sample_weights[keep].sum()
                objective = numerator / denominator
                if not bool(torch.isfinite(objective)):
                    raise FloatingPointError("Non-finite model loss.")
                objective.backward()
                optimizer.step()
                sums = torch.stack((numerator.detach(), denominator.detach())).cpu().tolist()
                loss_sums[model_index] += sums[0]
                weight_sums[model_index] += sums[1]
        selected += len(chosen[0])
        if phase == "correction":
            label_logits[indices] = update_label_logits(
                labels, *detached, originals, eligible[indices].to(device), correction).cpu()
        total += len(indices)
        correct += int(((detached[0] + detached[1]) / 2).argmax(1).eq(originals).sum())
        mean_loss = sum(n / d if d else 0.0 for n, d in zip(loss_sums, weight_sums)) / 2
        if step % config.training.log_interval == 0:
            progress.set_postfix(loss=f"{mean_loss:.4f}", accuracy=f"{correct / total:.3f}")
    if not bool(seen.all()):
        raise ValueError("Training loader omitted samples; drop_last must be False.")
    names = ("model_a", "model_b")
    losses_by_network = {name: n / d if d else None for name, n, d in zip(names, loss_sums, weight_sums)}
    return {"phase": phase, "loss": sum(value or 0.0 for value in losses_by_network.values()) / 2,
            "model_loss_by_network": losses_by_network,
            "model_loss_weight_sum": dict(zip(names, weight_sums)),
            "loss_reduction": "mean_of_per_network_selected_weighted_means",
            "model_class_weighting": correction.model_class_weighting,
            "selected_count": selected, "eligible_label_count": int(eligible.sum()),
            "warmup_fallback_batches": fallback_batches, "train_accuracy": correct / total}


def train_teacher(config, run_dir=None, *, resume=None, stop_after_epochs=None) -> Path:
    config.validate()
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be a positive integer.")
    run_dir, checkpoint = start_run(run_dir or config.run_dir / "teacher", resume)
    if checkpoint is None:
        atomic_write_json(run_dir / "config.json", to_dict(config))
    else:
        validate_checkpoint_history(run_dir, checkpoint)
        if checkpoint.get("config", {}).get("correction", {}).get("model_class_weighting") != config.correction.model_class_weighting:
            raise ValueError("Teacher class weighting provenance mismatch. Use a new runtime.name "
                             "instead of changing weighting during resume.")
    device = resolve_device(config.runtime.device)
    configure_runtime(config, device)
    data = build_data(config, report_path=run_dir / "data_audit.jsonl")
    fingerprint = dataset_fingerprint(data.sources, config)
    originals = data.sources["train"].labels.clone().long()
    classes = config.data.num_classes
    if bool(torch.bincount(originals, minlength=classes).eq(0).any()):
        raise ValueError("Every class must occur in the prepared original Train split.")
    weights, counts = teacher_model_weights(originals, classes, config.correction.model_class_weighting)
    if checkpoint is not None:
        if (checkpoint.get("kind") != "teacher" or checkpoint.get("dataset_fingerprint") != fingerprint
                or comparable_config(checkpoint["config"]) != comparable_config(to_dict(config))):
            raise ValueError("Resume checkpoint/config/data provenance mismatch.")
        validate_teacher_weighting(checkpoint, originals, classes, config.correction.model_class_weighting)
        validate_teacher_seed_metadata(checkpoint, config.correction.model_seeds)
    weighting = {"kind": "teacher", "strategy": config.correction.model_class_weighting,
                 "label_source": "original_train", "class_names": list(config.data.class_names),
                 "counts": counts.tolist(), "weights": weights.tolist(), "fixed": True,
                 "scope": "model_optimizer_all_phases", "reduction": "selected_weight_sum",
                 "selection_weighting": "none", "label_update_weighting": "none"}
    weights_path = run_dir / "class_weights.json"
    if checkpoint is None:
        atomic_write_json(weights_path, weighting)
    elif not weights_path.is_file() or json.loads(weights_path.read_text(encoding="utf-8")) != weighting:
        raise ValueError("Teacher class_weights.json does not match the saved fixed weighting policy.")
    weight_metadata = {"model_class_weighting": config.correction.model_class_weighting,
                       "class_weights": weights.tolist(), "class_counts": counts.tolist(),
                       "class_weight_label_reference": "original_train",
                       "class_weight_reduction": "selected_weight_sum"}
    model_weights = weights.to(device)
    seed_metadata = teacher_seed_metadata(config.correction.model_seeds)
    models = build_teacher_models(config, classes, device, initialize_pretrained=checkpoint is None)
    policy = PrecisionPolicy.from_config(config.training, device)
    if policy.channels_last:
        for model in models:
            model.to(memory_format=torch.channels_last)
    optimizers = [build_optimizer(model, config.training) for model in models]
    labels = F.one_hot(originals, classes).float() * config.correction.initial_label_logit
    groups, completed = None, 0
    if checkpoint is not None:
        for model, state in zip(models, checkpoint["models"], strict=True):
            model.load_state_dict(state)
        for optimizer, state in zip(optimizers, checkpoint["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        labels = checkpoint["label_logits"].float().cpu()
        if labels.shape != (len(originals), classes) or not bool(torch.isfinite(labels).all()):
            raise ValueError("Invalid resumed label bank.")
        if not torch.equal(checkpoint["original_labels"], originals):
            raise ValueError("Resumed original labels differ.")
        groups = checkpoint["curriculum_groups"]
        completed = checkpoint["completed_epochs"]
        if type(completed) is not int or not 0 <= completed <= config.training.epochs:
            raise ValueError("Invalid completed_epochs in checkpoint.")
        reconcile_metrics(run_dir / "metrics.jsonl", completed)
    target_epochs = config.training.epochs if stop_after_epochs is None else min(config.training.epochs, stop_after_epochs)
    with ExitStack() as stack:
        train_loader = stack.enter_context(managed_loader(data.train, config, device, training=True))
        val_loader = stack.enter_context(managed_loader(data.validation, config, device))
        fixed_loader = stack.enter_context(managed_loader(data.evaluation_train, config, device))
        metrics_log = stack.enter_context(JsonlWriter(run_dir / "metrics.jsonl", append=True))
        epochs = tqdm(range(completed, target_epochs), desc="Co Teacher", unit="epoch", mininterval=5)
        for epoch in epochs:
            seed_everything(config.seed + epoch * 100003)
            set_loader_epoch(train_loader, config, epoch)
            set_teacher_lr(optimizers, config, epoch)
            if epoch >= config.correction.warmup_epochs and groups is None:
                features = extract_features(models, fixed_loader, device, channels_last=policy.channels_last)
                groups = torch.from_numpy(build_curriculum(features, originals, config.correction,
                                                          seed=config.seed, work_dir=run_dir / "curriculum_tmp"))
                del features
            values = train_epoch(models, optimizers, train_loader, originals, labels, groups,
                                 config, device, epoch, model_class_weights=model_weights)
            logits = predict_logits(models, val_loader, device, channels_last=policy.channels_last)
            validation = evaluate_logits(logits, data.sources["val"].labels, class_names=config.data.class_names)
            completed = epoch + 1
            values.update(epoch=epoch, completed_epochs=completed, validation=validation,
                          validation_macro_f1=validation["macro_f1"], validation_label_reference="original")
            state = {"schema_version": 1, "kind": "teacher", "config": to_dict(config),
                     **weight_metadata, **seed_metadata,
                     "dataset_fingerprint": fingerprint, "source_hashes": source_hashes(config, "train"),
                     "models": [model.state_dict() for model in models],
                     "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                     "label_logits": labels, "original_labels": originals, "curriculum_groups": groups,
                     "completed_epochs": completed, "metrics": values,
                     "selection": {"method": "last", "criterion": "last_epoch"}}
            if completed == config.correction.correction_end_epoch:
                atomic_torch_save({**state, "selection": {"method": "fixed", "criterion": "end_of_label_updates"}},
                                  run_dir / "correction_end.pt")
            metrics_log.write(values)
            atomic_torch_save(state, run_dir / "last.pt")
            epochs.set_postfix(loss=f"{values['loss']:.4f}", accuracy=f"{validation['accuracy']:.3f}",
                               f1=f"{validation['macro_f1']:.3f}")
    status = "completed" if completed == config.training.epochs else "interrupted"
    atomic_write_json(run_dir / "run_summary.json", {"status": status, "kind": "teacher",
                     "completed_epochs": completed, "dataset_fingerprint": fingerprint,
                     "correction_checkpoint": "correction_end.pt", "classification_checkpoint": "last.pt",
                     "loss_variant": config.correction.loss_variant,
                     **weight_metadata, **seed_metadata},
                     overwrite=True)
    return run_dir
