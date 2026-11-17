"""Teacher, baseline and corrected-label Student training."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from math import cos, isfinite, pi
from typing import Any
import random

from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

from .checkpoints import CheckpointManager
from .config import ExperimentConfig
from .data import ExperimentData, build_experiment_data
from .evaluation import evaluate_classification
from .features import evaluate_epoch
from .model import SSRNetworks, build_ce_networks, build_ssr_networks, calibrate_batch_norm
from .neighbors import SelectedSampleSampler, SelectionResult
from .training import train_wce_epoch, train_epoch
from shared.artifacts import JsonlWriter, atomic_write_json
from shared.data import _file_hash, _resolve_array
from shared.metrics import class_weights
from shared.precision import PrecisionPolicy, full_precision
from shared.runtime import close_loader

@dataclass(frozen=True, slots=True)
class EpochLoaders:
    evaluation: DataLoader
    all_samples: DataLoader
    validation: DataLoader | None


def seed_everything(seed: int) -> None:
    """전역 config seed를 Python, NumPy, PyTorch, CUDA에 동일 적용."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_execution(
    config: ExperimentConfig, networks: SSRNetworks, device: torch.device,
) -> None:
    """Enable CUDA training optimizations without reducing selection precision."""
    policy = PrecisionPolicy.from_config(config.training, device)
    if device.type != "cuda":
        return
    torch.backends.cudnn.deterministic = config.runtime.deterministic
    torch.backends.cudnn.benchmark = not config.runtime.deterministic
    # BF16 is scoped to training. k-NN/threshold decisions stay FP32, not TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if policy.channels_last:
        networks.encoder.to(memory_format=torch.channels_last)




def _loader_options(
    config: ExperimentConfig,
    device: torch.device,
    *,
    evaluation: bool = False,
) -> dict[str, Any]:
    workers = (
        config.training.eval_num_workers if evaluation else config.training.num_workers
    )
    options: dict[str, Any] = {
        "batch_size": (
            config.training.eval_batch_size if evaluation else config.training.batch_size
        ),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options.update(
            prefetch_factor=config.training.prefetch_factor,
            multiprocessing_context="spawn",
            # Do not keep evaluation worker pools resident during training.
            persistent_workers=config.training.persistent_workers and not evaluation,
        )
    return options


def _build_epoch_loaders(
    data: ExperimentData,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochLoaders:
    options = _loader_options(config, device)
    eval_options = _loader_options(config, device, evaluation=True)
    return EpochLoaders(
        evaluation=DataLoader(data.evaluation_train, shuffle=False, **eval_options),
        all_samples=DataLoader(
            data.all_train,
            shuffle=True,
            drop_last=config.training.method == "ssr",
            **options,
        ),
        validation=(
            DataLoader(data.validation, shuffle=False, **eval_options)
            if data.validation is not None else None
        ),
    )


def _build_calibration_loader(
    data: ExperimentData,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    """Build a one-pass, non-augmented loader with isolated worker RNG state."""
    options = _loader_options(config, device)
    options["persistent_workers"] = False
    return DataLoader(
        data.calibration_train,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.seed),
        **options,
    )




def _build_selected_loader(
    data: ExperimentData,
    selection: SelectionResult,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    sampler = SelectedSampleSampler(selection, data.num_classes)
    options = _loader_options(config, device)
    # Selection lives in the parent sampler, never in worker-local Subset copies.
    loader = DataLoader(
        data.selected_train,
        sampler=sampler,
        drop_last=True,
        **options,
    )
    if len(loader) == 0:
        raise RuntimeError(
            "The class-balanced selected loader has no full batch; "
            "SSR cannot perform its supervised update."
        )
    return loader






def _build_optimizer(
    networks: SSRNetworks,
    config: ExperimentConfig,
) -> Optimizer:
    encoder_group: dict[str, Any] = {"params": networks.encoder.parameters()}
    if config.training.encoder_learning_rate is not None:
        encoder_group["lr"] = config.training.encoder_learning_rate
    modules = networks.all_modules() if config.training.method == "ssr" else (networks.encoder, networks.classifier)
    parameter_groups = [encoder_group, *({"params": module.parameters()} for module in modules[1:])]
    if config.training.optimizer == "adamw":
        use_fused = config.training.fused_optimizer and all(
            parameter.is_cuda
            for module in networks.all_modules()
            for parameter in module.parameters()
        )
        return AdamW(
            parameter_groups,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            fused=True if use_fused else None,
        )
    return SGD(
        parameter_groups,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        momentum=config.training.momentum,
    )


def _build_scheduler(
    optimizer: Optimizer,
    config: ExperimentConfig,
) -> LambdaLR:
    """Apply one cosine multiplier to every parameter group's own base LR."""

    minimum_ratio = config.training.scheduler_eta_min_ratio
    total_epochs = config.training.epochs

    def cosine_multiplier(completed_epochs: int) -> float:
        progress = min(completed_epochs, total_epochs) / total_epochs
        return minimum_ratio + (1.0 - minimum_ratio) * (
            1.0 + cos(pi * progress)
        ) / 2.0

    return LambdaLR(optimizer, lr_lambda=cosine_multiplier)


def _prepare_config(config: ExperimentConfig) -> ExperimentConfig:
    """Reserve a fresh output directory before writing any run artifact."""
    config.validate()
    config.run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        config.run_dir.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Run directory already exists: {config.run_dir}. Use --fresh to start again.") from error
    return config


def run(config: ExperimentConfig) -> float | None:
    """Train one new teacher, original-label baseline, or fixed-label student."""
    config = _prepare_config(config)
    use_ssr = config.training.method == "ssr"
    corrected = config.training.corrected_labels_path is not None
    label_reference = "teacher_corrected" if corrected else "original"
    primary_filename = "best_corrected_macro_f1.pt" if corrected else "best_macro_f1.pt"
    atomic_write_json(config.run_dir / "config.json", config)
    device = config.resolve_device()
    seed_everything(config.seed)
    data = build_experiment_data(
        config.data, structural_labels_enabled=config.structural_labels.enabled,
        augmentation=config.augmentation, training_method=config.training.method,
        corrected_labels_path=config.training.corrected_labels_path,
        corrected_validation_labels_path=config.training.corrected_validation_labels_path,
        corrected_test_labels_path=config.training.corrected_test_labels_path,
        report_path=config.run_dir / "data_audit.jsonl",
    )
    training_samples = int(data.noisy_labels.numel())
    if use_ssr:
        for name, count in (("training.batch_size", config.training.batch_size),
                            ("ssr.neighbors", config.ssr.neighbors),
                            ("structural_labels.neighbors", config.structural_labels.neighbors
                             if config.structural_labels.enabled else 0)):
            if count > training_samples:
                raise ValueError(f"{name}={count} exceeds training samples={training_samples}.")
    weights = None
    if not use_ssr:
        weights, counts = class_weights(data.noisy_labels, data.num_classes)
        atomic_write_json(config.run_dir / "class_weights.json", {
            "loss": "wce", "strategy": "inverse_sqrt",
            "label_source": "corrected_train" if corrected else "original_train",
            "class_names": list(data.class_names),
            "counts": counts.tolist(),
            "weights": weights.tolist(),
        })
    loaders = _build_epoch_loaders(data, config, device)
    noisy_labels = data.noisy_labels.to(device)
    networks = (build_ssr_networks if use_ssr else build_ce_networks)(
        config.model, device, data.num_classes)
    configure_execution(config, networks, device)
    if config.model.calibrate_initial_batch_norm and not config.model.pretrained:
        with full_precision(device):
            calibrate_batch_norm(_build_calibration_loader(data, config, device), networks.encoder, device)
    optimizer = _build_optimizer(networks, config)
    checkpoints = CheckpointManager(config.run_dir, networks, optimizer, config)
    if weights is not None:
        checkpoints.class_weights = weights.tolist()
        weights = weights.to(device)
    checkpoints.source_hashes = {
        "paths": _file_hash(_resolve_array(config.data.root, config.data.train.paths)),
        "labels": _file_hash(_resolve_array(config.data.root, config.data.train.labels)),
    }
    for name, provenance in (("training_targets", data.training_targets),
                             ("validation_targets", data.validation_targets),
                             ("test_targets", data.test_targets)):
        if provenance is not None:
            atomic_write_json(config.run_dir / f"{name}.json", provenance)
    scheduler = _build_scheduler(optimizer, config)
    best_score = best_epoch = None
    best_total_loss = float("inf")
    selected_loader = None
    last_epoch = -1
    with ExitStack() as stack:
        stack.callback(close_loader, loaders.all_samples)
        writer = stack.enter_context(JsonlWriter(config.run_dir / "metrics.jsonl"))
        progress = stack.enter_context(tqdm(range(config.training.epochs),
            desc="SSR+LSL" if use_ssr else "WCE", unit="epoch", dynamic_ncols=True, mininterval=5))
        for epoch in progress:
            supervision = evaluate_epoch(loaders.evaluation, networks, noisy_labels, config, device) if use_ssr else None
            if use_ssr:
                if selected_loader is None:
                    selected_loader = _build_selected_loader(data, supervision.selection, config, device)
                    stack.callback(close_loader, selected_loader)
                else:
                    selected_loader.sampler.update(supervision.selection)
                    if len(selected_loader) == 0:
                        raise RuntimeError("SSR selection has no full batch after updating the balanced sampler.")
                losses = train_epoch(selected_loader, loaders.all_samples,
                    supervision.selection.modified_labels, supervision.structural_targets,
                    networks, optimizer, config, device, epoch)
            else:
                losses = train_wce_epoch(loaders.all_samples, noisy_labels, networks, optimizer,
                                        config, device, epoch, class_weights=weights)
            scheduler.step()
            last_epoch = epoch
            total_loss = (losses.supervised_loss
                + config.ssr.feature_consistency_weight * losses.feature_consistency_loss
                + config.structural_labels.loss_weight * (losses.structural_loss or 0.0))
            if not isfinite(total_loss):
                raise FloatingPointError("Weighted training total_loss is not finite.")
            training_metrics = {"supervised_loss": losses.supervised_loss,
                "feature_consistency_loss": losses.feature_consistency_loss,
                "structural_loss": losses.structural_loss, "total_loss": total_loss}
            if use_ssr and total_loss < best_total_loss:
                best_total_loss = total_loss
                checkpoints.save("best_train_total_loss.pt", epoch, metrics=training_metrics,
                    selection={"method": "train_loss", "split": "train", "criterion": "total_loss",
                               "score": total_loss, "direction": "min", "aggregation": "epoch_mean",
                               "label_stage": "post_training_model"})
            validation = validation_corrected = primary_validation = None
            try:
                if loaders.validation is not None:
                    validation = evaluate_classification(loaders.validation, networks, device,
                        channels_last=config.training.channels_last, class_names=data.class_names,
                        **({"corrected_labels": data.validation_correction} if corrected else {}))
                    validation_corrected = validation.pop("corrected_label_metrics", None)
                    if corrected and validation_corrected is None:
                        raise RuntimeError("Corrected student requires corrected Validation metrics; original-label fallback is disabled.")
                    primary_validation = validation_corrected if corrected else validation
            finally:
                # Preserve the trained epoch even when validation raises.
                checkpoint_metrics = {**training_metrics, "validation": validation,
                                      "validation_corrected": validation_corrected}
                checkpoints.save("last.pt", epoch, metrics=checkpoint_metrics,
                    selection={"method": "last", "criterion": "last_epoch", "label_reference": label_reference})
            if primary_validation is not None:
                score = primary_validation["macro_f1"]
                if best_score is None or score > best_score:
                    best_score, best_epoch = score, epoch + 1
                    selection = {"method": "best", "split": "validation", "criterion": "macro_f1",
                                 "score": score, "label_reference": label_reference}
                    if corrected:
                        selection.update(labels_path=str(config.training.corrected_validation_labels_path),
                                         labels_sha256=data.validation_targets["artifact_sha256"])
                    checkpoints.save(primary_filename, epoch, metrics=checkpoint_metrics, selection=selection)
            values = {"epoch": epoch, "completed_epochs": epoch + 1, **training_metrics,
                      "validation_accuracy": primary_validation["accuracy"] if primary_validation else None,
                      "validation_macro_f1": primary_validation["macro_f1"] if primary_validation else None,
                      "selection_label_reference": label_reference, "best_validation_macro_f1": best_score,
                      "best_epoch": best_epoch}
            writer.write(values)
            progress.set_postfix(loss=f"{total_loss:.4f}",
                accuracy=f"{values['validation_accuracy']:.4f}" if primary_validation else "-",
                f1=f"{values['validation_macro_f1']:.4f}" if primary_validation else "-")
            del supervision
    atomic_write_json(config.run_dir / "run_summary.json", {
        "status": "completed", "completed_epochs": last_epoch + 1,
        "selection_label_reference": label_reference,
        "primary_checkpoint": primary_filename if primary_filename in checkpoints.records else None,
        "best_validation_macro_f1": best_score, "best_epoch": best_epoch,
        "checkpoints": checkpoints.records,
    })
    return best_score
