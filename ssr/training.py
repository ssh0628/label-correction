"""Per-epoch SSR+LSL and fixed-label WCE optimization."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from .config import ExperimentConfig
from .losses import (
    mixup_hard_label_views,
    negative_cosine_similarity,
    soft_cross_entropy,
    structural_mixup_loss,
)
from .model import SSRNetworks
from shared.precision import PrecisionPolicy

@dataclass(slots=True)
class _RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def value(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass(frozen=True, slots=True)
class TrainingLosses:
    supervised_loss: float
    feature_consistency_loss: float
    structural_loss: float | None


def _loss_value(loss: Tensor, name: str) -> float:
    value = loss.item()
    if not isfinite(value):
        raise FloatingPointError(f"Non-finite {name} loss; optimizer step was not applied.")
    return value


def train_epoch(
    selected_loader: DataLoader,
    all_samples_loader: DataLoader,
    modified_labels: Tensor,
    structural_targets: Tensor | None,
    networks: SSRNetworks,
    optimizer: Optimizer,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> TrainingLosses:
    networks.train()
    training = getattr(config, "training", None)
    precision = PrecisionPolicy.from_config(training, device)
    log_interval = getattr(training, "log_interval", 20)
    total_steps = len(all_samples_loader)
    expected_views = 3 if structural_targets is not None else 2
    supervised_average = _RunningAverage()
    consistency_average = _RunningAverage()
    structural_average = _RunningAverage() if structural_targets is not None else None
    selected_iterator = iter(selected_loader)
    all_iterator = iter(all_samples_loader)
    progress = tqdm(range(1, total_steps + 1), desc=f"Train {epoch + 1}", leave=False, mininterval=5)

    for step in progress:
        all_views, all_indices = next(all_iterator)
        try:
            selected_views, selected_indices = next(selected_iterator)
        except StopIteration:
            selected_iterator = iter(selected_loader)
            selected_views, selected_indices = next(selected_iterator)

        # Reject malformed view batches before any parameter/BN state changes.
        if len(all_views) != expected_views:
            if structural_targets is not None:
                raise RuntimeError("LSL requires [weak, strong, strong] all-sample views.")
            raise RuntimeError("SSR requires [weak, strong] all-sample views.")

        # Release the previous step's gradients before allocating activations.
        optimizer.zero_grad(set_to_none=True)
        first_selected = precision.to_device(selected_views[0])
        second_selected = precision.to_device(selected_views[1])
        selected_labels = modified_labels[selected_indices.to(device)]
        mixed_inputs, mixed_targets, _ = mixup_hard_label_views(
            first_selected,
            second_selected,
            selected_labels,
            num_classes=config.data.num_classes,
            alpha=config.ssr.mixup_alpha,
        )
        with precision.autocast():
            supervised_logits = networks.classifier(networks.encoder(mixed_inputs))
        supervised_loss = soft_cross_entropy(supervised_logits, mixed_targets)
        supervised_value = _loss_value(supervised_loss, "supervised")
        # Additive loss gradients accumulate at the same parameter values.
        # Backpropagate each branch before allocating the next encoder graph.
        supervised_loss.backward()
        supervised_average.update(supervised_value)
        del supervised_logits, supervised_loss, mixed_inputs, mixed_targets
        del first_selected, second_selected, selected_labels

        weak_view = precision.to_device(all_views[0])
        first_strong_view = precision.to_device(all_views[1])
        # SSR stops gradients through its weak target. Train-mode BN updates
        # and stochastic forward calls are still performed in the same order.
        # Separate AMP scopes prevent cached no-grad weights leaking to strong.
        with torch.no_grad(), precision.autocast():
            weak_projection = networks.projector(networks.encoder(weak_view))
        with precision.autocast():
            strong_projection = networks.projector(networks.encoder(first_strong_view))

        # p_weak는 loss에 직접 쓰이지 않지만
        # 공식 SSR의 predictor BN 갱신에 필요하다.
        with torch.no_grad(), precision.autocast():
            networks.predictor(weak_projection)
        with precision.autocast():
            strong_prediction = networks.predictor(strong_projection)
        consistency_loss = negative_cosine_similarity(
            strong_prediction,
            weak_projection,
        )
        consistency_value = _loss_value(consistency_loss, "feature-consistency")
        (config.ssr.feature_consistency_weight * consistency_loss).backward()
        consistency_average.update(consistency_value)
        del consistency_loss, strong_prediction, strong_projection, weak_projection
        del weak_view

        if structural_targets is not None:
            second_strong_view = precision.to_device(all_views[2])
            batch_structural_targets = structural_targets[all_indices.to(device)]
            with precision.autocast():
                current_structural_loss = structural_mixup_loss(
                    networks.encoder,
                    networks.classifier,
                    first_strong_view,
                    second_strong_view,
                    batch_structural_targets,
                    mixup_alpha=config.ssr.mixup_alpha,
                )
            structural_value = _loss_value(current_structural_loss, "structural")
            (config.structural_labels.loss_weight * current_structural_loss).backward()
            structural_average.update(structural_value)
            del current_structural_loss, batch_structural_targets, second_strong_view

        optimizer.step()
        del first_strong_view

        if progress.disable or (step % log_interval != 0 and step != total_steps):
            continue
        total = (supervised_average.value
                 + config.ssr.feature_consistency_weight * consistency_average.value
                 + config.structural_labels.loss_weight * (structural_average.value if structural_average else 0.0))
        progress.set_postfix(loss=f"{total:.4f}")

    return TrainingLosses(
        supervised_loss=supervised_average.value,
        feature_consistency_loss=consistency_average.value,
        structural_loss=(
            structural_average.value if structural_average is not None else None
        ),
    )


def train_wce_epoch(loader, labels, networks, optimizer, config, device, epoch, *,
                   class_weights: torch.Tensor) -> TrainingLosses:
    loss_kind = config.training.ce_loss
    if loss_kind != "wce":
        raise ValueError("Expected weighted cross entropy.")
    class_weights = torch.as_tensor(class_weights, device=device, dtype=torch.float32)
    if (class_weights.shape != (config.data.num_classes,) or not bool(torch.isfinite(class_weights).all())
            or bool((class_weights <= 0).any())):
        raise ValueError("WCE requires one positive finite weight per class.")
    networks.train()
    precision = PrecisionPolicy.from_config(config.training, device)
    loss_sum = count = 0
    iterator = iter(loader)
    progress = tqdm(range(1, len(loader) + 1), desc=f"{loss_kind.upper()} train {epoch + 1}", leave=False, mininterval=5)
    for step in progress:
        images, indices = next(iterator)
        if not isinstance(images, torch.Tensor) or images.shape[0] != len(indices):
            raise ValueError("WCE requires single-view images and stable training indices.")
        optimizer.zero_grad(set_to_none=True)
        inputs = precision.to_device(images)
        targets = labels[indices.to(labels.device)].to(device, non_blocking=True)
        with precision.autocast():
            logits = networks.classifier(networks.encoder(inputs))
        loss = F.cross_entropy(logits.float(), targets, weight=class_weights)
        value = _loss_value(loss, loss_kind.upper())
        loss.backward()
        optimizer.step()
        loss_sum += value * len(indices)
        count += len(indices)
        del inputs, targets, logits, loss
        if step % config.training.log_interval == 0 or step == len(loader):
            progress.set_postfix(loss=f"{loss_sum / count:.4f}")
    if not count:
        raise ValueError("WCE training loader is empty.")
    return TrainingLosses(loss_sum / count, 0.0, None)
