"""Co-Correcting's agreement selection and gradient-based label correction.

Adapted from the locally inspected JiarunLiu/Co-Correcting implementation
(Co-Correcting.py and Loss.py, Apache-2.0). No upstream package is imported.
The ``official`` loss preserves its class-mean KL/entropy and cross entropy
on label probabilities. ``paper`` uses class sums and negative log label
probabilities as written in the paper. Their hyperparameters are not
interchangeable. Numerical log-softmax replaces log(softmax) in both modes.
"""

from __future__ import annotations

from contextlib import nullcontext
from math import isfinite

import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm


def forget_rate_at(epoch: int, target: float, gradual: int) -> float:
    """Zero-based epoch schedule matching upstream's linspace warm-up.

``gradual=10`` reaches the target at epoch index 9 and then stays constant.
This ramp is separate from the stage before label correction starts.
"""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer.")
    if not isfinite(target) or not 0 <= target < 1:
        raise ValueError("forget_rate must be finite and in [0, 1).")
    if isinstance(gradual, bool) or not isinstance(gradual, int) or gradual < 1:
        raise ValueError("num_gradual must be a positive integer.")
    return target if gradual == 1 else target * min(epoch / (gradual - 1), 1.0)


def _validate_logits(logits: Tensor, label_logits: Tensor, originals: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("Model logits must have nonempty [batch, classes] shape.")
    if label_logits.shape != logits.shape:
        raise ValueError("Label logits must align with model logits.")
    if originals.shape != (len(logits),) or originals.dtype != torch.long:
        raise ValueError("Original labels must be int64 with one entry per sample.")
    if originals.device != logits.device or label_logits.device != logits.device:
        raise ValueError("Model logits, label logits and original labels must share a device.")
    if not bool(torch.isfinite(logits).all() & torch.isfinite(label_logits).all()):
        raise FloatingPointError("Model and label logits must be finite.")
    if bool(((originals < 0) | (originals >= logits.shape[1])).any()):
        raise ValueError("Original labels are outside the class range.")


def per_sample_loss(
    logits: Tensor,
    label_logits: Tensor,
    original_labels: Tensor,
    alpha: float,
    beta: float,
    variant: str = "official",
) -> Tensor:
    """Return the stage-III loss vector, retaining model and label gradients.

Set alpha=beta=0 for stage IV's classification-only loss. Stage I uses
ordinary hard-label cross entropy, before this function is needed.
"""
    _validate_logits(logits, label_logits, original_labels)
    if variant not in {"official", "paper"}:
        raise ValueError("loss_variant must be 'official' or 'paper'.")
    if any(not isfinite(x) or x < 0 for x in (alpha, beta)):
        raise ValueError("alpha and beta must be finite and non-negative.")
    # Loss arithmetic must remain FP32 even when the model forward uses BF16.
    log_p = F.log_softmax(logits.float(), dim=1)
    log_y = F.log_softmax(label_logits.float(), dim=1)
    p = log_p.exp()
    class_scale = logits.shape[1] if variant == "official" else 1
    classification = (p * (log_p - log_y)).sum(dim=1) / class_scale
    entropy = -(p * log_p).sum(dim=1) / class_scale
    if variant == "official":
        compatibility = F.cross_entropy(log_y.exp(), original_labels, reduction="none")
    else:
        compatibility = F.nll_loss(log_y, original_labels, reduction="none")
    return classification + alpha * compatibility + beta * entropy


@torch.no_grad()
def select_small_loss_indices(
    logits_a: Tensor,
    logits_b: Tensor,
    loss_a: Tensor,
    loss_b: Tensor,
    forget_rate: float,
) -> tuple[Tensor, Tensor]:
    """Return peer-selected training indices for A and B, respectively.

Only prediction-agreement samples are eligible. B's small-loss samples
train A; A's small-loss samples train B. An entirely disagreeing batch
returns empty indices. The training engine explicitly bootstraps such a
batch with hard-label CE during warmup; after warmup it skips optimizer.step()
(including weight decay), while label updates proceed independently. One
agreeing sample is kept if rounding would otherwise empty a nonempty set.
"""
    if (logits_a.ndim != 2 or logits_a.shape != logits_b.shape
            or logits_a.shape[1] < 2 or logits_a.device != logits_b.device):
        raise ValueError("Both model logits must have matching [batch, classes] shapes/devices.")
    if (loss_a.shape != (len(logits_a),) or loss_b.shape != loss_a.shape
            or loss_a.device != logits_a.device or loss_b.device != logits_a.device):
        raise ValueError("Per-sample losses must align with model logits.")
    if not isfinite(forget_rate) or not 0 <= forget_rate < 1:
        raise ValueError("forget_rate must be finite and in [0, 1).")
    if not all(bool(torch.isfinite(value).all()) for value in (logits_a, logits_b, loss_a, loss_b)):
        raise FloatingPointError("Selection inputs must be finite.")
    agreement = (logits_a.argmax(1) == logits_b.argmax(1)).nonzero(as_tuple=False).flatten()
    if not len(agreement):
        return agreement, agreement.clone()
    keep = max(1, int((1.0 - forget_rate) * len(agreement)))
    from_a = agreement[torch.argsort(loss_a[agreement], stable=True)[:keep]]
    from_b = agreement[torch.argsort(loss_b[agreement], stable=True)[:keep]]
    return from_b, from_a


def effective_label_lr(config, batch_size: int) -> float:
    """Actual minibatch-mean update multiplier, including a short last batch.

With scaling enabled, lambda/reference_batch is invariant as the batch
size changes. This is a defined adaptation of the upstream raw lambda.
"""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    value = float(config.label_lr)
    if not isfinite(value) or value <= 0:
        raise ValueError("label_lr must be finite and positive.")
    if config.scale_label_lr_with_batch:
        reference = config.label_lr_reference_batch_size
        if isinstance(reference, bool) or not isinstance(reference, int) or reference < 1:
            raise ValueError("label_lr_reference_batch_size must be a positive integer.")
        value *= batch_size / reference
    return value


def update_label_logits(
    label_logits: Tensor,
    logits_a: Tensor,
    logits_b: Tensor,
    original_labels: Tensor,
    eligible: Tensor,
    config,
) -> Tensor:
    """Return corrected FP32 label logits; input tensors are never mutated.

Uses the sum of two network label gradients, averaged over the *full*
minibatch before applying the curriculum mask. Thus a smaller eligible set
does not implicitly increase its update rate. No model gradient is created.
``config`` is the correction section of ExperimentConfig.
"""
    _validate_logits(logits_a, label_logits, original_labels)
    _validate_logits(logits_b, label_logits, original_labels)
    if eligible.shape != (len(label_logits),) or eligible.dtype != torch.bool:
        raise ValueError("Curriculum eligibility must be a batch-length boolean mask.")
    if eligible.device != label_logits.device:
        raise ValueError("Curriculum mask and label logits must share a device.")
    lr = effective_label_lr(config, len(label_logits))
    updated = label_logits.detach().float().clone()
    if not bool(eligible.any()):
        return updated
    autocast_off = (torch.autocast(device_type=label_logits.device.type, enabled=False)
                    if label_logits.device.type in {"cpu", "cuda"} else nullcontext())
    with torch.enable_grad(), autocast_off:
        labels = updated.clone().requires_grad_(True)
        loss_a = per_sample_loss(logits_a.detach().float(), labels, original_labels,
                                 config.alpha, config.beta, config.loss_variant).mean()
        loss_b = per_sample_loss(logits_b.detach().float(), labels, original_labels,
                                 config.alpha, config.beta, config.loss_variant).mean()
        gradient, = torch.autograd.grad(loss_a + loss_b, labels)
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Non-finite label-correction gradient.")
    updated[eligible] -= lr * gradient.detach()[eligible]
    if not bool(torch.isfinite(updated).all()):
        raise FloatingPointError("Non-finite corrected label logits; lower label_lr.")
    return updated


@torch.no_grad()
def optimize_holdout_labels(
    network_logits: Tensor,
    original_labels: Tensor,
    config,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    progress_label: str | None = None,
) -> tuple[Tensor, list[dict]]:
    """Extend Co's label update to heldouts, with both Teachers fixed.

    Cache each Teacher's logits once. Every epoch updates label logits only;
    it does not repeat image inference, train the models or select small-loss
    samples. Keep the two losses separate until their label gradients are summed.
    """
    if any(type(value) is not int or value < 1 for value in (epochs, batch_size)):
        raise ValueError("Holdout epochs and batch_size must be positive integers.")
    if (network_logits.ndim != 3 or network_logits.shape[1] != 2
            or not network_logits.is_floating_point()):
        raise ValueError("Expected separate Teacher logits [samples, 2, classes].")
    if not isfinite(config.initial_label_logit) or config.initial_label_logit <= 0:
        raise ValueError("initial_label_logit must be finite and positive.")
    # Cloning outside inference_mode also makes cached inference tensors usable
    # in the label-gradient calculation without creating model gradients.
    logits = network_logits.detach().float().clone()
    originals = original_labels.detach().to(logits.device)
    labels = logits.new_zeros((len(logits), logits.shape[-1]))
    _validate_logits(logits[:, 0], labels, originals)
    _validate_logits(logits[:, 1], labels, originals)
    labels.scatter_(1, originals[:, None], config.initial_label_logit)
    eligible = torch.ones(len(labels), device=labels.device, dtype=torch.bool)

    def objective():
        return sum(per_sample_loss(logits[:, index], labels, originals,
                                   config.alpha, config.beta, config.loss_variant)
                   for index in range(2)).mean().item()

    history = []
    loss_before = objective()
    progress = tqdm(range(epochs), desc=progress_label, disable=progress_label is None,
                    leave=False, mininterval=5)
    for epoch in progress:
        previous = labels.softmax(1)
        previous_targets = labels.argmax(1)
        for start in range(0, len(labels), batch_size):
            rows = slice(start, start + batch_size)
            labels[rows] = update_label_logits(
                labels[rows], logits[rows, 0], logits[rows, 1], originals[rows],
                eligible[rows], config,
            )
        targets = labels.argmax(1)
        loss = objective()
        row = {"epoch": epoch + 1, "loss_before": loss_before, "loss": loss,
               "changed_count": targets.ne(originals).sum().item(),
               "updated_count": targets.ne(previous_targets).sum().item(),
               "mean_probability_change": (labels.softmax(1) - previous).abs().sum(1).mean().item()}
        history.append(row)
        progress.set_postfix(loss=f"{loss:.5f}", changed=row["changed_count"])
        loss_before = loss
    return labels.cpu(), history
