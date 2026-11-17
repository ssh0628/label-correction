"""Center-ROI validation and fixed 27-ROI Test evaluation."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from itertools import tee
from pathlib import Path
from typing import Any
import hashlib
import json
import time

from torch import Tensor, nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from tqdm import tqdm
import torch

from .config import ModelConfig
from .features import _evaluation_images, _record_indices, _validate_model_outputs
from .model import SSRNetworks
from .pipeline import primary_checkpoint, validate_selection
from shared.artifacts import JsonlWriter
from shared.backbone import build_encoder
from shared.bbox import BBox
from shared.config import DataConfig, data_config_from_dict
from shared.data import SplitSource, _corrected_targets, prepare_dataset
from shared.metrics import _LABEL_DTYPES, _metrics_from_batches, _validate_logits, metrics_from_logits
from shared.precision import full_precision
from shared.roi import LETTERBOX_FILL
from shared.views import FixedViewsDataset

@torch.no_grad()
def evaluate_classification(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
    *,
    description: str = "Validation",
    channels_last: bool = False,
    class_names: tuple[str, ...] | None = None,
    corrected_labels: Tensor | None = None,
) -> dict[str, Any]:
    """Evaluate in full precision; one forward pass, streaming metrics by default.

    Networks remain in evaluation mode, matching the previous accuracy helper.
    FP16/BF16 inputs and logits are promoted to FP32; diagnostic FP64 is kept.
    Optional corrected labels are scored against the same logits, without a
    second model pass. Original scores remain at the top level; alternate scores
    live under corrected_label_metrics. Only logits/labels (not images) are cached.
    """
    device = torch.device(device)
    if corrected_labels is not None:
        if (
            not isinstance(loader.sampler, SequentialSampler) or loader.drop_last
            or type(loader.batch_sampler) is not BatchSampler
        ):
            raise ValueError("Corrected-label evaluation requires a sequential loader with drop_last=False.")
        corrected_labels = torch.as_tensor(corrected_labels).detach().cpu()
        if (
            corrected_labels.ndim != 1 or corrected_labels.numel() != len(loader.dataset)
            or corrected_labels.dtype not in _LABEL_DTYPES
        ):
            raise ValueError("Corrected labels must be a one-dimensional integer tensor matching the dataset.")
        classes = len(class_names) if class_names is not None else getattr(networks.classifier, "out_features", None)
        if (corrected_labels < 0).any().item() or (
            classes is not None and (corrected_labels >= classes).any().item()
        ):
            raise ValueError("Corrected labels must lie within the classifier's class range.")
    networks.eval()

    def batches() -> Iterable[tuple[Tensor, Tensor]]:
        for images, labels in tqdm(loader, desc=description, leave=False, mininterval=5):
            if images.size(0) == 0:
                raise ValueError("Evaluation batches must not be empty.")
            images = _evaluation_images(images, device, channels_last)
            features = networks.encoder(images)
            logits = networks.classifier(features)
            _validate_logits(logits, images.size(0))
            _validate_model_outputs(features, logits)
            yield logits, labels

    if corrected_labels is None:
        return _metrics_from_batches(batches(), device, class_names)

    original_batches, alternate_batches = tee(batches())
    result = _metrics_from_batches(original_batches, device, class_names)

    def corrected_batches() -> Iterable[tuple[Tensor, Tensor]]:
        offset = 0
        for logits, _ in alternate_batches:
            stop = offset + logits.size(0)
            yield logits, corrected_labels[offset:stop]
            offset = stop
        if offset != corrected_labels.numel():
            raise ValueError("Corrected-label evaluation did not cover the complete dataset.")

    result["corrected_label_metrics"] = _metrics_from_batches(corrected_batches(), device, class_names)
    return result


IMAGE_BATCH_SIZE = 16  # CPU 이미지 수; 이미지당 27개 view
FORWARD_BATCH_SIZE = 128  # GPU에 한 번에 넣는 view 수; OOM이면 감소
NUM_WORKERS = 4  # 27-view CPU 전처리; prefetch 1
DEVICE = "auto"  # auto / cuda / cuda:0 / cpu / mps

# - 기존 multi_roi/ConvNeXt/inference/total_test.py 평가 규칙
SCALES = (0.8, 1.0, 1.2)
SHIFT_RATIO = 0.15
VIEW_COUNT = 27
CENTER_VIEW = 13


def load_run_config(
    run_dir: Path,
) -> tuple[DataConfig, ModelConfig, bool]:
    """Restore inference settings from this run, not the current training CONFIG."""
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data = data_config_from_dict(saved["data"], legacy_roi_defaults=True)
    if not data.crop_bbox or data.test is None:
        raise ValueError("27-view evaluation requires saved crop_bbox=True and a test split.")
    model = ModelConfig(**saved["model"])
    model.validate()
    return data, model, bool(saved["training"].get("channels_last", False))


class MultiROIDataset(FixedViewsDataset):
    """SSR's fixed 27-view protocol, independent of saved training ROI ranges."""
    def __init__(self, source: SplitSource, boxes: tuple[BBox | None, ...], data: DataConfig):
        super().__init__(source, boxes, data, scales=SCALES, shift_ratio=SHIFT_RATIO)


def _load_checkpoint(path: Path, model: nn.Sequential, data: DataConfig, config: ModelConfig):
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "model_name": config.name, "num_classes": data.num_classes,
        "class_names": list(data.class_names),
    }
    expected_input = {
        key: value for key, value in asdict(data).items()
        if key in {"image_size", "label_offset", "crop_bbox", "crop_method", "missing_bbox"}
    }
    expected_input.update(mean=list(data.mean), std=list(data.std),
                          letterbox_fill=list(LETTERBOX_FILL), roi_padding_fill=0)
    for actual, reference in ((state, expected), (state.get("input_config", {}), expected_input)):
        for key, value in reference.items():
            if key in actual and actual[key] != value:
                raise ValueError(f"{path.name}: saved config/checkpoint mismatch for {key}.")
    model[0].load_state_dict(state["encoder"], strict=True)
    model[1].load_state_dict(state["classifier"], strict=True)
    # Do not retain optimizer, projector, or predictor tensors during inference.
    return {key: state.get(key) for key in ("cur_epoch", "completed_epochs", "selection")}


@torch.inference_mode()
def collect_view_logits(
    model: nn.Sequential, loader: DataLoader, device: torch.device, *,
    num_classes: int, forward_batch_size: int, channels_last: bool,
) -> Tensor:
    """Transfer only a bounded view chunk to GPU; retain small logits on CPU."""
    if forward_batch_size < 1:
        raise ValueError("forward_batch_size must be positive.")
    model.eval()
    result = torch.empty((len(loader.dataset), VIEW_COUNT, num_classes), dtype=torch.float32)
    seen = torch.zeros(len(loader.dataset), dtype=torch.bool)
    with full_precision(device):
        for views, indices in tqdm(loader, desc="27-view Test", leave=False, mininterval=5):
            if views.ndim != 5 or views.shape[1:3] != (VIEW_COUNT, 3):
                raise ValueError("Expected an image batch with shape [B, 27, 3, H, W].")
            indices = _record_indices(indices, views.size(0), seen)
            flat = views.flatten(0, 1)
            batch_logits = torch.empty((flat.size(0), num_classes), dtype=torch.float32)
            for start in range(0, flat.size(0), forward_batch_size):
                images = _evaluation_images(flat[start:start + forward_batch_size], device, channels_last)
                features = model[0](images)
                logits = model[1](features)
                if logits.shape != (images.size(0), num_classes) or not logits.is_floating_point():
                    raise ValueError("Classifier must return floating-point logits [B, num_classes].")
                _validate_model_outputs(features, logits)
                batch_logits[start:start + logits.size(0)] = logits.float().cpu()
            result[indices] = batch_logits.reshape(-1, VIEW_COUNT, num_classes)
    if not seen.all():
        raise RuntimeError("Evaluation loader did not cover every test sample.")
    return result


def run(
    run_dir: Path, *,
    image_batch_size: int = IMAGE_BATCH_SIZE, forward_batch_size: int = FORWARD_BATCH_SIZE,
    num_workers: int = NUM_WORKERS, device: str = DEVICE,
    correction_paths: dict[str, Path], output_dir: Path,
    checkpoint_path: Path | None = None,
) -> Path:
    if image_batch_size < 1 or forward_batch_size < 1 or num_workers < 0:
        raise ValueError("Batch sizes must be positive and num_workers nonnegative.")
    run_dir = Path(run_dir).expanduser().resolve()
    data, model_config, channels_last = load_run_config(run_dir)
    filename, expected_selection = primary_checkpoint(run_dir)
    checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else run_dir / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if any(not isinstance(name, str) or not name.strip() for name in correction_paths):
        raise ValueError("Correction reference names must be nonempty strings.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    target = torch.device(device)
    torch.manual_seed(0)
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "run_dir": str(run_dir), "data": asdict(data), "model": asdict(model_config),
        "view_count": VIEW_COUNT, "scales": SCALES, "shift_ratio": SHIFT_RATIO,
        "aggregation": "logit_avg", "center_view_index": CENTER_VIEW,
        "image_batch_size": image_batch_size, "forward_batch_size": forward_batch_size,
        "num_workers": num_workers, "precision": "fp32", "device": str(target),
        "channels_last": channels_last and target.type == "cuda",
        "metric_units": "rates in [0, 1]; loss in nats",
    }
    sources, boxes = prepare_dataset(data, split="test", report_path=output / "data_audit.jsonl")
    source = sources["test"]
    references, loaded = {}, {}
    for name, path in correction_paths.items():
        path = Path(path).expanduser().resolve()
        if path not in loaded:
            loaded[path] = _corrected_targets(data, source, path)
        labels, provenance = loaded[path]
        references[name] = (path, labels)
        settings.setdefault("corrected_references", {})[name] = provenance
    primary_reference = None
    if expected_selection["label_reference"] == "teacher_corrected":
        training = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["training"]
        expected_path = Path(training["corrected_test_labels_path"]).expanduser().resolve()
        primary_reference = next((name for name, (path, _) in references.items() if path == expected_path), None)
        # A copied historical run can retain server paths. Match an explicitly
        # provided Test artifact to the saved Train artifact's teacher identity.
        training_targets = run_dir / "training_targets.json"
        if primary_reference is None and training_targets.is_file():
            expected = json.loads(training_targets.read_text(encoding="utf-8")).get("metadata", {})
            teacher_hash = expected.get("checkpoint_sha256")
            if teacher_hash:
                primary_reference = next((name for name, provenance in
                    settings.get("corrected_references", {}).items()
                    if provenance.get("metadata", {}).get("checkpoint_sha256") == teacher_hash
                    and provenance.get("metadata", {}).get("correction_threshold") == expected.get("correction_threshold")), None)
        if primary_reference is None:
            labels, provenance = _corrected_targets(data, source, expected_path)
            primary_reference = "student_corrected"
            if primary_reference in references:
                raise ValueError("student_corrected reference does not match the student's Test correction.")
            references[primary_reference] = (expected_path, labels)
            settings.setdefault("corrected_references", {})[primary_reference] = provenance
    settings.update(checkpoint=checkpoint.name, selection_label_reference=expected_selection["label_reference"],
                    primary_test_reference=primary_reference or "original")
    settings["test_sample_count"] = len(source.paths)
    settings["test_paths_sha256"] = hashlib.sha256("\0".join(source.paths).encode("utf-8")).hexdigest()
    settings["test_labels_sha256"] = hashlib.sha256(source.labels.numpy().tobytes()).hexdigest()
    (output / "config.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    dataset = MultiROIDataset(source, boxes["test"], data)
    loader = DataLoader(
        dataset, batch_size=image_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=target.type == "cuda", persistent_workers=num_workers > 0,
        **({"prefetch_factor": 1, "multiprocessing_context": "spawn"} if num_workers else {}),
    )
    encoder, feature_dim = build_encoder(
        model_config.name, pretrained=False, drop_path_rate=model_config.drop_path_rate,
    )
    model = nn.Sequential(encoder, nn.Linear(feature_dim, data.num_classes)).to(target)
    if channels_last and target.type == "cuda":
        model[0].to(memory_format=torch.channels_last)
    started = time.perf_counter()
    metadata = _load_checkpoint(checkpoint, model, data, model_config)
    validate_selection(metadata["selection"], run_dir,
                        label_reference=expected_selection["label_reference"])
    logits = collect_view_logits(model, loader, target, num_classes=data.num_classes,
                                forward_batch_size=forward_batch_size, channels_last=channels_last and target.type == "cuda")
    averaged = logits.mean(1)
    center = metrics_from_logits(logits[:, CENTER_VIEW], source.labels, class_names=data.class_names)
    fused = metrics_from_logits(averaged, source.labels, class_names=data.class_names)
    corrected = {}
    for name, (labels_path, labels) in references.items():
        corrected[name] = {
            "labels_path": str(labels_path),
            "single_center": metrics_from_logits(logits[:, CENTER_VIEW], labels, class_names=data.class_names),
            "logit_avg": metrics_from_logits(averaged, labels, class_names=data.class_names),
        }
    primary_metrics = corrected[primary_reference]["logit_avg"] if primary_reference is not None else fused
    extra = {"corrected_references": corrected,
             "primary_label_reference": expected_selection["label_reference"],
             "primary_test_reference": primary_reference or "original",
             "primary_test_metrics": primary_metrics}
    if len(corrected) == 1:
        reference = next(iter(corrected.values()))
        extra.update(single_center_corrected=reference["single_center"], logit_avg_corrected=reference["logit_avg"])
    with JsonlWriter(output / "checkpoint_results.jsonl") as writer:
        writer.write({
            "filename": checkpoint.name, "status": "evaluated", **metadata,
            "epoch": metadata["cur_epoch"], "single_center": center, "logit_avg": fused,
            "test_reused_same_epoch": False, "seconds": time.perf_counter() - started, **extra,
        })
    return output
