"""SSR checkpoint construction and persistence."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from torch.optim import Optimizer

from .config import ExperimentConfig
from .model import SSRNetworks
from shared.artifacts import atomic_torch_save
from shared.roi import LETTERBOX_FILL

class CheckpointManager:
    """Persist consistently structured checkpoints for one training run."""

    def __init__(
        self,
        run_dir: Path,
        networks: SSRNetworks,
        optimizer: Optimizer,
        config: ExperimentConfig,
    ) -> None:
        self.run_dir = run_dir
        self.networks = networks
        self.optimizer = optimizer
        self.config = config
        self.records: dict[str, dict[str, Any]] = {}
        self.source_hashes: dict[str, str] | None = None
        self.class_weights: list[float] | None = None

    def save(
        self,
        filename: str,
        epoch: int,
        *,
        metrics: Mapping[str, Any] | None = None,
        selection: Mapping[str, Any] | None = None,
    ) -> None:
        if Path(filename).name != filename:
            raise ValueError("checkpoint filename must not contain a directory.")
        state: dict[str, Any] = {
            "cur_epoch": epoch,
            "completed_epochs": epoch + 1,
            "metric_units": "rates in [0, 1]; loss in nats",
            "model_name": self.config.model.name,
            "model_config": asdict(self.config.model),
            "training_method": self.config.training.method,
            "supervised_loss_kind": "mixup_ce" if self.config.training.method == "ssr" else self.config.training.ce_loss,
            "class_weights": self.class_weights,
            "ssr_config": asdict(self.config.ssr),
            "source_hashes": self.source_hashes,
            "class_names": list(self.config.data.class_names),
            "num_classes": self.config.data.num_classes,
            "input_config": {
                "image_size": self.config.data.image_size,
                "mean": list(self.config.data.mean),
                "std": list(self.config.data.std),
                "label_offset": self.config.data.label_offset,
                "crop_bbox": self.config.data.crop_bbox,
                "missing_bbox": self.config.data.missing_bbox,
                "crop_method": self.config.data.crop_method,
                "multi_roi": self.config.data.multi_roi,
                "roi_scales": list(self.config.data.roi_scales),
                "roi_shift_ratio": self.config.data.roi_shift_ratio,
                "letterbox_fill": list(LETTERBOX_FILL),
                "roi_padding_fill": 0,
                "evaluation_roi": "center" if self.config.data.crop_bbox else "full",
            },
            "structural_labels_enabled": self.config.structural_labels.enabled,
            "classifier": self.networks.classifier.state_dict(),
            "encoder": self.networks.encoder.state_dict(),
            "proj_head": self.networks.projector.state_dict(),
            "pred_head": self.networks.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if metrics is not None:
            state["metrics"] = dict(metrics)
        if selection is not None:
            state["selection"] = dict(selection)
        atomic_torch_save(state, self.run_dir / filename)
        self.records[filename] = {
            "filename": filename,
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "validation": dict(metrics or {}).get("validation"),
            "validation_corrected": dict(metrics or {}).get("validation_corrected"),
            "selection": dict(selection or {}),
        }
