"""Co-Correcting teacher, frozen-label WCE and saved-run configuration."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields, replace
from math import isfinite
from pathlib import Path
import json

from shared.config import AugmentationConfig, DataConfig, SplitConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output" / "co"
RETIRED_CORRECTION_FIELDS = {"export_checkpoint"}
RETIRED_PIPELINE_FIELDS = {"baseline_run_dir", "student_label_wave", "label_wave_window",
                           "label_wave_patience", "export_holdout_predictions"}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "convnextv2_tiny"
    pretrained: bool = True
    pretrained_weights: Path | None = None
    drop_path_rate: float = 0.2


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    # Teacher: 10 warm-up + 140 label correction + 50 fixed-label fine-tuning.
    epochs: int = 200
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 3e-5
    optimizer: str = "adamw"
    momentum: float = 0.9
    weight_decay: float = 0.1
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 16
    eval_num_workers: int = 16
    prefetch_factor: int = 2
    log_interval: int = 20
    # Co-Correcting reduces LR on entry to fixed-label fine-tuning.
    finetune_lr_ratio: float = 0.1
    student_min_lr_ratio: float = 1e-3

    def validate(self):
        positive = (self.epochs, self.batch_size, self.eval_batch_size, self.prefetch_factor, self.log_interval)
        if any(type(value) is not int or value < 1 for value in positive):
            raise ValueError("Epochs, batch sizes and schedule lengths must be positive integers.")
        if any(type(value) is not int or value < 0 for value in (self.num_workers, self.eval_num_workers)):
            raise ValueError("Worker counts must be nonnegative integers.")
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("Optimizer must be adamw or sgd.")
        if any(not isfinite(value) or value <= 0 for value in (self.learning_rate, self.encoder_learning_rate)):
            raise ValueError("Learning rates must be finite and positive.")
        if not isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative.")
        if any(not isfinite(value) or not 0 < value <= 1
               for value in (self.finetune_lr_ratio, self.student_min_lr_ratio)):
            raise ValueError("Learning rate ratios must be in (0,1].")


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    # Fixed original-Train weights affect only model optimization, in every phase.
    model_class_weighting: str = "inverse_sqrt"
    # None preserves the original shared RNG stream; a pair isolates each teacher.
    model_seeds: tuple[int, int] | None = None
    forget_rate: float = 0.10
    num_gradual: int = 10
    warmup_epochs: int = 10
    correction_end_epoch: int = 150
    alpha: float = 0.4
    beta: float = 0.1
    # Reference-batch learning rate: batch 512 uses effective lambda 2000.
    label_lr: float = 1000.0
    label_lr_reference_batch_size: int = 256
    scale_label_lr_with_batch: bool = True
    initial_label_logit: float = 10.0
    loss_variant: str = "official"
    curriculum_groups: int = 3
    pca_dimensions: int = 128
    density_percentile: float = 60.0
    curriculum_chunk_size: int = 512


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    output_root: Path = field(default_factory=lambda: OUTPUT_ROOT)
    name: str = "a1-a7-co-correcting"
    device: str = "auto"
    deterministic: bool = False


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    student_epochs: int = 300
    student_batch_size: int = 256
    evaluate_27_views: bool = True
    evaluation_image_batch_size: int = 16
    evaluation_view_chunk_size: int = 128
    holdout_threshold: float = 0.7


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @property
    def run_dir(self) -> Path:
        return self.runtime.output_root / self.runtime.name

    def validate(self, *, student: bool = False) -> None:
        self.training.validate()
        self.data.validate()
        self.augmentation.validate()
        if self.data.validation is None:
            raise ValueError("Validation is required for model selection.")
        if not self.runtime.output_root.is_absolute():
            raise ValueError("runtime.output_root must be absolute.")
        if (not self.runtime.name or Path(self.runtime.name).name != self.runtime.name
                or self.runtime.name in {".", ".."}):
            raise ValueError("runtime.name must be a single directory name.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        t, c, p = self.training, self.correction, self.pipeline
        positive_ints = (c.num_gradual, c.correction_end_epoch, c.label_lr_reference_batch_size,
                        c.curriculum_groups, c.pca_dimensions, c.curriculum_chunk_size,
                        p.student_epochs, p.student_batch_size, p.evaluation_image_batch_size, p.evaluation_view_chunk_size)
        if any(type(v) is not int or v < 1 for v in positive_ints):
            raise ValueError("Epochs, batch sizes, schedule lengths and chunk sizes must be positive integers.")
        if type(c.warmup_epochs) is not int or c.warmup_epochs < 0:
            raise ValueError("Worker counts/warmup must be nonnegative integers.")
        if not student and not c.warmup_epochs < c.correction_end_epoch < t.epochs:
            raise ValueError("Require warmup_epochs < correction_end_epoch < training.epochs.")
        if c.correction_end_epoch - c.warmup_epochs < c.curriculum_groups:
            raise ValueError("Correction phase must include every curriculum group.")
        if c.loss_variant not in {"official", "paper"}:
            raise ValueError("correction.loss_variant must be official or paper.")
        if c.model_class_weighting not in {"inverse_sqrt", "none"}:
            raise ValueError("correction.model_class_weighting must be inverse_sqrt or none.")
        if c.model_seeds is not None and (
                not isinstance(c.model_seeds, tuple) or len(c.model_seeds) != 2
                or any(type(seed) is not int or seed < 0 for seed in c.model_seeds)
                or c.model_seeds[0] == c.model_seeds[1]):
            raise ValueError("correction.model_seeds must be two distinct nonnegative integers or null.")
        for value in (c.label_lr, c.initial_label_logit):
            if not isfinite(value) or value <= 0:
                raise ValueError("Learning rates and initial_label_logit must be finite and positive.")
        for value in (c.alpha, c.beta):
            if not isfinite(value) or value < 0:
                raise ValueError("Loss weights and weight decay must be finite and nonnegative.")
        if not isfinite(c.forget_rate) or not 0 <= c.forget_rate < 1:
            raise ValueError("forget_rate must be in [0, 1).")
        if not isfinite(c.density_percentile) or not 0 < c.density_percentile < 100:
            raise ValueError("density_percentile must lie strictly between 0 and 100.")
        if not 0 <= p.holdout_threshold <= 1 or not isfinite(p.holdout_threshold):
            raise ValueError("holdout_threshold must be in [0,1].")
        if not self.model.name or not 0 <= self.model.drop_path_rate < 1:
            raise ValueError("Invalid model name/drop_path_rate.")
        # Initialization files are needed only by a fresh pretrained model.
        # Resume/evaluation restore full checkpoint weights and remain portable.


def student_config(config):
    """Use the Student epoch/batch schedule without changing Teacher parameters."""
    return replace(config, training=replace(config.training, epochs=config.pipeline.student_epochs,
                                            batch_size=config.pipeline.student_batch_size))


def to_dict(config: ExperimentConfig) -> dict:
    return json.loads(json.dumps(asdict(config), default=str))


def from_dict(payload: dict, *, validate: bool = True) -> ExperimentConfig:
    """Strict config parsing; reject typos rather than silently changing experiments."""
    classes = {"data": DataConfig, "augmentation": AugmentationConfig, "model": ModelConfig,
               "training": TrainingConfig, "correction": CorrectionConfig, "runtime": RuntimeConfig,
               "pipeline": PipelineConfig}
    allowed = {f.name for f in fields(ExperimentConfig)}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unknown top-level config fields.")
    result = dict(payload)
    paths = {"root", "image_root", "annotation_root", "pretrained_weights", "output_root"}
    tuples = {"class_names", "roi_scales", "mean", "std", "color_jitter"}
    for section, cls in classes.items():
        if section not in result:
            continue
        values = dict(result[section])
        retired = {"pipeline": RETIRED_PIPELINE_FIELDS, "correction": RETIRED_CORRECTION_FIELDS}.get(section, ())
        values = {key: value for key, value in values.items() if key not in retired}
        if set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown fields in {section}.")
        for key, value in values.items():
            if key in paths and value is not None:
                values[key] = Path(value).expanduser()
            elif section == "correction" and key == "model_seeds":
                if value is not None and not isinstance(value, (list, tuple)):
                    raise ValueError("correction.model_seeds must be two distinct nonnegative integers or null.")
                values[key] = None if value is None else tuple(value)
            elif key in tuples:
                values[key] = tuple(value)
            elif section == "data" and key in {"train", "validation", "test"} and value is not None:
                values[key] = SplitConfig(**value)
        result[section] = cls(**values)
    config = ExperimentConfig(**result)
    if validate:
        config.validate()
    return config


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        config = CONFIG
    else:
        path = Path(path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        for section, names in {"data": ("root", "image_root", "annotation_root"),
                               "runtime": ("output_root",), "model": ("pretrained_weights",)}.items():
            for name in names:
                value = payload.get(section, {}).get(name)
                if value is not None:
                    resolved = Path(value).expanduser()
                    payload[section][name] = str(resolved if resolved.is_absolute() else path.parent / resolved)
        config = from_dict(payload)
    config.validate()
    return config


CONFIG = ExperimentConfig()
