"""Typed configuration for SSR+LSL teachers and fixed-label WCE students."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, get_args

import torch

from shared.config import AugmentationConfig, DataConfig

PROJECT_ROOT = Path(__file__).resolve().parent  # - SSR backend
OUTPUT_ROOT = PROJECT_ROOT.parent / "output" / "ssr"  # - 학습 결과

# - H100 NVL: batch 256/1024, workers 16×2/32, prefetch 2; train 512 OOM
# - RTX 5080: train 16 / eval 64 / prefetch 1; 시작값
OptimizerName = Literal["sgd", "adamw"]
TrainingMethod = Literal["ssr", "ce"]
ClassificationLoss = Literal["ce", "wce"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Backbone 설정."""

    name: str = "convnextv2_tiny"  # - timm 모델명
    pretrained: bool = True  # - 사전학습 가중치
    drop_path_rate: float = 0.2  # - stochastic depth
    calibrate_initial_batch_norm: bool = True  # - scratch 모델 BN 보정

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("model.name must not be empty.")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("model.drop_path_rate must be in [0, 1).")


@dataclass(frozen=True, slots=True)
class SSRConfig:
    """SSR relabel, sample selection과 loss 설정."""

    relabel_threshold: float = 0.7  # - confidence > tau_r
    selection_threshold: float = 1.0  # - 1: 빈도 보정 k-NN 최대 점수 일치
    neighbors: int = 200  # - k-NN 이웃 수
    knn_chunks: int = 10  # - query 분할 수
    feature_consistency_weight: float = 1.0  # - consistency loss 가중치
    mixup_alpha: float = 4.0  # - mixup Beta(alpha, alpha)

    def validate(self) -> None:
        if not 0.0 <= self.relabel_threshold <= 1.0:
            raise ValueError("ssr.relabel_threshold must be in [0, 1].")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("ssr.selection_threshold must be in [0, 1].")
        if self.neighbors < 1:
            raise ValueError("ssr.neighbors must be positive.")
        if self.knn_chunks < 1:
            raise ValueError("ssr.knn_chunks must be positive.")
        if (
            not isfinite(self.feature_consistency_weight)
            or self.feature_consistency_weight < 0.0
        ):
            raise ValueError("ssr.feature_consistency_weight must not be negative.")
        if not isfinite(self.mixup_alpha) or self.mixup_alpha <= 0.0:
            raise ValueError("ssr.mixup_alpha must be positive.")


@dataclass(frozen=True, slots=True)
class StructuralLabelsConfig:
    """CVPR 2024 Learning with Structural Labels 설정."""

    enabled: bool = True  # - LSL 사용
    neighbors: int = 20  # - reverse k-NN 이웃 수
    knn_chunks: int = 10  # - query 분할 수
    loss_weight: float = 1.0  # - structural loss 가중치

    def validate(self) -> None:
        if self.neighbors < 1:
            raise ValueError("structural_labels.neighbors must be positive.")
        if self.knn_chunks < 1:
            raise ValueError("structural_labels.knn_chunks must be positive.")
        if not isfinite(self.loss_weight) or self.loss_weight < 0.0:
            raise ValueError("structural_labels.loss_weight must not be negative.")
        if self.enabled and self.loss_weight == 0.0:
            raise ValueError(
                "Disable structural_labels for the SSR ablation instead of using zero loss weight."
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """일반 이미지 데이터셋 학습 설정."""

    method: TrainingMethod = "ssr"  # - ssr / 고정 라벨 ce
    ce_loss: ClassificationLoss = "ce"  # - SSR: ce; 고정 라벨: wce
    corrected_labels_path: Path | None = None  # - ce 교정 Train NPZ
    corrected_validation_labels_path: Path | None = None  # - 교정 Val로 모델 선택
    corrected_test_labels_path: Path | None = None  # - 교정 Test로 최종 평가
    epochs: int = 300  # - 학습·cosine 길이
    batch_size: int = 256  # - 학습; mixup 입력은 2배
    eval_batch_size: int = 1024  # - FP32 평가
    amp: bool = True  # - CUDA 학습 BF16
    channels_last: bool = True  # - CUDA 메모리 배치
    fused_optimizer: bool = True  # - CUDA fused AdamW
    learning_rate: float = 1e-3  # - head 초기 LR
    encoder_learning_rate: float | None = 3e-5  # - encoder LR; None: head와 동일
    optimizer: OptimizerName = "adamw"  # - adamw / sgd
    momentum: float = 0.9  # - SGD momentum
    weight_decay: float = 0.1  # - weight decay
    scheduler_eta_min_ratio: float = 1e-3  # - 최저 LR / 초기 LR
    num_workers: int = 16  # - 학습 loader당 worker; 동시 32개
    eval_num_workers: int = 32  # - feature 추출·평가 worker
    prefetch_factor: int = 2  # - worker당 선행 batch
    persistent_workers: bool = True  # - 학습 worker 유지; 선택 인덱스만 갱신
    log_interval: int = 20  # - 진행 표시 step 간격

    def validate(self) -> None:
        if self.method not in get_args(TrainingMethod):
            raise ValueError("training.method must be 'ssr' or 'ce'.")
        if self.ce_loss not in get_args(ClassificationLoss):
            raise ValueError("training.ce_loss must be 'ce' or 'wce'.")
        if self.method == "ce" and self.ce_loss != "wce":
            raise ValueError("Fixed-label training requires WCE.")
        for name in ("corrected_labels_path", "corrected_validation_labels_path", "corrected_test_labels_path"):
            path = getattr(self, name)
            if path is not None and (self.method != "ce" or not Path(path).is_absolute()):
                raise ValueError(f"{name} requires CE and an absolute NPZ path.")
        if self.epochs < 1:
            raise ValueError("training.epochs must be positive.")
        if self.batch_size < 2:
            raise ValueError(
                "training.batch_size must be at least 2 for the SSR BatchNorm heads."
            )
        if self.eval_batch_size < 1:
            raise ValueError("training.eval_batch_size must be positive.")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive.")
        if self.encoder_learning_rate is not None and (
            not isfinite(self.encoder_learning_rate) or self.encoder_learning_rate <= 0.0
        ):
            raise ValueError("training.encoder_learning_rate must be positive when set.")
        if self.optimizer not in {"sgd", "adamw"}:
            raise ValueError("training.optimizer must be sgd or adamw.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("training.momentum must be in [0, 1).")
        if not isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("training.weight_decay must not be negative.")
        if not 0.0 <= self.scheduler_eta_min_ratio <= 1.0:
            raise ValueError("training.scheduler_eta_min_ratio must be in [0, 1].")
        if self.num_workers < 0 or self.eval_num_workers < 0:
            raise ValueError("training worker counts must not be negative.")
        if self.prefetch_factor < 1:
            raise ValueError("training.prefetch_factor must be positive.")
        if self.log_interval < 1:
            raise ValueError("training.log_interval must be positive.")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """실행·저장 설정."""

    device: str = "auto"  # - auto: CUDA → MPS → CPU
    deterministic: bool = False  # - False: cuDNN autotune
    output_root: Path = field(default_factory=lambda: OUTPUT_ROOT)  # - 결과 루트
    name: str = "teacher"  # - output_root/name

    def validate(self) -> None:
        if not self.output_root.is_absolute():
            raise ValueError("runtime.output_root must be an absolute path.")
        if (
            not self.name or self.name in {".", ".."}
            or Path(self.name).name != self.name or "/" in self.name or "\\" in self.name
        ):
            raise ValueError("runtime.name must be a single directory name.")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """이 설정 하나로 일반 데이터셋 실험 실행."""

    seed: int = 0  # - Python/NumPy/PyTorch 공통 seed
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssr: SSRConfig = field(default_factory=SSRConfig)
    structural_labels: StructuralLabelsConfig = field(default_factory=StructuralLabelsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_dir(self) -> Path:
        return self.runtime.output_root / self.runtime.name

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must not be negative.")
        self.data.validate()
        self.augmentation.validate()
        self.model.validate()
        self.ssr.validate()
        self.structural_labels.validate()
        self.training.validate()
        self.runtime.validate()
        if self.training.method == "ce" and self.structural_labels.enabled:
            raise ValueError("CE retraining requires structural_labels disabled.")
        corrections = (self.training.corrected_labels_path,
                       self.training.corrected_validation_labels_path,
                       self.training.corrected_test_labels_path)
        if any(path is not None for path in corrections) and not all(corrections):
            raise ValueError("Corrected retraining requires Train, Validation and Test correction files.")
        for name, split in (("corrected_validation_labels_path", self.data.validation),
                            ("corrected_test_labels_path", self.data.test)):
            if getattr(self.training, name) is not None and split is None:
                raise ValueError(f"{name} requires the corresponding dataset split.")

    def resolve_device(self) -> torch.device:
        if self.runtime.device != "auto":
            return torch.device(self.runtime.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


CONFIG = ExperimentConfig()
