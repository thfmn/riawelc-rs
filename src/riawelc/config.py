#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#
#  Licensed under the MIT License.
#  For details: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   RIAWELC — Welding Defect Classification & Segmentation Pipeline

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


@dataclass
class LossConfig:
    name: str = "categorical_crossentropy"


@dataclass
class CallbacksConfig:
    checkpoint_dir: str = "outputs/models/checkpoints"
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: Literal["auto", "min", "max"] = "min"
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 5
    lr_min: float = 1e-6
    early_stop_patience: int = 10
    restore_best_weights: bool = True
    mlflow_tracking: bool = True
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "riawelc_classification"


@dataclass
class ModelConfig:
    name: str = "efficientnetb0"
    input_shape: list[int] = field(default_factory=lambda: [227, 227, 1])
    num_classes: int = 4
    freeze_backbone: bool = True
    fine_tune_at: int = 120  # Start Block 5 (of 7)


@dataclass
class FineTuneConfig:
    epochs: int = 0
    learning_rate: float = 1e-4
    early_stop_patience: int = 15
    lr_reduce_patience: int = 7


@dataclass
class DataConfig:
    train_dir: str = "Dataset_partitioned/training"
    val_dir: str = "Dataset_partitioned/validation"
    test_dir: str = "Dataset_partitioned/testing"
    augmentation: bool = True
    augmentation_mode: str = "full"


@dataclass
class TrainingConfig:
    model_version: str = "v1"
    batch_size: int = 32
    epochs: int = 50
    seed: int = 9

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    fine_tune: FineTuneConfig = field(default_factory=FineTuneConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load a TrainingConfig from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        return cls(
            model_version=raw.get("model_version", "v1"),
            batch_size=raw.get("batch_size", 32),
            epochs=raw.get("epochs", 50),
            seed=raw.get("seed", 9),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            loss=LossConfig(**raw.get("loss", {})),
            callbacks=CallbacksConfig(**raw.get("callbacks", {})),
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            fine_tune=FineTuneConfig(**raw.get("fine_tune", {})),
        )

    @property
    def has_fine_tune_phase(self) -> bool:
        return self.fine_tune.epochs > 0

    @property
    def input_shape(self) -> tuple[int, int, int]:  # (H, W, C)
        return tuple(self.model.input_shape)  # type: ignore[return-value]

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.callbacks.checkpoint_dir)
