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

from pathlib import Path

import pytest

from riawelc.config import FineTuneConfig, TrainingConfig


class TestTrainingConfig:
    def test_from_yaml(self, tmp_config_yaml: Path) -> None:
        config = TrainingConfig.from_yaml(tmp_config_yaml)
        assert config.batch_size == 4
        assert config.epochs == 1
        assert config.seed == 9
        assert config.model.name == "efficientnetb0"

    def test_from_yaml_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_yaml("/nonexistent/config.yaml")

    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.batch_size == 32
        assert config.seed == 9
        assert config.model.num_classes == 4

    def test_input_shape_property(self, tmp_config_yaml: Path) -> None:
        config = TrainingConfig.from_yaml(tmp_config_yaml)
        assert config.input_shape == (227, 227, 1)

    def test_checkpoint_dir_property(self, tmp_config_yaml: Path) -> None:
        config = TrainingConfig.from_yaml(tmp_config_yaml)
        assert config.checkpoint_dir == Path("/tmp/test_checkpoints")

    def test_optimizer_config(self, tmp_config_yaml: Path) -> None:
        config = TrainingConfig.from_yaml(tmp_config_yaml)
        assert config.optimizer.name == "adamw"
        assert config.optimizer.learning_rate == 0.001

    def test_loss_config(self, tmp_config_yaml: Path) -> None:
        config = TrainingConfig.from_yaml(tmp_config_yaml)
        assert config.loss.name == "categorical_crossentropy"


class TestFineTuneConfig:
    def test_fine_tune_config_defaults(self) -> None:
        ft = FineTuneConfig()
        assert ft.epochs == 0
        assert ft.learning_rate == 1e-4
        assert ft.early_stop_patience == 15
        assert ft.lr_reduce_patience == 7

    def test_has_fine_tune_phase_false_by_default(self) -> None:
        config = TrainingConfig()
        assert config.has_fine_tune_phase is False

    def test_fine_tune_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "ft_config.yaml"
        yaml_path.write_text(
            """\
model_version: "v1"
batch_size: 32
epochs: 15
seed: 9

fine_tune:
  epochs: 25
  learning_rate: 0.0001
  early_stop_patience: 15
  lr_reduce_patience: 7

model:
  name: efficientnetb0
  input_shape: [227, 227, 1]
  num_classes: 4
  freeze_backbone: true
  fine_tune_at: 120
"""
        )
        config = TrainingConfig.from_yaml(yaml_path)
        assert config.has_fine_tune_phase is True
        assert config.fine_tune.epochs == 25
        assert config.fine_tune.learning_rate == 0.0001
        assert config.fine_tune.early_stop_patience == 15
        assert config.fine_tune.lr_reduce_patience == 7

    def test_backward_compat_no_fine_tune(self, tmp_config_yaml: Path) -> None:
        config = TrainingConfig.from_yaml(tmp_config_yaml)
        assert config.has_fine_tune_phase is False
        assert config.fine_tune.epochs == 0
