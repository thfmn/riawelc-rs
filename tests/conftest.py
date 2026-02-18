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

import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> np.ndarray:
    """A single 227x227x1 grayscale image in [0, 255]."""
    rng = np.random.default_rng(9)
    return rng.integers(0, 256, size=(227, 227, 1)).astype(np.float32)


@pytest.fixture
def sample_batch(sample_image: np.ndarray) -> np.ndarray:
    """A batch of 4 images, shape (4, 227, 227, 1)."""
    return np.stack([sample_image] * 4, axis=0)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """PNG-encoded 227x227 grayscale image bytes."""
    rng = np.random.default_rng(9)
    arr = (rng.random((227, 227)) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_model() -> MagicMock:
    """A mock Keras model that returns plausible predictions."""
    model = MagicMock()
    model.input_shape = (None, 227, 227, 1)
    model.output_shape = (None, 4)
    model.name = "test_model"
    model.predict.return_value = np.array([[0.1, 0.2, 0.6, 0.1]])
    model.layers = []
    return model


@pytest.fixture
def tmp_config_yaml(tmp_path: Path) -> Path:
    """Write a minimal config YAML for testing."""
    config = tmp_path / "test_config.yaml"
    config.write_text(
        """\
model_version: "v1"
batch_size: 4
epochs: 1
seed: 9

optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

loss:
  name: categorical_crossentropy

callbacks:
  checkpoint_dir: /tmp/test_checkpoints
  save_best_only: true
  monitor: val_loss
  mode: min
  lr_reduce_factor: 0.5
  lr_reduce_patience: 5
  lr_min: 0.000001
  early_stop_patience: 10
  restore_best_weights: true
  mlflow_tracking: false
  mlflow_tracking_uri: mlruns
  mlflow_experiment_name: test

model:
  name: efficientnetb0
  input_shape: [227, 227, 1]
  num_classes: 4
  freeze_backbone: true
  fine_tune_at: 100

data:
  train_dir: Dataset_partitioned/training
  val_dir: Dataset_partitioned/validation
  test_dir: Dataset_partitioned/testing
  augmentation: true
"""
    )
    return config
