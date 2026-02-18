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

import numpy as np
import pytest

from riawelc.config import ModelConfig
from riawelc.models.gradcam import generate_heatmap, overlay_heatmap


@pytest.fixture
def trained_model():
    """Build a small classifier for Grad-CAM testing."""
    from riawelc.models.classifier import build_efficientnet_classifier

    config = ModelConfig(
        name="efficientnetb0",
        input_shape=[227, 227, 1],
        num_classes=4,
        freeze_backbone=True,
        fine_tune_at=0,
    )
    return build_efficientnet_classifier(config)


class TestGenerateHeatmap:
    def test_output_shape(self, trained_model, sample_image: np.ndarray) -> None:
        heatmap = generate_heatmap(trained_model, sample_image)
        assert heatmap.shape == (227, 227)

    def test_output_range(self, trained_model, sample_image: np.ndarray) -> None:
        heatmap = generate_heatmap(trained_model, sample_image)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_with_target_class(self, trained_model, sample_image: np.ndarray) -> None:
        heatmap = generate_heatmap(trained_model, sample_image, target_class=0)
        assert heatmap.shape == (227, 227)

    def test_batch_input(self, trained_model, sample_image: np.ndarray) -> None:
        batch = np.expand_dims(sample_image, axis=0)
        heatmap = generate_heatmap(trained_model, batch)
        assert heatmap.shape == (227, 227)


class TestOverlayHeatmap:
    def test_output_shape(self, sample_image: np.ndarray) -> None:
        heatmap = np.random.rand(227, 227).astype(np.float32)
        overlay = overlay_heatmap(sample_image, heatmap)
        assert overlay.shape == (227, 227, 3)

    def test_output_dtype(self, sample_image: np.ndarray) -> None:
        heatmap = np.random.rand(227, 227).astype(np.float32)
        overlay = overlay_heatmap(sample_image, heatmap)
        assert overlay.dtype == np.float32
