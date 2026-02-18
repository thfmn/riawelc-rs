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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from riawelc.inference.predictor import WeldingDefectPredictor, segment_with_unet


@pytest.fixture
def mock_unet_model() -> MagicMock:
    model = MagicMock()
    model.name = "unet_efficientnetb0"
    model.predict.return_value = np.ones((1, 224, 224, 1), dtype=np.float32)
    return model


@pytest.fixture
def mock_classifier() -> MagicMock:
    model = MagicMock()
    model.input_shape = (None, 227, 227, 1)
    model.predict.return_value = np.array([[0.05, 0.1, 0.8, 0.05]])
    return model


class TestWeldingDefectPredictor:
    def test_predict_returns_correct_class(
        self, mock_classifier: MagicMock, sample_image_bytes: bytes
    ) -> None:
        with patch("riawelc.inference.predictor.generate_heatmap") as mock_gc:
            mock_gc.return_value = np.zeros((227, 227), dtype=np.float32)
            predictor = WeldingDefectPredictor(mock_classifier)
            result = predictor.predict(sample_image_bytes)

        assert result["class_name"] == "no_defect"
        assert result["confidence"] == pytest.approx(0.8)
        assert len(result["class_probabilities"]) == 4

    def test_predict_without_gradcam(
        self, mock_classifier: MagicMock, sample_image_bytes: bytes
    ) -> None:
        predictor = WeldingDefectPredictor(mock_classifier)
        result = predictor.predict(sample_image_bytes, gradcam=False)

        assert result["gradcam_base64"] is None
        assert result["class_name"] == "no_defect"

    def test_segment_returns_mask(
        self, mock_classifier: MagicMock, sample_image_bytes: bytes
    ) -> None:
        with patch("riawelc.inference.predictor.generate_heatmap") as mock_gc:
            mock_gc.return_value = np.ones((227, 227), dtype=np.float32)
            predictor = WeldingDefectPredictor(mock_classifier)
            result = predictor.segment(sample_image_bytes)

        assert "mask_base64" in result
        assert result["class_name"] == "no_defect"
        assert result["confidence"] == pytest.approx(0.8)


class TestSegmentWithUnet:
    def test_returns_mask_and_model_name(
        self, mock_unet_model: MagicMock, sample_image_bytes: bytes
    ) -> None:
        result = segment_with_unet(mock_unet_model, sample_image_bytes)
        assert "mask_base64" in result
        assert result["model_name"] == "unet_efficientnetb0"
        assert isinstance(result["mask_base64"], str)

    def test_binary_thresholding(
        self, mock_unet_model: MagicMock, sample_image_bytes: bytes
    ) -> None:
        # Values below 0.5 should produce a black mask
        mock_unet_model.predict.return_value = np.zeros(
            (1, 224, 224, 1), dtype=np.float32
        )
        result = segment_with_unet(mock_unet_model, sample_image_bytes)
        assert "mask_base64" in result

        # Decode and verify the mask is all zeros
        import base64
        import io

        from PIL import Image

        mask_bytes = base64.b64decode(result["mask_base64"])
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask_arr = np.array(mask_img)
        assert mask_arr.max() == 0

    def test_above_threshold_produces_white_mask(
        self, mock_unet_model: MagicMock, sample_image_bytes: bytes
    ) -> None:
        # Values above 0.5 should produce a white mask
        mock_unet_model.predict.return_value = np.ones(
            (1, 224, 224, 1), dtype=np.float32
        )
        result = segment_with_unet(mock_unet_model, sample_image_bytes)

        import base64
        import io

        from PIL import Image

        mask_bytes = base64.b64decode(result["mask_base64"])
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask_arr = np.array(mask_img)
        assert mask_arr.min() == 255
