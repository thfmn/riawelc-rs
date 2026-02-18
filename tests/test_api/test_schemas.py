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

from riawelc.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SegmentationResponse,
    UNetSegmentationResponse,
)


class TestPredictionResponse:
    def test_valid(self) -> None:
        resp = PredictionResponse(
            class_name="crack",
            confidence=0.95,
            class_probabilities={
                "crack": 0.95,
                "porosity": 0.03,
                "no_defect": 0.01,
                "lack_of_penetration": 0.01,
            },
        )
        assert resp.class_name == "crack"
        assert resp.gradcam_base64 is None

    def test_with_gradcam(self) -> None:
        resp = PredictionResponse(
            class_name="porosity",
            confidence=0.8,
            class_probabilities={"porosity": 0.8},
            gradcam_base64="base64data",
        )
        assert resp.gradcam_base64 == "base64data"


class TestSegmentationResponse:
    def test_valid(self) -> None:
        resp = SegmentationResponse(
            mask_base64="maskdata",
            class_name="crack",
            confidence=0.9,
        )
        assert resp.mask_base64 == "maskdata"


class TestUNetSegmentationResponse:
    def test_valid(self) -> None:
        resp = UNetSegmentationResponse(
            mask_base64="maskdata",
            model_name="unet_efficientnetb0",
        )
        assert resp.mask_base64 == "maskdata"
        assert resp.model_name == "unet_efficientnetb0"


class TestHealthResponse:
    def test_valid(self) -> None:
        resp = HealthResponse(status="ok", version="0.1.0")
        assert resp.status == "ok"


class TestModelInfoResponse:
    def test_valid(self) -> None:
        resp = ModelInfoResponse(
            model_name="efficientnetb0",
            input_shape=[227, 227, 1],
            num_classes=4,
            description="Test model",
        )
        assert resp.num_classes == 4
