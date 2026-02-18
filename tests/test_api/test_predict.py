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

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

_PREDICTOR_PATH = "riawelc.api.routes.predict.WeldingDefectPredictor"


def _make_mock_predictor(*_args, **_kwargs) -> MagicMock:
    """Return a mock WeldingDefectPredictor with plausible return values."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "class_name": "no_defect",
        "confidence": 0.6,
        "class_probabilities": {
            "crack": 0.1,
            "lack_of_penetration": 0.2,
            "no_defect": 0.6,
            "porosity": 0.1,
        },
        "gradcam_base64": "AAAA",
    }
    predictor.segment.return_value = {
        "mask_base64": "AAAA",
        "class_name": "no_defect",
        "confidence": 0.6,
    }
    return predictor


@pytest.mark.asyncio
async def test_predict_returns_valid_response(
    client: AsyncClient, override_models, sample_image_bytes: bytes
) -> None:
    """POST /predict returns a valid PredictionResponse with mocked model."""
    with patch(_PREDICTOR_PATH, side_effect=_make_mock_predictor):
        response = await client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "class_name" in data
    assert "confidence" in data
    assert "class_probabilities" in data
    assert isinstance(data["confidence"], float)
    assert isinstance(data["class_probabilities"], dict)


@pytest.mark.asyncio
async def test_segment_returns_valid_response(
    client: AsyncClient, override_models, sample_image_bytes: bytes
) -> None:
    """POST /segment returns a valid SegmentationResponse with mocked model."""
    with patch(_PREDICTOR_PATH, side_effect=_make_mock_predictor):
        response = await client.post(
            "/segment",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "mask_base64" in data
    assert "class_name" in data
    assert "confidence" in data


@pytest.mark.asyncio
async def test_predict_invalid_content_type_returns_415(
    client: AsyncClient, override_models
) -> None:
    """POST /predict with non-image content type returns 415."""
    response = await client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_predict_oversized_file_returns_413(
    client: AsyncClient, override_models
) -> None:
    """POST /predict with file exceeding max_upload_size returns 413."""
    from riawelc.api.dependencies import Settings

    small_settings = Settings(max_upload_size=100)
    with patch("riawelc.api.routes.predict.get_settings", return_value=small_settings):
        large_data = b"\x89PNG\r\n" + b"\x00" * 200
        response = await client.post(
            "/predict",
            files={"file": ("big.png", large_data, "image/png")},
        )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_concurrent_predict_requests_do_not_crash(
    client: AsyncClient, override_models, sample_image_bytes: bytes
) -> None:
    """Multiple concurrent /predict requests serialize correctly without crashing."""
    with patch(_PREDICTOR_PATH, side_effect=_make_mock_predictor):
        tasks = [
            client.post(
                "/predict",
                files={"file": ("test.png", sample_image_bytes, "image/png")},
            )
            for _ in range(5)
        ]
        responses = await asyncio.gather(*tasks)
    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert "class_name" in data


@pytest.mark.asyncio
async def test_model_info_returns_valid_response(
    client: AsyncClient, override_models
) -> None:
    """GET /model/info returns valid ModelInfoResponse."""
    response = await client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "input_shape" in data
    assert "num_classes" in data
    assert data["num_classes"] == 4
