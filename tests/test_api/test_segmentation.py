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

from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from riawelc.api.main import create_app


@pytest.fixture
def mock_seg_model() -> MagicMock:
    model = MagicMock()
    model.name = "unet_efficientnetb0"
    model.predict.return_value = np.ones((1, 224, 224, 1), dtype=np.float32)
    return model


@pytest.fixture
def app(mock_seg_model: MagicMock):
    application = create_app()
    # Override both segmentation model dependencies
    from riawelc.api.dependencies import get_seg_augmented_model, get_seg_baseline_model

    application.dependency_overrides[get_seg_baseline_model] = lambda: mock_seg_model
    application.dependency_overrides[get_seg_augmented_model] = lambda: mock_seg_model
    return application


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_baseline_returns_200(
    client: AsyncClient, sample_image_bytes: bytes
) -> None:
    response = await client.post(
        "/segment/unet/baseline",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "mask_base64" in data
    assert "model_name" in data


@pytest.mark.asyncio
async def test_augmented_returns_200(
    client: AsyncClient, sample_image_bytes: bytes
) -> None:
    response = await client.post(
        "/segment/unet/augmented",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "mask_base64" in data
    assert "model_name" in data


@pytest.mark.asyncio
async def test_invalid_content_type_returns_415(client: AsyncClient) -> None:
    response = await client.post(
        "/segment/unet/baseline",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
