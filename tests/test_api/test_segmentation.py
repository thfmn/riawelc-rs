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

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_baseline_returns_200(
    client: AsyncClient, override_models, sample_image_bytes: bytes
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
    client: AsyncClient, override_models, sample_image_bytes: bytes
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
async def test_invalid_content_type_returns_415(
    client: AsyncClient, override_models
) -> None:
    response = await client.post(
        "/segment/unet/baseline",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
