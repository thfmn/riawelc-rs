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

import pytest
from httpx import AsyncClient

from riawelc.api import dependencies


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_response_schema(client: AsyncClient) -> None:
    response = await client.get("/health")
    data = response.json()
    assert isinstance(data["status"], str)
    assert isinstance(data["version"], str)


@pytest.mark.asyncio
async def test_ready_returns_503_when_classifier_not_loaded(client: AsyncClient) -> None:
    """Without classifier model cached, /ready should return 503."""
    original = dependencies._model_cache
    dependencies._model_cache = None
    try:
        response = await client.get("/ready")
    finally:
        dependencies._model_cache = original
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert "models" in data
    assert data["models"]["classifier"] == "not_loaded"


@pytest.mark.asyncio
async def test_ready_returns_200_when_classifier_loaded(client: AsyncClient) -> None:
    """With classifier model cached, /ready should return 200."""
    original = dependencies._model_cache
    dependencies._model_cache = MagicMock()
    try:
        response = await client.get("/ready")
    finally:
        dependencies._model_cache = original
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["models"]["classifier"] == "loaded"


@pytest.mark.asyncio
async def test_ready_reports_per_model_status(client: AsyncClient) -> None:
    """The /ready endpoint should report status for both models."""
    orig_cls = dependencies._model_cache
    orig_base = dependencies._seg_baseline_cache

    dependencies._model_cache = MagicMock()
    dependencies._seg_baseline_cache = MagicMock()
    try:
        response = await client.get("/ready")
    finally:
        dependencies._model_cache = orig_cls
        dependencies._seg_baseline_cache = orig_base

    assert response.status_code == 200
    data = response.json()
    assert data["models"]["classifier"] == "loaded"
    assert data["models"]["seg_baseline"] == "loaded"
