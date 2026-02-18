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

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from riawelc.api.dependencies import get_model, get_seg_baseline_model, get_settings
from riawelc.api.main import create_app


def _make_app(rate_limit: int = 5):
    """Create a fresh app with the given rate limit setting."""
    with patch.dict(
        "os.environ",
        {"RIAWELC_RATE_LIMIT": str(rate_limit), "RIAWELC_API_KEY": ""},
        clear=False,
    ):
        get_settings.cache_clear()
        app = create_app()
    # Override model dependencies so tests don't need real model files
    app.dependency_overrides[get_model] = lambda: MagicMock(name="mock_classifier")
    app.dependency_overrides[get_seg_baseline_model] = lambda: MagicMock(name="mock_seg")
    return app


@pytest_asyncio.fixture
async def limited_client():
    """Client with rate limit = 3 requests/minute for fast testing."""
    app = _make_app(rate_limit=3)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def unlimited_client():
    """Client with rate limiting disabled (rate_limit=0)."""
    app = _make_app(rate_limit=0)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_requests_within_limit_succeed(limited_client: AsyncClient) -> None:
    """Requests within the rate limit should pass through normally."""
    # /predict will fail with 422 (no file), but NOT 429 — that's the point
    for _ in range(3):
        resp = await limited_client.post("/predict")
        assert resp.status_code != 429


@pytest.mark.asyncio
async def test_exceeding_limit_returns_429(limited_client: AsyncClient) -> None:
    """Exceeding the rate limit should return 429 with Retry-After header."""
    # Exhaust the limit (3 requests)
    for _ in range(3):
        await limited_client.post("/predict")

    # 4th request should be rate-limited
    resp = await limited_client.post("/predict")
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
    assert "Rate limit exceeded" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_rate_limit_disabled_when_zero(unlimited_client: AsyncClient) -> None:
    """When rate_limit=0, no rate limiting should occur."""
    for _ in range(10):
        resp = await unlimited_client.post("/predict")
        assert resp.status_code != 429


@pytest.mark.asyncio
async def test_health_not_rate_limited(limited_client: AsyncClient) -> None:
    """Health and ready endpoints should not be rate-limited."""
    # Exhaust the limit on inference endpoints first
    for _ in range(3):
        await limited_client.post("/predict")

    # Verify inference is now rate-limited
    resp = await limited_client.post("/predict")
    assert resp.status_code == 429

    # Health/ready should still work
    resp_health = await limited_client.get("/health")
    assert resp_health.status_code == 200

    resp_ready = await limited_client.get("/ready")
    assert resp_ready.status_code != 429
