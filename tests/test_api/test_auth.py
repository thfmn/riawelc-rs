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

from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from riawelc.api.dependencies import get_settings
from riawelc.api.main import create_app


def _make_app(api_key: str = ""):
    """Create a fresh app with the given API key setting."""
    with patch.dict("os.environ", {"RIAWELC_API_KEY": api_key}, clear=False):
        get_settings.cache_clear()
        app = create_app()
    return app


@pytest_asyncio.fixture
async def auth_client():
    """Client with API key auth enabled (key = 'test-secret')."""
    app = _make_app(api_key="test-secret")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def noauth_client():
    """Client with API key auth disabled (empty key)."""
    app = _make_app(api_key="")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_valid_key_returns_200(auth_client: AsyncClient) -> None:
    """Request with valid API key should reach the endpoint."""
    resp = await auth_client.get("/health", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_missing_key_returns_401(auth_client: AsyncClient) -> None:
    """Request without API key should be rejected with 401."""
    resp = await auth_client.get("/model/info")
    assert resp.status_code == 401
    assert "Missing API key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_wrong_key_returns_403(auth_client: AsyncClient) -> None:
    """Request with wrong API key should be rejected with 403."""
    resp = await auth_client.get("/model/info", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 403
    assert "Invalid API key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_health_exempt_from_auth(auth_client: AsyncClient) -> None:
    """Health and ready endpoints should be accessible without API key."""
    resp_health = await auth_client.get("/health")
    assert resp_health.status_code == 200

    resp_ready = await auth_client.get("/ready")
    # ready may be 503 (no model loaded) but should NOT be 401/403
    assert resp_ready.status_code not in (401, 403)


@pytest.mark.asyncio
async def test_docs_exempt_from_auth(auth_client: AsyncClient) -> None:
    """Docs and OpenAPI endpoints should be accessible without API key."""
    resp_openapi = await auth_client.get("/openapi.json")
    assert resp_openapi.status_code == 200


@pytest.mark.asyncio
async def test_auth_disabled_when_key_empty(noauth_client: AsyncClient) -> None:
    """When RIAWELC_API_KEY is empty, all endpoints are accessible without a key."""
    resp_health = await noauth_client.get("/health")
    assert resp_health.status_code == 200

    resp_openapi = await noauth_client.get("/openapi.json")
    assert resp_openapi.status_code == 200
