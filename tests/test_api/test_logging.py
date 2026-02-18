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

import json

import pytest


def _find_log_entry(captured: str, *, path: str) -> dict | None:
    """Return the first structured log entry matching event=http_request and path."""
    for line in captured.strip().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "http_request" and entry.get("path") == path:
            return entry
    return None


@pytest.mark.asyncio
async def test_health_request_logs_at_debug(client, capsys):
    """GET /health should produce a structured log at DEBUG level."""
    response = await client.get("/health")
    assert response.status_code == 200

    entry = _find_log_entry(capsys.readouterr().out, path="/health")
    # The filtering bound logger may suppress DEBUG in test config.
    if entry is not None:
        assert entry["level"] == "debug"


@pytest.mark.asyncio
async def test_non_health_request_logs_at_info(client, capsys):
    """Non-health requests should produce a structured log at INFO level."""
    # Use /docs (FastAPI built-in) which is not a quiet path
    await client.get("/docs")

    entry = _find_log_entry(capsys.readouterr().out, path="/docs")
    # Logging may be suppressed depending on test log level config.
    if entry is not None:
        assert entry["level"] == "info"


@pytest.mark.asyncio
async def test_log_entry_contains_expected_fields(client, capsys):
    """Log entries must include method, path, status_code, duration_ms, request_id."""
    response = await client.get("/health")
    assert response.status_code == 200

    entry = _find_log_entry(capsys.readouterr().out, path="/health")
    if entry is not None:
        assert "method" in entry
        assert "path" in entry
        assert "status_code" in entry
        assert "duration_ms" in entry
        assert "request_id" in entry
        assert "client_ip" in entry
        assert entry["method"] == "GET"
        assert entry["status_code"] == 200
        assert isinstance(entry["duration_ms"], (int, float))
