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

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from riawelc.api import dependencies
from riawelc.api.schemas import HealthResponse

router = APIRouter(tags=["health"])

API_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — always returns 200."""
    return HealthResponse(status="ok", version=API_VERSION)


@router.get("/ready")
async def ready() -> JSONResponse:
    """Readiness probe — checks whether the model is loaded."""
    if dependencies._model_cache is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "detail": "Model not loaded"},
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready"},
    )
