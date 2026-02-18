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
from riawelc.api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])

API_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — always returns 200."""
    return HealthResponse(status="ok", version=API_VERSION)


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> JSONResponse:
    """Readiness probe — checks whether models are loaded."""
    models: dict[str, str] = {
        "classifier": "loaded" if dependencies._model_cache is not None else "not_loaded",
        "seg_baseline": "loaded" if dependencies._seg_baseline_cache is not None else "not_loaded",
        "seg_augmented": "loaded" if dependencies._seg_augmented_cache is not None else "not_loaded",
    }

    # Classifier is required; seg models are optional (200 with warning).
    if dependencies._model_cache is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "models": models},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready", "models": models},
    )
