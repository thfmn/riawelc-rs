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

import uuid
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from riawelc.api.auth import APIKeyMiddleware
from riawelc.api.dependencies import get_settings
from riawelc.api.logging_middleware import LoggingMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to each request and response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return structured JSON for unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "unhandled_exception",
        request_id=request_id,
        path=request.url.path,
        error=str(exc),
        exc_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


def configure_middleware(app: FastAPI) -> None:
    """Register all middleware and exception handlers on the app."""
    settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID", "X-API-Key", "Authorization"],
    )

    app.add_middleware(RequestIDMiddleware)

    # API-6: API key authentication (runs after request-id is attached)
    app.add_middleware(APIKeyMiddleware)

    # API-11: Structured request/response logging (outermost — sees final response)
    app.add_middleware(LoggingMiddleware)

    app.add_exception_handler(Exception, global_exception_handler)  # type: ignore[arg-type]
