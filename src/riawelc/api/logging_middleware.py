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

import time
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request
    from starlette.responses import Response

logger = structlog.get_logger(__name__)

# Paths that are logged at DEBUG level instead of INFO.
_QUIET_PATHS: frozenset[str] = frozenset({"/health", "/ready"})


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured request/response logging for every HTTP request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        request_id = getattr(request.state, "request_id", None)
        client_ip = request.client.host if request.client else None

        log_kwargs = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "request_id": request_id,
            "client_ip": client_ip,
            "content_type": request.headers.get("content-type"),
        }

        if request.url.path in _QUIET_PATHS:
            logger.debug("http_request", **log_kwargs)
        else:
            logger.info("http_request", **log_kwargs)

        return response
