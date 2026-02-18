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

import hmac

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from riawelc.api.dependencies import get_settings

logger = structlog.get_logger(__name__)

_EXEMPT_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces API key authentication via the X-API-Key header.

    Behaviour:
    - If ``settings.api_key`` is empty the middleware is a no-op (local dev).
    - Requests to exempt paths (health, ready, docs) are always passed through.
    - Missing ``X-API-Key`` header → 401 Unauthorized.
    - Wrong key → 403 Forbidden.
    """

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        settings = get_settings()

        # Auth disabled when api_key is empty
        if not settings.api_key:
            return await call_next(request)

        # Exempt paths (health, docs, etc.)
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")

        if api_key is None:
            logger.warning("auth_missing_key", path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing API key. Provide X-API-Key header."},
            )

        if not hmac.compare_digest(api_key, settings.api_key):
            logger.warning("auth_invalid_key", path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)
