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

import math
import time
from collections import defaultdict

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from riawelc.api.dependencies import get_settings

logger = structlog.get_logger(__name__)

# Paths that are subject to rate limiting (inference endpoints).
_RATE_LIMITED_PREFIXES = ("/predict", "/segment")

# Sliding window duration in seconds.
_WINDOW_SECONDS = 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client-IP sliding window rate limiter for inference endpoints.

    Behaviour:
    - If ``settings.rate_limit`` is 0, rate limiting is disabled.
    - Only applies to paths starting with ``/predict`` or ``/segment``.
    - Tracks request timestamps per client IP in a sliding 60-second window.
    - Returns 429 with ``Retry-After`` header when the limit is exceeded.
    """

    def __init__(self, app, **kwargs):  # noqa: ANN001, ANN003
        super().__init__(app, **kwargs)
        # {client_ip: [timestamp, ...]}
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        settings = get_settings()

        # Disabled when rate_limit is 0
        if settings.rate_limit <= 0:
            return await call_next(request)

        # Only rate-limit inference endpoints
        path = request.url.path
        if not any(path.startswith(prefix) for prefix in _RATE_LIMITED_PREFIXES):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - _WINDOW_SECONDS

        # Prune old entries outside the sliding window
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > window_start]
        timestamps = self._requests[client_ip]

        if len(timestamps) >= settings.rate_limit:
            # Calculate Retry-After: seconds until the oldest request in the
            # window expires.
            retry_after = math.ceil(timestamps[0] - window_start)
            retry_after = max(retry_after, 1)
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                path=path,
                count=len(timestamps),
                limit=settings.rate_limit,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Max {settings.rate_limit} requests per minute.",
                },
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        return await call_next(request)
