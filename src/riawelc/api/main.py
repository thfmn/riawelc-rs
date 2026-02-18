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

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from riawelc.api.dependencies import (
    clear_model_cache,
    clear_seg_model_cache,
    get_model,
    get_seg_baseline_model,
    get_settings,
)
from riawelc.api.middleware import configure_middleware
from riawelc.api.routes.health import router as health_router
from riawelc.api.routes.predict import router as predict_router
from riawelc.api.routes.segmentation import router as segmentation_router

logger = structlog.get_logger(__name__)

_LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _configure_logging() -> None:
    """Set up structlog with JSON output for production."""
    settings = get_settings()
    level = _LOG_LEVEL_MAP.get(settings.log_level.lower(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle for the application."""
    logger.info("startup", msg="Loading model...")
    try:
        get_model()
        logger.info("startup", msg="Classification model loaded successfully")
    except Exception:
        logger.warning("startup", msg="Classification model not found - running without model")
    try:
        get_seg_baseline_model()
        logger.info("startup", msg="Baseline segmentation model loaded successfully")
    except Exception:
        logger.warning(
            "startup",
            msg="Baseline segmentation model not found - endpoint will fail on use",
        )
    yield
    logger.info("shutdown", msg="Cleaning up resources")
    clear_model_cache()
    clear_seg_model_cache()


def _configure_otel(app: FastAPI) -> None:
    """Initialize OpenTelemetry tracing when enabled via settings."""
    settings = get_settings()
    if not settings.otel_enabled:
        return

    try:
        import os

        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        service_name = os.environ.get("OTEL_SERVICE_NAME", "riawelc-api")
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Try to use OTLP exporter if installed; fall back to no exporter.
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except ImportError:
            logger.warning(
                "otel_setup",
                msg="OTLP exporter not installed — traces will not be exported",
            )

        from opentelemetry import trace

        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("otel_setup", msg="OpenTelemetry tracing enabled")
    except Exception:
        logger.warning(
            "otel_setup",
            msg="Failed to initialize OpenTelemetry — running without tracing",
        )


def create_app() -> FastAPI:
    """Application factory for the RIAWELC API."""
    _configure_logging()

    app = FastAPI(
        title="RIAWELC — Welding Defect Classification API",
        description="Classification and weakly-supervised segmentation of welding radiographs",
        version="0.1.0",
        lifespan=lifespan,
    )

    configure_middleware(app)

    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(segmentation_router)

    _configure_otel(app)

    return app
