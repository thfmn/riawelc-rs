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

import asyncio
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import keras


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    model_path: str = field(
        default_factory=lambda: os.environ.get(
            "RIAWELC_MODEL_PATH", "outputs/models/checkpoints/efficientnetb0/v1/fine_tune/best.keras"
        )
    )
    host: str = field(default_factory=lambda: os.environ.get("RIAWELC_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("RIAWELC_PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.environ.get("RIAWELC_LOG_LEVEL", "info"))
    cors_origins: list[str] = field(
        default_factory=lambda: os.environ.get(
            "RIAWELC_CORS_ORIGINS",
            "http://localhost:1420,http://localhost:8000,http://localhost:5173",
        ).split(",")
    )
    max_upload_size: int = field(
        default_factory=lambda: int(
            os.environ.get("RIAWELC_MAX_UPLOAD_SIZE", str(10 * 1024 * 1024))
        )
    )
    seg_baseline_path: str = field(
        default_factory=lambda: os.environ.get(
            "RIAWELC_SEG_BASELINE_PATH",
            "outputs/models/checkpoints/unet_efficientnetb0/v1/best.keras",
        )
    )
    seg_augmented_path: str = field(
        default_factory=lambda: os.environ.get(
            "RIAWELC_SEG_AUGMENTED_PATH",
            "outputs/models/checkpoints/unet_efficientnetb0_augmented/v1/best.keras",
        )
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("RIAWELC_API_KEY", "")
    )
    rate_limit: int = field(
        default_factory=lambda: int(os.environ.get("RIAWELC_RATE_LIMIT", "30"))
    )
    otel_enabled: bool = field(
        default_factory=lambda: os.environ.get("OTEL_ENABLED", "false").lower() == "true"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


_model_cache: keras.Model | None = None
_seg_baseline_cache: keras.Model | None = None
_seg_augmented_cache: keras.Model | None = None


def get_model() -> keras.Model:
    """Load and cache the Keras classification model."""
    global _model_cache  # noqa: PLW0603
    if _model_cache is None:
        import keras as _keras

        settings = get_settings()
        _model_cache = _keras.models.load_model(settings.model_path)
    return _model_cache


def get_seg_baseline_model() -> keras.Model:
    """Load and cache the baseline U-Net segmentation model."""
    global _seg_baseline_cache  # noqa: PLW0603
    if _seg_baseline_cache is None:
        import keras as _keras

        settings = get_settings()
        _seg_baseline_cache = _keras.models.load_model(
            settings.seg_baseline_path, compile=False
        )
    return _seg_baseline_cache


def get_seg_augmented_model() -> keras.Model:
    """Load and cache the augmented U-Net segmentation model."""
    global _seg_augmented_cache  # noqa: PLW0603
    if _seg_augmented_cache is None:
        import keras as _keras

        settings = get_settings()
        _seg_augmented_cache = _keras.models.load_model(
            settings.seg_augmented_path, compile=False
        )
    return _seg_augmented_cache


def clear_model_cache() -> None:
    """Clear the cached model (used during shutdown)."""
    global _model_cache  # noqa: PLW0603
    _model_cache = None


def clear_seg_model_cache() -> None:
    """Clear the cached segmentation models (used during shutdown)."""
    global _seg_baseline_cache, _seg_augmented_cache  # noqa: PLW0603
    _seg_baseline_cache = None
    _seg_augmented_cache = None


# --- Inference lock (API-8) ---
# Module-level lock to serialize TF/Keras inference calls.
# asyncio.Lock() is safe to create at module level in Python 3.10+
# (no longer binds to a specific event loop on creation).
_inference_lock = asyncio.Lock()


async def run_inference(fn: Any, *args: Any) -> Any:
    """Run a CPU-bound inference function under the inference lock.

    Acquires ``_inference_lock`` so that only one TensorFlow prediction
    runs at a time (TF is not thread-safe), then offloads the blocking
    call to a worker thread via ``asyncio.to_thread`` so the event loop
    is not blocked.
    """
    async with _inference_lock:
        return await asyncio.to_thread(fn, *args)
