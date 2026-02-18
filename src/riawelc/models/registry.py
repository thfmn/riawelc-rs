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

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import keras

    from riawelc.config import ModelConfig

_REGISTRY: dict[str, dict] = {}

BuilderFn = Callable[["ModelConfig"], "keras.Model"]


def register_model(
    name: str,
    *,
    description: str = "",
) -> Callable[[BuilderFn], BuilderFn]:
    """Decorator that registers a model builder function."""

    def decorator(fn: BuilderFn) -> BuilderFn:
        _REGISTRY[name] = {
            "builder": fn,
            "description": description,
        }
        return fn

    return decorator


def build_model(name: str, config: ModelConfig) -> keras.Model:
    """Build a model by registry name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown model {name!r}. Available: {available}")
    return _REGISTRY[name]["builder"](config)


def list_models() -> list[str]:
    """Return sorted list of registered model names.
    Used in 01_train_classifier.py to resolve model names.
    """
    return sorted(_REGISTRY)


def get_model_info(name: str) -> dict:
    """Return metadata for a registered model."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown model {name!r}. Available: {available}")
    entry = _REGISTRY[name]
    return {
        "name": name,
        "description": entry["description"],
        "builder": entry["builder"].__qualname__,
    }
