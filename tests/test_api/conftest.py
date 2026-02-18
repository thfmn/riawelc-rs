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

from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from riawelc.api.dependencies import get_model, get_seg_augmented_model, get_seg_baseline_model
from riawelc.api.main import create_app


@pytest.fixture
def mock_classifier() -> MagicMock:
    """Mock Keras classification model returning shape (1, 4)."""
    model = MagicMock()
    model.name = "efficientnetb0"
    model.input_shape = (None, 227, 227, 1)
    model.output_shape = (None, 4)
    model.predict.return_value = np.array([[0.1, 0.2, 0.6, 0.1]], dtype=np.float32)
    model.layers = []
    return model


@pytest.fixture
def mock_seg_model() -> MagicMock:
    """Mock U-Net segmentation model returning shape (1, 224, 224, 1)."""
    model = MagicMock()
    model.name = "unet_efficientnetb0"
    model.predict.return_value = np.ones((1, 224, 224, 1), dtype=np.float32)
    return model


@pytest.fixture
def app():
    """Create a fresh FastAPI app instance without loading real models."""
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP client wired to the ASGI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def override_models(app, mock_classifier, mock_seg_model):
    """Apply dependency overrides for all three model dependencies."""
    app.dependency_overrides[get_model] = lambda: mock_classifier
    app.dependency_overrides[get_seg_baseline_model] = lambda: mock_seg_model
    app.dependency_overrides[get_seg_augmented_model] = lambda: mock_seg_model
    yield
    app.dependency_overrides.clear()
