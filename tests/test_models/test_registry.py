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

import pytest

from riawelc.models.registry import build_model, get_model_info, list_models


def test_list_models_includes_builtin() -> None:
    # Importing classifier registers models
    import riawelc.models.classifier  # noqa: F401

    models = list_models()
    assert "efficientnetb0" in models
    assert "resnet50v2" in models


def test_get_model_info() -> None:
    import riawelc.models.classifier  # noqa: F401

    info = get_model_info("efficientnetb0")
    assert info["name"] == "efficientnetb0"
    assert "description" in info


def test_build_unknown_model_raises() -> None:
    from riawelc.config import ModelConfig

    with pytest.raises(KeyError, match="Unknown model"):
        build_model("nonexistent_model", ModelConfig())


def test_get_model_info_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        get_model_info("nonexistent_model")
