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
from keras import layers

from riawelc.config import ModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        name="efficientnetb0",
        input_shape=[227, 227, 1],
        num_classes=4,
        freeze_backbone=True,
        fine_tune_at=120,  # Start Block 5 (of 7)
    )


class TestEfficientNetB0:
    def test_build_output_shape(self, model_config: ModelConfig) -> None:
        from riawelc.models.classifier import build_efficientnet_classifier

        model = build_efficientnet_classifier(model_config)
        assert model.output_shape == (None, 4)

    def test_build_input_shape(self, model_config: ModelConfig) -> None:
        from riawelc.models.classifier import build_efficientnet_classifier

        model = build_efficientnet_classifier(model_config)
        assert model.input_shape == (None, 227, 227, 1)

    def test_model_name(self, model_config: ModelConfig) -> None:
        from riawelc.models.classifier import build_efficientnet_classifier

        model = build_efficientnet_classifier(model_config)
        assert model.name == "efficientnetb0"


class TestUnfreezeBackbone:
    def test_unfreeze_makes_upper_layers_trainable(self) -> None:
        from riawelc.models.classifier import build_efficientnet_classifier, unfreeze_backbone

        config = ModelConfig(
            name="efficientnetb0",
            input_shape=[227, 227, 1],
            num_classes=4,
            freeze_backbone=True,
            fine_tune_at=0,  # Fully frozen backbone
        )
        model = build_efficientnet_classifier(config)
        backbone = model.layers[2]

        # Before unfreeze: all backbone layers should be non-trainable
        for layer in backbone.layers:
            assert layer.trainable is False

        # Unfreeze upper layers
        unfreeze_backbone(model, fine_tune_at=120)

        # After unfreeze: layers >= 120 should be trainable (except BatchNorm)
        for layer in backbone.layers[120:]:
            if isinstance(layer, layers.BatchNormalization):
                assert layer.trainable is False
            else:
                assert layer.trainable is True

        # Layers below fine_tune_at should remain frozen
        for layer in backbone.layers[:120]:
            assert layer.trainable is False


class TestResNet50V2:
    def test_build_output_shape(self) -> None:
        from riawelc.models.classifier import build_resnet50v2_classifier

        config = ModelConfig(
            name="resnet50v2",
            input_shape=[227, 227, 1],
            num_classes=4,
            freeze_backbone=True,
            fine_tune_at=154,  # Start conv5 (of 5)
        )
        model = build_resnet50v2_classifier(config)
        assert model.output_shape == (None, 4)

    def test_build_input_shape(self) -> None:
        from riawelc.models.classifier import build_resnet50v2_classifier

        config = ModelConfig(
            name="resnet50v2",
            input_shape=[227, 227, 1],
            num_classes=4,
            freeze_backbone=True,
            fine_tune_at=154,
        )
        model = build_resnet50v2_classifier(config)
        assert model.input_shape == (None, 227, 227, 1)
