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

from typing import TYPE_CHECKING

import keras
from keras import layers

from riawelc.models.registry import register_model

if TYPE_CHECKING:
    from riawelc.config import ModelConfig

SEED = 9


def _grayscale_to_rgb(inputs: keras.KerasTensor) -> keras.KerasTensor:
    """Expand single-channel input to 3-channel using a 1x1 convolution."""
    return layers.Conv2D(
        filters=3,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="ones",
        trainable=False,
        name="grayscale_to_rgb",
    )(inputs)


def _freeze_backbone(
    backbone: keras.Model,
    *,
    freeze: bool,
    fine_tune_at: int,
) -> None:
    """Freeze or partially freeze backbone layers."""
    if not freeze:
        backbone.trainable = True
        return

    backbone.trainable = False
    if fine_tune_at > 0:
        for layer in backbone.layers[fine_tune_at:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True


def unfreeze_backbone(model: keras.Model, fine_tune_at: int) -> None:
    """Unfreeze upper backbone layers for fine-tuning phase."""
    backbone = model.layers[2]  # Input -> grayscale_to_rgb -> backbone
    _freeze_backbone(backbone, freeze=True, fine_tune_at=fine_tune_at)


@register_model(
    "efficientnetb0",
    description="EfficientNetB0 with ImageNet weights for grayscale classification",
)
def build_efficientnet_classifier(config: ModelConfig) -> keras.Model:
    """Build an EfficientNetB0 classifier using the Functional API."""
    h, w, c = config.input_shape
    inputs = layers.Input(shape=(h, w, c), name="input_image")

    x = _grayscale_to_rgb(inputs) if c == 1 else inputs

    backbone = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(h, w, 3),
    )
    _freeze_backbone(
        backbone,
        freeze=config.freeze_backbone,
        fine_tune_at=config.fine_tune_at,
    )

    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, seed=SEED, name="dropout")(x)
    outputs = layers.Dense(config.num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0")


@register_model(
    "resnet50v2",
    description="ResNet50V2 with ImageNet weights for grayscale classification",
)
def build_resnet50v2_classifier(config: ModelConfig) -> keras.Model:
    """Build a ResNet50V2 classifier using the Functional API."""
    h, w, c = config.input_shape
    inputs = layers.Input(shape=(h, w, c), name="input_image")

    x = _grayscale_to_rgb(inputs) if c == 1 else inputs
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="resnet_preprocess")(x)

    backbone = keras.applications.ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(h, w, 3),
    )
    _freeze_backbone(
        backbone,
        freeze=config.freeze_backbone,
        fine_tune_at=config.fine_tune_at,
    )

    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, seed=SEED, name="dropout")(x)
    outputs = layers.Dense(config.num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="resnet50v2")
