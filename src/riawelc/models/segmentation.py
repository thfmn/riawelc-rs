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

import keras
import tensorflow as tf
from keras import layers


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Compute Dice coefficient between predictions and ground truth."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    return (2.0 * intersection + smooth) / denom


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice loss: 1 - Dice coefficient."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def dice_bce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combined Dice + Binary Cross-Entropy loss."""
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_loss(y_true, y_pred) + tf.reduce_mean(bce)


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


def _decoder_block(
    x: keras.KerasTensor,
    skip: keras.KerasTensor | None,
    filters: int,
    stage: int,
) -> keras.KerasTensor:
    """Decoder block: UpSampling2D + optional skip concatenation + 2x Conv2D-BN-ReLU."""
    x = layers.UpSampling2D(size=(2, 2), name=f"decoder{stage}_upsample")(x)

    if skip is not None:
        x = layers.Concatenate(name=f"decoder{stage}_concat")([x, skip])

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"decoder{stage}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"decoder{stage}_bn1")(x)
    x = layers.ReLU(name=f"decoder{stage}_relu1")(x)

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"decoder{stage}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"decoder{stage}_bn2")(x)
    x = layers.ReLU(name=f"decoder{stage}_relu2")(x)

    return x


# EfficientNetB0 layer names for skip connections (with 224x224 input):
#   block1a_project_bn  -> 112x112, 16 ch
#   block2b_add         -> 56x56,   24 ch
#   block3b_add         -> 28x28,   40 ch
#   block5c_add         -> 14x14,  112 ch
#   top_activation      -> 7x7,   1280 ch  (bottleneck)
_SKIP_LAYER_NAMES = [
    "block1a_project_bn",  # 112x112
    "block2b_add",         # 56x56
    "block3b_add",         # 28x28
    "block5c_add",         # 14x14
]
_BOTTLENECK_LAYER = "top_activation"  # 7x7
_DECODER_FILTERS = [256, 128, 64, 32, 16]


def build_unet_segmentation(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = 1,
    activation: str = "sigmoid",
) -> keras.Model:
    """Build a U-Net segmentation model with EfficientNetB0 encoder (native Keras 3).

    Uses ``keras.applications.EfficientNetB0`` as encoder and extracts skip
    connections at 4 intermediate resolutions.  A lightweight decoder
    upsamples back to the input resolution.

    For single-channel (grayscale) inputs the model prepends a frozen 1x1
    convolution that replicates the channel to RGB so the ImageNet-pretrained
    encoder receives 3-channel data.
    """
    h, w, c = input_shape
    inp = layers.Input(shape=input_shape, name="grayscale_input")

    # Grayscale -> RGB for ImageNet backbone
    x = _grayscale_to_rgb(inp) if c == 1 else inp

    # --- Encoder ---
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(h, w, 3),
    )

    # Build a multi-output model to extract skip features + bottleneck
    skip_outputs = [backbone.get_layer(name).output for name in _SKIP_LAYER_NAMES]
    bottleneck_output = backbone.get_layer(_BOTTLENECK_LAYER).output

    encoder = keras.Model(
        inputs=backbone.input,
        outputs=skip_outputs + [bottleneck_output],
        name="efficientnetb0_encoder",
    )

    features = encoder(x)  # list: [skip0, skip1, skip2, skip3, bottleneck]
    skips = features[:-1]  # 4 skip connections (112, 56, 28, 14)
    x = features[-1]       # bottleneck (7x7)

    # --- Decoder ---
    # Stage 1: 7 -> 14, skip from block5c_add (14x14)
    x = _decoder_block(x, skips[3], _DECODER_FILTERS[0], stage=1)
    # Stage 2: 14 -> 28, skip from block3b_add (28x28)
    x = _decoder_block(x, skips[2], _DECODER_FILTERS[1], stage=2)
    # Stage 3: 28 -> 56, skip from block2b_add (56x56)
    x = _decoder_block(x, skips[1], _DECODER_FILTERS[2], stage=3)
    # Stage 4: 56 -> 112, skip from block1a_project_bn (112x112)
    x = _decoder_block(x, skips[0], _DECODER_FILTERS[3], stage=4)
    # Stage 5: 112 -> 224, no skip (backbone has no 224x224 feature map)
    x = _decoder_block(x, None, _DECODER_FILTERS[4], stage=5)

    # --- Output ---
    output = layers.Conv2D(
        num_classes, 1, activation=activation, name="segmentation_output"
    )(x)

    return keras.Model(inputs=inp, outputs=output, name="unet_efficientnetb0")


def compute_iou(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> float:
    """Compute Intersection over Union for binary masks."""
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred_bin, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    iou = (intersection + 1e-7) / (union + 1e-7)
    return float(iou.numpy())
