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
import numpy as np
import tensorflow as tf


def _resolve_layer(
    model: keras.Model,
    layer_name: str,
) -> keras.layers.Layer:
    """Resolve a layer by name, searching nested submodels if needed."""
    try:
        return model.get_layer(layer_name)
    except ValueError:
        pass
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            try:
                return layer.get_layer(layer_name)
            except ValueError:
                continue
    raise ValueError(f"Layer {layer_name!r} not found in model or its submodels.")


def find_target_layer(model: keras.Model, layer_name: str | None = None) -> str:
    """Find the target (last) 2D convolutional layer for Grad-CAM.

    If layer_name is provided, validate it exists (including in nested
    submodels). Otherwise find the last Conv2D layer.
    """
    if layer_name is not None:
        _resolve_layer(model, layer_name)  # raises ValueError if not found
        return layer_name

    # Search top-level and nested submodels for last Conv2D
    last_conv: str | None = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = layer.name
        elif isinstance(layer, keras.Model):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, keras.layers.Conv2D):
                    last_conv = sub_layer.name

    if last_conv is None:
        raise ValueError("No Conv2D layer found in model for Grad-CAM.")
    return last_conv


def _build_grad_model(
    model: keras.Model,
    layer_name: str,
) -> keras.Model:
    """Build a model that outputs both target layer activations and predictions.

    Handles target layers nested inside submodel backbones by extending
    the submodel to expose the target output, then re-chaining the head layers.
    """
    # Fast path: target layer is at the top level
    try:
        target = model.get_layer(layer_name)
        return keras.Model(
            inputs=model.input,
            outputs=[target.output, model.output],
        )
    except (ValueError, KeyError):
        pass

    # Find which submodel contains the target layer
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            try:
                layer.get_layer(layer_name)
                backbone = layer
                break
            except ValueError:
                continue

    if backbone is None:
        raise ValueError(f"Layer {layer_name!r} not found in model or its submodels.")

    target = backbone.get_layer(layer_name)

    # Create extended backbone that outputs both the target and final
    backbone_ext = keras.Model(
        inputs=backbone.input,
        outputs=[target.output, backbone.output],
    )

    # Rebuild the full model graph using the extended backbone:
    # Apply pre-backbone layers, then backbone_ext, then post-backbone layers
    x = model.input
    conv_out = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if layer is backbone:
            conv_out, x = backbone_ext(x)
            continue
        x = layer(x)

    if conv_out is None:
        raise ValueError(f"Could not build grad model for layer {layer_name!r}.")

    return keras.Model(inputs=model.input, outputs=[conv_out, x])


def generate_heatmap(
    model: keras.Model,
    image: np.ndarray,
    target_class: int | None = None,
    target_layer: str | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image.

    Supports models with nested submodel backbones (e.g. EfficientNet wrapped
    inside a classifier).

    Args:
        model: Trained Keras model.
        image: Input image as numpy array, shape (H, W, C) or (1, H, W, C).
        target_class: Class index to explain. If None, uses the predicted class.
        target_layer: Name of the conv layer to use. If None, uses the last Conv2D.

    Returns:
        Heatmap as numpy array, shape (H, W), values in [0, 1].
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)  # Adds single "batch" to Pos. 0

    image_tensor = tf.cast(image, tf.float32)
    layer_name = find_target_layer(model, target_layer)
    grad_model = _build_grad_model(model, layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if target_class is None:
            target_class = tf.argmax(predictions[0])
        class_score = predictions[:, target_class]

    grads = tape.gradient(class_score, conv_outputs)

    # Global average pooling of gradients -> channel weights
    weights = tf.reduce_mean(grads, axis=(1, 2))

    # Weighted combination of feature maps
    cam = tf.reduce_sum(conv_outputs * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)
    cam = tf.nn.relu(cam)

    # Resize to input image dimensions
    input_h, input_w = image.shape[1], image.shape[2]
    cam = tf.image.resize(cam[..., tf.newaxis], (input_h, input_w))[0, :, :, 0]

    # Normalize to [0, 1]
    cam_max = tf.reduce_max(cam)
    if cam_max > 0:
        cam = cam / cam_max

    return cam.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original image using a jet colormap.

    Args:
        image: Original image, shape (H, W) or (H, W, C), values in [0, 255] or uint8.
        heatmap: Heatmap from generate_heatmap(), shape (H, W), values in [0, 1].
        alpha: Blending factor for the heatmap overlay.

    Returns:
        Blended RGB image as float32 numpy array, shape (H, W, 3), values in [0, 1].
    """
    import matplotlib.pyplot as plt

    # Apply jet colormap (returns RGBA, take RGB)
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3].astype(np.float32)

    # Normalize image to float [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # Ensure 3-channel
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    overlay = (1.0 - alpha) * img + alpha * heatmap_colored
    return np.clip(overlay, 0.0, 1.0)


def batch_generate(
    model: keras.Model,
    images: np.ndarray,
    target_layer: str | None = None,
) -> list[np.ndarray]:
    """Generate Grad-CAM heatmaps for a batch of images.

    Args:
        model: Trained Keras model.
        images: Batch of images, shape (N, H, W, C).
        target_layer: Name of the conv layer to use.

    Returns:
        List of heatmaps, each shape (H, W) with values in [0, 1].
    """
    return [
        generate_heatmap(model, images[i], target_layer=target_layer) for i in range(len(images))
    ]
