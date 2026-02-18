#!/usr/bin/env python3

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

"""Hugging Face Spaces wrapper for the RIAWELC Welding Defect Inspector.

This module adapts the Gradio demo for deployment on Hugging Face Spaces.
Models are loaded from paths configured via environment variables or from
bundled weights within the Space repository.

Environment variables:
    MODEL_PATH:     Path to the EfficientNetB0 classification model (.keras).
                    Default: models/classifier/best.keras
    SEG_MODEL_PATH: Path to the U-Net segmentation model (.keras).
                    Default: models/segmentation/best.keras
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["crack", "lack_of_penetration", "no_defect", "porosity"]

# Model paths — configurable via env vars for HF Spaces persistent storage
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "models/classifier/best.keras",
)
SEG_MODEL_PATH = os.environ.get(
    "SEG_MODEL_PATH",
    "models/segmentation/best.keras",
)

# --------------------------------------------------------------------------- #
# Global model handles
# --------------------------------------------------------------------------- #
_model: tf.keras.Model | None = None
_seg_model: tf.keras.Model | None = None


# --------------------------------------------------------------------------- #
# Grad-CAM utilities (inlined to avoid cross-package imports)
# --------------------------------------------------------------------------- #
def _find_target_layer(model: keras.Model) -> str:
    """Find the last Conv2D layer for Grad-CAM, including nested submodels."""
    last_conv: str | None = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = layer.name
        elif isinstance(layer, keras.Model):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, keras.layers.Conv2D):
                    last_conv = sub_layer.name
    if last_conv is None:
        msg = "No Conv2D layer found in model for Grad-CAM."
        raise ValueError(msg)
    return last_conv


def _build_grad_model(model: keras.Model, layer_name: str) -> keras.Model:
    """Build a model that outputs both target layer activations and predictions."""
    # Fast path: target layer at top level
    try:
        target = model.get_layer(layer_name)
        return keras.Model(
            inputs=model.input,
            outputs=[target.output, model.output],
        )
    except (ValueError, KeyError):
        pass

    # Find submodel containing the target layer
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
        msg = f"Layer {layer_name!r} not found in model or its submodels."
        raise ValueError(msg)

    target = backbone.get_layer(layer_name)
    backbone_ext = keras.Model(
        inputs=backbone.input,
        outputs=[target.output, backbone.output],
    )

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
        msg = f"Could not build grad model for layer {layer_name!r}."
        raise ValueError(msg)

    return keras.Model(inputs=model.input, outputs=[conv_out, x])


def generate_heatmap(model: keras.Model, image: np.ndarray) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image."""
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    image_tensor = tf.cast(image, tf.float32)
    layer_name = _find_target_layer(model)
    grad_model = _build_grad_model(model, layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        target_class = tf.argmax(predictions[0])
        class_score = predictions[:, target_class]

    grads = tape.gradient(class_score, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(
        conv_outputs * weights[:, tf.newaxis, tf.newaxis, :], axis=-1,
    )
    cam = tf.nn.relu(cam)

    input_h, input_w = image.shape[1], image.shape[2]
    cam = tf.image.resize(cam[..., tf.newaxis], (input_h, input_w))[0, :, :, 0]

    cam_max = tf.reduce_max(cam)
    if cam_max > 0:
        cam = cam / cam_max

    return cam.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original image using a jet colormap."""
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3].astype(np.float32)

    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    overlay = (1.0 - alpha) * img + alpha * heatmap_colored
    return np.clip(overlay, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Segmentation overlay
# --------------------------------------------------------------------------- #
def create_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """Overlay a predicted mask on the original image using a red tint."""
    img = original.squeeze()
    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    mask_2d = (mask.squeeze() > 0.5).astype(np.float32)

    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
    else:
        rgb = img.astype(np.float32)
    red_overlay = np.zeros_like(rgb)
    red_overlay[:, :, 0] = 255.0

    for c in range(3):
        rgb[:, :, c] = np.where(
            mask_2d > 0,
            rgb[:, :, c] * (1 - alpha) + red_overlay[:, :, c] * alpha,
            rgb[:, :, c],
        )

    return np.clip(rgb, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_models() -> None:
    """Load classification and segmentation models from configured paths."""
    global _model, _seg_model  # noqa: PLW0603

    if Path(MODEL_PATH).exists():
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded classification model from {MODEL_PATH}")
    else:
        print(f"WARNING: Classification model not found: {MODEL_PATH}")

    if SEG_MODEL_PATH and Path(SEG_MODEL_PATH).exists():
        _seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)
        print(f"Loaded segmentation model from {SEG_MODEL_PATH}")
    elif SEG_MODEL_PATH:
        print(f"WARNING: Segmentation model not found: {SEG_MODEL_PATH}")


# --------------------------------------------------------------------------- #
# Preprocessing & prediction
# --------------------------------------------------------------------------- #
def preprocess(image: np.ndarray) -> np.ndarray:
    """Preprocess uploaded image for model input."""
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    elif image.shape[-1] == 3:
        image = np.mean(image, axis=-1, keepdims=True).astype(np.float32)

    input_shape = _model.input_shape[1:3]
    img = tf.image.resize(image, input_shape).numpy()
    img = img.astype(np.float32)
    return img


def predict(image: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Run classification + Grad-CAM + optional segmentation."""
    if _model is None:
        blank = np.zeros(image.shape[:2], dtype=np.uint8)
        return {}, image, blank

    orig_h, orig_w = image.shape[:2]
    img = preprocess(image)
    batch = np.expand_dims(img, axis=0)

    preds = _model.predict(batch, verbose=0)[0]
    confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    # Grad-CAM at model resolution, resized to original, overlaid on original
    heatmap = generate_heatmap(_model, img)
    heatmap_full = tf.image.resize(
        heatmap[..., np.newaxis], (orig_h, orig_w),
    ).numpy()[:, :, 0]
    overlay = overlay_heatmap(image, heatmap_full, alpha=0.4)

    if _seg_model is not None:
        seg_input_shape = _seg_model.input_shape[1:3]
        seg_img = tf.image.resize(img, seg_input_shape).numpy()
        seg_batch = np.expand_dims(seg_img, axis=0)
        seg_pred = _seg_model.predict(seg_batch, verbose=0)[0]
        seg_mask = tf.image.resize(seg_pred, (orig_h, orig_w)).numpy()
        seg_overlay = create_overlay(image, seg_mask)
    else:
        seg_overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    return confidences, overlay, seg_overlay


# --------------------------------------------------------------------------- #
# Example loading
# --------------------------------------------------------------------------- #
def _load_examples() -> list[list[str]]:
    """Load pre-selected example images if available."""
    examples_dir = Path(__file__).parent / "examples"
    meta_path = examples_dir / "metadata.json"

    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        metadata = json.load(f)

    order = [
        "crack.png",
        "lack_of_penetration.png",
        "porosity.png",
        "no_defect.png",
        "borderline_crack_vs_lp.png",
    ]

    examples = []
    for filename in order:
        img_path = examples_dir / filename
        if img_path.exists() and filename in metadata:
            examples.append([str(img_path.resolve())])

    return examples


def _example_labels() -> list[str]:
    """Return human-readable labels for the example images."""
    examples_dir = Path(__file__).parent / "examples"
    meta_path = examples_dir / "metadata.json"

    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        metadata = json.load(f)

    order = [
        "crack.png",
        "lack_of_penetration.png",
        "porosity.png",
        "no_defect.png",
        "borderline_crack_vs_lp.png",
    ]

    labels = []
    for filename in order:
        if filename in metadata:
            labels.append(metadata[filename]["label"])

    return labels


# --------------------------------------------------------------------------- #
# Gradio interface
# --------------------------------------------------------------------------- #
def create_interface() -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    examples_dir = Path(__file__).parent / "examples"
    if examples_dir.exists():
        gr.set_static_paths(paths=[str(examples_dir.resolve())])

    with gr.Blocks(title="RIAWELC — Welding Defect Inspector") as demo:
        gr.Markdown("# RIAWELC — Welding Defect Inspector")
        gr.Markdown(
            "Upload a radiographic weld image or select an example below "
            "for defect classification, Grad-CAM localization, and segmentation.",
        )

        # Define outputs early so gr.Examples can reference them
        label_output = gr.Label(
            label="Classification", num_top_classes=4, render=False,
        )
        gradcam_output = gr.Image(label="Grad-CAM Overlay", render=False)
        seg_output = gr.Image(label="Segmentation Overlay", render=False)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Radiograph", type="numpy")
                submit_btn = gr.Button("Analyze", variant="primary")

                examples = _load_examples()
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=input_image,
                        outputs=[label_output, gradcam_output, seg_output],
                        fn=predict,
                        cache_examples=False,
                        label="Example Radiographs",
                        example_labels=_example_labels(),
                    )

            with gr.Column():
                label_output.render()
                gradcam_output.render()
                seg_output.render()

        submit_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=[label_output, gradcam_output, seg_output],
        )

        gr.Markdown("---")
        gr.Markdown(
            "**Models**: EfficientNetB0 (classification) + U-Net (segmentation) "
            "| **Dataset**: RIAWELC-RS (21,964 radiographic images)",
        )

    return demo


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    load_models()
    interface = create_interface()
    interface.launch()
else:
    # HF Spaces imports app.py and expects a Gradio interface at module level
    load_models()
    demo = create_interface()
