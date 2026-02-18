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

"""Gradio demo for welding defect classification and segmentation.

Usage:
    python demo/gradio_app.py
    python demo/gradio_app.py --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from riawelc.models.gradcam import generate_heatmap, overlay_heatmap

CLASS_NAMES = ["crack", "lack_of_penetration", "no_defect", "porosity"]

_model: tf.keras.Model | None = None
_seg_model: tf.keras.Model | None = None


def load_models(model_path: str, seg_path: str | None = None) -> None:
    global _model, _seg_model
    _model = tf.keras.models.load_model(model_path)
    if seg_path and Path(seg_path).exists():
        _seg_model = tf.keras.models.load_model(seg_path, compile=False)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Preprocess uploaded image for model input."""
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    elif image.shape[-1] == 3:
        image = np.mean(image, axis=-1, keepdims=True).astype(np.float32)

    input_shape = _model.input_shape[1:3]
    img = tf.image.resize(image, input_shape).numpy()

    # Gradio delivers uint8, cast to float32 [0, 255]
    img = img.astype(np.float32)

    return img


def predict(image: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Run classification + Grad-CAM + optional segmentation."""
    if _model is None:
        blank = np.zeros(image.shape[:2], dtype=np.uint8)
        return {}, image, blank

    img = preprocess(image)
    batch = np.expand_dims(img, axis=0)

    preds = _model.predict(batch, verbose=0)[0]
    confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    heatmap = generate_heatmap(_model, img)
    overlay = overlay_heatmap(img, heatmap, alpha=0.4)

    if _seg_model is not None:
        seg_pred = _seg_model.predict(batch, verbose=0)[0]
        seg_mask = (seg_pred[:, :, 0] * 255).astype(np.uint8)
    else:
        seg_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    return confidences, overlay, seg_mask


def _load_examples() -> list[list[str]]:
    """Load pre-selected example images as file path strings.

    File paths are used so that ``is_static_file()`` can match them against
    the directory registered via ``gr.set_static_paths()``, serving images
    directly without copying to the Gradio cache.
    """
    examples_dir = Path(__file__).parent / "examples"
    meta_path = examples_dir / "metadata.json"

    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        metadata = json.load(f)

    # Fixed display order: clear classes first, then borderline
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


def create_interface() -> gr.Blocks:
    examples_dir = Path(__file__).parent / "examples"
    gr.set_static_paths(paths=[str(examples_dir.resolve())])

    with gr.Blocks(title="RIAWELC — Welding Defect Inspector") as demo:
        gr.Markdown("# RIAWELC — Welding Defect Inspector")
        gr.Markdown("Upload a radiographic weld image or select an example below for defect classification, Grad-CAM localization, and segmentation.")

        # Define outputs early (unrendered) so gr.Examples can reference them
        label_output = gr.Label(label="Classification", num_top_classes=4, render=False)
        gradcam_output = gr.Image(label="Grad-CAM Overlay", render=False)
        seg_output = gr.Image(label="Segmentation Mask", render=False)

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
                        cache_examples=True,
                        cache_mode="lazy",
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
        gr.Markdown("**Model**: EfficientNetB0 transfer learning | **Dataset**: RIAWELC (24,407 radiographic images)")

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio demo.")
    parser.add_argument("--model-path", type=str, default="outputs/models/checkpoints/efficientnetb0/v1/best.keras")
    parser.add_argument("--seg-model-path", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _patch_gradio_run_route_bug() -> None:
    """Work around Gradio 6.5.1 bug in the /run/{api_name} route handler.

    The handler passes ``route_path="/gradio_api/api/{api_name}"`` to
    ``get_request_origin()``, but the actual request URL uses ``/run/`` not
    ``/api/``.  This causes the route-path stripping to fail, leaving the
    full endpoint path in the root URL.  File-serving URLs then get the
    endpoint path prepended, producing 404s for example input images.

    The fix: monkey-patch ``get_request_origin`` so that when the standard
    stripping fails, it also tries replacing ``/api/`` with ``/run/`` in the
    route path.
    """
    from gradio import route_utils

    _original = route_utils.get_request_origin

    def _patched(request, route_path):  # type: ignore[no-untyped-def]
        result = _original(request, route_path)
        if "/api/" in route_path:
            alt_path = route_path.replace("/api/", "/run/")
            alt_result = _original(request, alt_path)
            # Use whichever is shorter — proper stripping yields a shorter URL.
            if len(str(alt_result)) < len(str(result)):
                return alt_result
        return result

    route_utils.get_request_origin = _patched


def main() -> None:
    _patch_gradio_run_route_bug()
    args = parse_args()
    load_models(args.model_path, args.seg_model_path)
    demo = create_interface()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
