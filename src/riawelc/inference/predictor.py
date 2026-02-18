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

import base64
import io
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from riawelc.data import CLASS_NAMES
from riawelc.models.gradcam import generate_heatmap, overlay_heatmap

if TYPE_CHECKING:
    import keras

INPUT_SIZE: tuple[int, int] = (227, 227)
UNET_INPUT_SIZE: tuple[int, int] = (224, 224)


def segment_with_unet(model: keras.Model, image_bytes: bytes) -> dict[str, Any]:
    """Run U-Net segmentation on an image.

    Standalone function (not a method on WeldingDefectPredictor) because
    U-Net uses 224x224 input while the classifier uses 227x227.

    Args:
        model: A loaded U-Net Keras model.
        image_bytes: Raw image file bytes (PNG or JPEG).

    Returns:
        Dict with mask_base64 (PNG) and model_name.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize(UNET_INPUT_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32)
    preprocessed = arr.reshape(1, *UNET_INPUT_SIZE, 1)

    prediction = model.predict(preprocessed, verbose=0)
    mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255

    pil_mask = Image.fromarray(mask)
    buffer = io.BytesIO()
    pil_mask.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "mask_base64": mask_base64,
        "model_name": model.name,
    }


class WeldingDefectPredictor:
    """Inference pipeline for welding defect classification and segmentation."""

    def __init__(self, model: keras.Model) -> None:
        self._model = model

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Load image bytes and preprocess to model input format.

        Returns:
            Array of shape (1, 227, 227, 1), float32 in [0, 255].
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
        arr = np.array(image, dtype=np.float32)
        return arr.reshape(1, *INPUT_SIZE, 1)

    def predict(self, image_bytes: bytes, *, gradcam: bool = True) -> dict[str, Any]:
        """Run classification on an image.

        Args:
            image_bytes: Raw image file bytes (PNG or JPEG).
            gradcam: Whether to generate a Grad-CAM overlay.

        Returns:
            Dict with class_name, confidence, class_probabilities, and
            optionally gradcam_base64.
        """
        preprocessed = self._preprocess(image_bytes)
        predictions = self._model.predict(preprocessed, verbose=0)
        probs = predictions[0]

        class_idx = int(np.argmax(probs))
        class_name = CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])
        class_probabilities = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

        gradcam_base64: str | None = None
        if gradcam:
            heatmap = generate_heatmap(self._model, preprocessed, target_class=class_idx)
            overlay = overlay_heatmap(preprocessed[0], heatmap)
            gradcam_base64 = self._encode_image(overlay)

        return {
            "class_name": class_name,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "gradcam_base64": gradcam_base64,
        }

    def segment(self, image_bytes: bytes) -> dict[str, Any]:
        """Run classification and produce a thresholded Grad-CAM mask.

        Uses the Grad-CAM heatmap as a weakly-supervised segmentation proxy.

        Returns:
            Dict with mask_base64 (PNG), class_name, confidence.
        """
        preprocessed = self._preprocess(image_bytes)
        predictions = self._model.predict(preprocessed, verbose=0)
        probs = predictions[0]

        class_idx = int(np.argmax(probs))
        class_name = CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])

        heatmap = generate_heatmap(self._model, preprocessed, target_class=class_idx)

        # Threshold heatmap to binary mask
        mask = (heatmap > 0.5).astype(np.uint8) * 255
        mask_base64 = self._encode_image(mask)

        return {
            "mask_base64": mask_base64,
            "class_name": class_name,
            "confidence": confidence,
        }

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """Encode a numpy image array to base64 PNG string."""
        if image.dtype != np.uint8:
            image = np.clip(image * 255 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
