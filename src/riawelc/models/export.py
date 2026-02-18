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

import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


def export_to_savedmodel(model: keras.Model, path: str | Path) -> Path:
    """Export a Keras model to SavedModel format."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def export_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    """Convert a SavedModel to ONNX format using tf2onnx."""
    import tf2onnx

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(str(model_path))
    input_signature = [
        tf.TensorSpec(model.input_shape, tf.float32, name="input"),
    ]

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(output_path),
    )

    return output_path


def validate_onnx_output(
    tf_model_path: str | Path,
    onnx_path: str | Path,
    test_images: np.ndarray,
) -> dict[str, float]:
    """Compare ONNX model output against TF model on the same inputs.

    Returns dict with max_diff and mean_diff. Asserts max_diff < 1e-5.
    """
    import onnxruntime as ort

    tf_model = keras.models.load_model(str(tf_model_path))
    tf_preds = tf_model.predict(test_images, verbose=0)

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_preds = session.run(None, {input_name: test_images.astype(np.float32)})[0]

    max_diff = float(np.max(np.abs(tf_preds - onnx_preds)))
    mean_diff = float(np.mean(np.abs(tf_preds - onnx_preds)))

    return {"max_diff": max_diff, "mean_diff": mean_diff}


def benchmark_inference(
    model_path: str | Path,
    test_images: np.ndarray,
    n_runs: int = 100,
    backend: str = "tensorflow",
) -> dict[str, float]:
    """Benchmark inference latency.

    Args:
        model_path: Path to model (SavedModel dir or ONNX file).
        test_images: Test images to run inference on.
        n_runs: Number of inference runs for timing.
        backend: "tensorflow" or "onnx".

    Returns:
        Dict with mean_latency_ms and std_latency_ms.
    """
    if backend == "onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name

        def predict_fn(imgs: np.ndarray) -> list:
            return session.run(None, {input_name: imgs.astype(np.float32)})
    else:
        model = keras.models.load_model(str(model_path))

        def predict_fn(imgs: np.ndarray) -> np.ndarray:
            return model.predict(imgs, verbose=0)

    # Warmup
    predict_fn(test_images[:1])

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(test_images)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "n_runs": n_runs,
        "n_images": len(test_images),
        "backend": backend,
    }
