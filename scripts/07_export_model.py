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

"""Export trained models to ONNX (and optionally TensorRT).

Usage:
    python scripts/07_export_model.py --format onnx \\
        --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras
    python scripts/07_export_model.py --format onnx \\
        --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras \\
        --validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from riawelc.models.export import (
    benchmark_inference,
    export_to_onnx,
    export_to_savedmodel,
    validate_onnx_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to ONNX or TensorRT.")
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "tensorrt"],
        required=True,
        help="Export format.",
    )
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Path to Keras model.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/exported"), help="Output directory.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate ONNX output against TF.",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(str(args.model_path))
    input_shape = model.input_shape[1:]

    savedmodel_path = args.output_dir / "savedmodel"
    print(f"Exporting to SavedModel at {savedmodel_path}")
    export_to_savedmodel(model, savedmodel_path)

    if args.format == "onnx":
        onnx_path = args.output_dir / "model.onnx"
        print(f"Converting to ONNX at {onnx_path}")
        export_to_onnx(savedmodel_path, onnx_path, opset=args.opset)
        print(f"ONNX model saved to {onnx_path}")

        if args.validate:
            print("\nValidating ONNX output against TF...")
            test_images = np.random.randint(0, 256, size=(5, *input_shape)).astype(np.float32)
            result = validate_onnx_output(savedmodel_path, onnx_path, test_images)
            print(f"  Max diff: {result['max_diff']:.2e}")
            print(f"  Mean diff: {result['mean_diff']:.2e}")
            if result["max_diff"] < 1e-5:
                print("  PASS: Output match within tolerance.")
            else:
                print("  WARNING: Output diff exceeds 1e-5 tolerance.")

        if args.benchmark:
            print("\nBenchmarking inference latency...")
            test_images = np.random.randint(0, 256, size=(10, *input_shape)).astype(np.float32)

            tf_bench = benchmark_inference(savedmodel_path, test_images, backend="tensorflow")
            tf_mean = tf_bench['mean_latency_ms']
            tf_std = tf_bench['std_latency_ms']
            print(f"  TF:   {tf_mean:.1f}ms +/- {tf_std:.1f}ms")

            onnx_bench = benchmark_inference(onnx_path, test_images, backend="onnx")
            ox_mean = onnx_bench['mean_latency_ms']
            ox_std = onnx_bench['std_latency_ms']
            print(f"  ONNX: {ox_mean:.1f}ms +/- {ox_std:.1f}ms")

    elif args.format == "tensorrt":
        print("TensorRT export requires ONNX model as input.")
        print("Run with --format onnx first, then use trtexec or TensorRT Python API.")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
