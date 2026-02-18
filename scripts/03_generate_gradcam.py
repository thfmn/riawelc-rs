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

"""Generate Grad-CAM heatmaps for test set images.

Usage:
    python scripts/03_generate_gradcam.py --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras --config configs/gradcam.yaml
    python scripts/03_generate_gradcam.py --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras --num-samples 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from riawelc.models.gradcam import generate_heatmap, overlay_heatmap

CLASS_NAMES = ["crack", "lack_of_penetration", "no_defect", "porosity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model.")
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset_partitioned/testing"), help="Test images dir.")
    parser.add_argument("--config", type=str, default="configs/gradcam.yaml", help="Grad-CAM config YAML.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradcam"), help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Max samples per class (default: all).")
    parser.add_argument("--target-layer", type=str, default=None, help="Override target conv layer name.")
    return parser.parse_args()


def load_gradcam_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    args = parse_args()
    gc_config = load_gradcam_config(args.config)

    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(str(args.model_path))

    target_layer = args.target_layer
    if target_layer is None and "target_layers" in gc_config:
        for layer_name in gc_config["target_layers"].values():
            try:
                model.get_layer(layer_name)
                target_layer = layer_name
                break
            except ValueError:
                continue

    overlay_dir = args.output_dir / "overlays"
    heatmap_dir = args.output_dir / "heatmaps"
    raw_heatmap_dir = args.output_dir / "heatmaps_raw"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    raw_heatmap_dir.mkdir(parents=True, exist_ok=True)

    alpha = gc_config.get("output", {}).get("alpha", 0.4)
    total_generated = 0

    for class_dir in sorted(args.data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        image_paths = sorted(class_dir.glob("*.png"))

        if args.num_samples is not None:
            image_paths = image_paths[: args.num_samples]

        print(f"\nProcessing {class_name}: {len(image_paths)} images")

        (overlay_dir / class_name).mkdir(parents=True, exist_ok=True)
        (heatmap_dir / class_name).mkdir(parents=True, exist_ok=True)
        (raw_heatmap_dir / class_name).mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            img = tf.keras.utils.load_img(str(img_path), color_mode="grayscale")
            img_array = tf.keras.utils.img_to_array(img)

            input_shape = model.input_shape[1:3]
            if img_array.shape[:2] != tuple(input_shape):
                img_array = tf.image.resize(img_array, input_shape).numpy()

            heatmap = generate_heatmap(model, img_array, target_layer=target_layer)
            overlay = overlay_heatmap(img_array, heatmap, alpha=alpha)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_array[:, :, 0].astype(np.uint8), cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Heatmap")
            axes[1].axis("off")
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
            plt.tight_layout()
            fig.savefig(overlay_dir / class_name / img_path.name, dpi=100)
            plt.close(fig)

            plt.imsave(str(heatmap_dir / class_name / img_path.name), heatmap, cmap="jet")

            raw_path = raw_heatmap_dir / class_name / img_path.name
            Image.fromarray((heatmap * 255).astype(np.uint8), mode="L").save(raw_path)

            total_generated += 1

    print(f"\nDone. Generated {total_generated} heatmaps in {args.output_dir}")


if __name__ == "__main__":
    main()
