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

"""Generate pseudo-masks from Grad-CAM heatmaps via thresholding and morphological cleanup.

Usage:
    python scripts/04_generate_pseudomasks.py
    python scripts/04_generate_pseudomasks.py --heatmap-dir outputs/gradcam/heatmaps --threshold 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-masks from Grad-CAM heatmaps.")
    parser.add_argument("--heatmap-dir", type=Path, default=Path("outputs/gradcam/heatmaps_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pseudomasks_v2"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (0-1).")
    parser.add_argument("--kernel-size", type=int, default=5, help="Morphological kernel size.")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum connected component area in pixels.")
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True, help="Fill holes in binary masks.")
    return parser.parse_args()


def create_pseudomask(
    heatmap_path: Path,
    threshold: float,
    kernel_size: int,
    min_area: int,
    fill_holes: bool = False,
) -> np.ndarray:
    """Convert a heatmap image to a binary pseudo-mask."""
    heatmap_img = np.array(Image.open(heatmap_path).convert("L"))
    heatmap_norm = heatmap_img.astype(np.float32) / 255.0

    binary = (heatmap_norm >= threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if fill_holes:
        # Pad with a 1-pixel background border so (0,0) is guaranteed to be
        # background, even when the foreground touches the image edge.
        h, w = binary.shape
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = binary
        flood_mask = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(padded, flood_mask, (0, 0), 1)
        # After flood: exterior background → 1, foreground → 1, holes → 0
        holes = (padded[1:-1, 1:-1] == 0).astype(np.uint8)
        binary = binary | holes

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 1

    return mask * 255


def main() -> None:
    args = parse_args()
    total = 0

    for class_dir in sorted(args.heatmap_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        out_class_dir = args.output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for heatmap_path in sorted(class_dir.glob("*.png")):
            mask = create_pseudomask(heatmap_path, args.threshold, args.kernel_size, args.min_area, args.fill_holes)
            Image.fromarray(mask).save(out_class_dir / heatmap_path.name)
            total += 1

    print(f"Generated {total} pseudo-masks in {args.output_dir}")


if __name__ == "__main__":
    main()
