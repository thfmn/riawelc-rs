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

"""Evaluate segmentation model: IoU, Dice, per-class breakdown, visual comparison grid.

Produces 5 artifacts:
  - segmentation_grid.png    8-row × 4-col grid (Original, GT, Predicted, Overlay)
  - per_sample_metrics.csv   filename, class, iou, dice for every test sample
  - per_class_metrics.csv    per-class and overall IoU/Dice mean, std, count
  - summary_statistics.csv   mean, std, median, min, max per class per metric
  - iou_histogram.png        IoU distribution with mean line

Usage:
    python scripts/06_evaluate_segmentation.py \
        --model-path outputs/models/checkpoints/unet_efficientnetb0/v1/best.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from riawelc.models.segmentation import compute_iou, dice_coefficient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation model.")
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to segmentation model."
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("Dataset_partitioned/testing")
    )
    parser.add_argument(
        "--mask-dir", type=Path, default=Path("outputs/pseudomasks")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: derived from model path).",
    )
    parser.add_argument(
        "--num-visualize",
        type=int,
        default=8,
        help="Number of samples for visual grid.",
    )
    return parser.parse_args()


def create_overlay(original: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a predicted mask on the original image using a red tint.

    Args:
        original: Grayscale image array, shape (H, W) or (H, W, 1), uint8 range.
        mask: Binary mask array, shape (H, W) or (H, W, 1), values in [0, 1].
        alpha: Blend strength of the mask overlay.

    Returns:
        RGB image array (H, W, 3) with red-tinted mask overlay, uint8.
    """
    img = original.squeeze()
    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    mask_2d = (mask.squeeze() > 0.5).astype(np.float32)

    rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
    red_overlay = np.zeros_like(rgb)
    red_overlay[:, :, 0] = 255.0

    for c in range(3):
        rgb[:, :, c] = np.where(
            mask_2d > 0,
            rgb[:, :, c] * (1 - alpha) + red_overlay[:, :, c] * alpha,
            rgb[:, :, c],
        )

    return np.clip(rgb, 0, 255).astype(np.uint8)


def select_diverse_samples(
    df: pd.DataFrame, num_samples: int = 8
) -> list[int]:
    """Select diverse samples: best, worst, and median IoU per class.

    Picks up to 3 samples per class (best, worst, median IoU), then trims
    to ``num_samples``. Returns indices into the original DataFrame.
    """
    selected: list[int] = []
    for _cls, group in df.groupby("class"):
        sorted_group = group.sort_values("iou")
        indices = sorted_group.index.tolist()
        if len(indices) >= 3:
            selected.append(indices[0])  # worst
            selected.append(indices[len(indices) // 2])  # median
            selected.append(indices[-1])  # best
        elif len(indices) in (1, 2):
            selected.extend(indices)

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)

    return unique[:num_samples]


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Derive output directory from model path if not explicitly set.

    E.g. ``...checkpoints/unet_efficientnetb0/v1/best.keras``
    → ``outputs/evaluation/segmentation/unet_efficientnetb0``
    """
    if args.output_dir is not None:
        return args.output_dir
    # Walk up from best.keras → v1 → model_name
    model_name = args.model_path.parent.parent.name
    return Path("outputs/evaluation/segmentation") / model_name


def main() -> None:
    args = parse_args()
    args.output_dir = _resolve_output_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(str(args.model_path), compile=False)
    input_shape = model.input_shape[1:3]

    # --- Load data ---
    images, masks, names, classes = [], [], [], []
    for class_dir in sorted(args.test_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name == "no_defect":
            continue
        mask_class_dir = args.mask_dir / class_dir.name
        if not mask_class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*.png")):
            mask_path = mask_class_dir / img_path.name
            if not mask_path.exists():
                continue

            img = tf.keras.utils.load_img(
                str(img_path), color_mode="grayscale", target_size=input_shape
            )
            img_arr = tf.keras.utils.img_to_array(img)

            mask = tf.keras.utils.load_img(
                str(mask_path), color_mode="grayscale", target_size=input_shape
            )
            mask_arr = tf.keras.utils.img_to_array(mask) / 255.0

            images.append(img_arr)
            masks.append(mask_arr)
            names.append(f"{class_dir.name}/{img_path.name}")
            classes.append(class_dir.name)

    if not images:
        print("No test image-mask pairs found.")
        sys.exit(1)

    images_np = np.array(images)
    masks_np = np.array(masks)

    print(f"Loaded {len(images)} image-mask pairs across {len(set(classes))} classes")

    # --- Predict ---
    preds = model.predict(images_np, verbose=1)

    # --- Per-sample metrics ---
    records = []
    for i in range(len(images)):
        iou = compute_iou(masks_np[i], preds[i])
        dice = float(dice_coefficient(masks_np[i], preds[i]).numpy())
        records.append({
            "filename": names[i],
            "class": classes[i],
            "iou": round(iou, 4),
            "dice": round(dice, 4),
        })

    df = pd.DataFrame(records)

    # --- Export CSVs ---
    # 1. Per-sample metrics
    df.to_csv(args.output_dir / "per_sample_metrics.csv", index=False)
    print(f"\nPer-sample metrics saved ({len(df)} samples)")

    # 2. Per-class metrics
    class_agg = df.groupby("class").agg(
        iou_mean=("iou", "mean"),
        iou_std=("iou", "std"),
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        count=("iou", "count"),
    )
    overall = pd.DataFrame(
        {
            "iou_mean": [df["iou"].mean()],
            "iou_std": [df["iou"].std()],
            "dice_mean": [df["dice"].mean()],
            "dice_std": [df["dice"].std()],
            "count": [len(df)],
        },
        index=pd.Index(["overall"], name="class"),
    )
    per_class = pd.concat([class_agg, overall])
    per_class = per_class.round(4)
    per_class.to_csv(args.output_dir / "per_class_metrics.csv")
    print("Per-class metrics saved")

    # 3. Summary statistics
    summary_rows = []
    for cls in sorted(df["class"].unique()):
        subset = df[df["class"] == cls]
        for metric in ["iou", "dice"]:
            summary_rows.append({
                "class": cls,
                "metric": metric,
                "mean": round(subset[metric].mean(), 4),
                "std": round(subset[metric].std(), 4),
                "median": round(subset[metric].median(), 4),
                "min": round(subset[metric].min(), 4),
                "max": round(subset[metric].max(), 4),
            })
    for metric in ["iou", "dice"]:
        summary_rows.append({
            "class": "overall",
            "metric": metric,
            "mean": round(df[metric].mean(), 4),
            "std": round(df[metric].std(), 4),
            "median": round(df[metric].median(), 4),
            "min": round(df[metric].min(), 4),
            "max": round(df[metric].max(), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "summary_statistics.csv", index=False)
    print("Summary statistics saved")

    # --- Print results ---
    print(f"\nMean IoU:  {df['iou'].mean():.4f} (+/- {df['iou'].std():.4f})")
    print(f"Mean Dice: {df['dice'].mean():.4f} (+/- {df['dice'].std():.4f})")
    print("\nPer-class breakdown:")
    print(per_class.to_string())

    # --- IoU histogram ---
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(df["iou"], bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    mean_iou = df["iou"].mean()
    ax_hist.axvline(
        mean_iou, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_iou:.3f}"
    )
    ax_hist.set_xlabel("IoU")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("IoU Distribution Across Test Samples")
    ax_hist.legend()
    plt.tight_layout()
    fig_hist.savefig(args.output_dir / "iou_histogram.png", dpi=150)
    plt.close(fig_hist)
    print(f"\nIoU histogram saved to {args.output_dir / 'iou_histogram.png'}")

    # --- Visual grid (4 columns: Original, GT, Predicted, Overlay) ---
    vis_indices = select_diverse_samples(df, num_samples=args.num_visualize)
    n_vis = len(vis_indices)

    fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
    if n_vis == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(vis_indices):
        original = images_np[idx, :, :, 0]
        gt_mask = masks_np[idx, :, :, 0]
        pred_mask = preds[idx, :, :, 0]
        overlay = create_overlay(images_np[idx], preds[idx])

        axes[row, 0].imshow(original.astype(np.uint8), cmap="gray")
        axes[row, 0].set_title(f"{df.iloc[idx]['class']}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_mask, cmap="gray")
        axes[row, 1].set_title("Ground Truth", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_mask, cmap="gray")
        axes[row, 2].set_title(f"Predicted (IoU={df.iloc[idx]['iou']:.3f})", fontsize=9)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Overlay", fontsize=9)
        axes[row, 3].axis("off")

    plt.tight_layout()
    fig.savefig(args.output_dir / "segmentation_grid.png", dpi=150)
    plt.close(fig)
    print(f"Visual grid saved to {args.output_dir / 'segmentation_grid.png'}")


if __name__ == "__main__":
    main()
