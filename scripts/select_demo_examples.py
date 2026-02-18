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

"""Select clear and borderline demo examples for the Gradio frontend.

Picks one high-confidence, high-IoU example per defect class, one clear
no_defect example, and one borderline Crack-vs-LP sample.  Copies the
selected images into ``demo/examples/`` with descriptive filenames and
writes a metadata JSON alongside them.

Usage:
    python scripts/select_demo_examples.py
    python scripts/select_demo_examples.py --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from riawelc.config import TrainingConfig

CLASS_NAMES = ["crack", "lack_of_penetration", "no_defect", "porosity"]
DISPLAY_NAMES = {
    "crack": "Crack",
    "lack_of_penetration": "Lack of Penetration",
    "no_defect": "No Defect",
    "porosity": "Porosity",
}

CONFIG_PATH = "configs/efficientnetb0_baseline.yaml"
MODEL_PATH = "outputs/models/checkpoints/efficientnetb0/v1/fine_tune/best.keras"
SEG_METRICS_PATH = "outputs/evaluation/segmentation/per_sample_metrics.csv"
OUTPUT_DIR = Path("demo/examples")

SEED = 42
# Minimum IoU to consider a defect example "good" for the demo
MIN_IOU = 0.70
# Number of top candidates to randomly sample from
TOP_N = 15


def collect_file_paths(test_dir: str | Path) -> list[str]:
    """Return sorted file paths matching the keras directory order."""
    test_dir = Path(test_dir)
    paths: list[str] = []
    for cls_name in CLASS_NAMES:
        cls_dir = test_dir / cls_name
        cls_paths = sorted(str(p) for p in cls_dir.glob("*.png"))
        paths.extend(cls_paths)
    return paths


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    config = TrainingConfig.from_yaml(CONFIG_PATH)
    image_size = (config.input_shape[0], config.input_shape[1])

    # ── Load segmentation metrics ────────────────────────────────────
    seg_df = pd.read_csv(SEG_METRICS_PATH)
    print(f"Loaded segmentation metrics: {len(seg_df)} samples")

    # ── Load classifier and run inference on test set ────────────────
    print(f"Loading classifier from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.data.test_dir,
        label_mode="categorical",
        color_mode="grayscale",
        image_size=image_size,
        batch_size=32,
        shuffle=False,
        seed=config.seed,
        class_names=CLASS_NAMES,
    )
    test_ds = test_ds.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    file_paths = collect_file_paths(config.data.test_dir)
    print(f"Found {len(file_paths)} test images")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds, verbose=1)

    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_pred_conf = np.max(y_pred_probs, axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict] = {}

    # ── Select one clear example per class ───────────────────────────
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        correct_mask = (y_true_labels == cls_idx) & (y_pred_labels == cls_idx)
        correct_indices = np.where(correct_mask)[0]
        confidences = y_pred_conf[correct_indices]

        if cls_name == "no_defect":
            # No segmentation metrics for no_defect; just pick high-confidence
            sorted_order = np.argsort(-confidences)
            top_indices = correct_indices[sorted_order[:TOP_N]]
            chosen_idx = int(random.choice(top_indices))
        else:
            # Cross-reference with segmentation IoU
            high_iou_files = set(
                seg_df.loc[
                    (seg_df["class"] == cls_name) & (seg_df["iou"] >= MIN_IOU),
                    "filename",
                ].tolist()
            )

            candidates = []
            for i, global_idx in enumerate(correct_indices):
                fpath = file_paths[global_idx]
                rel_name = f"{cls_name}/{Path(fpath).name}"
                if rel_name in high_iou_files:
                    candidates.append((global_idx, confidences[i]))

            if not candidates:
                # Fallback: just pick high-confidence
                print(f"  Warning: no high-IoU candidates for {cls_name}, using confidence only")
                sorted_order = np.argsort(-confidences)
                top_indices = correct_indices[sorted_order[:TOP_N]]
                chosen_idx = int(random.choice(top_indices))
            else:
                # Sort by confidence descending, take top-N, pick randomly
                candidates.sort(key=lambda x: -x[1])
                top_candidates = candidates[:TOP_N]
                chosen_idx = int(random.choice(top_candidates)[0])

        src_path = file_paths[chosen_idx]
        dst_name = f"{cls_name}.png"
        dst_path = OUTPUT_DIR / dst_name
        Image.open(src_path).convert("RGB").save(dst_path)

        conf = float(y_pred_conf[chosen_idx])
        probs = {CLASS_NAMES[j]: float(y_pred_probs[chosen_idx][j]) for j in range(4)}

        # Get IoU if available
        rel_name = f"{cls_name}/{Path(src_path).name}"
        iou_match = seg_df.loc[seg_df["filename"] == rel_name, "iou"]
        iou = float(iou_match.iloc[0]) if len(iou_match) > 0 else None
        dice_match = seg_df.loc[seg_df["filename"] == rel_name, "dice"]
        dice = float(dice_match.iloc[0]) if len(dice_match) > 0 else None

        display = DISPLAY_NAMES[cls_name]
        print(f"  {display}: {Path(src_path).name}  conf={conf:.4f}  IoU={iou}  Dice={dice}")

        metadata[dst_name] = {
            "label": display,
            "source": src_path,
            "confidence": round(conf, 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
            "iou": round(iou, 4) if iou is not None else None,
            "dice": round(dice, 4) if dice is not None else None,
        }

    # ── Select borderline Crack vs LP example ────────────────────────
    print("\nSearching for borderline Crack vs LP example...")

    crack_idx = CLASS_NAMES.index("crack")
    lp_idx = CLASS_NAMES.index("lack_of_penetration")

    borderline_candidates = []
    for cls_idx_iter in [crack_idx, lp_idx]:
        correct_mask = (y_true_labels == cls_idx_iter) & (y_pred_labels == cls_idx_iter)
        correct_indices = np.where(correct_mask)[0]

        for global_idx in correct_indices:
            probs = y_pred_probs[global_idx]
            winner_prob = probs[cls_idx_iter]
            # Check if the runner-up is the other class (crack <-> LP)
            other_idx = lp_idx if cls_idx_iter == crack_idx else crack_idx
            other_prob = probs[other_idx]

            # Runner-up must be the other class in the crack/LP pair
            sorted_prob_indices = np.argsort(-probs)
            if sorted_prob_indices[1] != other_idx:
                continue

            gap = winner_prob - other_prob
            borderline_candidates.append({
                "global_idx": int(global_idx),
                "true_cls": cls_idx_iter,
                "winner_prob": float(winner_prob),
                "other_prob": float(other_prob),
                "gap": float(gap),
            })

    # Sort by smallest gap (most borderline)
    borderline_candidates.sort(key=lambda x: x["gap"])

    if borderline_candidates:
        chosen = borderline_candidates[0]
        chosen_idx = chosen["global_idx"]
        src_path = file_paths[chosen_idx]
        dst_name = "borderline_crack_vs_lp.png"
        dst_path = OUTPUT_DIR / dst_name
        Image.open(src_path).convert("RGB").save(dst_path)

        true_name = DISPLAY_NAMES[CLASS_NAMES[chosen["true_cls"]]]
        other_name = "Lack of Penetration" if chosen["true_cls"] == crack_idx else "Crack"
        probs = {CLASS_NAMES[j]: float(y_pred_probs[chosen_idx][j]) for j in range(4)}

        print(f"  Borderline: {Path(src_path).name}")
        print(f"    True: {true_name} ({chosen['winner_prob']:.4f}) "
              f"vs {other_name} ({chosen['other_prob']:.4f})")
        print(f"    Gap: {chosen['gap']:.4f}")

        metadata[dst_name] = {
            "label": f"Borderline: {true_name} vs {other_name}",
            "source": src_path,
            "true_class": CLASS_NAMES[chosen["true_cls"]],
            "confidence": round(chosen["winner_prob"], 4),
            "runner_up_class": CLASS_NAMES[lp_idx if chosen["true_cls"] == crack_idx else crack_idx],
            "runner_up_prob": round(chosen["other_prob"], 4),
            "gap": round(chosen["gap"], 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        }
    else:
        print("  Warning: no borderline Crack vs LP example found")

    # ── Save metadata ────────────────────────────────────────────────
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExamples saved to {OUTPUT_DIR}/")
    print(f"Metadata saved to {meta_path}")
    print(f"Total examples: {len(metadata)}")


if __name__ == "__main__":
    main()
