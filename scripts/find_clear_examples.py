#!/usr/bin/env python3
"""Find clearly and correctly classified test-set examples for each category.

Loads the trained EfficientNetB0 model, runs inference on the test set,
and selects the top-N highest-confidence correct predictions per class.
Saves a visualization grid and prints file paths.

Usage:
    python scripts/find_clear_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
OUTPUT_DIR = Path("outputs/evaluation/clear_examples")
TOP_N = 5  # number of examples per class


def collect_file_paths(test_dir: str | Path) -> list[str]:
    """Return sorted file paths matching the order keras uses internally."""
    test_dir = Path(test_dir)
    paths: list[str] = []
    for cls_name in CLASS_NAMES:
        cls_dir = test_dir / cls_name
        cls_paths = sorted(str(p) for p in cls_dir.glob("*.png"))
        paths.extend(cls_paths)
    return paths


def main() -> None:
    config = TrainingConfig.from_yaml(CONFIG_PATH)
    image_size = (config.input_shape[0], config.input_shape[1])

    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load test dataset WITHOUT shuffling, matching keras directory order
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

    # Collect file paths in same order as keras
    file_paths = collect_file_paths(config.data.test_dir)
    print(f"Found {len(file_paths)} test images")

    # Get predictions
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds, verbose=1)

    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_pred_conf = np.max(y_pred_probs, axis=1)

    # For each class, find correctly classified examples sorted by confidence
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(CLASS_NAMES), TOP_N + 1,
        figsize=(3 * TOP_N + 2, 3.2 * len(CLASS_NAMES)),
        gridspec_kw={"width_ratios": [1.2] + [1] * TOP_N},
    )

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        # Indices where true == predicted == this class
        correct_mask = (y_true_labels == cls_idx) & (y_pred_labels == cls_idx)
        correct_indices = np.where(correct_mask)[0]
        confidences = y_pred_conf[correct_indices]

        # Sort by confidence descending
        sorted_order = np.argsort(-confidences)
        top_indices = correct_indices[sorted_order[:TOP_N]]
        top_confidences = confidences[sorted_order[:TOP_N]]

        display_name = DISPLAY_NAMES[cls_name]
        print(f"\n{'='*60}")
        print(f"  {display_name} — Top {TOP_N} correct predictions")
        print(f"{'='*60}")

        # Label column
        label_ax = axes[cls_idx, 0]
        label_ax.text(0.5, 0.5, display_name, fontsize=13, fontweight="bold",
                      ha="center", va="center", transform=label_ax.transAxes)
        label_ax.axis("off")

        for rank, (idx, conf) in enumerate(zip(top_indices, top_confidences)):
            fpath = file_paths[idx]
            fname = Path(fpath).name
            print(f"  [{rank+1}] {fname}  (confidence: {conf:.4f})")

            # Load and display image
            img = tf.io.read_file(fpath)
            img = tf.image.decode_png(img, channels=1)
            img_np = img.numpy().squeeze()

            ax = axes[cls_idx, rank + 1]
            ax.imshow(img_np, cmap="gray")
            ax.set_title(f"{conf:.2%}", fontsize=10)
            ax.axis("off")

    fig.suptitle("Top-5 Highest-Confidence Correct Predictions per Class (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "clear_examples_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrid saved to {out_path}")

    # ── Borderline correct predictions (confidence closest to 50%) ──────
    find_borderline_examples(
        y_true_labels, y_pred_labels, y_pred_conf, y_pred_probs, file_paths,
    )


def find_borderline_examples(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    y_pred_conf: np.ndarray,
    y_pred_probs: np.ndarray,
    file_paths: list[str],
) -> None:
    """Find correctly classified examples where the model was least sure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOP_BORDERLINE = 5

    print(f"\n{'#'*70}")
    print(f"  BORDERLINE CORRECT PREDICTIONS (confidence closest to 50%)")
    print(f"{'#'*70}")

    # ── Per-class borderline examples ───────────────────────────────────
    # Collect across all classes to see which classes have borderline cases
    all_borderline: list[dict] = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        correct_mask = (y_true_labels == cls_idx) & (y_pred_labels == cls_idx)
        correct_indices = np.where(correct_mask)[0]
        confidences = y_pred_conf[correct_indices]

        # Sort by distance from 0.5 (ascending = closest to 50% first)
        dist_from_half = np.abs(confidences - 0.5)
        sorted_order = np.argsort(dist_from_half)

        display_name = DISPLAY_NAMES[cls_name]
        print(f"\n{'='*60}")
        print(f"  {display_name} — Most borderline correct predictions")
        print(f"{'='*60}")

        n_found = min(TOP_BORDERLINE, len(sorted_order))
        for rank in range(n_found):
            idx = correct_indices[sorted_order[rank]]
            conf = confidences[sorted_order[rank]]
            probs = y_pred_probs[idx]
            fpath = file_paths[idx]
            fname = Path(fpath).name

            # Find the runner-up class
            sorted_probs = np.argsort(-probs)
            runner_up_idx = sorted_probs[1]
            runner_up_name = DISPLAY_NAMES[CLASS_NAMES[runner_up_idx]]
            runner_up_prob = probs[runner_up_idx]

            print(f"  [{rank+1}] {fname}")
            print(f"       Winner:    {display_name} = {conf:.4f}")
            print(f"       Runner-up: {runner_up_name} = {runner_up_prob:.4f}")
            print(f"       All probs: {dict(zip(CLASS_NAMES, [f'{p:.4f}' for p in probs]))}")

            all_borderline.append({
                "idx": idx,
                "true_cls": cls_idx,
                "conf": conf,
                "runner_up_cls": runner_up_idx,
                "runner_up_prob": runner_up_prob,
                "probs": probs,
                "fpath": fpath,
            })

    # ── Global top-10 most borderline (any class) ──────────────────────
    all_borderline.sort(key=lambda d: abs(d["conf"] - 0.5))
    top_global = all_borderline[:10]

    print(f"\n{'='*60}")
    print(f"  GLOBAL TOP-10 MOST BORDERLINE (any class)")
    print(f"{'='*60}")
    for rank, item in enumerate(top_global):
        fname = Path(item["fpath"]).name
        true_name = DISPLAY_NAMES[CLASS_NAMES[item["true_cls"]]]
        runner_name = DISPLAY_NAMES[CLASS_NAMES[item["runner_up_cls"]]]
        print(f"  [{rank+1}] {fname}")
        print(f"       True class: {true_name} ({item['conf']:.4f}) "
              f"vs {runner_name} ({item['runner_up_prob']:.4f})")

    # ── Visualization: image + probability bar chart side by side ───────
    n_show = min(8, len(top_global))
    fig, axes = plt.subplots(n_show, 2, figsize=(10, 2.8 * n_show),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    if n_show == 1:
        axes = axes[np.newaxis, :]

    bar_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    short_names = ["CR", "LP", "ND", "PO"]

    for row, item in enumerate(top_global[:n_show]):
        # Image
        img = tf.io.read_file(item["fpath"])
        img = tf.image.decode_png(img, channels=1)
        img_np = img.numpy().squeeze()

        ax_img = axes[row, 0]
        ax_img.imshow(img_np, cmap="gray")
        true_name = DISPLAY_NAMES[CLASS_NAMES[item["true_cls"]]]
        ax_img.set_title(f"True: {true_name}", fontsize=10, fontweight="bold")
        ax_img.axis("off")

        # Probability bar chart
        ax_bar = axes[row, 1]
        probs = item["probs"]
        bars = ax_bar.barh(short_names, probs, color=bar_colors)
        ax_bar.set_xlim(0, 1)
        ax_bar.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        # Annotate bars with probabilities
        for bar, prob in zip(bars, probs):
            if prob > 0.05:
                ax_bar.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
                            f"{prob:.1%}", ha="right", va="center", fontsize=9,
                            fontweight="bold", color="white")
            elif prob > 0.005:
                ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{prob:.1%}", ha="left", va="center", fontsize=8)

        ax_bar.set_xlabel("Probability", fontsize=9)
        fname = Path(item["fpath"]).stem
        ax_bar.set_title(fname, fontsize=8, color="gray")

    fig.suptitle("Most Borderline Correct Predictions (Test Set)\nCorrectly classified but model was least confident",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "borderline_examples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBorderline grid saved to {out_path}")


if __name__ == "__main__":
    main()
