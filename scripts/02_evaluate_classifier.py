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

"""Evaluate a trained classifier on the test set.

Generates confusion matrix, classification report, and per-class metrics CSV.

Usage:
    python scripts/02_evaluate_classifier.py \\
        --model efficientnetb0 --config configs/efficientnetb0_baseline.yaml
    python scripts/02_evaluate_classifier.py \\
        --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras \\
        --config configs/efficientnetb0_baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from riawelc.config import TrainingConfig
from riawelc.data.loader import create_datasets

CLASS_NAMES = ["crack", "lack_of_penetration", "no_defect", "porosity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (resolves checkpoint path).",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Direct path to model checkpoint.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: derived from model name/path).",
    )
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace, config: TrainingConfig) -> Path:
    if args.model_path is not None:
        return args.model_path
    if args.model is not None:
        base = Path(config.callbacks.checkpoint_dir)
        return base / args.model / config.model_version / "best.keras"
    print("ERROR: Provide either --model or --model-path.")
    sys.exit(1)


def _resolve_output_dir(args: argparse.Namespace, model_path: Path) -> Path:
    """Derive output directory from model path if not explicitly set.

    E.g. ``...checkpoints/efficientnetb0/v1/best.keras``
    → ``outputs/evaluation/efficientnetb0``
    """
    if args.output_dir is not None:
        return args.output_dir
    model_name = model_path.parent.parent.name
    return Path("outputs/evaluation") / model_name


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_yaml(args.config)
    model_path = resolve_model_path(args, config)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    args.output_dir = _resolve_output_dir(args, model_path)

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    _, _, test_ds = create_datasets(config)

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds, verbose=1)

    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true_labels, y_pred_labels,
        target_names=CLASS_NAMES, output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(args.output_dir / "classification_report.csv")
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "model_path": str(model_path),
    }
    for cls in CLASS_NAMES:
        if cls in report:
            metrics[f"{cls}_f1"] = report[cls]["f1-score"]
            metrics[f"{cls}_precision"] = report[cls]["precision"]
            metrics[f"{cls}_recall"] = report[cls]["recall"]

    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
