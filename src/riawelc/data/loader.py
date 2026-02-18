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

from pathlib import Path

import keras
import tensorflow as tf

from riawelc.config import TrainingConfig
from riawelc.data import CLASS_NAMES


def _load_split(
    directory: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Load a single dataset split via ``image_dataset_from_directory``."""
    return keras.utils.image_dataset_from_directory(
        directory,
        label_mode="categorical",
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        class_names=CLASS_NAMES,
    )


def create_datasets(
    config: TrainingConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train / validation / test tf.data.Dataset pipelines.

    Returns a tuple (train_ds, val_ds, test_ds) with images as float32
    in [0, 255] and labels one-hot encoded.
    """
    image_size = (config.input_shape[0], config.input_shape[1])
    batch_size = config.batch_size
    seed = config.seed

    train_ds = _load_split(
        config.data.train_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )
    val_ds = _load_split(
        config.data.val_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )
    test_ds = _load_split(
        config.data.test_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )

    # Cast uint8 → float32, keeping [0, 255] range
    train_ds = train_ds.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ds = test_ds.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def compute_class_weights(config: TrainingConfig) -> dict[int, float]:
    """Compute per-class weights inversely proportional to class frequency.

    Uses the training split directory to count images per class.  Returns a
    dict mapping class index (int) → weight (float) suitable for
    model.fit(class_weight=...).
    """
    train_dir = Path(config.data.train_dir)
    counts: list[int] = []
    for name in CLASS_NAMES:
        class_dir = train_dir / name
        n = len(list(class_dir.glob("*.png")))
        counts.append(n)

    total = sum(counts)
    n_classes = len(CLASS_NAMES)
    weights: dict[int, float] = {}
    for idx, count in enumerate(counts):
        weights[idx] = total / (n_classes * count) if count > 0 else 1.0

    return weights
