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

from typing import Literal

import keras
import numpy as np
import tensorflow as tf

from riawelc.config import TrainingConfig
from riawelc.training.callbacks import build_callbacks


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compile_model(model: keras.Model, config: TrainingConfig) -> keras.Model:
    """Compile a Keras model with optimizer and loss from config."""
    opt_config = config.optimizer

    if opt_config.name.lower() == "adamw":
        optimizer = keras.optimizers.AdamW(
            learning_rate=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
        )
    elif opt_config.name.lower() == "adam":
        optimizer = keras.optimizers.Adam(
            learning_rate=opt_config.learning_rate,
        )
    else:
        optimizer = keras.optimizers.get(opt_config.name)

    model.compile(
        optimizer=optimizer,
        loss=config.loss.name,
        metrics=["accuracy"],
    )
    return model


def recompile_for_fine_tune(model: keras.Model, config: TrainingConfig) -> keras.Model:
    """Recompile a model with a lower learning rate for fine-tuning."""
    opt_config = config.optimizer
    lr = config.fine_tune.learning_rate

    if opt_config.name.lower() == "adamw":
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=opt_config.weight_decay,
        )
    elif opt_config.name.lower() == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = keras.optimizers.get(opt_config.name)

    model.compile(
        optimizer=optimizer,
        loss=config.loss.name,
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: TrainingConfig,
    class_weights: dict[int, float] | None = None,
    *,
    epochs_override: int | None = None,
    phase: str | None = None,
    run_suffix: str | None = None,
    config_file: str | None = None,
    tracking: Literal["mlflow", "vertex", "both", "none"] = "mlflow",
    project: str | None = None,
    location: str | None = None,
) -> keras.callbacks.History:
    """Train a model with the given datasets and config."""
    set_seeds(config.seed)

    callbacks = build_callbacks(
        config,
        phase=phase,
        run_suffix=run_suffix,
        config_file=config_file,
        tracking=tracking,
        project=project,
        location=location,
    )
    epochs = epochs_override if epochs_override is not None else config.epochs

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    return history
