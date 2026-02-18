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

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import keras

from riawelc.config import TrainingConfig

logger = logging.getLogger(__name__)


class MLflowCallback(keras.callbacks.Callback):
    """Log training metrics, params, and artifacts to MLflow."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        *,
        run_name: str | None = None,
        config: TrainingConfig | None = None,
        tags: dict[str, str] | None = None,
    ):
        super().__init__()
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._config = config
        self._tags = tags or {}
        self._mlflow = None

    def _ensure_mlflow(self) -> None:
        if self._mlflow is None:
            import mlflow

            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._mlflow = mlflow

    def on_train_begin(self, logs: dict | None = None) -> None:
        self._ensure_mlflow()
        self._mlflow.start_run(run_name=self._run_name, nested=True)

        if self._tags:
            self._mlflow.set_tags(self._tags)

        if self._config is not None:
            flat_params = _flatten_config(self._config)
            self._mlflow.log_params(flat_params)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs and self._mlflow:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in logs.items()},
                step=epoch,
            )

    def on_train_end(self, logs: dict | None = None) -> None:
        if not self._mlflow or not self._mlflow.active_run():
            return

        if self.model is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                summary_path = Path(tmpdir) / "model_summary.json"
                summary_lines: list[str] = []
                self.model.summary(print_fn=summary_lines.append)
                summary_path.write_text(
                    json.dumps(
                        {
                            "model_name": self.model.name,
                            "total_params": int(self.model.count_params()),
                            "summary": "\n".join(summary_lines),
                        },
                        indent=2,
                    )
                )
                self._mlflow.log_artifact(str(summary_path))

        self._mlflow.end_run()


class VertexAICallback(keras.callbacks.Callback):
    """Log training metrics to Vertex AI Experiments."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        params: dict[str, str | float | int],
        *,
        project: str | None = None,
        location: str | None = None,
    ):
        super().__init__()
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._params = params
        self._project = project
        self._location = location
        self._aiplatform = None
        self._active = False

    def _ensure_aiplatform(self) -> None:
        if self._aiplatform is None:
            from google.cloud import aiplatform

            self._aiplatform = aiplatform
            self._default_project = os.environ.get("CLOUD_ML_PROJECT_ID")
            self._default_location = os.environ.get("CLOUD_ML_REGION")

    def on_train_begin(self, logs: dict | None = None) -> None:
        self._ensure_aiplatform()
        project = self._project or self._default_project
        location = self._location or self._default_location
        try:
            self._aiplatform.init(
                experiment=self._experiment_name,
                project=project,
                location=location,
            )
            self._aiplatform.start_run(run=self._run_name)
            if self._params:
                self._aiplatform.log_params(self._params)
            self._active = True
        except Exception as exc:
            logger.warning(
                "Vertex AI Experiments init failed (project=%s, location=%s): %s. "
                "Continuing without Vertex AI tracking.",
                project,
                location,
                exc,
            )
            self._active = False

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs and self._active:
            self._aiplatform.log_metrics(
                {k: float(v) for k, v in logs.items()},
            )

    def on_train_end(self, logs: dict | None = None) -> None:
        if self._active:
            self._aiplatform.end_run()


def _flatten_config(config: TrainingConfig) -> dict[str, str]:
    """Flatten a TrainingConfig dataclass into a flat string dict for MLflow params."""
    flat: dict[str, str] = {}
    for key, value in asdict(config).items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = str(sub_value)
        else:
            flat[key] = str(value)
    return flat


def build_callbacks(
    config: TrainingConfig,
    *,
    phase: str | None = None,
    run_suffix: str | None = None,
    config_file: str | None = None,
    tracking: Literal["mlflow", "vertex", "both", "none"] = "mlflow",
    project: str | None = None,
    location: str | None = None,
) -> list[keras.callbacks.Callback]:
    """Build the list of Keras callbacks from training config.

    Args:
        run_suffix: Optional suffix appended to run names (e.g. "ht_1" from
            a config filename like ``efficientnetb0_ht_1.yaml``).
        config_file: Config YAML path, logged as a tag/param for traceability.
    """
    cb_config = config.callbacks
    callbacks: list[keras.callbacks.Callback] = []

    checkpoint_dir = Path(cb_config.checkpoint_dir) / config.model.name / config.model_version
    if phase is not None:
        checkpoint_dir = checkpoint_dir / phase
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best.keras"),
            monitor=cb_config.monitor,
            mode=cb_config.mode,
            save_best_only=cb_config.save_best_only,
            verbose=1,
        )
    )

    early_stop_patience = cb_config.early_stop_patience
    lr_reduce_patience = cb_config.lr_reduce_patience
    if phase == "fine_tune":
        early_stop_patience = config.fine_tune.early_stop_patience
        lr_reduce_patience = config.fine_tune.lr_reduce_patience

    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=cb_config.monitor,
            factor=cb_config.lr_reduce_factor,
            patience=lr_reduce_patience,
            min_lr=cb_config.lr_min,
            verbose=1,
        )
    )

    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=cb_config.monitor,
            patience=early_stop_patience,
            restore_best_weights=cb_config.restore_best_weights,
            verbose=1,
        )
    )

    run_name = config.model.name
    if phase is not None:
        run_name = f"{config.model.name}-{phase}"
    if run_suffix is not None:
        run_name = f"{run_name}-{run_suffix}"

    # Tags logged to MLflow for every run
    tags: dict[str, str] = {}
    if config_file is not None:
        tags["config_file"] = config_file
    if phase is not None:
        tags["phase"] = phase

    if tracking in ("mlflow", "both"):
        callbacks.append(
            MLflowCallback(
                tracking_uri=cb_config.mlflow_tracking_uri,
                experiment_name=cb_config.mlflow_experiment_name,
                run_name=run_name,
                config=config,
                tags=tags,
            )
        )

    if tracking in ("vertex", "both"):
        flat_params = _flatten_config(config)
        if config_file is not None:
            flat_params["config_file"] = config_file
        if phase is not None:
            flat_params["phase"] = phase
        callbacks.append(
            VertexAICallback(
                experiment_name=cb_config.mlflow_experiment_name,
                run_name=run_name,
                params=flat_params,
                project=project,
                location=location,
            )
        )

    return callbacks
