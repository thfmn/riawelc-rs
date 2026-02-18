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
from unittest.mock import MagicMock, patch

import pytest

from riawelc.config import TrainingConfig
from riawelc.training.callbacks import (
    MLflowCallback,
    VertexAICallback,
    _flatten_config,
    build_callbacks,
)


@pytest.fixture
def config(tmp_config_yaml: Path) -> TrainingConfig:
    return TrainingConfig.from_yaml(tmp_config_yaml)


class TestFlattenConfig:
    def test_returns_flat_dict(self, config: TrainingConfig) -> None:
        flat = _flatten_config(config)
        assert isinstance(flat, dict)
        assert all(isinstance(v, str) for v in flat.values())

    def test_nested_keys_are_dotted(self, config: TrainingConfig) -> None:
        flat = _flatten_config(config)
        assert "optimizer.learning_rate" in flat
        assert "model.name" in flat
        assert "callbacks.monitor" in flat

    def test_top_level_keys_present(self, config: TrainingConfig) -> None:
        flat = _flatten_config(config)
        assert "seed" in flat
        assert "epochs" in flat


class TestBuildCallbacksTrackingModes:
    def test_tracking_none_excludes_mlflow_and_vertex(self, config: TrainingConfig) -> None:
        callbacks = build_callbacks(config, tracking="none")
        types = [type(cb).__name__ for cb in callbacks]
        assert "MLflowCallback" not in types
        assert "VertexAICallback" not in types

    def test_tracking_mlflow_includes_mlflow(self, config: TrainingConfig) -> None:
        callbacks = build_callbacks(config, tracking="mlflow")
        types = [type(cb).__name__ for cb in callbacks]
        assert "MLflowCallback" in types
        assert "VertexAICallback" not in types

    def test_tracking_vertex_includes_vertex(self, config: TrainingConfig) -> None:
        callbacks = build_callbacks(config, tracking="vertex")
        types = [type(cb).__name__ for cb in callbacks]
        assert "VertexAICallback" in types
        assert "MLflowCallback" not in types

    def test_tracking_both_includes_both(self, config: TrainingConfig) -> None:
        callbacks = build_callbacks(config, tracking="both")
        types = [type(cb).__name__ for cb in callbacks]
        assert "MLflowCallback" in types
        assert "VertexAICallback" in types

    def test_always_includes_checkpoint_earlystop_reducelr(self, config: TrainingConfig) -> None:
        for mode in ("mlflow", "vertex", "both", "none"):
            callbacks = build_callbacks(config, tracking=mode)
            types = [type(cb).__name__ for cb in callbacks]
            assert "ModelCheckpoint" in types
            assert "EarlyStopping" in types
            assert "ReduceLROnPlateau" in types

    def test_run_name_includes_phase(self, config: TrainingConfig) -> None:
        callbacks = build_callbacks(config, phase="feature_extraction", tracking="mlflow")
        mlflow_cbs = [cb for cb in callbacks if isinstance(cb, MLflowCallback)]
        assert len(mlflow_cbs) == 1
        assert mlflow_cbs[0]._run_name == "efficientnetb0-feature_extraction"

    def test_run_name_without_phase(self, config: TrainingConfig) -> None:
        callbacks = build_callbacks(config, tracking="mlflow")
        mlflow_cbs = [cb for cb in callbacks if isinstance(cb, MLflowCallback)]
        assert len(mlflow_cbs) == 1
        assert mlflow_cbs[0]._run_name == "efficientnetb0"

    def test_vertex_receives_project_and_location(self, config: TrainingConfig) -> None:
        callbacks = build_callbacks(
            config,
            tracking="vertex",
            project="my-project",
            location="europe-west3",
        )
        vertex_cbs = [cb for cb in callbacks if isinstance(cb, VertexAICallback)]
        assert len(vertex_cbs) == 1
        assert vertex_cbs[0]._project == "my-project"
        assert vertex_cbs[0]._location == "europe-west3"


class TestMLflowCallback:
    @patch("riawelc.training.callbacks.MLflowCallback._ensure_mlflow")
    def test_on_train_begin_starts_run(self, mock_ensure: MagicMock) -> None:
        cb = MLflowCallback(
            tracking_uri="mlruns",
            experiment_name="test",
            run_name="test-run",
        )
        cb._mlflow = MagicMock()
        cb.on_train_begin()
        cb._mlflow.start_run.assert_called_once_with(run_name="test-run", nested=True)

    @patch("riawelc.training.callbacks.MLflowCallback._ensure_mlflow")
    def test_on_train_begin_logs_params(
        self, mock_ensure: MagicMock, config: TrainingConfig
    ) -> None:
        cb = MLflowCallback(
            tracking_uri="mlruns",
            experiment_name="test",
            run_name="test-run",
            config=config,
        )
        cb._mlflow = MagicMock()
        cb.on_train_begin()
        cb._mlflow.log_params.assert_called_once()
        logged = cb._mlflow.log_params.call_args[0][0]
        assert "optimizer.learning_rate" in logged

    def test_on_epoch_end_logs_metrics(self) -> None:
        cb = MLflowCallback(tracking_uri="mlruns", experiment_name="test")
        cb._mlflow = MagicMock()
        cb.on_epoch_end(epoch=0, logs={"loss": 0.5, "val_loss": 0.6})
        cb._mlflow.log_metrics.assert_called_once_with({"loss": 0.5, "val_loss": 0.6}, step=0)

    def test_on_epoch_end_skips_when_no_logs(self) -> None:
        cb = MLflowCallback(tracking_uri="mlruns", experiment_name="test")
        cb._mlflow = MagicMock()
        cb.on_epoch_end(epoch=0, logs=None)
        cb._mlflow.log_metrics.assert_not_called()


class TestVertexAICallback:
    @patch("riawelc.training.callbacks.VertexAICallback._ensure_aiplatform")
    def test_on_train_begin_inits_and_starts_run(self, mock_ensure: MagicMock) -> None:
        cb = VertexAICallback(
            experiment_name="test-exp",
            run_name="test-run",
            params={"seed": "9"},
            project="my-project",
            location="us-central1",
        )
        cb._aiplatform = MagicMock()
        cb._default_project = "auto-detected-project"
        cb._default_location = "auto-detected-location"
        cb.on_train_begin()

        cb._aiplatform.init.assert_called_once_with(
            experiment="test-exp",
            project="my-project",
            location="us-central1",
        )
        cb._aiplatform.start_run.assert_called_once_with(run="test-run")
        cb._aiplatform.log_params.assert_called_once_with({"seed": "9"})
        assert cb._active is True

    @patch("riawelc.training.callbacks.VertexAICallback._ensure_aiplatform")
    def test_on_train_begin_falls_back_to_defaults(self, mock_ensure: MagicMock) -> None:
        cb = VertexAICallback(
            experiment_name="test-exp",
            run_name="test-run",
            params={},
        )
        cb._aiplatform = MagicMock()
        cb._default_project = "auto-detected-project"
        cb._default_location = "europe-west3"
        cb.on_train_begin()

        cb._aiplatform.init.assert_called_once_with(
            experiment="test-exp",
            project="auto-detected-project",
            location="europe-west3",
        )

    @patch("riawelc.training.callbacks.VertexAICallback._ensure_aiplatform")
    def test_on_train_begin_graceful_fallback_on_error(self, mock_ensure: MagicMock) -> None:
        cb = VertexAICallback(
            experiment_name="test-exp",
            run_name="test-run",
            params={},
        )
        cb._aiplatform = MagicMock()
        cb._aiplatform.init.side_effect = PermissionError("ACCESS_TOKEN_SCOPE_INSUFFICIENT")
        cb._default_project = "my-project"
        cb._default_location = "us-central1"
        cb.on_train_begin()

        assert cb._active is False

    def test_on_epoch_end_logs_metrics(self) -> None:
        cb = VertexAICallback(
            experiment_name="test-exp",
            run_name="test-run",
            params={},
        )
        cb._aiplatform = MagicMock()
        cb._active = True
        cb.on_epoch_end(epoch=1, logs={"loss": 0.3, "accuracy": 0.9})
        cb._aiplatform.log_metrics.assert_called_once_with(
            {"loss": 0.3, "accuracy": 0.9},
        )

    def test_on_epoch_end_skips_when_not_active(self) -> None:
        cb = VertexAICallback(
            experiment_name="test-exp",
            run_name="test-run",
            params={},
        )
        cb._aiplatform = MagicMock()
        cb._active = False
        cb.on_epoch_end(epoch=1, logs={"loss": 0.3, "accuracy": 0.9})
        cb._aiplatform.log_metrics.assert_not_called()

    def test_on_train_end_ends_run(self) -> None:
        cb = VertexAICallback(
            experiment_name="test-exp",
            run_name="test-run",
            params={},
        )
        cb._aiplatform = MagicMock()
        cb._active = True
        cb.on_train_end()
        cb._aiplatform.end_run.assert_called_once()

    def test_on_train_end_skips_when_not_active(self) -> None:
        cb = VertexAICallback(
            experiment_name="test-exp",
            run_name="test-run",
            params={},
        )
        cb._aiplatform = MagicMock()
        cb._active = False
        cb.on_train_end()
        cb._aiplatform.end_run.assert_not_called()
