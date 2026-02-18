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

"""Train classification models for welding defect detection.

Usage:
    python scripts/01_train_classifier.py \\
        --model efficientnetb0 --config configs/efficientnetb0_baseline.yaml
    python scripts/01_train_classifier.py \\
        --model resnet50v2 --config configs/resnet50v2_baseline.yaml
    python scripts/01_train_classifier.py --model all --dry-run
    python scripts/01_train_classifier.py \\
        --model efficientnetb0 --config configs/efficientnetb0_baseline.yaml \\
        --tracking vertex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf

from riawelc.config import ModelConfig, TrainingConfig
from riawelc.data.loader import compute_class_weights, create_datasets
from riawelc.models.classifier import unfreeze_backbone
from riawelc.models.registry import build_model, list_models
from riawelc.training.train import compile_model, recompile_for_fine_tune, set_seeds, train_model

CONFIGS_DIR = Path("configs")


def print_gpu_info() -> None:
    """Print GPU availability and device info for debugging."""
    gpus = tf.config.list_physical_devices("GPU")
    print("=" * 60)
    print("GPU Diagnostics")
    print("=" * 60)
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  Built with CUDA:    {tf.test.is_built_with_cuda()}")
    print(f"  GPUs available:     {len(gpus)}")
    for gpu in gpus:
        print(f"    - {gpu.name} ({gpu.device_type})")
    if not gpus:
        print("  WARNING: No GPUs detected. Training will run on CPU.")
        print("  If this is a GPU job, check NVIDIA_VISIBLE_DEVICES env var.")
    print("=" * 60)


def resolve_model_names(model_arg: str) -> list[str]:
    """Resolve --model argument to a list of validated model names."""
    registered = list_models()
    if model_arg == "all":
        if not registered:
            print("ERROR: No models registered.")
            sys.exit(1)
        return registered

    names = [n.strip() for n in model_arg.split(",")]
    for name in names:
        if name not in registered:
            available = ", ".join(registered) or "(none)"
            print(f"ERROR: Model '{name}' not registered. Available: {available}")
            sys.exit(1)
    return names


def resolve_config_path(model_name: str, explicit_config_path: str | None) -> Path:
    """Resolve which config YAML to use for a model."""
    if explicit_config_path is not None:
        path = Path(explicit_config_path)
        if not path.exists():
            print(f"ERROR: Config file not found: {path}")
            sys.exit(1)
        return path

    model_config_path = CONFIGS_DIR / f"{model_name}_baseline.yaml"
    if model_config_path.exists():
        return model_config_path

    print(f"ERROR: No config found for '{model_name}'. Tried: {model_config_path}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classification models for welding defect detection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g. efficientnetb0), comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML. If omitted, uses configs/{model}_baseline.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build models and print summaries without training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root output directory (default: outputs).",
    )
    parser.add_argument(
        "--tracking",
        type=str,
        choices=["mlflow", "vertex", "both", "none"],
        default="mlflow",
        help="Experiment tracking backend (default: mlflow).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory. Appends /training, /validation, /testing.",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="Override MLflow tracking URI from config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory from config.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a Phase 1 checkpoint, skip feature extraction, run fine-tune only.",
    )
    return parser.parse_args()


def derive_run_suffix(model_name: str, config_path: Path) -> str | None:
    """Derive a run name suffix from the config filename.

    Examples:
        efficientnetb0_baseline.yaml  -> None
        efficientnetb0_ht_1.yaml      -> "ht_1"
        resnet50v2_ht_2.yaml          -> "ht_2"
    """
    stem = config_path.stem  # e.g. "efficientnetb0_ht_1"
    prefix = f"{model_name}_"
    if stem.startswith(prefix):
        remainder = stem[len(prefix):]
        if remainder and remainder != "baseline":
            return remainder
    return None


def apply_cli_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply CLI argument overrides to the loaded config."""
    if args.data_dir is not None:
        config.data.train_dir = str(args.data_dir / "training")
        config.data.val_dir = str(args.data_dir / "validation")
        config.data.test_dir = str(args.data_dir / "testing")

    if args.mlflow_uri is not None:
        config.callbacks.mlflow_tracking_uri = args.mlflow_uri

    if args.checkpoint_dir is not None:
        config.callbacks.checkpoint_dir = str(args.checkpoint_dir)

    return config


def main() -> None:
    args = parse_args()
    print_gpu_info()
    model_names = resolve_model_names(args.model)

    for model_name in model_names:
        config_path = resolve_config_path(model_name, args.config)
        config = TrainingConfig.from_yaml(config_path)
        config = apply_cli_overrides(config, args)
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name} | Config: {config_path}")
        print(f"Tracking: {args.tracking}")
        print(f"{'=' * 60}")

        run_suffix = derive_run_suffix(model_name, config_path)
        config_file = str(config_path)
        set_seeds(config.seed)

        if args.dry_run:
            if config.has_fine_tune_phase:
                # Phase 1: fully frozen backbone
                phase1_model_config = ModelConfig(**{**vars(config.model), "fine_tune_at": 0})
                model = build_model(model_name, phase1_model_config)
                model.summary()
                print(
                    f"\n  Phase 1 (feature extraction): {config.epochs} epochs, "
                    f"lr={config.optimizer.learning_rate}"
                )
                print(
                    f"  Phase 2 (fine-tune): {config.fine_tune.epochs} epochs, "
                    f"lr={config.fine_tune.learning_rate}, "
                    f"unfreeze at layer {config.model.fine_tune_at}"
                )
            else:
                model = build_model(model_name, config.model)
                model.summary()
            print(f"\n[DRY RUN] Skipping training for {model_name}.")
            continue

        train_ds, val_ds, _ = create_datasets(config)
        class_weights = compute_class_weights(config)

        if args.resume_from is not None:
            # Skip Phase 1, load checkpoint, go straight to fine-tuning
            if not args.resume_from.exists():
                print(f"ERROR: Checkpoint not found at {args.resume_from}")
                sys.exit(1)
            print(f"\nResuming from checkpoint: {args.resume_from}")
            model = tf.keras.models.load_model(str(args.resume_from))
            unfreeze_backbone(model, config.model.fine_tune_at)
            model = recompile_for_fine_tune(model, config)

            print(f"\n--- Fine-tuning ({config.fine_tune.epochs} epochs) ---")
            history = train_model(
                model,
                train_ds,
                val_ds,
                config,
                class_weights=class_weights,
                epochs_override=config.fine_tune.epochs,
                phase="fine_tune",
                run_suffix=run_suffix,
                config_file=config_file,
                tracking=args.tracking,
            )
            best_val_loss = min(history.history["val_loss"])
            best_val_acc = max(history.history["val_accuracy"])
            print(
                f"Fine-tuning done. Best val_loss: {best_val_loss:.4f}, "
                f"Best val_acc: {best_val_acc:.4f}"
            )

        elif config.has_fine_tune_phase:
            # Phase 1: feature extraction (fully frozen backbone)
            phase1_model_config = ModelConfig(**{**vars(config.model), "fine_tune_at": 0})
            model = build_model(model_name, phase1_model_config)
            model = compile_model(model, config)

            print(f"\n--- Phase 1: Feature extraction ({config.epochs} epochs) ---")
            history = train_model(
                model,
                train_ds,
                val_ds,
                config,
                class_weights=class_weights,
                phase="feature_extraction",
                run_suffix=run_suffix,
                config_file=config_file,
                tracking=args.tracking,
            )
            best_val_loss = min(history.history["val_loss"])
            best_val_acc = max(history.history["val_accuracy"])
            print(
                f"Phase 1 done. Best val_loss: {best_val_loss:.4f}, "
                f"Best val_acc: {best_val_acc:.4f}"
            )

            # Phase 2: fine-tuning (unfreeze upper backbone layers)
            unfreeze_backbone(model, config.model.fine_tune_at)
            model = recompile_for_fine_tune(model, config)

            print(f"\n--- Phase 2: Fine-tuning ({config.fine_tune.epochs} epochs) ---")
            history = train_model(
                model,
                train_ds,
                val_ds,
                config,
                class_weights=class_weights,
                epochs_override=config.fine_tune.epochs,
                phase="fine_tune",
                run_suffix=run_suffix,
                config_file=config_file,
                tracking=args.tracking,
            )
            best_val_loss = min(history.history["val_loss"])
            best_val_acc = max(history.history["val_accuracy"])
            print(
                f"Phase 2 done. Best val_loss: {best_val_loss:.4f}, "
                f"Best val_acc: {best_val_acc:.4f}"
            )
        else:
            # Single-phase training (backward compatible)
            model = build_model(model_name, config.model)
            model = compile_model(model, config)
            history = train_model(
                model,
                train_ds,
                val_ds,
                config,
                class_weights=class_weights,
                run_suffix=run_suffix,
                config_file=config_file,
                tracking=args.tracking,
            )

            best_val_loss = min(history.history["val_loss"])
            best_val_acc = max(history.history["val_accuracy"])
            print(
                f"\nTraining complete. Best val_loss: {best_val_loss:.4f}, "
                f"Best val_acc: {best_val_acc:.4f}"
            )


if __name__ == "__main__":
    main()
