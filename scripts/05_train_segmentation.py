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

"""Train U-Net segmentation model on pseudo-masks.

Usage:
    python scripts/05_train_segmentation.py --config configs/segmentation_unet.yaml
    python scripts/05_train_segmentation.py --config configs/segmentation_unet.yaml --dry-run
    python scripts/05_train_segmentation.py --config configs/segmentation_unet.yaml \\
        --data-dir /gcs/$GCS_DATA_BUCKET \\
        --mask-dir /gcs/$GCS_ARTIFACTS_BUCKET/outputs/pseudomasks \\
        --tracking vertex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from riawelc.config import TrainingConfig
from riawelc.data.augmentation import augment_segmentation_pair
from riawelc.models.segmentation import build_unet_segmentation, dice_bce_loss, dice_coefficient
from riawelc.training.callbacks import build_callbacks
from riawelc.training.train import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net segmentation on pseudo-masks.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to segmentation config YAML."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build model and print summary only."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory. Appends /training, /validation, /testing.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Override pseudo-mask directory (default: outputs/pseudomasks).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory from config.",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="Override MLflow tracking URI from config.",
    )
    parser.add_argument(
        "--tracking",
        type=str,
        default="mlflow",
        choices=["mlflow", "vertex", "both", "none"],
        help="Experiment tracking backend (default: mlflow).",
    )
    return parser.parse_args()


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


def load_segmentation_data(
    image_dir: Path,
    mask_dir: Path,
    input_shape: tuple[int, int],
    batch_size: int,
    augment: bool = False,
    augmentation_mode: str = "full",
) -> tf.data.Dataset:
    """Load image-mask pairs for segmentation training."""
    image_paths = []
    mask_paths = []

    for class_dir in sorted(image_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name == "no_defect":
            continue
        mask_class_dir = mask_dir / class_dir.name
        if not mask_class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*.png")):
            mask_path = mask_class_dir / img_path.name
            if mask_path.exists():
                image_paths.append(str(img_path))
                mask_paths.append(str(mask_path))

    def load_pair(img_path: tf.Tensor, mask_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, input_shape)
        img = tf.cast(img, tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, input_shape, method="nearest")
        mask = tf.cast(mask, tf.float32) / 255.0
        return img, mask

    if not image_paths:
        raise RuntimeError(
            f"No image-mask pairs found.\n"
            f"  image_dir: {image_dir}\n"
            f"  mask_dir:  {mask_dir}\n"
            f"Ensure Grad-CAM heatmaps and pseudo-masks were generated for the "
            f"training split (not just testing):\n"
            f"  python scripts/03_generate_gradcam.py --data-dir {image_dir} ...\n"
            f"  python scripts/04_generate_pseudomasks.py"
        )

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.shuffle(len(image_paths), seed=9)
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug_mode = augmentation_mode
        ds = ds.map(
            lambda img, msk: augment_segmentation_pair(img, msk, mode=aug_mode),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


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
    config = TrainingConfig.from_yaml(args.config)
    config = apply_cli_overrides(config, args)
    set_seeds(config.seed)

    input_shape = tuple(config.model.input_shape[:2])
    model = build_unet_segmentation(
        input_shape=tuple(config.model.input_shape),
        num_classes=config.model.num_classes,
    )

    if args.dry_run:
        model.summary()
        print("\n[DRY RUN] Skipping training.")
        return

    if args.mask_dir is not None:
        mask_dir = args.mask_dir
    else:
        raw = yaml.safe_load(open(args.config))
        mask_dir = Path(raw.get("segmentation", {}).get("mask_dir", "outputs/pseudomasks"))
    image_dir = Path(config.data.train_dir)

    train_ds = load_segmentation_data(
        image_dir, mask_dir, input_shape, config.batch_size,
        augment=config.data.augmentation,
        augmentation_mode=config.data.augmentation_mode,
    )

    val_image_dir = Path(config.data.val_dir)
    val_ds = load_segmentation_data(val_image_dir, mask_dir, input_shape, config.batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        ),
        loss=dice_bce_loss,
        metrics=[dice_coefficient],
    )

    callbacks = build_callbacks(
        config,
        tracking=args.tracking,
        config_file=args.config,
        extra_tags={"mask_dir": str(mask_dir)},
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("Segmentation training complete.")


if __name__ == "__main__":
    main()
