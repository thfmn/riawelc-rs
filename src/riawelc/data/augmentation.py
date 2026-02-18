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

import keras
import tensorflow as tf
from keras import layers

from riawelc.config import TrainingConfig

# Module-level RNG for segmentation augmentation.  Using a persistent
# tf.random.Generator ensures that each call to augment_segmentation_pair()
# inside tf.data.Dataset.map() receives different random seeds (the
# generator's internal state advances on every make_seeds() call).
_seg_aug_rng = tf.random.Generator.from_seed(9)


def build_augmentation_pipeline(config: TrainingConfig) -> keras.Model:
    """Build a Keras augmentation model for grayscale radiographic images.

    Applies random spatial and intensity transforms suitable for welding
    radiograph classification.  The returned model expects float32 inputs
    in [0, 255] with shape (batch, H, W, 1) and only augments during
    training (training=True).
    """
    seed = config.seed
    height, width, channels = config.input_shape

    inputs = layers.Input(shape=(height, width, channels), name="aug_input")

    # No obvious orientation can be determined from the images
    x = layers.RandomFlip("horizontal_and_vertical", seed=seed, name="random_flip")(inputs)
    x = layers.RandomRotation(
        0.05,
        fill_mode="constant",
        fill_value=0.0,
        seed=seed,
        name="random_rotation",
    )(x)
    x = layers.RandomZoom(
        (-0.1, 0.1),
        fill_mode="constant",
        fill_value=0.0,
        seed=seed,
        name="random_zoom",
    )(x)
    x = layers.RandomContrast(0.1, seed=seed, name="random_contrast")(x)
    x = layers.GaussianNoise(0.01, seed=seed, name="gaussian_noise")(x)

    return keras.Model(inputs, x, name="augmentation_pipeline")


def augment_segmentation_pair(
    image: tf.Tensor, mask: tf.Tensor, seed: int = 9, mode: str = "full"
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentation to an image-mask pair for segmentation.

    Spatial transforms (flip, rotation) are applied identically to both
    image and mask using shared seeds.  Intensity transforms (brightness,
    contrast) are applied to the image only when ``mode="full"``.

    Parameters
    ----------
    image : tf.Tensor
        Float32 image tensor with shape (H, W, 1), values in [0, 255].
    mask : tf.Tensor
        Float32 binary mask tensor with shape (H, W, 1), values in [0, 1].
    seed : int
        Base seed used to derive per-call random seeds.
    mode : str
        ``"full"`` applies spatial + intensity transforms (default).
        ``"spatial"`` applies only spatial transforms.
    """
    # Draw fresh seeds from the module-level RNG so each mapped sample
    # receives a different random augmentation.
    seed_hflip = _seg_aug_rng.make_seeds(2)[0]
    seed_vflip = _seg_aug_rng.make_seeds(2)[0]
    seed_rot = _seg_aug_rng.make_seeds(2)[0]
    seed_bright = _seg_aug_rng.make_seeds(2)[0]
    seed_contrast = _seg_aug_rng.make_seeds(2)[0]

    # --- Spatial transforms (applied identically to image AND mask) ---

    # Random horizontal flip
    image = tf.image.stateless_random_flip_left_right(image, seed=seed_hflip)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=seed_hflip)

    # Random vertical flip
    image = tf.image.stateless_random_flip_up_down(image, seed=seed_vflip)
    mask = tf.image.stateless_random_flip_up_down(mask, seed=seed_vflip)

    # Random rotation ±5% of a full turn (~±18 degrees)
    angle = tf.random.stateless_uniform([], seed=seed_rot, minval=-0.05, maxval=0.05)
    angle_rad = angle * 2.0 * 3.141592653589793
    image = _rotate_fill(image, angle_rad, fill_value=0.0)
    mask = _rotate_fill(mask, angle_rad, fill_value=0.0)

    # --- Intensity transforms (image ONLY, skipped in spatial mode) ---

    if mode == "full":
        image = tf.image.stateless_random_brightness(image, max_delta=0.1, seed=seed_bright)
        image = tf.image.stateless_random_contrast(image, lower=0.9, upper=1.1, seed=seed_contrast)

    return image, mask


def _rotate_fill(tensor: tf.Tensor, angle: tf.Tensor, fill_value: float = 0.0) -> tf.Tensor:
    """Rotate a single (H, W, C) tensor by *angle* radians, filling with *fill_value*."""
    # tfa-free rotation via contrib-style raw transform
    # tf.raw_ops.ImageProjectiveTransformV3 expects a batch dimension
    img = tf.expand_dims(tensor, 0)
    h = tf.cast(tf.shape(tensor)[0], tf.float32)
    w = tf.cast(tf.shape(tensor)[1], tf.float32)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    cx = w / 2.0
    cy = h / 2.0
    # Inverse affine (maps output coords -> input coords)
    #  x' = cos*(x-cx) + sin*(y-cy) + cx
    #  y' = -sin*(x-cx) + cos*(y-cy) + cy
    transform = tf.stack([
        cos_a, sin_a, cx - cos_a * cx - sin_a * cy,
        -sin_a, cos_a, cy + sin_a * cx - cos_a * cy,
        0.0, 0.0,
    ])
    transform = tf.reshape(transform, [1, 8])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img,
        transforms=transform,
        output_shape=tf.shape(tensor)[:2],
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=fill_value,
    )
    return tf.squeeze(out, 0)
