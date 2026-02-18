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

import numpy as np
import pytest
import tensorflow as tf

from riawelc.data import augmentation as aug_module
from riawelc.data.augmentation import augment_segmentation_pair

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seg_image() -> tf.Tensor:
    """Synthetic 224x224x1 float32 image in [0, 255] with a gradient pattern."""
    rng = np.random.default_rng(42)
    arr = rng.uniform(0, 255, size=(224, 224, 1)).astype(np.float32)
    return tf.constant(arr)


@pytest.fixture
def seg_mask() -> tf.Tensor:
    """Synthetic 224x224x1 binary mask: white rectangle on black background."""
    mask = np.zeros((224, 224, 1), dtype=np.float32)
    mask[50:180, 60:170, 0] = 1.0
    return tf.constant(mask)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAugmentSegmentationPair:
    def test_shape_preservation(self, seg_image: tf.Tensor, seg_mask: tf.Tensor) -> None:
        """Output shapes must match input shapes."""
        aug_img, aug_msk = augment_segmentation_pair(seg_image, seg_mask)
        assert aug_img.shape == seg_image.shape
        assert aug_msk.shape == seg_mask.shape

    def test_output_dtype(self, seg_image: tf.Tensor, seg_mask: tf.Tensor) -> None:
        """Output dtype must remain float32."""
        aug_img, aug_msk = augment_segmentation_pair(seg_image, seg_mask)
        assert aug_img.dtype == tf.float32
        assert aug_msk.dtype == tf.float32

    def test_mask_values_in_range(self, seg_image: tf.Tensor, seg_mask: tf.Tensor) -> None:
        """Mask values should stay in [0, 1] after augmentation."""
        aug_img, aug_msk = augment_segmentation_pair(seg_image, seg_mask)
        assert float(tf.reduce_min(aug_msk)) >= 0.0
        assert float(tf.reduce_max(aug_msk)) <= 1.0

    def test_image_no_nan(self, seg_image: tf.Tensor, seg_mask: tf.Tensor) -> None:
        """Output image must not contain NaN values."""
        aug_img, aug_msk = augment_segmentation_pair(seg_image, seg_mask)
        assert not tf.reduce_any(tf.math.is_nan(aug_img))
        assert not tf.reduce_any(tf.math.is_nan(aug_msk))

    def test_image_values_reasonable(self, seg_image: tf.Tensor, seg_mask: tf.Tensor) -> None:
        """Image values should stay in a reasonable range after augmentation."""
        aug_img, _ = augment_segmentation_pair(seg_image, seg_mask)
        # brightness/contrast may shift values slightly outside [0,255]
        # but they shouldn't explode
        assert float(tf.reduce_min(aug_img)) > -50.0
        assert float(tf.reduce_max(aug_img)) < 300.0

    def test_spatial_consistency(self) -> None:
        """Spatial transforms must affect image and mask identically.

        Strategy: create an image and mask that are identical (both binary).
        After a spatial-only transform, they should still be identical.
        We run multiple seeds to increase the chance of triggering a flip/rotation.
        """
        pattern = np.zeros((224, 224, 1), dtype=np.float32)
        # Asymmetric L-shape so flips are detectable
        pattern[20:100, 20:40, 0] = 1.0   # vertical bar
        pattern[80:100, 20:120, 0] = 1.0   # horizontal bar

        image = tf.constant(pattern * 255.0)  # scale to image range
        mask = tf.constant(pattern)

        # Run several times — the module-level RNG advances each call
        for _ in range(5):
            aug_img, aug_msk = augment_segmentation_pair(image, mask)
            # Normalise the augmented image back to [0,1] range for comparison
            aug_img_norm = aug_img / 255.0
            # Due to intensity transforms on image only, we can't compare values
            # directly, but we CAN compare where pixels are nonzero (spatial layout)
            img_nonzero = tf.cast(aug_img_norm > 0.01, tf.float32)
            msk_nonzero = tf.cast(aug_msk > 0.01, tf.float32)
            # The nonzero patterns should be identical (same spatial transform)
            diff = tf.reduce_sum(tf.abs(img_nonzero - msk_nonzero))
            # Allow a small tolerance for interpolation artefacts at edges
            assert float(diff) < 224 * 2  # at most ~2 rows of edge pixels

    def test_mask_mostly_binary(self, seg_image: tf.Tensor, seg_mask: tf.Tensor) -> None:
        """Most mask pixels should remain 0 or 1 after augmentation."""
        aug_module._seg_aug_rng = tf.random.Generator.from_seed(9)
        aug_img, aug_msk = augment_segmentation_pair(seg_image, seg_mask)
        binary_mask = tf.logical_or(
            tf.abs(aug_msk) < 0.05,
            tf.abs(aug_msk - 1.0) < 0.05,
        )
        binary_fraction = float(tf.reduce_mean(tf.cast(binary_mask, tf.float32)))
        # At least 90% of pixels should be near 0 or 1
        assert binary_fraction > 0.90

    def test_determinism_with_same_seed(
        self, seg_image: tf.Tensor, seg_mask: tf.Tensor
    ) -> None:
        """Resetting the module-level RNG to the same seed reproduces output."""
        aug_module._seg_aug_rng = tf.random.Generator.from_seed(123)
        out_img_1, out_msk_1 = augment_segmentation_pair(seg_image, seg_mask)

        aug_module._seg_aug_rng = tf.random.Generator.from_seed(123)
        out_img_2, out_msk_2 = augment_segmentation_pair(seg_image, seg_mask)

        np.testing.assert_array_equal(out_img_1.numpy(), out_img_2.numpy())
        np.testing.assert_array_equal(out_msk_1.numpy(), out_msk_2.numpy())

    def test_different_seeds_differ(
        self, seg_image: tf.Tensor, seg_mask: tf.Tensor
    ) -> None:
        """Consecutive calls with advancing RNG produce different results."""
        aug_module._seg_aug_rng = tf.random.Generator.from_seed(9)
        out_img_1, _ = augment_segmentation_pair(seg_image, seg_mask)
        out_img_2, _ = augment_segmentation_pair(seg_image, seg_mask)

        assert not np.array_equal(out_img_1.numpy(), out_img_2.numpy())

    def test_intensity_transforms_image_only(self) -> None:
        """An all-zero mask stays all-zero (spatial transforms of zeros = zeros).

        This verifies that intensity transforms (brightness, contrast) are
        not applied to the mask.
        """
        flat_img = tf.constant(np.full((224, 224, 1), 128.0, dtype=np.float32))
        zero_mask = tf.constant(np.zeros((224, 224, 1), dtype=np.float32))

        aug_module._seg_aug_rng = tf.random.Generator.from_seed(9)
        _, out_mask = augment_segmentation_pair(flat_img, zero_mask)

        np.testing.assert_array_equal(
            out_mask.numpy(),
            np.zeros((224, 224, 1), dtype=np.float32),
        )
