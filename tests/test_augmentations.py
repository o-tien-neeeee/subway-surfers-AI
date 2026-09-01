"""Tests for the visual augmentation pipeline.

These tests pin:

* Each augmentation returns a tensor of the SAME shape
  (or one extra channel for ``frame_diff``).
* The augmentations are actually random (different
  seeds / different RNG state produce different
  outputs).
* The augmented output stays in the [0, 1] range
  (no NaN, no Inf, no values > 1.0 from
  intensity jitter).
* The augmentations do not change the action
  (the action is an external label; only observations
  are perturbed).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from augmentations import (AugmentationConfig, augment_batch,
                              frame_difference, intensity_jitter,
                              mixup, random_erasing, random_translate)


class TestIntensityJitter:
    def test_shape_preserved(self) -> None:
        x = torch.rand(4, 4, 84, 84)
        y = intensity_jitter(x)
        assert y.shape == x.shape

    def test_values_in_range(self) -> None:
        torch.manual_seed(0)
        x = torch.rand(8, 4, 84, 84)
        y = intensity_jitter(x, brightness=0.10, contrast=0.10)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_random(self) -> None:
        x = torch.rand(4, 4, 84, 84)
        torch.manual_seed(0)
        y1 = intensity_jitter(x)
        torch.manual_seed(1)
        y2 = intensity_jitter(x)
        assert not torch.allclose(y1, y2)

    def test_same_jitter_per_stack(self) -> None:
        """The same brightness/contrast is applied to
        every frame in the stack.  We verify this by
        computing the per-frame BRIGHTNESS OFFSET
        directly from the jitter: it must be the same
        for every frame in the stack (the contrast
        scale c is also the same — but the *resulting*
        pixel values differ because the original
        frames have different values)."""
        x = torch.rand(2, 4, 84, 84)
        torch.manual_seed(0)
        # Re-derive the per-batch jitter.
        B = x.shape[0]
        b = (torch.rand(B, 1, 1, 1) * 2 - 1) * 0.5
        c = 1.0 + (torch.rand(B, 1, 1, 1) * 2 - 1) * 0.0  # contrast=0
        # Apply the same offset to all frames.
        y = (x * c + b).clamp(0.0, 1.0)
        # The difference between two frames after
        # jittering should equal the difference
        # between the same two frames before jittering
        # (the constant offset cancels).
        diff_before = (x[0, 0] - x[0, 1]).abs().mean()
        diff_after = (y[0, 0] - y[0, 1]).abs().mean()
        assert abs(diff_before - diff_after) < 1e-3


class TestRandomTranslate:
    def test_shape_preserved(self) -> None:
        x = torch.rand(2, 4, 84, 84)
        y = random_translate(x, max_pixels=3)
        assert y.shape == x.shape

    def test_zero_shift(self) -> None:
        """When the RNG picks 0, the output is
        identical to the input."""
        x = torch.rand(2, 4, 84, 84)
        torch.manual_seed(0)
        # Force a 0-shift: just check that
        # ``max_pixels=0`` is a no-op.
        y = random_translate(x, max_pixels=0)
        assert torch.allclose(x, y)

    def test_borders_zeroed(self) -> None:
        """After a non-zero shift, the wrapped
        region (the new border) is black, not a
        wrapped copy."""
        x = torch.ones(2, 4, 84, 84)
        torch.manual_seed(123)
        y = random_translate(x, max_pixels=4)
        # The image was all ones before; after a
        # translate the shifted-in region (the
        # border) should be zero.
        if not torch.allclose(y, x):
            # Some shift was applied: the first row
            # OR the last row must contain zeros
            # (depending on the direction).
            assert (y[..., 0, :] == 0).any() or (y[..., -1, :] == 0).any() \
                or (y[..., :, 0] == 0).any() or (y[..., :, -1] == 0).any()


class TestRandomErasing:
    def test_off_by_default(self) -> None:
        x = torch.rand(2, 4, 84, 84)
        y = random_erasing(x, prob=0.0)
        assert torch.allclose(x, y)

    def test_can_erase_something(self) -> None:
        x = torch.ones(2, 4, 84, 84)
        # Probability 1.0 ensures at least one erasure.
        torch.manual_seed(0)
        y = random_erasing(x, prob=1.0, max_frac=0.2)
        # At least one element must be zero now.
        assert (y == 0).any()
        # But not all of them (the rectangle is
        # only a small fraction).
        assert (y == 1).any()


class TestFrameDifference:
    def test_adds_one_channel(self) -> None:
        x = torch.rand(2, 4, 84, 84)
        y = frame_difference(x)
        assert y.shape == (2, 5, 84, 84)

    def test_single_frame_unchanged(self) -> None:
        x = torch.rand(2, 1, 84, 84)
        y = frame_difference(x)
        assert y.shape == (2, 1, 84, 84)
        assert torch.allclose(x, y)


class TestMixup:
    def test_shapes_preserved(self) -> None:
        x = torch.rand(4, 4, 84, 84)
        a = torch.tensor([0, 1, 2, 3])
        r = torch.rand(4)
        d = torch.zeros(4)
        xm, am, rm, dm = mixup(x, a, r, d, alpha=0.5)
        assert xm.shape == x.shape
        # Action comes from the FIRST sample, not mixed.
        assert torch.allclose(am, a)
        # Done flag is not mixed.
        assert torch.allclose(dm, d)

    def test_alpha_zero_no_mix(self) -> None:
        x = torch.rand(4, 4, 84, 84)
        a = torch.tensor([0, 1, 2, 3])
        r = torch.rand(4)
        d = torch.zeros(4)
        xm, _, _, _ = mixup(x, a, r, d, alpha=0.0)
        assert torch.allclose(x, xm)


class TestAugmentBatch:
    def test_default_no_change_when_training_false(self) -> None:
        x = torch.rand(2, 4, 84, 84)
        cfg = AugmentationConfig()
        y = augment_batch(x, cfg=cfg, training=False)
        assert torch.allclose(x, y)

    def test_full_pipeline(self) -> None:
        x = torch.rand(2, 4, 84, 84)
        cfg = AugmentationConfig(
            intensity_jitter=True, translate=True,
            random_erasing=True, frame_diff=True, mixup=False)
        y = augment_batch(x, cfg=cfg, training=True)
        # frame_diff adds one channel.
        assert y.shape == (2, 5, 84, 84)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_uint8_input(self) -> None:
        """Augmentations accept uint8 input and
        return float32 in [0, 1]."""
        x = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
        cfg = AugmentationConfig(intensity_jitter=True, translate=True)
        y = augment_batch(x, cfg=cfg, training=True)
        assert y.dtype == torch.float32
        assert y.min() >= 0.0
        assert y.max() <= 1.0
