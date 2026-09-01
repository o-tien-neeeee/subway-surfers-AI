"""Tests for the improved encoder building blocks.

These tests pin:

* Each block preserves the (B, C, H, W) shape contract
  (or squeezes to (B, C) for the pool).
* :class:`LayerScale` multiplies by the per-channel
  gamma without changing the shape.
* :class:`AttentionPool2d` returns a (B, C) vector
  and is invariant to the input spatial size (it
  works for any H, W).
* :func:`init_module` does not break the parameter
  shapes (the parameters stay the same shape and
  the orthogonal init is non-degenerate).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from encoder_blocks import (AttentionPool2d, DepthwiseSeparableBlockV2,
                              ImprovedConvBlock, LayerScale,
                              init_module, orthogonal_init_)


class TestLayerScale:
    def test_shape_preserved(self) -> None:
        ls = LayerScale(channels=8, init_value=0.1)
        x = torch.randn(2, 8, 4, 4)
        y = ls(x)
        assert y.shape == x.shape

    def test_initial_scale_value(self) -> None:
        ls = LayerScale(channels=4, init_value=0.5)
        x = torch.ones(1, 4, 2, 2)
        y = ls(x)
        # All ones * 0.5 = 0.5
        assert torch.allclose(y, torch.full_like(x, 0.5))

    def test_disable_via_init_one(self) -> None:
        ls = LayerScale(channels=4, init_value=1.0)
        x = torch.randn(1, 4, 2, 2)
        y = ls(x)
        assert torch.allclose(x, y)


class TestImprovedConvBlock:
    def test_shape_preserved(self) -> None:
        b = ImprovedConvBlock(3, 8, kernel=3, stride=1)
        x = torch.randn(2, 3, 16, 16)
        y = b(x)
        assert y.shape == (2, 8, 16, 16)

    def test_stride_changes_spatial(self) -> None:
        b = ImprovedConvBlock(3, 8, kernel=3, stride=2)
        x = torch.randn(2, 3, 16, 16)
        y = b(x)
        assert y.shape == (2, 8, 8, 8)

    def test_no_layerscale_when_disabled(self) -> None:
        b = ImprovedConvBlock(3, 8, kernel=3, use_layerscale=False)
        # No LayerScale param should be present.
        param_names = [n for n, _ in b.named_parameters()]
        assert not any("layerscale" in n for n in param_names)


class TestDepthwiseSeparableBlockV2:
    def test_shape_preserved(self) -> None:
        b = DepthwiseSeparableBlockV2(4, 8, stride=1)
        x = torch.randn(2, 4, 16, 16)
        y = b(x)
        assert y.shape == (2, 8, 16, 16)

    def test_stride_changes_spatial(self) -> None:
        b = DepthwiseSeparableBlockV2(4, 8, stride=2)
        x = torch.randn(2, 4, 16, 16)
        y = b(x)
        assert y.shape == (2, 8, 8, 8)


class TestAttentionPool2d:
    def test_returns_bc(self) -> None:
        # 8 channels divides cleanly by 4 heads.
        pool = AttentionPool2d(channels=8, n_heads=4)
        x = torch.randn(2, 8, 16, 16)
        y = pool(x)
        assert y.shape == (2, 8)

    def test_invariant_to_spatial_size(self) -> None:
        """The attention pool should work for any H, W."""
        pool = AttentionPool2d(channels=8, n_heads=4)
        pool.eval()
        # Same channels, different spatial sizes.
        with torch.no_grad():
            y1 = pool(torch.randn(1, 8, 8, 8))
            y2 = pool(torch.randn(1, 8, 16, 16))
            y3 = pool(torch.randn(1, 8, 32, 32))
        # All should be (1, 8).
        assert y1.shape == (1, 8)
        assert y2.shape == (1, 8)
        assert y3.shape == (1, 8)


class TestInitModule:
    def test_orthogonal_init_2d(self) -> None:
        t = torch.empty(8, 16)
        orthogonal_init_(t, gain=1.0)
        # Orthogonal columns: t @ t.T should be close to I.
        product = t @ t.t()
        eye = torch.eye(8)
        assert torch.allclose(product, eye, atol=1e-5)

    def test_init_module_does_not_break_shapes(self) -> None:
        b = ImprovedConvBlock(3, 8, kernel=3)
        # Snapshot the original weights.
        before = {n: p.clone() for n, p in b.named_parameters()}
        init_module(b, gain=1.0)
        # All shapes preserved.
        for n, p in b.named_parameters():
            assert p.shape == before[n].shape
        # At least one weight changed (otherwise the
        # init was a no-op, which would mean the
        # initial values were already correct, which
        # is statistically unlikely).
        any_changed = any(
            not torch.allclose(p, before[n])
            for n, p in b.named_parameters())
        assert any_changed
