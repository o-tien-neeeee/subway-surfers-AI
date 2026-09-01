"""Tests for the AvgL1Norm module (TD7's normaliser).

These tests pin:
* AvgL1Norm divides each component by the
  average L1 norm of the vector.
* The output has unit mean L1 norm along the
  normalised dimension.
* The wrapped Linear + AvgL1Norm layer
  produces the right shape.
* The functional form matches the module.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from avg_l1_norm import (AvgL1Norm, AvgL1NormLinear,
                            avg_l1_normalize)


class TestAvgL1Norm:
    def test_output_shape(self) -> None:
        norm = AvgL1Norm(dim=-1)
        x = torch.randn(8, 16)
        y = norm(x)
        assert y.shape == x.shape

    def test_unit_l1_norm(self) -> None:
        """After normalisation, the mean L1 norm
        along the chosen dim should be ~1."""
        torch.manual_seed(0)
        norm = AvgL1Norm(dim=-1)
        x = torch.randn(64, 32) * 5.0  # big numbers
        y = norm(x)
        # mean(|y|) along the last dim should be ~1.
        l1 = y.abs().mean(dim=-1)
        assert torch.allclose(l1, torch.ones_like(l1), atol=1e-3)

    def test_handles_zero_input(self) -> None:
        """Zero input should not cause a
        divide-by-zero (the eps handles it)."""
        norm = AvgL1Norm(dim=-1, eps=1e-6)
        x = torch.zeros(4, 8)
        y = norm(x)
        # All zeros should remain zero.
        assert torch.allclose(y, torch.zeros_like(y))

    def test_normalisation_preserves_sign(self) -> None:
        """Positive inputs stay positive, negative
        inputs stay negative."""
        norm = AvgL1Norm(dim=-1)
        x = torch.tensor([[1.0, 2.0, -3.0, 4.0]])
        y = norm(x)
        assert (y[0, :2] > 0).all()
        assert y[0, 2] < 0
        assert y[0, 3] > 0


class TestAvgL1NormLinear:
    def test_shape_preserved(self) -> None:
        layer = AvgL1NormLinear(in_features=8, out_features=16)
        x = torch.randn(4, 8)
        y = layer(x)
        assert y.shape == (4, 16)

    def test_output_l1_normalised(self) -> None:
        """The output of AvgL1NormLinear should
        have unit mean L1 norm along the last
        dim."""
        torch.manual_seed(0)
        layer = AvgL1NormLinear(in_features=8, out_features=32)
        x = torch.randn(16, 8) * 10.0
        y = layer(x)
        l1 = y.abs().mean(dim=-1)
        assert torch.allclose(l1, torch.ones_like(l1), atol=1e-3)


class TestFunctional:
    def test_functional_matches_module(self) -> None:
        torch.manual_seed(0)
        norm = AvgL1Norm(dim=-1)
        x = torch.randn(4, 8)
        y_mod = norm(x)
        y_func = avg_l1_normalize(x, dim=-1)
        assert torch.allclose(y_mod, y_func)
