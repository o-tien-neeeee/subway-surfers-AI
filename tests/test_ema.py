"""Tests for the EMA (exponential moving average) module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from ema import EMA


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))


class TestEMA:
    def test_initial_shadow_matches_source(self) -> None:
        """The shadow should start as a copy of
        the source's parameters."""
        net = _TinyNet()
        ema = EMA(net, decay=0.9)
        assert torch.allclose(ema.shadow["w"], net.w.detach())

    def test_update_moves_shadow(self) -> None:
        """After one update, the shadow should
        be a weighted average of the old shadow
        and the new source — per-element."""
        net = _TinyNet()
        ema = EMA(net, decay=0.9)
        # Change the source parameters.
        with torch.no_grad():
            net.w.fill_(10.0)
        ema.update()
        # Shadow[i] = 0.9 * shadow_init[i] + 0.1 *
        # source[i] = 0.9 * [1, 2, 3] + 0.1 * 10
        #            = [1.9, 2.8, 3.7]
        assert torch.allclose(ema.shadow["w"],
                                torch.tensor([1.9, 2.8, 3.7]),
                                atol=1e-5)

    def test_install_restore_round_trip(self) -> None:
        """``install`` swaps in the EMA weights
        and ``restore`` puts the originals back."""
        net = _TinyNet()
        ema = EMA(net, decay=0.5)
        # Move the EMA.
        with torch.no_grad():
            net.w.fill_(5.0)
        ema.update()
        # Shadow = 0.5 * [1, 2, 3] + 0.5 * 5
        #        = [3.0, 3.5, 4.0]
        # Install the EMA.
        ema.install()
        assert torch.allclose(net.w, torch.tensor([3.0, 3.5, 4.0]))
        # Restore.
        ema.restore()
        # The original (5.0) is back.
        assert torch.allclose(net.w, torch.tensor([5.0, 5.0, 5.0]))

    def test_state_dict_round_trip(self) -> None:
        """The EMA's state dict should round-trip
        through ``load_state_dict``."""
        net = _TinyNet()
        ema = EMA(net, decay=0.9)
        with torch.no_grad():
            net.w.fill_(2.0)
        ema.update()
        # Shadow = 0.9 * [1, 2, 3] + 0.1 * 2
        #        = [1.1, 2.0, 2.9]
        sd = ema.state_dict()
        # Now build a new EMA and load.
        ema2 = EMA(net, decay=0.5)
        ema2.load_state_dict(sd)
        assert torch.allclose(ema2.shadow["w"],
                                torch.tensor([1.1, 2.0, 2.9]),
                                atol=1e-5)

    def test_many_updates_converge(self) -> None:
        """After many updates, the shadow should
        converge to the source."""
        net = _TinyNet()
        ema = EMA(net, decay=0.9)
        with torch.no_grad():
            net.w.fill_(100.0)
        for _ in range(1000):
            ema.update()
        # The shadow should be very close to 100.
        diff = (ema.shadow["w"] - 100.0).abs().max().item()
        assert diff < 1.0
