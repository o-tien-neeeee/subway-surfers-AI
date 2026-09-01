"""Tests for the ImprovedQuantileDuelingDQN.

These tests pin:

* The forward shape contract (``[B, F, H, W]`` ->
  ``[B, A, N]``) for the standard and attention
  pool profiles.
* The LayerScale initial values match the config.
* The orthogonal init produces a sensible weight
  distribution (not all zeros, not all ones).
* The parameter count for each profile is in the
  expected range (no runaway growth).
"""

from __future__ import annotations

import pytest
import torch

from improved_dqn import (IMPROVED_PROFILES,
                            ImprovedQuantileDuelingDQN,
                            build_improved_agent, mid_quantiles)


class TestImprovedQuantileDuelingDQN:
    @pytest.mark.parametrize("profile", list(IMPROVED_PROFILES))
    def test_forward_shape(self, profile: str) -> None:
        net = ImprovedQuantileDuelingDQN(profile, in_frames=4,
                                            size=84, num_quantiles=11,
                                            noisy=False)
        x = torch.rand(2, 4, 84, 84)
        y = net(x)
        assert y.shape == (2, 5, 11)

    @pytest.mark.parametrize("profile", list(IMPROVED_PROFILES))
    def test_q_values_shape(self, profile: str) -> None:
        net = ImprovedQuantileDuelingDQN(profile, in_frames=4,
                                            size=84, num_quantiles=11,
                                            noisy=False)
        x = torch.rand(2, 4, 84, 84)
        q = net.q_values(x)
        assert q.shape == (2, 5)

    def test_layer_scale_initial_values(self) -> None:
        """The LayerScale gamma should be initialised
        to ``layerscale_init`` (1e-2 by default for
        the 4-block encoder)."""
        net = ImprovedQuantileDuelingDQN(
            "improved_strict_lite", in_frames=4, size=84,
            num_quantiles=11, noisy=False)
        for m in net.modules():
            if isinstance(m, torch.nn.Module) and \
                    hasattr(m, "gamma") and m.gamma.ndim == 1:
                # Each LayerScale param should be ~1e-2.
                assert (m.gamma.abs() < 5e-1).all()
                assert (m.gamma.abs() > 1e-4).all()

    def test_orthogonal_init_produces_nonzero_weights(self) -> None:
        """The orthogonal init should give the
        dueling head's value-stream Linear weights a
        non-trivial distribution (not all zeros)."""
        net = ImprovedQuantileDuelingDQN(
            "improved_strict_lite", in_frames=4, size=84,
            num_quantiles=11, noisy=False)
        # The first Linear in the value stream.
        w = net.head.value_stream[0].weight
        assert w.abs().sum() > 0
        # The columns should be roughly unit-norm
        # (the orthogonal init's defining property).
        col_norms = w.norm(dim=0)
        assert (col_norms > 0.5).all()
        assert (col_norms < 2.0).all()

    @pytest.mark.parametrize("profile", list(IMPROVED_PROFILES))
    def test_param_count_in_expected_range(self, profile: str) -> None:
        """The improved profiles should stay in a
        reasonable parameter range — not 10k (too
        small to be useful) and not 1M (too slow on
        CPU)."""
        net = ImprovedQuantileDuelingDQN(profile, in_frames=4,
                                            size=84, num_quantiles=51,
                                            noisy=True)
        n = net.count_params()
        assert 30_000 < n < 1_000_000, (
            f"profile {profile} has {n} params, outside the "
            f"30k-1M range")

    def test_with_frame_diff_channel(self) -> None:
        """When the input has F+1 channels (after the
        frame-difference augmentation) the network
        still works because we pass
        ``in_frames_override``."""
        net = ImprovedQuantileDuelingDQN(
            "improved_strict_lite", in_frames=4, size=84,
            num_quantiles=11, noisy=False,
            in_frames_override=5)
        x = torch.rand(2, 5, 84, 84)
        y = net(x)
        assert y.shape == (2, 5, 11)

    def test_noisy_nets_make_resampling(self) -> None:
        """Two forward passes with the noisy layers
        in train mode should give different outputs
        (because the factorised Gaussian noise is
        resampled each call)."""
        net = ImprovedQuantileDuelingDQN(
            "improved_strict_lite", in_frames=4, size=84,
            num_quantiles=11, noisy=True)
        net.train()
        x = torch.rand(2, 4, 84, 84)
        with torch.no_grad():
            y1 = net.q_values(x)
            y2 = net.q_values(x)
        # The two outputs must differ (otherwise the
        # noise is not being sampled).
        assert not torch.allclose(y1, y2)


class TestMidQuantiles:
    def test_mid_quantiles_range(self) -> None:
        taus = mid_quantiles(11)
        assert taus.shape == (11,)
        assert taus[0] > 0
        assert taus[-1] < 1
        # Spacing is uniform.
        diffs = (taus[1:] - taus[:-1])
        assert torch.allclose(diffs, diffs[0:1])


class TestBuildImprovedAgent:
    def test_factory(self) -> None:
        net = build_improved_agent("improved_strict_lite",
                                      num_quantiles=11, seed=0)
        assert isinstance(net, ImprovedQuantileDuelingDQN)
        assert net.profile == "improved_strict_lite"
