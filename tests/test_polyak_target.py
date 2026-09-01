"""Tests for the Polyak (soft) target update.

The Polyak update rule is::

    target = (1 - tau) * target + tau * online

with ``tau=0.005`` (TD3 default) by default.  This
test pins:

* The legacy path (hard copy every N steps) still
  works.
* The new path (Polyak every step) is enabled by
  ``cfg.polyak_target=True``.
* After a Polyak step the target is *strictly between*
  the old target and the new online (the convex
  combination property).
* The mixing coefficient matches the config.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from agent_distributional import DistributionalDoubleDQNAgent
from config import RLConfig


class _ToyOnline(nn.Module):
    """A minimal two-parameter network for the
    polyak test (much smaller than QR-DQN)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))


class _ToyTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([10.0, 20.0, 30.0]))


def _make_agent_with_polyak(polyak: bool, tau: float = 0.005) -> DistributionalDoubleDQNAgent:
    cfg = RLConfig(profile="strict_lite", gamma=0.99,
                    num_quantiles=11,
                    learning_rate=1e-4,
                    target_update_every=1000,
                    polyak_target=polyak, polyak_tau=tau)
    agent = DistributionalDoubleDQNAgent(
        "strict_lite", cfg, in_frames=4, size=84,
        num_quantiles=11, device="cpu", seed=0)
    # Replace the QR-DQN head with a tiny toy network
    # so we can verify the polyak math directly.
    agent.online = _ToyOnline()
    agent.target = _ToyTarget()
    return agent


class TestPolyakTarget:
    def test_legacy_hard_copy_unchanged(self) -> None:
        """The default (polyak_target=False) keeps
        the original hard-copy behaviour."""
        agent = _make_agent_with_polyak(polyak=False)
        # Step 1: maybe_sync_target should NOT sync
        # (because update_count is 0).
        synced = agent.maybe_sync_target()
        assert synced is False
        # The target is unchanged.
        assert torch.allclose(agent.target.w,
                                torch.tensor([10.0, 20.0, 30.0]))
        # Force a sync at the hard-update step.
        agent.update_count = 1000
        synced = agent.maybe_sync_target()
        assert synced is True
        assert torch.allclose(agent.target.w,
                                torch.tensor([1.0, 2.0, 3.0]))

    def test_polyak_convex_combination(self) -> None:
        """Polyak update is a strict convex combination
        of the old target and the new online."""
        agent = _make_agent_with_polyak(polyak=True, tau=0.1)
        # update_count must be > 0 for maybe_sync_target
        # to do anything (matches the parent's check).
        agent.update_count = 1
        old_target = agent.target.w.clone()
        new_online = agent.online.w.clone()
        # Sync.
        agent.maybe_sync_target()
        # The new target is (1 - 0.1) * old + 0.1 * online.
        expected = 0.9 * old_target + 0.1 * new_online
        assert torch.allclose(agent.target.w, expected)

    def test_polyak_small_tau_stays_close_to_old(self) -> None:
        """With tau=0.005 the target only nudges 0.5%
        toward the online net, so the target stays
        close to the OLD target."""
        agent = _make_agent_with_polyak(polyak=True, tau=0.005)
        agent.update_count = 1
        old_target = agent.target.w.clone()
        agent.maybe_sync_target()
        diff = (agent.target.w - old_target).abs().max().item()
        # The online net is [1, 2, 3] and the old
        # target is [10, 20, 30]; the per-step diff
        # per dim is 0.005 * 9 = 0.045 (for the first
        # dim, where the gap is 9).  The third dim has
        # a gap of 27, so the diff is 0.005 * 27 = 0.135.
        assert diff < 0.2

    def test_polyak_many_steps_converges(self) -> None:
        """After many Polyak steps the target should
        converge to the online net (the half-life is
        ~log(0.5)/log(1-tau) = 138 steps for tau=0.005;
        1000 steps is plenty)."""
        agent = _make_agent_with_polyak(polyak=True, tau=0.1)
        agent.update_count = 0
        for _ in range(1000):
            agent.update_count += 1
            agent.maybe_sync_target()
        # After 1000 steps with tau=0.1, the target
        # should be (almost) the online net.
        # The remaining error is (1 - tau)^N = 0.9^1000 ≈ 0
        diff = (agent.target.w - agent.online.w).abs().max().item()
        assert diff < 1e-3
