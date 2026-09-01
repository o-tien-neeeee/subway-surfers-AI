"""Tests for the ``disable_exploration_after_bc`` flag.

The audit_bc_then_rl.py proved that even 15% ε-greedy
*after* BC destroys the BC-pretrained policy (30s → 14.6s).
The fix is the ``disable_exploration_after_bc: bool = True``
default in :class:`config.RLConfig`, which makes
:meth:`agent.effective_epsilon` return 0.0 when BC has
produced a policy.

These tests pin both the old ("epsilon_after_bc cap")
and new ("zero epsilon after BC") behaviours.
"""

from __future__ import annotations

import pytest

from agent import effective_epsilon, epsilon_for_frame
from config import RLConfig


class TestEffectiveEpsilon:
    def test_no_bc_uses_normal_schedule(self) -> None:
        cfg = RLConfig(profile="strict_lite", gamma=0.99,
                        disable_exploration_after_bc=True,
                        epsilon_start=1.0, epsilon_end=0.05,
                        epsilon_decay_frames=50_000)
        eps = effective_epsilon(0, cfg, bc_pretrained=0.0)
        # Frame 0 = epsilon_start = 1.0 (full
        # exploration before any learning).
        assert abs(eps - 1.0) < 0.01

    def test_with_bc_and_disable_flag_returns_zero(self) -> None:
        cfg = RLConfig(profile="strict_lite", gamma=0.99,
                        disable_exploration_after_bc=True,
                        epsilon_start=1.0, epsilon_end=0.05,
                        epsilon_decay_frames=50_000,
                        epsilon_after_bc=0.15)
        eps = effective_epsilon(0, cfg, bc_pretrained=1.0)
        # BC + disable flag = pure exploitation.
        assert eps == 0.0

    def test_with_bc_and_disable_flag_off_uses_cap(self) -> None:
        """Legacy path: epsilon_after_bc caps exploration
        at 0.15 (the old behaviour)."""
        cfg = RLConfig(profile="strict_lite", gamma=0.99,
                        disable_exploration_after_bc=False,
                        epsilon_start=1.0, epsilon_end=0.05,
                        epsilon_decay_frames=50_000,
                        epsilon_after_bc=0.15)
        eps = effective_epsilon(0, cfg, bc_pretrained=1.0)
        # At frame 0 the normal schedule says 1.0;
        # the cap brings it down to 0.15.
        assert abs(eps - 0.15) < 0.01

    def test_with_bc_decays_below_cap(self) -> None:
        cfg = RLConfig(profile="strict_lite", gamma=0.99,
                        disable_exploration_after_bc=False,
                        epsilon_start=1.0, epsilon_end=0.0,
                        epsilon_decay_frames=1000,
                        epsilon_after_bc=0.15)
        # Mid-way through the schedule the normal
        # epsilon should be below the cap.
        eps = effective_epsilon(500, cfg, bc_pretrained=1.0)
        normal = epsilon_for_frame(500, cfg)
        assert eps < normal
        assert eps <= 0.15
