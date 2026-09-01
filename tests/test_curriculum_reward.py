"""Tests for the survival-curriculum reward.

The agent gets a tiny one-shot bonus the first time it crosses each
survival-time threshold.  These tests pin the per-episode semantics:
one pay per milestone, re-armed on episode start, ignored when the
config is empty or the bonus is zero.
"""

from __future__ import annotations

import time

from config import BotConfig, NOOP, LEFT
from rewards import SurvivalRewardCalculator


def _hz(ts: float, detected: bool = False):
    from horizon_detector import HorizonResult
    return HorizonResult(
        frame_id=int(ts * 30), ts=ts,
        change_score=0.0, raw_score=0.0, changed_ratio=0.0,
        detected=detected, confidence=0.0,
    )


class TestCurriculumReward:
    def test_no_milestones_pays_nothing(self) -> None:
        cfg = BotConfig().reward
        cfg.curriculum_milestones = ()
        cfg.curriculum_bonus = 0.5
        rc = SurvivalRewardCalculator(cfg)
        rc.begin_episode(0.0)
        for t in (1.0, 5.0, 20.0, 60.0):
            bd = rc.step(t, NOOP, _hz(t))
            assert bd.curriculum == 0.0

    def test_bonus_zero_disables_milestones(self) -> None:
        cfg = BotConfig().reward
        cfg.curriculum_milestones = (2.0, 5.0)
        cfg.curriculum_bonus = 0.0
        rc = SurvivalRewardCalculator(cfg)
        rc.begin_episode(0.0)
        for t in (2.5, 6.0):
            assert rc.step(t, NOOP, _hz(t)).curriculum == 0.0

    def test_first_crossing_pays_bonus(self) -> None:
        cfg = BotConfig().reward
        cfg.curriculum_milestones = (2.0, 5.0, 10.0)
        cfg.curriculum_bonus = 0.5
        rc = SurvivalRewardCalculator(cfg)
        rc.begin_episode(0.0)
        # before the first milestone: nothing
        assert rc.step(1.5, NOOP, _hz(1.5)).curriculum == 0.0
        # crossing 2s: +0.5
        assert rc.step(2.5, NOOP, _hz(2.5)).curriculum == pytest_approx(0.5)
        # same milestone again: nothing
        assert rc.step(3.0, NOOP, _hz(3.0)).curriculum == 0.0
        assert rc.step(4.9, NOOP, _hz(4.9)).curriculum == 0.0
        # crossing 5s: +0.5
        assert rc.step(5.1, NOOP, _hz(5.1)).curriculum == pytest_approx(0.5)
        # crossing 10s: +0.5
        assert rc.step(10.5, NOOP, _hz(10.5)).curriculum == pytest_approx(0.5)
        # past the schedule: nothing
        assert rc.step(60.0, NOOP, _hz(60.0)).curriculum == 0.0

    def test_two_milestones_in_one_frame_pay_both(self) -> None:
        cfg = BotConfig().reward
        cfg.curriculum_milestones = (2.0, 5.0)
        cfg.curriculum_bonus = 0.5
        rc = SurvivalRewardCalculator(cfg)
        rc.begin_episode(0.0)
        bd = rc.step(6.0, NOOP, _hz(6.0))
        assert bd.curriculum == pytest_approx(1.0), (
            "two milestones in one frame must pay both")

    def test_re_arms_on_new_episode(self) -> None:
        cfg = BotConfig().reward
        cfg.curriculum_milestones = (2.0,)
        cfg.curriculum_bonus = 0.5
        rc = SurvivalRewardCalculator(cfg)
        rc.begin_episode(0.0)
        assert rc.step(2.5, NOOP, _hz(2.5)).curriculum == pytest_approx(0.5)
        rc.begin_episode(0.0)
        assert rc.step(2.5, NOOP, _hz(2.5)).curriculum == pytest_approx(0.5), (
            "the same milestone must re-arm for the next episode")

    def test_milestone_counter_increments(self) -> None:
        cfg = BotConfig().reward
        cfg.curriculum_milestones = (2.0, 5.0, 10.0)
        cfg.curriculum_bonus = 0.5
        rc = SurvivalRewardCalculator(cfg)
        rc.begin_episode(0.0)
        rc.step(2.5, NOOP, _hz(2.5))
        rc.step(5.0, NOOP, _hz(5.0))
        rc.step(11.0, NOOP, _hz(11.0))
        assert rc.milestones_crossed == 3

    def test_curriculum_does_not_crash_with_uninitialised_start(self) -> None:
        """Calling step() before begin_episode is allowed by legacy code paths."""
        cfg = BotConfig().reward
        cfg.curriculum_milestones = (2.0,)
        cfg.curriculum_bonus = 0.5
        rc = SurvivalRewardCalculator(cfg)
        # No begin_episode() — should be a no-op rather than a crash.
        bd = rc.step(time.monotonic(), NOOP, _hz(time.monotonic()))
        assert bd.curriculum == 0.0


def pytest_approx(x: float):
    """Local alias to avoid pulling pytest into this tiny test module."""
    import pytest
    return pytest.approx(x)
