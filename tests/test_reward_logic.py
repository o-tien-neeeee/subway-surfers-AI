"""Reward logic tests: survival rewards, clipping, pending hazards, hacking."""

from __future__ import annotations

import numpy as np
import pytest

from config import RewardConfig
from horizon_detector import HorizonResult
from rewards import PendingHazardTracker, SurvivalRewardCalculator


def hz(frame_id: int, ts: float, score: float, detected: bool,
       confidence: float = 0.9) -> HorizonResult:
    return HorizonResult(frame_id=frame_id, ts=ts, change_score=score,
                         raw_score=score, changed_ratio=0.1 if detected else 0.0,
                         detected=detected, confidence=confidence if detected else 0.0)


class TestSurvivalRewards:
    def test_alive_reward_proportional_to_real_time(self) -> None:
        cfg = RewardConfig(alive_per_frame=0.02, nominal_fps=30)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(10.0)
        # one perfect frame interval at 30fps
        r = calc.step(ts=10.0 + 1 / 30, action=0, horizon=hz(1, 10.033, 0.0, False))
        assert r.alive == pytest.approx(0.02, rel=0.05)
        # adaptive skipping: 4 frame intervals -> 4x reward, not 4 frames*0.02... 
        r4 = calc.step(ts=10.0 + 5 / 30, action=0, horizon=hz(2, 10.166, 0.0, False))
        assert r4.alive == pytest.approx(4 * 0.02, rel=0.05)

    def test_reward_clock_tolerance_to_frame_drops(self) -> None:
        cfg = RewardConfig(alive_per_frame=0.02, nominal_fps=30)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        r = calc.step(ts=0.2, action=0, horizon=hz(1, 0.2, 0.0, False))
        assert r.alive == pytest.approx(0.02 * 6, rel=0.05)

    def test_death_penalty_once(self) -> None:
        # v1.18: death_penalty softened to -5.0 (was -10.0) so
        # the gradient signal in the synthetic env can actually
        # compete with the survival reward.  See audit_pipeline
        # for the evidence.
        calc = SurvivalRewardCalculator(RewardConfig())
        calc.begin_episode(0.0)
        r1 = calc.step(ts=0.1, action=0, horizon=hz(1, 0.1, 0, False), died=True)
        assert r1.death == -5.0
        assert r1.total < 0
        r2 = calc.step(ts=0.2, action=0, horizon=hz(2, 0.2, 0, False), died=True)
        assert r2.death == 0.0, "death penalty must be applied exactly once"

    def test_total_clipped_to_bounds(self) -> None:
        # v1.18: alive_per_frame default bumped to 0.5 (was 0.02)
        # so the test exercises the new magnitude, but the bound
        # is still the upper clip.
        calc = SurvivalRewardCalculator(RewardConfig())
        calc.begin_episode(0.0)
        r = calc.step(ts=0.01, action=0, horizon=hz(1, 0.01, 0, False), died=True)
        assert r.death == -5.0
        # Total should be -5 (death) + tiny alive component.  With
        # alive_per_frame=0.5 and dt=0.01 the alive component is
        # 0.5 * 0.01 * 30 = 0.15, so total ≈ -4.85.  Well above
        # the lower clip so it is not clipped.
        assert r.total == pytest.approx(-5.0 + 0.15, abs=0.05)
        assert r.clipped is False
        # absurdly large alive component must clip at max
        r2 = calc.step(ts=1000.0, action=0, horizon=hz(2, 1000.0, 0, False))
        assert r2.total <= calc.cfg.reward_clip_max + 1e-9

    def test_negative_clip(self) -> None:
        cfg = RewardConfig(death_penalty=-100.0)  # beyond clip min
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        r = calc.step(ts=0.01, action=0, horizon=hz(1, 0.01, 0, False), died=True)
        assert r.total == cfg.reward_clip_min
        assert r.clipped is True


class TestPendingHazard:
    def test_dodge_after_hazard_bonus(self) -> None:
        cfg = RewardConfig(hazard_bonus=0.1, hazard_resolve_frames=2,
                           hazard_expiry_s=1.0)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        # hazard appears
        r0 = calc.step(ts=0.03, action=3, horizon=hz(10, 0.03, 20.0, True))
        assert r0.hazard == 0.0, "bonus is pending, not immediate"
        # next 2 frames: action recorded + horizon quiet -> resolve
        r1 = calc.step(ts=0.06, action=0, horizon=hz(11, 0.06, 1.0, False))
        r2 = calc.step(ts=0.09, action=0, horizon=hz(12, 0.09, 1.0, False))
        assert r2.hazard == pytest.approx(0.1)

    def test_noop_gets_no_bonus(self) -> None:
        cfg = RewardConfig(hazard_bonus=0.1, hazard_resolve_frames=2)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        calc.step(ts=0.03, action=0, horizon=hz(10, 0.03, 20.0, True))
        calc.step(ts=0.06, action=0, horizon=hz(11, 0.06, 1.0, False))
        r = calc.step(ts=0.09, action=0, horizon=hz(12, 0.09, 1.0, False))
        assert r.hazard == 0.0

    def test_hazard_not_resolving_gives_no_bonus(self) -> None:
        cfg = RewardConfig(hazard_bonus=0.1, hazard_resolve_frames=2,
                           hazard_expiry_s=5.0)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        calc.step(ts=0.03, action=1, horizon=hz(10, 0.03, 20.0, True))
        calc.step(ts=0.06, action=0, horizon=hz(11, 0.06, 20.0, True))
        r = calc.step(ts=0.09, action=0, horizon=hz(12, 0.09, 20.0, True))
        assert r.hazard == 0.0, "unresolved hazard must not pay out"

    def test_old_events_expire(self) -> None:
        cfg = RewardConfig(hazard_bonus=0.1, hazard_resolve_frames=2,
                           hazard_expiry_s=0.1)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        calc.step(ts=0.03, action=1, horizon=hz(10, 0.03, 20.0, True))
        calc.step(ts=0.06, action=0, horizon=hz(11, 0.06, 0.0, False))
        r = calc.step(ts=0.5, action=0, horizon=hz(12, 0.5, 0.0, False))
        assert r.hazard == 0.0, "expired events pay nothing"
        assert calc.hazards.stats()["hazards_expired"] >= 1

    def test_no_future_information_leak(self) -> None:
        """At the moment of the hazard frame the reward must not depend on
        whether the hazard LATER resolves — checked by comparing two worlds."""
        def world(dodge: bool, resolves: bool) -> float:
            cfg = RewardConfig(hazard_bonus=0.1, hazard_resolve_frames=2)
            calc = SurvivalRewardCalculator(cfg)
            calc.begin_episode(0.0)
            action = 3 if dodge else 0
            r0 = calc.step(ts=0.03, action=action,
                           horizon=hz(10, 0.03, 20.0, True))
            det = False if resolves else True
            calc.step(ts=0.06, action=0, horizon=hz(11, 0.06, 5.0, det))
            calc.step(ts=0.09, action=0, horizon=hz(12, 0.09, 5.0, det))
            return r0.total

        a = world(dodge=True, resolves=True)
        b = world(dodge=True, resolves=False)
        c = world(dodge=False, resolves=True)
        assert a == b == c, "reward at hazard onset must be identical in all worlds"

    def test_overlapping_hazards_do_not_double_pay(self) -> None:
        cfg = RewardConfig(hazard_bonus=0.1, hazard_resolve_frames=1)
        tracker = PendingHazardTracker(cfg)
        tracker.register(hz(1, 1.0, 30.0, True))
        tracker.register(hz(2, 1.03, 30.0, True))  # extends, no new event
        assert len(tracker.events) == 1
        tracker.on_frame(hz(3, 1.06, 0.0, False), 2)
        assert tracker.bonuses_granted == 1

    def test_stats_tracking(self) -> None:
        cfg = RewardConfig(hazard_bonus=0.1, hazard_resolve_frames=1,
                           hazard_expiry_s=0.05)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        calc.step(ts=0.03, action=2, horizon=hz(1, 0.03, 30.0, True))
        calc.step(ts=0.06, action=0, horizon=hz(2, 0.06, 0.0, False))
        s = calc.hazards.stats()
        assert s["hazards_total"] == 1
        assert s["hazards_resolved"] == 1


class TestPixelDiffAblation:
    def test_disabled_by_default(self) -> None:
        calc = SurvivalRewardCalculator(RewardConfig())
        calc.begin_episode(0.0)
        a = np.zeros((84, 84), dtype=np.uint8)
        b = np.full((84, 84), 255, dtype=np.uint8)
        r = calc.step(ts=0.03, action=0, horizon=hz(1, 0.03, 0, False),
                      ground_gray=b, prev_ground_gray=a)
        assert r.pixel_diff == 0.0

    def test_tightly_clipped_when_enabled(self) -> None:
        cfg = RewardConfig(use_pixel_diff_reward=True, pixel_diff_clip=0.01)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        a = np.zeros((84, 84), dtype=np.uint8)
        b = np.full((84, 84), 255, dtype=np.uint8)  # full-screen flash
        r = calc.step(ts=0.03, action=0, horizon=hz(1, 0.03, 0, False),
                      ground_gray=b, prev_ground_gray=a)
        assert r.pixel_diff == pytest.approx(0.01)

    def test_reward_hacking_resistance(self) -> None:
        """A 'hacker' flashing the screen every frame must not farm more
        reward than the clip allows; total stays within bounds."""
        cfg = RewardConfig(use_pixel_diff_reward=True, pixel_diff_clip=0.01,
                           alive_per_frame=0.02,
                           # disable curriculum for this test so the bound
                           # is the same as before the curriculum change.
                           curriculum_milestones=(),
                           curriculum_bonus=0.0)
        calc = SurvivalRewardCalculator(cfg)
        calc.begin_episode(0.0)
        rng = np.random.default_rng(0)
        prev = rng.integers(0, 256, (84, 84), dtype=np.uint8)
        total = 0.0
        for i in range(300):  # 10 seconds of full-screen flashing
            cur = rng.integers(0, 256, (84, 84), dtype=np.uint8)
            r = calc.step(ts=(i + 1) / 30.0, action=0,
                          horizon=hz(i, (i + 1) / 30.0, 0, False),
                          ground_gray=cur, prev_ground_gray=prev)
            prev = cur
            total += r.total
            assert r.total <= cfg.reward_clip_max
        # pixel component can contribute at most 300*0.01 = 3.0 total
        assert total <= 0.02 * 300 + 3.0 + 1e-6

    def test_log_components(self) -> None:
        calc = SurvivalRewardCalculator(RewardConfig())
        calc.begin_episode(0.0)
        r = calc.step(ts=0.03, action=0, horizon=hz(1, 0.03, 0, False))
        d = r.to_dict()
        assert set(d) == {"alive", "death", "hazard", "pixel_diff",
                          "curriculum", "total", "clipped"}
