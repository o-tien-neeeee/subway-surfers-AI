"""Action scheduler tests: cadence, danger override, expiry, buffering."""

from __future__ import annotations

import pytest

from action_scheduler import ActionScheduler
from config import SchedulerConfig


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += ms / 1000.0


def make_scheduler(**kw) -> tuple[ActionScheduler, FakeClock]:
    clock = FakeClock()
    cfg = SchedulerConfig(min_decision_frames=2, max_decision_frames=4,
                          danger_decision_frames=1, buffer_size=1)
    s = ActionScheduler(cfg, now=clock, cooldown_ms=kw.pop("cooldown_ms", 110.0),
                        **kw)
    s.ttl_ms = kw.pop("ttl_ms", 140.0)
    return s, clock


class TestCadence:
    def test_normal_cadence_low_load(self) -> None:
        s, _clock = make_scheduler()
        s.set_load(0.1)  # low load -> min frames (2)
        gates = [s.on_frame(danger=False) for _ in range(7)]
        assert gates == [False, True, False, True, False, True, False]

    def test_normal_cadence_high_load(self) -> None:
        s, _ = make_scheduler()
        s.set_load(0.95)  # high load -> max frames (4)
        gates = [s.on_frame(danger=False) for _ in range(9)]
        true_idx = [i for i, g in enumerate(gates) if g]
        assert true_idx == [3, 7]

    def test_danger_decides_next_frame(self) -> None:
        s, _ = make_scheduler()
        s.set_load(0.95)
        assert s.on_frame(danger=False) is False
        assert s.on_frame(danger=True) is True, "danger must allow immediate action"
        assert s.on_frame(danger=False) is False
        assert s.on_frame(danger=False) is False
        assert s.on_frame(danger=True) is True


class TestBuffering:
    def test_submit_pop_executes(self) -> None:
        s, _clock = make_scheduler()
        s.submit(3)  # JUMP
        planned = s.pop_executable()
        assert planned is not None and planned.action == 3
        assert planned.action_step == 1
        assert s.stats.executed == 1

    def test_buffer_is_bounded(self) -> None:
        s, _ = make_scheduler()
        for a in (1, 2, 3, 4):
            s.submit(a)
        # buffer_size=1 -> only the newest survives
        planned = s.pop_executable()
        assert planned is not None and planned.action == 4

    def test_expired_actions_dropped(self) -> None:
        s, clock = make_scheduler()
        s.submit(2)
        clock.advance(500.0)  # way past TTL
        assert s.pop_executable() is None
        assert s.stats.expired == 1

    def test_duplicate_suppression_within_cooldown(self) -> None:
        s, clock = make_scheduler(cooldown_ms=110.0)
        s.submit(1)
        assert s.pop_executable().action == 1
        clock.advance(50.0)  # inside cooldown
        s.submit(1)
        assert s.pop_executable() is None
        assert s.stats.suppressed_duplicates == 1
        clock.advance(100.0)  # outside cooldown
        s.submit(1)
        assert s.pop_executable() is not None

    def test_noop_not_duplicate_suppressed(self) -> None:
        s, clock = make_scheduler()
        s.submit(0)
        assert s.pop_executable().action == 0
        clock.advance(10)
        s.submit(0)
        assert s.pop_executable() is not None

    def test_interrupt_drops_buffered(self) -> None:
        s, _ = make_scheduler()
        s.submit(2)
        s.interrupt()
        assert s.pop_executable() is None
        assert s.stats.interrupts == 1

    def test_danger_drops_buffered_normal_actions(self) -> None:
        s, _ = make_scheduler()
        s.submit(4)  # buffered normal decision
        s.on_frame(danger=True)  # interrupts buffer and opens a decision slot
        assert s.pop_executable() is None
        assert s.stats.interrupts >= 1
        assert s.stats.danger_overrides == 1

    def test_invalid_action_rejected(self) -> None:
        s, _ = make_scheduler()
        assert s.submit(9) is False
        assert s.submit(-1) is False

    def test_reset_episode(self) -> None:
        s, _clock = make_scheduler()
        s.submit(1)
        s.pop_executable()
        s.reset_episode()
        s.submit(1)  # cooldown cleared
        assert s.pop_executable() is not None

    def test_action_step_counter_independent_of_env_frames(self) -> None:
        s, clock = make_scheduler()
        s.set_load(0.1)
        gates = 0
        actions = [1, 2]  # distinct actions: different keys are never duplicates
        for i in range(4):
            if s.on_frame(danger=False):
                s.submit(actions[gates])
                gates += 1
                s.pop_executable()
            clock.advance(33)  # one frame interval
        assert gates == 2, "cadence=2 -> two decision slots in four frames"
        assert s.action_step == 2, "both decisions execute"
        # an IDENTICAL rapid decision inside the cooldown is suppressed
        clock.advance(33)
        s.submit(2)  # same as the last executed action, within cooldown
        s.pop_executable()
        assert s.action_step == 2 and s.stats.suppressed_duplicates == 1
        assert s.snapshot()["action_step"] == 2


class TestConfidenceAwareCadence:
    """§9: cadence depends on CPU load AND horizon confidence together."""

    def test_high_load_low_confidence_slow(self) -> None:
        s, _ = make_scheduler()
        s.set_signal(0.95, 0.2)
        assert s.cadence == s.cfg.max_decision_frames

    def test_high_load_high_confidence_fast(self) -> None:
        s, _ = make_scheduler()
        s.set_signal(0.95, 0.9)  # hazard brewing: safety outranks thrift
        assert s.cadence == s.cfg.min_decision_frames

    def test_low_load_any_confidence_fast(self) -> None:
        s, _ = make_scheduler()
        s.set_signal(0.1, 0.0)
        assert s.cadence == s.cfg.min_decision_frames
        s.set_signal(0.1, 0.9)
        assert s.cadence == s.cfg.min_decision_frames

    def test_confidence_at_threshold_is_fast(self) -> None:
        s, _ = make_scheduler()
        s.set_signal(0.95, s.cfg.fast_confidence)
        assert s.cadence == s.cfg.min_decision_frames

    def test_confidence_none_behaves_like_set_load(self) -> None:
        s1, _ = make_scheduler()
        s2, _ = make_scheduler()
        s1.set_load(0.95)
        s2.set_signal(0.95, None)
        assert s1.cadence == s2.cadence

    def test_snapshot_reports_signal(self) -> None:
        s, _ = make_scheduler()
        s.set_signal(0.42, 0.83)
        snap = s.snapshot()
        assert snap["load"] == pytest.approx(0.42)
        assert snap["confidence"] == pytest.approx(0.83)
        s.set_signal(0.1, None)
        assert s.snapshot()["confidence"] is None

    def test_actor_passes_confidence_through(self) -> None:
        # integration seam: BotActor calls set_signal(load, hz.confidence);
        # simulate a rising-confidence sequence and check cadence reacts.
        s, _ = make_scheduler()
        for conf in (0.1, 0.5, 0.76):
            s.set_signal(0.95, conf)
        assert s.cadence == s.cfg.min_decision_frames
