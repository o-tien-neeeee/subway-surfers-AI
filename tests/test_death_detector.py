"""Death detector + respawn controller tests with synthetic frames.

Covers: debounce ladder (ALIVE -> POSSIBLE_EVENT -> DEAD_CANDIDATE ->
DEAD_CONFIRMED), false-positive resistance (single-frame flash), recovery
stability requirement, UNKNOWN on missing patch, respawn click cadence,
timeout -> FAILED (never clicks forever), and recovery after stable frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import DeathConfig
from death_detector import (
    ColorAnchorDeathDetector,
    DeathState,
    RespawnController,
    StagnationDetector,
    synthetic_patch,
)

ALIVE = (200, 60, 60)
DEAD = (40, 40, 200)


def make_cfg(**kw) -> DeathConfig:
    base = dict(
        threshold=25.0, confirm_frames=3, stable_frames=5,
        respawn_interval_s=0.8, respawn_timeout_s=3.0,
        anchor_fx=0.05, anchor_fy=0.05, anchor_baseline_rgb=ALIVE,
    )
    base.update(kw)
    return DeathConfig(**base)


class TestColorAnchorDeathDetector:
    def test_requires_calibration(self) -> None:
        with pytest.raises(ValueError):
            ColorAnchorDeathDetector(DeathConfig())

    def test_alive_stays_alive(self) -> None:
        det = ColorAnchorDeathDetector(make_cfg())
        for i in range(20):
            r = det.update(synthetic_patch(ALIVE, noise=2.0), i, float(i) / 30)
        assert r.state is DeathState.ALIVE
        assert r.distance < 25.0

    def test_death_ladder_order(self) -> None:
        det = ColorAnchorDeathDetector(make_cfg(confirm_frames=3))
        r1 = det.update(synthetic_patch(DEAD), 1, 0.1)
        assert r1.state is DeathState.POSSIBLE_EVENT
        r2 = det.update(synthetic_patch(DEAD), 2, 0.2)
        assert r2.state is DeathState.DEAD_CANDIDATE
        r3 = det.update(synthetic_patch(DEAD), 3, 0.3)
        assert r3.state is DeathState.DEAD_CONFIRMED
        assert det.is_dead()

    def test_one_frame_flash_is_not_death(self) -> None:
        det = ColorAnchorDeathDetector(make_cfg(confirm_frames=3))
        det.update(synthetic_patch(ALIVE), 1, 0.1)
        det.update(synthetic_patch(DEAD), 2, 0.2)  # flash (animation artifact)
        r = det.update(synthetic_patch(ALIVE), 3, 0.3)
        assert r.state is DeathState.ALIVE
        assert not det.is_dead(), "single changed frame must not confirm death"

    def test_two_frames_below_confirm_still_candidate(self) -> None:
        det = ColorAnchorDeathDetector(make_cfg(confirm_frames=4))
        det.update(synthetic_patch(DEAD), 1, 0.1)
        det.update(synthetic_patch(DEAD), 2, 0.2)
        det.update(synthetic_patch(DEAD), 3, 0.3)
        assert det.state is DeathState.DEAD_CANDIDATE
        assert not det.is_dead()

    def test_recovery_requires_stability(self) -> None:
        det = ColorAnchorDeathDetector(make_cfg(confirm_frames=2, stable_frames=4))
        det.update(synthetic_patch(DEAD), 1, 0.1)
        det.update(synthetic_patch(DEAD), 2, 0.2)
        assert det.is_dead()
        r = det.update(synthetic_patch(ALIVE), 3, 0.3)
        assert r.state is DeathState.DEAD_CONFIRMED  # recovering, not yet stable
        for i in range(4, 8):
            r = det.update(synthetic_patch(ALIVE), i, i / 30.0)
        assert r.state is DeathState.ALIVE
        assert det.is_alive_stable()

    def test_missing_patch_is_unknown(self) -> None:
        det = ColorAnchorDeathDetector(make_cfg())
        r = det.update(None, 1, 0.1)
        assert r.state is DeathState.UNKNOWN
        assert r.distance == -1.0
        assert r.reason == "invalid_or_missing_patch"

    def test_threshold_is_configurable(self) -> None:
        near = (210, 70, 70)  # distance ~17 from ALIVE
        det = ColorAnchorDeathDetector(make_cfg(threshold=10.0, confirm_frames=2))
        det.update(synthetic_patch(near), 1, 0.1)
        r = det.update(synthetic_patch(near), 2, 0.2)
        assert r.state is DeathState.DEAD_CONFIRMED, "tighter threshold must fire"

    def test_reason_logging(self) -> None:
        det = ColorAnchorDeathDetector(make_cfg())
        det.update(synthetic_patch(DEAD), 1, 0.1)
        r = det.last_result
        assert "off_baseline" in r.reason or r.reason == "first_off_baseline"
        assert r.distance > 25.0


class TestRespawnController:
    def _run(self, ctl, states, t0=0.0, step=0.1):
        results = [ctl.update(s, t0 + i * step) for i, s in enumerate(states)]
        return results

    def test_clicks_on_interval_then_recovers(self) -> None:
        clicks: list[float] = []

        def click(t: float) -> bool:
            clicks.append(t)
            return True

        ctl = RespawnController(click, make_cfg(respawn_interval_s=0.8,
                                                stable_frames=3,
                                                respawn_timeout_s=30.0),
                                now=lambda: 0.0)
        ctl.start()
        t = 0.0

        def tick(state, dt=0.1):
            nonlocal t
            t += dt
            return ctl.update(state, t)

        # dead frames -> clicks at ~0.8s cadence
        r = tick(DeathState.DEAD_CONFIRMED)   # t=0.1 -> click #1
        assert r.action == "CLICK"
        for _ in range(8):                    # cooldown window
            r = tick(DeathState.DEAD_CONFIRMED)
        assert r.action == "WAIT"
        r = tick(DeathState.DEAD_CONFIRMED)   # well past 0.8s -> click #2
        assert r.action == "CLICK"
        # anchor stabilises
        r = tick(DeathState.ALIVE)
        assert r.action == "WAIT"
        r = tick(DeathState.ALIVE)
        assert r.action == "WAIT"
        r = tick(DeathState.ALIVE)
        assert r.action == "RECOVERED"
        assert not ctl.active
        assert len(clicks) == 2

    def test_timeout_fails_instead_of_clicking_forever(self) -> None:
        clicks = []
        ctl = RespawnController(lambda t: clicks.append(t) or True,
                                make_cfg(respawn_timeout_s=1.0,
                                         respawn_interval_s=0.1),
                                now=lambda: 0.0)
        ctl.start()
        t = 0.0
        last = None
        for i in range(50):
            t += 0.1
            last = ctl.update(DeathState.DEAD_CONFIRMED, t)
            if last.action == "FAILED":
                break
        assert last.action == "FAILED"
        assert not ctl.active
        assert len(clicks) <= 12, "must stop clicking after the timeout"

    def test_inactive_controller_waits(self) -> None:
        ctl = RespawnController(lambda t: True, make_cfg())
        r = ctl.update(DeathState.DEAD_CONFIRMED, 1.0)
        assert r.action == "WAIT" and r.detail == "inactive"


class TestStagnationDetector:
    def test_flags_long_stagnation(self) -> None:
        det = StagnationDetector(timeout_s=2.0)
        frame = np.full((84, 84), 100, dtype=np.uint8)
        det.update(frame, 0.0)
        assert not det.update(frame, 1.0)
        assert det.update(frame, 2.5), "frozen screen must eventually flag"

    def test_motion_resets_timer(self) -> None:
        det = StagnationDetector(timeout_s=1.0)
        base = np.zeros((84, 84), dtype=np.uint8)
        det.update(base, 0.0)
        det.update(base, 0.5)
        moving = base.copy()
        moving[10:20, 10:20] = 255
        assert not det.update(moving, 1.5)
        assert not det.update(base, 1.9)


class TestEndToEndDeathTransition:
    def test_full_death_then_respawn_cycle_with_synthetic_frames(self) -> None:
        """DEAD_CONFIRMED -> clicks -> stable -> ALIVE (whole §7 flow)."""
        det = ColorAnchorDeathDetector(make_cfg(confirm_frames=2, stable_frames=3))
        ctl = RespawnController(lambda t: True,
                                make_cfg(respawn_interval_s=0.5, stable_frames=3),
                                now=lambda: 0.0)
        # ... alive
        for i in range(5):
            det.update(synthetic_patch(ALIVE, noise=1.0), i, i / 30.0)
        assert det.state is DeathState.ALIVE
        # death
        det.update(synthetic_patch(DEAD), 10, 10 / 30.0)
        det.update(synthetic_patch(DEAD), 11, 11 / 30.0)
        assert det.state is DeathState.DEAD_CONFIRMED
        ctl.start()
        # respawn clicks until anchor returns
        t = [0.0]
        clicked = 0
        for i in range(12, 60):
            t[0] += 0.2
            r = ctl.update(det.state, t[0])
            if r.action == "CLICK":
                clicked += 1
                # game restarts after the click: anchor flips back to ALIVE
            if clicked >= 1 and i > 16:
                det.update(synthetic_patch(ALIVE), i, t[0])
            else:
                det.update(synthetic_patch(DEAD), i, t[0])
            if r.action == "RECOVERED":
                break
        assert ctl.update(det.state, t[0] + 1.0).action in ("WAIT", "RECOVERED")
        assert det.is_alive_stable() or det.state is DeathState.ALIVE
        assert clicked >= 1
