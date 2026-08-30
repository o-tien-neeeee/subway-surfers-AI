"""Process lifecycle tests: startup, stale-frame dropping, emergency stop.

Runs the real multi-process architecture (spawn) against the synthetic
game source with dry-run input — no display, no Chrome, no real keys.
"""

from __future__ import annotations

import time

import pytest

from config import BotConfig
from ipc import SharedFrameRing


def headless_cfg() -> BotConfig:
    cfg = BotConfig()
    cfg.capture.source = "fake"
    cfg.region.left, cfg.region.top = 0, 0
    cfg.region.width, cfg.region.height = 480, 800
    cfg.death.anchor_fx = 30 / 480
    cfg.death.anchor_fy = 30 / 800
    cfg.death.anchor_baseline_rgb = (206, 66, 66)
    cfg.death.anchor_baseline_std = 2.0
    cfg.input.respawn_fx, cfg.input.respawn_fy = 0.5, 0.6
    cfg.rl.warmup_transitions = 10_000_000  # no training in these tests
    cfg.seed = 11
    return cfg


@pytest.fixture(scope="module")
def app_module():
    from app import BotApplication

    return BotApplication


class TestFrameRing:
    def test_latest_wins_and_stale_dropped(self) -> None:
        ring = SharedFrameRing(4, 64, 48, 3)
        import numpy as np

        img = np.zeros((64, 48, 3), dtype=np.uint8)
        for fid in range(10):
            img[:] = fid
            assert ring.write(img, fid, float(fid))
        fr = ring.read_latest()
        assert fr.frame_id == 9, "reader must get the newest complete frame"
        assert int(fr.image.mean()) == 9

    def test_reader_never_sees_older_frame(self) -> None:
        import numpy as np

        ring = SharedFrameRing(3, 32, 32, 3)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        prev_id = -1
        for fid in range(50):
            img[:] = fid % 256
            ring.write(img, fid, float(fid))
            fr = ring.read_latest()
            if fr is not None:
                assert fr.frame_id >= prev_id
                prev_id = fr.frame_id
        assert prev_id >= 40, "writer laps must not go backwards for the reader"

    def test_size_mismatch_refused(self) -> None:
        import numpy as np

        ring = SharedFrameRing(2, 10, 10, 3)
        assert ring.write(np.zeros((5, 5, 3), dtype=np.uint8), 0, 0.0) is False
        assert ring.read_latest() is None

    def test_memory_constant(self) -> None:
        ring = SharedFrameRing(4, 100, 100, 3)
        n = ring.nbytes()
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for fid in range(500):
            ring.write(img, fid, float(fid))
        assert ring.nbytes() == n

    def test_counters(self) -> None:
        ring = SharedFrameRing(2, 8, 8, 3)
        ring.note_drop(3)
        assert ring.counters()["dropped"] == 3


class TestProcessLifecycle:
    def test_start_and_clean_shutdown(self, app_module) -> None:
        app = app_module(headless_cfg(), input_backend="dry_run")
        app.start(with_learner=False)
        try:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if app.ring.latest_frame_id() > 5:
                    break
                time.sleep(0.1)
            assert app.ring.latest_frame_id() > 5, "capture worker produced no frames"
            assert app.workers_alive()["actor"] is True
        finally:
            app.shutdown(timeout_s=15)
        alive = app.workers_alive()
        assert not any(alive.values()), f"processes still alive: {alive}"

    def test_emergency_stop_sets_events_and_releases(self, app_module) -> None:
        app = app_module(headless_cfg(), input_backend="dry_run")
        app.start(with_learner=False)
        try:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if app.ring.latest_frame_id() > 2:
                    break
                time.sleep(0.1)
            app.emergency_stop()  # F8-equivalent path
            assert app.events["emergency"].is_set()
            assert app.events["stop"].is_set()
        finally:
            app.shutdown(timeout_s=15)
        assert not any(app.workers_alive().values())

    def test_learner_shutdown_saves_state(self, app_module, tmp_path) -> None:
        cfg = headless_cfg()
        cfg.paths.checkpoints_dir = str(tmp_path / "ckpt")
        cfg.rl.warmup_transitions = 1
        app = app_module(cfg, input_backend="dry_run")
        app.start(with_learner=True)
        try:
            deadline = time.monotonic() + 25.0
            while time.monotonic() < deadline:
                if app.counters.learner_update_step.value > 0:
                    break
                time.sleep(0.2)
            assert app.counters.learner_update_step.value > 0, \
                "learner performed no updates"
        finally:
            app.shutdown(timeout_s=30)
        ckpt_dir = tmp_path / "ckpt" / cfg.rl.profile
        assert (ckpt_dir / "latest_model.pth").exists(), "shutdown must save model"
        assert (ckpt_dir / "buffer.pkl").exists(), "shutdown must save buffer"

    def test_shutdown_is_idempotent(self, app_module) -> None:
        app = app_module(headless_cfg(), input_backend="dry_run")
        app.start(with_learner=False)
        app.shutdown(timeout_s=15)
        app.shutdown(timeout_s=5)  # second call must not hang or raise
        assert not any(app.workers_alive().values())


class TestWorkerCrashResilience:
    def test_gui_side_survives_actor_crash(self, app_module) -> None:
        """Killing the actor hard must leave the app queryable (GUI stays)."""
        app = app_module(headless_cfg(), input_backend="dry_run")
        app.start(with_learner=False)
        try:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if app.ring.latest_frame_id() > 2:
                    break
                time.sleep(0.1)
            assert app.actor_proc is not None
            app.actor_proc.terminate()
            app.actor_proc.join(timeout=5)
            alive = app.workers_alive()
            assert alive["actor"] is False
            assert alive["capture"] is True, "capture must survive actor death"
            # metrics drain still works
            assert isinstance(app.drain_metrics(16), list)
        finally:
            app.shutdown(timeout_s=15)


class TestHotkeyFallback:
    def test_emergency_hotkey_degrades_without_input_system(self) -> None:
        from safety_watchdog import EmergencyHotkey

        events = {"emergency": __import__("multiprocessing").Event()}
        hk = EmergencyHotkey("f8", events)
        # On CI there is no input subsystem: must degrade, not crash.
        assert hk.available in (True, False)
        hk.stop()  # hooks unregistered
