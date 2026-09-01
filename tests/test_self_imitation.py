"""Tests for the self-imitation recorder.

The recorder is the *policy* side of DQfD-style self-imitation: it
decides which finished episodes are worth saving for re-training.
These tests pin the decision rule (factor x rolling mean) and the
on-disk format (identical to human demos so the dataset can train on
both pools without special-casing).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from self_imitation import SELF_DIR_NAME, SelfImitationConfig, SelfImitationRecorder


def _recorder(tmp_path, **overrides) -> SelfImitationRecorder:
    cfg = SelfImitationConfig(**overrides)
    return SelfImitationRecorder(cfg, tmp_path)


class TestSelfImitationGate:
    def test_disabled_saves_nothing(self, tmp_path) -> None:
        rec = _recorder(tmp_path, enabled=False)
        decision = rec.note_episode(1, 10.0)
        assert decision["saved"] is False
        assert decision["reason"] == "disabled"

    def test_non_positive_survival_rejected(self, tmp_path) -> None:
        rec = _recorder(tmp_path, factor=0.5)
        for surv in (0.0, -1.0):
            d = rec.note_episode(1, surv)
            assert d["saved"] is False
            assert d["reason"] == "non_positive_survival"

    def test_factor_zero_never_saves(self, tmp_path) -> None:
        rec = _recorder(tmp_path, factor=0.0, min_episodes_before_save=0)
        # warm up the rolling window so the gate is otherwise open
        for eid, surv in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
            rec.note_episode(eid, surv)
        d = rec.note_episode(5, 100.0)  # huge survival
        assert d["saved"] is False
        assert d["reason"] == "factor_zero"

    def test_warming_up_keeps_the_gate_closed(self, tmp_path) -> None:
        rec = _recorder(tmp_path, factor=0.5, min_episodes_before_save=3)
        for eid in (1, 2):
            d = rec.note_episode(eid, 100.0)  # huge survival
            assert d["saved"] is False
            assert "warming_up" in d["reason"]

    def test_factor_compares_to_rolling_mean(self, tmp_path) -> None:
        rec = _recorder(tmp_path, factor=1.2, min_episodes_before_save=0)
        # The threshold is computed AFTER the current episode's
        # survival is appended to the rolling window, so the
        # comparison is "is this episode better than 1.2x the running
        # mean INCLUDING itself?".  Once the window is full the
        # contribution of any one episode is small enough that the
        # rule behaves like the intuitive "1.2x the average of the
        # last 20 episodes".
        for eid in range(1, 21):
            rec.note_episode(eid, 10.0)
        # The window is full at mean=10.0; now feed one more episode.
        d = rec.note_episode(21, 12.0)
        # After append, window mean = (19*10 + 12) / 20 = 10.1.
        # threshold = 1.2 * 10.1 = 12.12.  12.0 is below -> reject.
        assert d["saved"] is False
        assert d["reason"] == "below_threshold"
        d = rec.note_episode(22, 13.0)
        # Window mean = (18*10 + 12 + 13) / 20 = 10.25.
        # threshold = 1.2 * 10.25 = 12.3.  13.0 > 12.3 -> save.
        assert d["saved"] is True
        assert d["threshold"] == pytest.approx(12.3, rel=1e-6)

    def test_min_episodes_can_be_zero(self, tmp_path) -> None:
        rec = _recorder(tmp_path, factor=0.5, min_episodes_before_save=0)
        # First call is allowed; the rolling window has nothing so
        # the mean is undefined — the gate falls through and saves it
        # (the first real episode IS the baseline, by definition).
        d = rec.note_episode(1, 5.0)
        # rolling mean of 1 sample is 5.0; threshold = 0.5 * 5.0 = 2.5.
        # 5.0 > 2.5 -> saved.
        assert d["saved"] is True


def pytest_approx(x, rel=1e-9):
    return pytest.approx(x, rel=rel)


class TestSelfImitationSave:
    def test_save_writes_standard_npz(self, tmp_path) -> None:
        rec = _recorder(tmp_path, factor=0.5, min_episodes_before_save=0)
        # build a tiny "episode"
        n = 12
        frames = np.zeros((n, 84, 84), dtype=np.uint8)
        actions = np.array([0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0], dtype=np.int64)
        ts = np.arange(n, dtype=np.float64) / 30.0
        done = np.zeros(n, dtype=bool)
        done[-1] = True
        path = rec.save_episode(1, 5.0, frames, actions, ts, done)
        assert path is not None and path.exists()
        # lives under <out>/self/
        assert SELF_DIR_NAME in str(path)
        # The on-disk format must be loadable as a human demo.
        data = np.load(path, allow_pickle=False)
        assert data["frames"].shape == (n, 84, 84)
        assert (data["actions"] == actions).all()
        assert (data["done"] == done).all()
        meta = json.loads(str(data["meta"]))
        assert meta["self_imitation"]["survival_s"] == 5.0

    def test_rotate_removes_oldest(self, tmp_path) -> None:
        rec = _recorder(tmp_path, factor=0.1, min_episodes_before_save=0,
                         max_episodes=3)
        # Save 5 episodes with the same data — the recorder should
        # keep only the last 3 on disk.
        for i in range(5):
            frames = np.full((2, 84, 84), i, np.uint8)
            actions = np.zeros(2, np.int64)
            ts = np.arange(2, dtype=np.float64)
            done = np.array([False, True])
            rec.save_episode(i, 5.0, frames, actions, ts, done)
        on_disk = list((tmp_path / SELF_DIR_NAME).glob("*.npz"))
        assert len(on_disk) == 3, (
            f"rotate should keep only the newest 3, got {len(on_disk)}")

    def test_empty_episode_saves_nothing(self, tmp_path) -> None:
        rec = _recorder(tmp_path)
        assert rec.save_episode(1, 0.0,
                                 np.zeros((0, 84, 84), np.uint8),
                                 np.zeros(0, np.int64),
                                 np.zeros(0, np.float64),
                                 np.zeros(0, bool)) is None

    def test_stats_reflect_counts(self, tmp_path) -> None:
        rec = _recorder(tmp_path, factor=0.5, min_episodes_before_save=0)
        # After one episode the rolling mean is 5.0 and the threshold
        # is 0.5 * 5.0 = 2.5.  Episode 2 with survival 1.0 is below
        # the threshold and is NOT counted in self_episodes_saved
        # (the counter only increments on actual save_episode calls).
        rec.note_episode(1, 5.0)
        rec.note_episode(2, 1.0)
        s = rec.stats()
        assert s["self_episodes_seen"] == 2
        # saved counter reflects files written, not gate decisions
        assert s["self_episodes_saved"] == 0
        assert s["self_on_disk"] == 0
        # now actually save one
        n_frames = 2
        rec.save_episode(1, 5.0,
                          np.zeros((n_frames, 84, 84), np.uint8),
                          np.zeros(n_frames, np.int64),
                          np.arange(n_frames, dtype=np.float64),
                          np.array([False, True]))
        s = rec.stats()
        assert s["self_episodes_saved"] == 1
        assert s["self_on_disk"] == 1
