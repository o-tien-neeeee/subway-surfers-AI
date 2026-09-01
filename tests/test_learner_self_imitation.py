"""Tests for the learner's self-imitation wiring.

The recorder is unit-tested in :mod:`tests.test_self_imitation`;
these tests pin the *integration* in the learner: the rolling
recent-transition ring, the gate decision, and the BC auto-trigger.
"""

from __future__ import annotations

import queue

import numpy as np

from config import BotConfig
from ipc import SharedCounters, SharedWeights
from learner_worker import Learner
from models import weight_size_for_profile
from replay_buffer import NStepTransition


def _make_learner(tmp_path, **reward_overrides) -> tuple[Learner, SharedCounters]:
    cfg = BotConfig()
    cfg.paths.checkpoints_dir = str(tmp_path / "ckpt")
    cfg.paths.demos_dir = str(tmp_path / "demos")
    for k, v in reward_overrides.items():
        setattr(cfg.reward, k, v)
    weights = SharedWeights(weight_size_for_profile(
        "quality_cpu", cfg.perception.frame_stack))
    counters = SharedCounters()
    learner = Learner(cfg, weights, counters, queue.Queue(),
                      str(tmp_path / "ckpt"))
    return learner, counters


def _make_transition(action: int = 1, frame_value: int = 0) -> NStepTransition:
    obs = np.full((4, 84, 84), frame_value, dtype=np.uint8)
    return NStepTransition(
        obs=obs, next_obs=obs, action=action, reward=0.1, done=False,
        span=3, gamma_pow=0.99 ** 3,
        obs_env_ids=(0, 1, 2, 3),
        next_env_ids=(1, 2, 3, 4),
    )


class TestLearnerSelfImitationGate:
    def test_learner_constructs_with_recorder(self, tmp_path) -> None:
        learner, _ = _make_learner(tmp_path)
        assert learner.self_imitation is not None
        # Default cfg has the gate on (factor > 0).
        assert learner.self_imitation.cfg.enabled is True
        # And the directory is under the configured demos dir.
        assert str(learner.self_imitation.out_dir).endswith("self")

    def test_disabled_factor_turns_recorder_off(self, tmp_path) -> None:
        learner, _ = _make_learner(tmp_path, self_imitation_factor=0.0)
        # factor == 0 disables the gate (the constructor reads it
        # and sets enabled = factor > 0).
        assert learner.self_imitation.cfg.enabled is False
        assert learner.self_imitation.cfg.factor == 0.0

    def test_recent_transitions_collect(self, tmp_path) -> None:
        learner, _ = _make_learner(tmp_path)
        # Drain the transition queue into the buffer AND the
        # recent-ring; the ring mirrors the buffer's tail so the
        # self-imitation gate can dump the latest episode.
        items = [_make_transition(action=1) for _ in range(5)]
        learner.add_transitions(items)
        assert len(learner._recent_transitions) == 5


class TestConsumeRecentEpisode:
    def test_returns_none_on_empty_ring(self, tmp_path) -> None:
        learner, _ = _make_learner(tmp_path)
        assert learner._consume_recent_episode() is None

    def test_extracts_frames_actions_timestamps(self, tmp_path) -> None:
        learner, _ = _make_learner(tmp_path)
        for a in (1, 2, 3, 4):
            learner._recent_transitions.append(_make_transition(action=a))
        ep = learner._consume_recent_episode()
        assert ep is not None
        assert ep["frames"].shape == (4, 84, 84)
        assert ep["actions"].tolist() == [1, 2, 3, 4]
        assert ep["done"].tolist()[-1] is True
        assert ep["done"].tolist()[:-1] == [False] * 3


class TestMaybeSelfImitate:
    def test_below_threshold_does_not_save(self, tmp_path) -> None:
        learner, _ = _make_learner(tmp_path, self_imitation_factor=2.0)
        # Fill the rolling mean at 10.0, threshold = 2.0 * 10 = 20.
        for i in range(25):
            learner.self_imitation.note_episode(i, 10.0)
        # Make some recent transitions.
        learner.add_transitions([_make_transition(action=1) for _ in range(3)])
        learner._maybe_self_imitate(episode_id=999, survival_s=5.0)
        # 5.0 is well below 20.0; nothing should land on disk.
        on_disk = list(learner.self_imitation.out_dir.glob("*.npz"))
        assert on_disk == []
        # And the recent ring was cleared.
        assert len(learner._recent_transitions) == 0

    def test_above_threshold_saves_episode(self, tmp_path) -> None:
        learner, _ = _make_learner(tmp_path, self_imitation_factor=0.5)
        # Fill the rolling mean at 5.0; threshold = 0.5 * 5 = 2.5.
        for i in range(25):
            learner.self_imitation.note_episode(i, 5.0)
        # Seed the recent ring with a few transitions.
        for a in (1, 2, 3):
            learner.add_transitions([_make_transition(action=a)])
        learner._maybe_self_imitate(episode_id=42, survival_s=8.0)
        # 8.0 > 2.5 (or whatever the post-append mean gives, see
        # test_self_imitation for the exact threshold arithmetic) ->
        # an episode file lands in <demos>/self/.
        on_disk = list(learner.self_imitation.out_dir.glob("*.npz"))
        assert len(on_disk) == 1
        # And the recent ring was cleared afterwards.
        assert len(learner._recent_transitions) == 0

    def test_no_recent_data_but_gate_open_saves_nothing(self, tmp_path) -> None:
        """Without recent transitions the gate can still fire (by
        the recorded number) but the save_episode call gets nothing
        to write — that must NOT crash, and must NOT leave a corrupt
        file on disk."""
        learner, _ = _make_learner(tmp_path, self_imitation_factor=0.5)
        for i in range(25):
            learner.self_imitation.note_episode(i, 5.0)
        # No add_transitions() — the recent ring is empty.
        learner._maybe_self_imitate(episode_id=99, survival_s=10.0)
        on_disk = list(learner.self_imitation.out_dir.glob("*.npz"))
        assert on_disk == []

    def test_recent_ring_is_cleared_on_rejection(self, tmp_path) -> None:
        learner, _ = _make_learner(tmp_path, self_imitation_factor=2.0)
        for i in range(25):
            learner.self_imitation.note_episode(i, 10.0)
        for a in (1, 2):
            learner.add_transitions([_make_transition(action=a)])
        learner._maybe_self_imitate(episode_id=1, survival_s=1.0)
        # 1.0 is way below 2.0 * 10 = 20 -> rejected.
        # The recent ring must still be cleared.
        assert len(learner._recent_transitions) == 0
