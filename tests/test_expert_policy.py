"""Tests for the expert policy.

The expert policy is the "knowing the answer" baseline
used to generate behaviour-cloning demonstrations.  These
tests pin that the expert is actually optimal (achieves
the full episode length) and that the ``collect_demonstration``
method produces a valid dataset of the right shape."""

from __future__ import annotations

import numpy as np

from expert_policy import ExpertPolicy
from learnable_env import LearnableEnv, LearnableEnvConfig


class TestExpertPolicy:
    def test_stays_in_safe_lane(self) -> None:
        # obs = [player_one_hot, obstacle_one_hot, tti]
        obs = np.array([0, 1, 0, 0, 0, 1, 0.5], dtype=np.float32)
        # Player in lane 0, obstacle in lane 1 — safe, NOOP.
        assert ExpertPolicy().act(obs) == 0

    def test_moves_out_of_targeted_lane(self) -> None:
        # Player in lane 1 (targeted), obstacle in lane 1.
        obs = np.array([0, 1, 0, 0, 1, 0, 0.3], dtype=np.float32)
        # Move LEFT.
        assert ExpertPolicy().act(obs) == 1
        # Same but in lane 2 (targeted) — must move LEFT.
        obs = np.array([0, 0, 1, 0, 0, 1, 0.3], dtype=np.float32)
        assert ExpertPolicy().act(obs) == 1
        # Player in lane 0 (targeted), must move RIGHT.
        obs = np.array([1, 0, 0, 1, 0, 0, 0.3], dtype=np.float32)
        assert ExpertPolicy().act(obs) == 2

    def test_waits_when_time_is_plenty(self) -> None:
        # Obstacle is far — no need to move yet.
        obs = np.array([0, 1, 0, 0, 1, 0, 1.0], dtype=np.float32)
        assert ExpertPolicy().act(obs) == 0

    def test_collect_demonstration_optimal(self) -> None:
        cfg = LearnableEnvConfig(
            obstacle_period=30, approach_time=15, max_steps=900)
        env = LearnableEnv(cfg, seed=0)
        expert = ExpertPolicy()
        obs, act, next_obs = expert.collect_demonstration(env)
        # Optimal policy survives the full episode.
        assert obs.shape[0] >= 899
        assert obs.shape[1] == 7
        assert act.shape[0] == obs.shape[0]
        assert next_obs.shape == obs.shape
        # No NOOP when in danger zone (last 5 frames before
        # impact): the expert must always move out.
        for i in range(obs.shape[0]):
            tti = obs[i, 6]
            if tti < 0.4:
                p = int(np.argmax(obs[i, :3]))
                o = int(np.argmax(obs[i, 3:6]))
                if p == o:
                    assert act[i] in (1, 2), (
                        f"expert failed to dodge at frame {i}: "
                        f"player={p}, obstacle={o}, tti={tti}")
