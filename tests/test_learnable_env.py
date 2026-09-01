"""Tests for the learnable synthetic env (audit-only)."""

from __future__ import annotations

import numpy as np

from learnable_env import LearnableEnv, LearnableEnvConfig


class TestLearnableEnv:
    def test_observation_shape(self) -> None:
        env = LearnableEnv(seed=0)
        obs = env.reset()
        assert obs.shape == (7,)
        assert obs.dtype == np.float32

    def test_starts_in_middle_lane(self) -> None:
        env = LearnableEnv(seed=0)
        env.reset()
        assert env.player_lane == 1
        obs = env.observation()
        assert obs[1] == 1.0  # player in lane 1

    def test_left_moves_to_lane_0(self) -> None:
        env = LearnableEnv(seed=0)
        env.reset()
        obs, _, _, _ = env.step(1)  # LEFT
        assert env.player_lane == 0

    def test_right_moves_to_lane_2(self) -> None:
        env = LearnableEnv(seed=0)
        env.reset()
        obs, _, _, _ = env.step(2)  # RIGHT
        assert env.player_lane == 2

    def test_dies_when_in_targeted_lane(self) -> None:
        # Obstacle_period=30, approach=15.  Spawn index 0 →
        # lane 0, impact at t=15.  Player starts in lane 1;
        # we steer them into lane 0 and HOLD there.
        env = LearnableEnv(LearnableEnvConfig(
            obstacle_period=30, approach_time=15, max_steps=900), seed=0)
        env.reset()
        # First step(s) to move to lane 0; then NOOP until the
        # obstacle impacts.
        env.step(1)  # t=0 -> lane 0
        for _ in range(20):
            obs, r, done, info = env.step(0)  # NOOP
            if done:
                break
        assert env.dead, "should die if sitting in targeted lane"

    def test_optimal_policy_survives_full_episode(self) -> None:
        # The optimal policy is: when the upcoming obstacle
        # is in lane 0, move to lane 1 or 2; when it's in
        # lane 1, move to lane 0 or 2; when it's in lane 2,
        # move to lane 0 or 1.  The simplest implementation
        # is: if player_lane == next_lane, move left.
        env = LearnableEnv(LearnableEnvConfig(
            obstacle_period=30, approach_time=15, max_steps=900), seed=0)
        env.reset()
        for t in range(900):
            # The "next spawn" is at index next_spawn_idx
            if env.next_spawn_idx < len(env.spawns):
                _, _, impact = env.spawns[env.next_spawn_idx]
                if impact - env.t <= 5 and env.next_spawn_idx < len(env.spawns):
                    target = env.spawns[env.next_spawn_idx][0]
                    if env.player_lane == target:
                        # Move out — choose left if not in 0.
                        a = 1 if env.player_lane > 0 else 2
                    else:
                        a = 0
                else:
                    a = 0
            else:
                a = 0
            _, _, done, _ = env.step(a)
            if done:
                break
        # Optimal policy should survive the full 30s.
        assert env.t >= 899, f"optimal policy only survived {env.t} frames"

    def test_random_policy_dies_quickly(self) -> None:
        env = LearnableEnv(seed=0)
        env.reset()
        np.random.seed(0)
        for t in range(900):
            a = int(np.random.randint(0, 5))
            _, _, done, _ = env.step(a)
            if done:
                break
        # Random should die well before 30s.
        assert env.t < 600, f"random survived too long: {env.t} frames"
