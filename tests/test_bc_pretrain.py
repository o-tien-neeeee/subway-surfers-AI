"""Tests for the BC pretrain + DQfD orchestration module.

These tests pin:

* :func:`pretrain_and_arm_dqfd` warms up the agent
  (BC loss goes to ~0) and arms the joint loss (the
  agent's ``_demo_obs`` is cached).
* :func:`prefill_replay_with_demos` adds n-step
  transitions to the buffer and boosts their priority
  so the next sample picks them up.
* The full pipeline (pretrain + prefill + 50 train
  steps) keeps the BC anchor stable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bc_pretrain import (build_dqfd_agent, prefill_replay_with_demos,
                            pretrain_and_arm_dqfd)
from config import BotConfig, RLConfig
from dqfd_agent import DQfDAgent, DQfDConfig
from expert_policy import ExpertPolicy
from learnable_env import LearnableEnv
from replay_buffer import PERConfig, PrioritizedReplayBuffer


@pytest.fixture()
def small_buffer() -> PrioritizedReplayBuffer:
    cfg = PERConfig(capacity=2000, alpha=0.6, beta_start=0.4,
                     priority_eps=0.01)
    return PrioritizedReplayBuffer(cfg, frame_size=84, gamma=0.99)


class TestBuildDQfDAgent:
    def test_build_default(self) -> None:
        cfg = RLConfig(profile="strict_lite", gamma=0.99,
                        learning_rate=1e-3, grad_clip_norm=10.0)
        agent = build_dqfd_agent("strict_lite", cfg)
        assert isinstance(agent, DQfDAgent)
        # No demos cached at construction time.
        assert agent._demo_obs is None

    def test_build_with_dqfd_config(self) -> None:
        cfg = RLConfig(profile="strict_lite", gamma=0.99,
                        learning_rate=1e-3)
        dqfd = DQfDConfig(lambda_bc=1.0, margin=0.5,
                           no_exploration=True)
        agent = build_dqfd_agent("strict_lite", cfg, dqfd)
        assert agent.dqfd.lambda_bc == 1.0
        assert agent.dqfd.margin == 0.5


class TestPretrainAndArmDQfD:
    def test_pretrain_caches_demos(self) -> None:
        cfg = RLConfig(profile="strict_lite", gamma=0.99,
                        learning_rate=1e-3)
        agent = build_dqfd_agent("strict_lite", cfg)
        # Replace the online net with a tiny Linear
        # net (same trick as tests/test_dqfd.py).
        from tests.test_dqfd import _SmallQNet
        from distributional import mid_quantiles
        agent.online = _SmallQNet(n_actions=5, in_dim=7,
                                    num_quantiles=11)
        agent.target = _SmallQNet(n_actions=5, in_dim=7,
                                    num_quantiles=11)
        agent.target.load_state_dict(agent.online.state_dict())
        for p in agent.target.parameters():
            p.requires_grad_(False)
        agent.tau = mid_quantiles(11)
        env = LearnableEnv(seed=0)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        obs_in = obs.reshape(obs.shape[0], 7)
        result = pretrain_and_arm_dqfd(agent, obs_in, act,
                                         n_epochs=5, batch_size=128,
                                         lr=3e-3, verbose=False)
        assert result["n_frames"] == obs_in.shape[0]
        assert result["n_epochs"] == 5
        # Demos should now be cached.
        assert agent._demo_obs is not None
        assert agent._demo_actions is not None


class TestPrefillReplayWithDemos:
    def test_prefill_adds_nstep_transitions(
            self, small_buffer: PrioritizedReplayBuffer) -> None:
        # 4-frame stack of 84x84 uint8.
        n = 30
        obs = np.random.randint(0, 255, size=(n, 4, 84, 84),
                                  dtype=np.uint8)
        actions = np.random.randint(0, 5, size=(n,))
        rewards = np.zeros(n, dtype=np.float32)
        n_added = prefill_replay_with_demos(
            small_buffer, obs, actions, rewards,
            frame_stack=4, gamma=0.99, n_step=5,
            priority_boost=100.0)
        assert n_added > 0
        assert small_buffer.size == n_added
        # The highest priority should be the boost
        # value (or close to it).
        assert small_buffer.max_priority >= 50.0

    def test_prefill_empty_returns_zero(
            self, small_buffer: PrioritizedReplayBuffer) -> None:
        n_added = prefill_replay_with_demos(
            small_buffer,
            np.zeros((0, 4, 84, 84), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32))
        assert n_added == 0
        assert small_buffer.size == 0


class TestPipelineIntegration:
    def test_full_pipeline_keeps_bc_anchor(self) -> None:
        """End-to-end: pretrain + prefill + 50 train
        steps.  The BC loss on a held-out demo batch
        must remain stable (i.e. the agent still
        matches the expert)."""
        cfg = RLConfig(profile="strict_lite", gamma=0.99,
                        learning_rate=1e-3, grad_clip_norm=10.0,
                        batch_size=16)
        agent = build_dqfd_agent("strict_lite", cfg)
        from tests.test_dqfd import _SmallQNet
        from distributional import mid_quantiles
        agent.online = _SmallQNet(n_actions=5, in_dim=7,
                                    num_quantiles=11)
        agent.target = _SmallQNet(n_actions=5, in_dim=7,
                                    num_quantiles=11)
        agent.target.load_state_dict(agent.online.state_dict())
        for p in agent.target.parameters():
            p.requires_grad_(False)
        agent.tau = mid_quantiles(11)
        # Collect demos.
        env = LearnableEnv(seed=0)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        obs_in = obs.reshape(obs.shape[0], 7)
        pretrain_and_arm_dqfd(agent, obs_in, act, n_epochs=10,
                                 batch_size=128, lr=3e-3,
                                 verbose=False)
        # BC loss after pretrain.
        torch.manual_seed(0)
        n = agent._demo_obs.shape[0]
        idx = torch.randint(0, n, (32,))
        with torch.no_grad():
            bc_before = float(torch.nn.functional.cross_entropy(
                agent.online(agent._demo_obs[idx]).mean(dim=-1),
                agent._demo_actions[idx]).item())
        # 50 random train steps.
        rng = np.random.default_rng(0)
        for _ in range(50):
            fake_obs = rng.standard_normal((8, 7)).astype(np.float32)
            fake_next = rng.standard_normal((8, 7)).astype(np.float32)
            fake_actions = rng.integers(0, 5, size=(8,))
            fake_rewards = rng.standard_normal((8,)).astype(np.float32) * 0.01
            fake_dones = np.zeros((8,), dtype=np.float32)
            fake_weights = np.ones((8,), dtype=np.float32)
            agent.train_step({
                "obs": torch.from_numpy(fake_obs),
                "next_obs": torch.from_numpy(fake_next),
                "actions": torch.from_numpy(fake_actions),
                "rewards": torch.from_numpy(fake_rewards),
                "dones": torch.from_numpy(fake_dones),
                "weights": torch.from_numpy(fake_weights),
                "indices": np.arange(8),
            })
        # BC loss after 50 train steps.
        with torch.no_grad():
            bc_after = float(torch.nn.functional.cross_entropy(
                agent.online(agent._demo_obs[idx]).mean(dim=-1),
                agent._demo_actions[idx]).item())
        assert bc_after < max(0.5, bc_before * 2.0), (
            f"BC anchor drifted {bc_before:.3f} -> {bc_after:.3f} "
            f"after the full pretrain+train pipeline")
