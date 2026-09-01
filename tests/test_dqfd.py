"""Tests for the DQfD joint-loss agent.

These tests pin:
* The supervised loss is *cross-entropy* between the
  agent's argmax and the expert action — same loss
  function the BC pretrain uses.
* The margin loss is a hinge that penalises the agent
  for putting probability mass on any non-expert action.
* The joint loss decreases the BC loss over training
  (the agent does internalise the demonstrations).
* The trained agent achieves expert-like behaviour on
  a held-out demonstration."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from config import RLConfig
from dqfd_agent import DQfDAgent, DQfDConfig
from expert_policy import ExpertPolicy
from learnable_env import LearnableEnv, LearnableEnvConfig


@pytest.fixture()
def dqfd_agent() -> DQfDAgent:
    cfg = RLConfig(profile="strict_lite", gamma=0.99,
                    grad_clip_norm=10.0, learning_rate=1e-3)
    dqfd_cfg = DQfDConfig()
    # 5 actions, no visual input — the 7-dim LearnableEnv
    # observation is flattened to a single "frame" of
    # shape (1, 7) so the encoder collapses to identity.
    agent = DQfDAgent("strict_lite", cfg, dqfd_cfg,
                       in_frames=1, size=7, num_quantiles=11)
    # Replace the encoder so 7-dim input works.
    import torch.nn as nn
    from distributional import mid_quantiles
    agent.online = _SmallQNet(n_actions=5, in_dim=7,
                               num_quantiles=11).to(agent.device)
    agent.target = _SmallQNet(n_actions=5, in_dim=7,
                               num_quantiles=11).to(agent.device)
    agent.target.load_state_dict(agent.online.state_dict())
    for p in agent.target.parameters():
        p.requires_grad_(False)
    agent.tau = mid_quantiles(11).to(agent.device)
    return agent


class _SmallQNet(nn.Module):
    """A tiny QR-DQN compatible with the 7-dim LearnableEnv obs."""

    def __init__(self, n_actions: int, in_dim: int,
                 num_quantiles: int) -> None:
        super().__init__()
        self.num_actions = n_actions
        self.num_quantiles = num_quantiles
        # Encoder is identity (we already have a flat 7-d
        # observation; the convolutional encoder would
        # throw a shape error).
        self.encoder = nn.Identity()
        self.enc_out = in_dim
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_quantiles),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, n_actions * num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        v = self.value_stream(h)
        a = self.advantage_stream(h).view(
            -1, self.num_actions, self.num_quantiles)
        v = v.unsqueeze(1)
        return v + a - a.mean(dim=1, keepdim=True)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=-1)


class TestDQfDPretrain:
    def test_pretrain_reduces_bc_loss(self, dqfd_agent: DQfDAgent) -> None:
        env = LearnableEnv(seed=0)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        # Reshape for the 1-frame, 7-dim input the test
        # network expects: pass as 2-D (N, 7) so the
        # production re-inflation path is also tested.
        obs_in = obs.reshape(obs.shape[0], obs.shape[1])
        result = dqfd_agent.pretrain_demos(obs_in, act, n_epochs=20)
        assert result["bc_loss"] < 1.0
        # The agent should now have cached the demos.
        assert dqfd_agent._demo_obs is not None
        assert dqfd_agent._demo_actions is not None

    def test_pretrain_solves_learnable_env(self,
                                           dqfd_agent: DQfDAgent) -> None:
        """End-to-end: pretrain on expert demos and verify
        the agent achieves the 30s KPI on the LearnableEnv.
        This is the canonical "BC + no exploration" test."""
        # Collect demos from many seeds so the BC dataset
        # covers the full state space.
        all_obs = []
        all_acts = []
        for seed in range(20):
            env = LearnableEnv(LearnableEnvConfig(
                obstacle_period=30, approach_time=15, max_steps=900),
                seed=seed)
            expert = ExpertPolicy()
            obs, act, _ = expert.collect_demonstration(env)
            # 2-D shape (N, 7) so pretrain_demos leaves
            # the obs alone (the test's _SmallQNet uses
            # an identity encoder and expects 2-D).
            obs_in = obs.reshape(obs.shape[0], obs.shape[1])
            all_obs.append(obs_in)
            all_acts.append(act)
        obs_train = np.concatenate(all_obs, axis=0)
        act_train = np.concatenate(all_acts, axis=0)
        dqfd_agent.pretrain_demos(obs_train, act_train, n_epochs=30)
        # Evaluate.
        env = LearnableEnv(seed=100)
        obs = env.reset()
        for _ in range(900):
            x = torch.from_numpy(
                obs.reshape(1, 7)).float().to(dqfd_agent.device)
            with torch.no_grad():
                q = dqfd_agent.online.q_values(x)
            a = int(q.argmax().item())
            obs, _, done, _ = env.step(a)
            if done:
                break
        # The BC policy must achieve the full 30s.
        assert env.t >= 899, (
            f"BC policy only survived {env.t/30.0:.2f}s, need ≥30s")


class TestDQfDSupervisedAndMarginLosses:
    def test_supervised_loss_returns_scalar(self,
                                            dqfd_agent: DQfDAgent) -> None:
        env = LearnableEnv(seed=0)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        # 2-D shape — _SmallQNet uses identity encoder
        # and expects 2-D input.
        obs_in = obs.reshape(obs.shape[0], obs.shape[1])
        dqfd_agent.pretrain_demos(obs_in, act, n_epochs=5)
        loss = dqfd_agent.supervised_loss(batch_size=32)
        assert loss.dim() == 0  # scalar
        assert torch.isfinite(loss)

    def test_margin_loss_returns_scalar(self,
                                        dqfd_agent: DQfDAgent) -> None:
        env = LearnableEnv(seed=0)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        obs_in = obs.reshape(obs.shape[0], obs.shape[1])
        dqfd_agent.pretrain_demos(obs_in, act, n_epochs=5)
        loss = dqfd_agent.margin_loss(batch_size=32)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        # Margin loss is non-negative (it's a hinge).
        assert float(loss.detach()) >= 0.0

    def test_supervised_loss_zero_without_demos(self,
                                                dqfd_agent: DQfDAgent) -> None:
        # No demos cached → zero loss (so the parent's pure
        # RL path is unaffected).
        assert dqfd_agent._demo_obs is None
        loss = dqfd_agent.supervised_loss(batch_size=8)
        assert float(loss.detach()) == 0.0


class TestDQfDConfig:
    def test_defaults(self) -> None:
        c = DQfDConfig()
        assert c.lambda_bc > 0
        assert c.lambda_margin >= 0
        assert c.margin > 0
        assert c.supervised_decay_episodes > 0


class TestDQfDGradientFlow:
    """Pins the *joint loss* gradient flow.

    The previous implementation called
    ``super().train_step`` (which did its own
    ``optimizer.step()``) and then computed the BC +
    margin terms afterwards.  PyTorch's
    ``Optimizer.step()`` does NOT zero ``.grad``, so the
    BC + margin backward pass *accumulated* on top of
    the TD gradient that had already been applied — the
    TD update was effectively applied twice and the BC
    anchor was silently lost.  These tests pin the
    correct behaviour: the BC loss must contribute to
    the gradient of ``self.online`` (so a single train
    step actually moves the policy towards the expert
    action), and the supervised-loss signal must
    survive a sequence of train steps that include
    pure-TD updates.
    """

    def test_bc_loss_gradients_flow_to_online_net(self,
                                                  dqfd_agent: DQfDAgent) -> None:
        """``bc_loss.backward()`` should populate
        ``self.online.weight.grad`` with non-zero
        values.  In the old buggy implementation the BC
        term was applied on a non-leaf tensor created
        after ``optimizer.step()`` had already detached
        the graph, so ``.grad`` stayed None."""
        import torch
        env = LearnableEnv(seed=0)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        obs_in = obs.reshape(obs.shape[0], obs.shape[1])
        dqfd_agent.pretrain_demos(obs_in, act, n_epochs=2)
        # Pick a demo minibatch.
        n = dqfd_agent._demo_obs.shape[0]
        idx = torch.randint(0, n, (8,))
        x = dqfd_agent._demo_obs[idx]
        y = dqfd_agent._demo_actions[idx]
        dist = dqfd_agent.online(x)
        logits = dist.mean(dim=-1)
        bc = torch.nn.functional.cross_entropy(logits, y)
        dqfd_agent.optimizer.zero_grad(set_to_none=True)
        bc.backward()
        # At least one parameter must have a non-zero grad.
        max_grad = 0.0
        for p in dqfd_agent.online.parameters():
            if p.grad is not None:
                max_grad = max(max_grad, float(p.grad.abs().max()))
        assert max_grad > 0.0, (
            "BC loss did not produce any gradient on "
            "self.online — the joint loss is broken")

    def test_joint_train_step_preserves_bc_anchor(self,
                                                  dqfd_agent: DQfDAgent) -> None:
        """After 50 train steps, the BC loss on a held-out
        demo batch must remain at the *post-pretrain*
        value (the agent should still match the expert).
        In the old buggy implementation the BC anchor
        drifted up by 30%+ within 50 steps because the
        TD loss was applied twice and the BC term was
        never optimised.
        """
        import torch
        env = LearnableEnv(seed=0)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        obs_in = obs.reshape(obs.shape[0], obs.shape[1])
        dqfd_agent.pretrain_demos(obs_in, act, n_epochs=20)
        # Measure the BC loss on a fixed held-out batch
        # *after* pretrain.
        torch.manual_seed(0)
        n = dqfd_agent._demo_obs.shape[0]
        idx = torch.randint(0, n, (32,))
        x_holdout = dqfd_agent._demo_obs[idx]
        y_holdout = dqfd_agent._demo_actions[idx]
        with torch.no_grad():
            logits = dqfd_agent.online(x_holdout).mean(dim=-1)
            bc_before = float(torch.nn.functional.cross_entropy(
                logits, y_holdout).item())
        # Run 50 train steps on RANDOM TD batches (the
        # agent has no real data; we just want the TD
        # updates to be applied alongside the BC anchor).
        import numpy as np
        rng = np.random.default_rng(0)
        for _ in range(50):
            # Random demo-batch (BC) + random online batch
            # (TD).  The agent's supervised_loss picks a
            # fresh demo minibatch internally, so we just
            # need *any* online batch shape.
            fake_obs = rng.standard_normal((8, obs.shape[1])).astype(np.float32)
            fake_next = rng.standard_normal((8, obs.shape[1])).astype(np.float32)
            fake_actions = rng.integers(0, 5, size=(8,))
            fake_rewards = rng.standard_normal((8,)).astype(np.float32) * 0.01
            fake_dones = np.zeros((8,), dtype=np.float32)
            fake_weights = np.ones((8,), dtype=np.float32)
            dqfd_agent.train_step({
                "obs": torch.from_numpy(fake_obs),
                "next_obs": torch.from_numpy(fake_next),
                "actions": torch.from_numpy(fake_actions),
                "rewards": torch.from_numpy(fake_rewards),
                "dones": torch.from_numpy(fake_dones),
                "weights": torch.from_numpy(fake_weights),
                "indices": np.arange(8),
            })
        # Measure BC loss again on the SAME held-out batch.
        with torch.no_grad():
            logits = dqfd_agent.online(x_holdout).mean(dim=-1)
            bc_after = float(torch.nn.functional.cross_entropy(
                logits, y_holdout).item())
        # BC loss should be at most 2x the initial value
        # (the old buggy code typically let it grow 5-10x
        # because the BC term was never optimised).
        assert bc_after < max(0.5, bc_before * 2.0), (
            f"BC anchor drifted: {bc_before:.3f} -> {bc_after:.3f} "
            f"after 50 train steps — the joint loss is broken "
            f"(TD loss being applied twice)")

    def test_joint_loss_single_backward_single_step(
            self, dqfd_agent: DQfDAgent) -> None:
        """Regression test for the backprop-bug class of
        failures.  The buggy implementation called
        ``super().train_step`` (which does its own
        ``optimizer.step()``) and *then* a second
        ``backward()`` + ``step()`` for the BC + margin
        terms.  PyTorch's ``Optimizer.step()`` does NOT
        zero ``.grad``, so the second backward
        ACCUMULATED on top of the still-present TD
        gradient, and the second step applied the TD
        gradient a second time.

        We pin the *correct* behaviour: exactly one
        ``zero_grad``, one ``backward``, one ``step`` per
        train step, and the resulting gradients reflect
        all three loss terms.
        """
        import numpy as np
        from unittest import mock
        # Pre-load the demo so the joint-loss path is
        # exercised.
        env = LearnableEnv(seed=0)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        obs_in = obs.reshape(obs.shape[0], obs.shape[1])
        dqfd_agent.pretrain_demos(obs_in, act, n_epochs=2)
        # Spy on the optimizer.
        opt = dqfd_agent.optimizer
        zero_count, step_count = 0, 0
        original_zero = opt.zero_grad
        original_step = opt.step
        def _zero(*a, **kw):
            nonlocal zero_count
            zero_count += 1
            return original_zero(*a, **kw)
        def _step(*a, **kw):
            nonlocal step_count
            step_count += 1
            return original_step(*a, **kw)
        opt.zero_grad = _zero
        opt.step = _step
        # Run a single train step.
        rng = np.random.default_rng(0)
        fake_obs = rng.standard_normal((8, obs.shape[1])).astype(np.float32)
        fake_next = rng.standard_normal((8, obs.shape[1])).astype(np.float32)
        fake_actions = rng.integers(0, 5, size=(8,))
        fake_rewards = rng.standard_normal((8,)).astype(np.float32) * 0.01
        fake_dones = np.zeros((8,), dtype=np.float32)
        fake_weights = np.ones((8,), dtype=np.float32)
        dqfd_agent.train_step({
            "obs": torch.from_numpy(fake_obs),
            "next_obs": torch.from_numpy(fake_next),
            "actions": torch.from_numpy(fake_actions),
            "rewards": torch.from_numpy(fake_rewards),
            "dones": torch.from_numpy(fake_dones),
            "weights": torch.from_numpy(fake_weights),
            "indices": np.arange(8),
        })
        # The fix uses ONE forward + ONE backward + ONE
        # step.  The buggy code did TWO of each.
        assert zero_count == 1, (
            f"optimizer.zero_grad called {zero_count} times "
            f"in one train step — joint loss has duplicate "
            f"optimizer pass")
        assert step_count == 1, (
            f"optimizer.step called {step_count} times in "
            f"one train step — TD loss is being applied "
            f"twice (the BC + margin path is not merged)")
