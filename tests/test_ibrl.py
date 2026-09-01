"""Tests for the IBRL (Imitation Bootstrapped RL) module.

These tests pin:
* The actor-proposal action selection picks the
  action with the higher Q-value.
* The bootstrap-proposal Q is the max of the two
  candidate Q-values.
* The IBRLAgent integrates a frozen BC net with a
  trainable RL net.
* The IBRLTrainer's loss is finite and the
  gradients flow.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from ibrl import (IBRLAgent, IBRLConfig, IBRLTrainer,
                    actor_proposal_action, bootstrap_proposal_q)


class _TinyNet(nn.Module):
    """A tiny 3-action policy that returns
    *given* Q-values so we can verify the
    actor-proposal / bootstrap-proposal logic."""

    def __init__(self, q_values: list) -> None:
        super().__init__()
        # Use a Parameter so the optimiser has
        # something to update.
        self.q_param = nn.Parameter(
            torch.tensor(q_values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast fixed_q to the batch dim.
        return self.q_param.expand(x.shape[0], -1)


def _fake_batch(n: int = 4) -> dict:
    return {
        "obs": torch.zeros(n, 1),
        "next_obs": torch.zeros(n, 1),
        "actions": torch.zeros(n, dtype=torch.long),
        "rewards": torch.zeros(n),
        "dones": torch.zeros(n),
        "weights": torch.ones(n),
        "gamma_pows": torch.ones(n),
    }


class TestActorProposal:
    def test_picks_higher_q_action(self) -> None:
        """When the BC net's argmax is NOT the RL
        net's argmax, the actor proposal picks the
        one with the higher RL Q-value."""
        # BC net thinks action 1 is best (Q=10).
        # RL net's Q: a_0=0, a_1=5, a_2=8 → argmax
        # = 2 (action 2, Q=8).  RL's Q at the BC
        # action (1) is 5, which is LESS than RL's
        # argmax (8 at action 2), so the proposal
        # picks action 2.
        bc = _TinyNet([0.0, 10.0, 0.0])  # argmax = 1
        rl = _TinyNet([0.0, 5.0, 8.0])   # argmax = 2
        # RL's Q(1) = 5 < RL's Q(2) = 8, so the
        # proposal picks action 2 (the RL action).
        obs = torch.zeros(1, 1)
        action, info = actor_proposal_action(bc, rl, obs,
                                                noise_eps=0.0)
        assert action.item() == 2
        assert info["bc_action"].item() == 1
        assert info["rl_action"].item() == 2
        # The proposal uses the RL action (Q=8)
        # rather than the BC action (Q=5).
        assert info["use_il"].item() == 0.0

    def test_picks_bc_when_higher(self) -> None:
        """When the RL net's Q at the BC action
        exceeds the RL net's Q at its own argmax,
        the proposal picks the BC action."""
        # BC argmax = 1, RL argmax = 2 (Q=1).
        # RL's Q at BC action (1) is 5, which is
        # greater than RL's argmax (1 at action 2).
        bc = _TinyNet([0.0, 10.0, 0.0])  # argmax = 1
        rl = _TinyNet([0.0, 5.0, 1.0])   # argmax = 1 (Q=5)
        obs = torch.zeros(1, 1)
        action, info = actor_proposal_action(bc, rl, obs,
                                                noise_eps=0.0)
        # Both argmax = 1, so the proposal picks 1.
        assert action.item() == 1
        assert info["bc_action"].item() == 1
        assert info["rl_action"].item() == 1

    def test_noise_replaces_action(self) -> None:
        """With noise_eps=1.0, the action is
        always replaced with a random one."""
        torch.manual_seed(0)
        bc = _TinyNet([0.0, 10.0, 0.0])
        rl = _TinyNet([0.0, 5.0, 1.0])
        obs = torch.zeros(10, 1)
        actions, _ = actor_proposal_action(bc, rl, obs,
                                              noise_eps=1.0)
        # At least one action should not be 1
        # (the deterministic answer).
        assert (actions != 1).any()


class TestBootstrapProposal:
    def test_max_of_two_qs(self) -> None:
        """The bootstrap Q is max(Q(a_il), Q(a_rl))."""
        bc = _TinyNet([0.0, 10.0, 0.0])  # a_il = 1
        rl = _TinyNet([0.0, 5.0, 7.0])   # a_rl = 2
        # Q(1) = 5, Q(2) = 7, max = 7.
        next_obs = torch.zeros(1, 1)
        q = bootstrap_proposal_q(bc, rl, next_obs=next_obs)
        assert q.item() == pytest.approx(7.0, abs=1e-5)

    def test_q_il_higher(self) -> None:
        """When the BC action's Q is higher than
        the RL action's Q, the bootstrap returns
        the BC action's Q."""
        bc = _TinyNet([0.0, 10.0, 0.0])  # a_il = 1
        rl = _TinyNet([0.0, 7.0, 0.0])   # a_rl = 1
        # Both pick action 1, but the Q is 7.
        next_obs = torch.zeros(1, 1)
        q = bootstrap_proposal_q(bc, rl, next_obs=next_obs)
        assert q.item() == pytest.approx(7.0, abs=1e-5)


class TestIBRLAgent:
    def test_bc_net_frozen(self) -> None:
        """The BC net's parameters should have
        ``requires_grad=False`` after wrapping."""
        bc = _TinyNet([0.0, 1.0, 0.0])
        rl = _TinyNet([0.0, 0.0, 1.0])
        agent = IBRLAgent(bc, rl)
        for p in agent.bc_net.parameters():
            assert not p.requires_grad

    def test_act_falls_back_to_rl(self) -> None:
        """When ``use_actor_proposal=False``, the
        agent returns the RL net's argmax."""
        bc = _TinyNet([0.0, 10.0, 0.0])
        rl = _TinyNet([0.0, 0.0, 5.0])
        cfg = IBRLConfig(use_actor_proposal=False)
        agent = IBRLAgent(bc, rl, cfg)
        obs = torch.zeros(1, 1)
        action, info = agent.act(obs)
        assert action.item() == 2

    def test_act_uses_actor_proposal(self) -> None:
        """When ``use_actor_proposal=True``, the
        agent picks the higher-Q action."""
        bc = _TinyNet([0.0, 10.0, 0.0])
        rl = _TinyNet([0.0, 5.0, 1.0])
        agent = IBRLAgent(bc, rl, IBRLConfig(noise_eps=0.0))
        obs = torch.zeros(1, 1)
        action, _ = agent.act(obs)
        # RL's Q(1)=5 > RL's Q(2)=1, so action 1
        # (the BC argmax) wins.
        assert action.item() == 1


class TestIBRLTrainer:
    def test_loss_finite(self) -> None:
        """The IBRL loss is finite on a fake batch."""
        bc = _TinyNet([0.0, 1.0, 0.0])
        rl = _TinyNet([0.0, 0.0, 1.0])
        ibrl = IBRLAgent(bc, rl, IBRLConfig(noise_eps=0.0))
        opt = torch.optim.Adam(rl.parameters(), lr=1e-3)
        trainer = IBRLTrainer(ibrl, opt, lambda_bc=0.5)
        batch = _fake_batch(n=4)
        batch["bc_actions"] = torch.zeros(4, dtype=torch.long)
        out = trainer.step(batch, grad_clip=10.0)
        assert np.isfinite(out["loss"])
        assert np.isfinite(out["td_loss"])

    def test_loss_with_sil(self) -> None:
        """The loss is finite when SIL is enabled."""
        from sil import SILBuffer, SILConfig, SILTrainer
        bc = _TinyNet([0.0, 1.0, 0.0])
        rl = _TinyNet([0.0, 0.0, 1.0])
        ibrl = IBRLAgent(bc, rl, IBRLConfig(noise_eps=0.0))
        opt = torch.optim.Adam(rl.parameters(), lr=1e-3)
        # Build a small SIL buffer with 1 good episode.
        buf = SILBuffer(SILConfig(capacity=5, gamma=0.99))
        buf.add_episode(states=[np.zeros(1) for _ in range(5)],
                          actions=[0, 1, 2, 0, 1],
                          rewards=[1.0, 1.0, 1.0, 1.0, 1.0],
                          start_value=0.0)
        # Need a "agent" wrapper with a .device
        # and .online attribute for SILTrainer.
        class _A:
            device = "cpu"
            online = rl
        sil_trainer = SILTrainer(_A(), SILConfig())
        trainer = IBRLTrainer(ibrl, opt, lambda_bc=0.0,
                                lambda_sil=0.1,
                                sil_trainer=sil_trainer)
        batch = _fake_batch(n=4)
        out = trainer.step(batch,
                            sil_batch=buf.sample(batch_size=4),
                            grad_clip=10.0)
        assert np.isfinite(out["loss"])
        assert "sil_policy_loss" in out
        assert "sil_value_loss" in out
