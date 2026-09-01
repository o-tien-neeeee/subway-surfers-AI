"""Tests for the IBRLDQNAgent (full v1.23.0 integration)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from ibrl_agent import IBRLDQNAgent, IBRLDQNConfig
from ibrl import IBRLConfig
from sil import SILConfig
from auto_entropy import AutoEntropyConfig


class _TinyBC(nn.Module):
    """A small BC net that always returns the
    same Q-values."""

    def __init__(self) -> None:
        super().__init__()
        self.q_param = nn.Parameter(
            torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_param.expand(x.shape[0], -1)


class _TinyRL(nn.Module):
    """A small RL net with a learnable parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.q_param = nn.Parameter(
            torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class TestIBRLDQNAgent:
    def test_build_freezes_bc(self) -> None:
        cfg = IBRLDQNConfig(use_ema=False, use_sil=False,
                              use_auto_entropy=False)
        agent = IBRLDQNAgent(cfg)
        bc = _TinyBC()
        rl = _TinyRL()
        agent.build(bc, rl)
        for p in agent.bc_net.parameters():
            assert not p.requires_grad

    def test_act_uses_actor_proposal(self) -> None:
        cfg = IBRLDQNConfig(use_ema=False, use_sil=False,
                              use_auto_entropy=False,
                              ibrl=IBRLConfig(noise_eps=0.0))
        agent = IBRLDQNAgent(cfg)
        agent.build(_TinyBC(), _TinyRL())
        obs = torch.zeros(1, 1)
        action, info = agent.act(obs)
        # BC argmax = 1 (Q_bc[1] = 1, others 0).
        # RL argmax = 2 (Q_rl[2] = 1, others 0).
        # Q_rl(1) = 0, Q_rl(2) = 1 → use action 2
        # (the RL action has the higher RL Q).
        assert action.item() == 2
        assert info["bc_action"].item() == 1
        assert info["rl_action"].item() == 2
        assert info["use_il"].item() == 0.0

    def test_train_step_runs(self) -> None:
        cfg = IBRLDQNConfig(use_ema=False, use_sil=False,
                              use_auto_entropy=False)
        agent = IBRLDQNAgent(cfg)
        agent.build(_TinyBC(), _TinyRL())
        batch = _fake_batch(n=4)
        batch["bc_actions"] = torch.zeros(4, dtype=torch.long)
        out = agent.train_step(batch)
        assert "loss" in out
        assert "td_loss" in out
        assert np.isfinite(out["loss"])

    def test_train_step_with_sil(self) -> None:
        cfg = IBRLDQNConfig(use_ema=False, use_sil=True,
                              use_auto_entropy=False,
                              sil=SILConfig(capacity=5, gamma=0.99))
        agent = IBRLDQNAgent(cfg)
        agent.build(_TinyBC(), _TinyRL())
        # Add an episode to the SIL buffer.
        agent.add_episode(
            states=[np.zeros(1) for _ in range(5)],
            actions=[0, 1, 2, 0, 1],
            rewards=[1.0, 1.0, 1.0, 1.0, 1.0],
            start_value=0.0)
        assert agent.sil_stats()["n"] == 5
        batch = _fake_batch(n=4)
        batch["bc_actions"] = torch.zeros(4, dtype=torch.long)
        out = agent.train_step(batch)
        assert np.isfinite(out["loss"])
        assert "sil_policy_loss" in out
        assert "sil_value_loss" in out

    def test_train_step_with_auto_entropy(self) -> None:
        cfg = IBRLDQNConfig(use_ema=False, use_sil=False,
                              use_auto_entropy=True,
                              auto_entropy=AutoEntropyConfig(
                                  n_actions=5))
        agent = IBRLDQNAgent(cfg)
        agent.build(_TinyBC(), _TinyRL())
        batch = _fake_batch(n=4)
        batch["bc_actions"] = torch.zeros(4, dtype=torch.long)
        out = agent.train_step(batch)
        assert "alpha" in out
        assert np.isfinite(out["alpha"])

    def test_ema_install_restore(self) -> None:
        cfg = IBRLDQNConfig(use_ema=True, use_sil=False,
                              use_auto_entropy=False)
        agent = IBRLDQNAgent(cfg)
        agent.build(_TinyBC(), _TinyRL())
        # EMA starts as a copy of the RL net.  Install.
        agent.eval_mode()
        for p in agent.rl_net.parameters():
            # The EMA values are the original
            # [0, 0, 1, 0, 0].
            assert torch.allclose(p, torch.tensor(
                [0.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-5)
        # Restore.
        agent.train_mode()
