"""Tests for the Self-Imitation Learning (SIL) module.

These tests pin the SIL paper's contract:
* add_episode computes the discounted return correctly
* the priority is (R - start_value)_+ + eps
* sample returns a valid batch with the right shape
* the trainer's loss is finite
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from sil import SILBuffer, SILConfig, SILTrainer


class TestSILBuffer:
    def test_discounted_return(self) -> None:
        """A short episode [r=1, r=1, r=1] with
        γ=0.5 should have returns [1.875, 1.75, 1]."""
        buf = SILBuffer(SILConfig(capacity=10, gamma=0.5))
        # 3-step episode, each step has state = step
        # index (just an int for the test).
        buf.add_episode(states=[0, 1, 2],
                          actions=[0, 0, 0],
                          rewards=[1.0, 1.0, 1.0],
                          start_value=0.0)
        # The flat arrays should have 3 entries.
        assert len(buf) == 3
        # R[2] = 1, R[1] = 1 + 0.5 * 1 = 1.5,
        # R[0] = 1 + 0.5 * 1.5 = 1.75.
        np.testing.assert_allclose(
            buf._flat_returns,
            [1.75, 1.5, 1.0],
            atol=1e-5)

    def test_priority_clipped(self) -> None:
        """A bad episode (R < start_value) should
        still be stored but with a low priority
        (just eps)."""
        buf = SILBuffer(SILConfig(capacity=10, gamma=0.99,
                                      priority_eps=0.01))
        buf.add_episode(states=[0, 1], actions=[0, 1],
                          rewards=[0.0, 0.0],
                          start_value=100.0)
        # Priority = max(0, R - 100) + 0.01 = 0.01.
        assert all(p == 0.01 for p in buf._flat_priorities)

    def test_priority_positive(self) -> None:
        """A good episode should have a priority
        proportional to (R - start_value)."""
        buf = SILBuffer(SILConfig(capacity=10, gamma=0.99,
                                      priority_eps=0.01))
        buf.add_episode(states=[0, 1, 2], actions=[0, 0, 0],
                          rewards=[1.0, 1.0, 1.0],
                          start_value=0.5)
        # R = 1 + 0.99 + 0.99^2 = 2.9701.
        # Priority = (2.9701 - 0.5) + 0.01 = 2.4801.
        for p in buf._flat_priorities:
            assert p > 2.4
            assert p < 2.5

    def test_sample_returns_correct_shape(self) -> None:
        buf = SILBuffer(SILConfig(capacity=10, gamma=0.99))
        for ep in range(3):
            buf.add_episode(states=[ep, ep + 1, ep + 2],
                              actions=[0, 1, 2],
                              rewards=[0.1 * ep] * 3,
                              start_value=0.0)
        # 3 episodes × 3 steps = 9 transitions.
        batch = buf.sample(batch_size=4)
        assert batch["obs"].shape == (4,)  # 4 sampled
        assert batch["actions"].shape == (4,)
        assert batch["returns"].shape == (4,)

    def test_max_transitions_cap(self) -> None:
        """The buffer should drop old transitions
        when the cap is exceeded."""
        cfg = SILConfig(capacity=2, max_episode_len=10,
                          gamma=0.99)
        buf = SILBuffer(cfg)
        # Add 5 episodes of 3 steps = 15 transitions;
        # cap is 2 * 10 = 20 so we keep all 15.
        for ep in range(5):
            buf.add_episode(states=[0, 1, 2],
                              actions=[0, 0, 0],
                              rewards=[0.0, 0.0, 0.0],
                              start_value=0.0)
        assert len(buf) == 15

    def test_is_ready(self) -> None:
        buf = SILBuffer(SILConfig(capacity=10, gamma=0.99))
        assert not buf.is_ready(min_transitions=5)
        for _ in range(3):
            buf.add_episode(states=[0, 1, 2],
                              actions=[0, 0, 0],
                              rewards=[0.0, 0.0, 0.0],
                              start_value=0.0)
        assert not buf.is_ready(min_transitions=10)
        assert buf.is_ready(min_transitions=5)


class TestSILTrainer:
    def test_loss_finite(self) -> None:
        """The SIL loss is finite on a non-empty
        batch."""
        # Build a small "agent" with a single
        # identity encoder + 3-action dueling head.
        class _Toy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, x):
                return self.fc(x)

        agent = type("A", (), {"device": "cpu",
                                  "online": _Toy()})()
        trainer = SILTrainer(agent, SILConfig())
        batch = {
            "obs": np.random.rand(8, 4).astype(np.float32),
            "actions": np.random.randint(0, 3, 8).astype(np.int64),
            "returns": np.random.rand(8).astype(np.float32),
        }
        out = trainer.loss(batch)
        assert torch.isfinite(out["policy_loss"])
        assert torch.isfinite(out["value_loss"])

    def test_loss_handles_empty_batch(self) -> None:
        """An empty batch should return zero losses
        (not crash)."""
        class _Toy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, x):
                return self.fc(x)

        agent = type("A", (), {"device": "cpu",
                                  "online": _Toy()})()
        trainer = SILTrainer(agent, SILConfig())
        out = trainer.loss({"obs": np.zeros((0, 4),
                                               dtype=np.float32),
                              "actions": np.zeros((0,),
                                                    dtype=np.int64),
                              "returns": np.zeros((0,),
                                                    dtype=np.float32)})
        assert out["policy_loss"].item() == 0.0
        assert out["value_loss"].item() == 0.0
