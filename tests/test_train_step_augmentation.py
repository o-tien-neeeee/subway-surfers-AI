"""Tests for the augmentation integration in
``DistributionalDoubleDQNAgent.train_step``.

These tests pin:

* When ``cfg.augment_obs=False`` the train_step
  produces the same loss as the no-augmentation
  baseline (the network sees the raw frames).
* When ``cfg.augment_obs=True`` the train_step
  produces a *different* loss from the no-
  augmentation baseline (the network sees shifted
  frames) — i.e. the augmentation is actually
  applied.
* The gradient still flows back through the
  augmented observation (i.e. ``loss.backward()``
  does not fail when the obs has been shifted).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent_distributional import DistributionalDoubleDQNAgent
from config import RLConfig


def _make_agent(augment: bool) -> DistributionalDoubleDQNAgent:
    cfg = RLConfig(profile="strict_lite", gamma=0.99,
                    num_quantiles=11,
                    learning_rate=1e-4,
                    target_update_every=1000,
                    polyak_target=False,
                    augment_obs=augment,
                    augment_translate_px=3,
                    augment_intensity=0.10,
                    augment_frame_diff=False)
    return DistributionalDoubleDQNAgent(
        "strict_lite", cfg, in_frames=4, size=84,
        num_quantiles=11, device="cpu", seed=42)


def _fake_batch() -> dict:
    """A small (B=2) batch of fake transitions."""
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 256, size=(2, 4, 84, 84), dtype=np.uint8)
    next_obs = rng.integers(0, 256, size=(2, 4, 84, 84),
                              dtype=np.uint8)
    return {
        "obs": obs,
        "next_obs": next_obs,
        "actions": np.array([0, 1], dtype=np.int64),
        "rewards": np.array([0.0, 0.0], dtype=np.float32),
        "dones": np.array([0.0, 0.0], dtype=np.float32),
        "weights": np.array([1.0, 1.0], dtype=np.float32),
        "gamma_pows": np.array([0.99, 0.99], dtype=np.float32),
        "indices": np.array([0, 1]),
    }


class TestTrainStepAugmentation:
    def test_no_aug_loss_is_deterministic(self) -> None:
        """With augment_obs=False, two train_steps on
        the same data with the same seed give the
        same loss (the network sees the raw obs)."""
        torch.manual_seed(0)
        agent = _make_agent(augment=False)
        batch = _fake_batch()
        loss1 = agent.train_step(batch)["loss"]
        # Re-init the agent to reset the optimizer's
        # momentum (the loss is otherwise different
        # because the optimiser state has moved).
        torch.manual_seed(0)
        agent = _make_agent(augment=False)
        loss2 = agent.train_step(batch)["loss"]
        assert abs(loss1 - loss2) < 1e-5

    def test_aug_changes_loss(self) -> None:
        """With augment_obs=True the loss is
        different from the no-augmentation baseline
        (because the network sees shifted frames)."""
        torch.manual_seed(0)
        agent_no_aug = _make_agent(augment=False)
        loss_no_aug = agent_no_aug.train_step(_fake_batch())["loss"]
        torch.manual_seed(0)
        agent_aug = _make_agent(augment=True)
        loss_aug = agent_aug.train_step(_fake_batch())["loss"]
        # The losses must differ (the augmentation
        # actually changes the input).
        assert abs(loss_no_aug - loss_aug) > 1e-3

    def test_aug_backward_works(self) -> None:
        """The loss.backward() inside train_step
        must not raise when augment_obs=True (the
        augmentation is differentiable? No — it is
        not — but it must not break the chain
        rule)."""
        agent = _make_agent(augment=True)
        # Should not raise.
        agent.train_step(_fake_batch())
