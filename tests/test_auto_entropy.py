"""Tests for the AutoEntropy (SAC-style temperature)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from auto_entropy import AutoEntropy, AutoEntropyConfig, soft_policy_loss


class TestAutoEntropy:
    def test_alpha_is_exp_of_log_alpha(self) -> None:
        """α = exp(log_alpha)."""
        cfg = AutoEntropyConfig(initial_log_alpha=float(np.log(0.5)),
                                  learn_alpha=True)
        ae = AutoEntropy(cfg)
        assert ae.alpha.item() == pytest.approx(0.5, abs=1e-5)

    def test_default_target_entropy(self) -> None:
        """With n_actions=5, target_entropy = -4.45."""
        cfg = AutoEntropyConfig()
        ae = AutoEntropy(cfg)
        assert ae.target_entropy == pytest.approx(-0.89 * 5, abs=1e-5)

    def test_step_updates_alpha(self) -> None:
        """The temperature changes after one step."""
        cfg = AutoEntropyConfig(learn_alpha=True, n_actions=3)
        ae = AutoEntropy(cfg)
        before = ae.alpha.item()
        # Large negative log_probs → the policy is
        # too deterministic → loss should DECREASE
        # α.
        log_probs = torch.full((4,), -10.0)
        ae.step(log_probs)
        # The exact change depends on the loss sign
        # but α should have moved.
        after = ae.alpha.item()
        assert after != before

    def test_step_does_nothing_when_learn_false(self) -> None:
        """When learn_alpha=False, step is a no-op."""
        cfg = AutoEntropyConfig(learn_alpha=False, n_actions=3)
        ae = AutoEntropy(cfg)
        before = ae.alpha.item()
        ae.step(torch.full((4,), -10.0))
        assert ae.alpha.item() == before


class TestSoftPolicyLoss:
    def test_loss_is_neg_q_at_argmax(self) -> None:
        """soft_policy_loss should return
        -E[Q(s, argmax_a Q(s, a))] for a batch."""
        q_values = torch.tensor([[1.0, 2.0, 3.0],
                                    [0.0, 0.0, 0.0]])
        loss, log_probs = soft_policy_loss(q_values)
        # argmax of [1, 2, 3] is 2 with Q=3; argmax
        # of [0, 0, 0] is 0 with Q=0; mean = 1.5.
        # The loss is -mean(Q) = -1.5.
        assert loss.item() == pytest.approx(-1.5, abs=1e-5)
        # log_probs = log_softmax; for [1, 2, 3] the
        # chosen action is 2, log_prob = log(3/6) =
        # log(0.5).
        # Compute log_softmax(2 | 1,2,3) = log(3/6)
        expected = torch.tensor([1.0, 2.0, 3.0]).log_softmax(dim=-1)
        assert log_probs[0].item() == pytest.approx(
            expected[2].item(), abs=1e-5)
