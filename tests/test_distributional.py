"""Tests for the distributional QR-DQN components.

These pin:
* Quantile grid construction.
* QuantileDuelingDQN output shape and inference reduction.
* Quantile Huber loss is finite, differentiable, and shrinks
  the predicted quantile toward the target.
* ``project_distribution`` correctly applies Bellman shift and
  the terminal mask.
* An end-to-end training step shows the loss decreases.
"""

from __future__ import annotations

import torch

from distributional import (QuantileDuelingDQN, mid_quantiles,
                             project_distribution, quantile_huber_loss)


class TestMidQuantiles:
    def test_correct_count(self) -> None:
        qs = mid_quantiles(51)
        assert qs.shape == (51,)
        assert torch.allclose(qs[0], torch.tensor(1.0 / 102.0))
        assert torch.allclose(qs[-1], torch.tensor(101.0 / 102.0))

    def test_in_unit_interval(self) -> None:
        qs = mid_quantiles(32)
        assert (qs > 0).all() and (qs < 1).all()
        # Uniform mid-quantile grid; consecutive deltas are equal.
        diffs = qs[1:] - qs[:-1]
        assert torch.allclose(diffs, diffs.new_full(diffs.shape, diffs[0]),
                              atol=1e-6)


class TestQuantileDuelingDQN:
    def test_output_shape(self) -> None:
        net = QuantileDuelingDQN("strict_lite", in_frames=4, size=84,
                                 num_quantiles=51)
        x = torch.zeros(2, 4, 84, 84)
        out = net(x)
        assert out.shape == (2, 5, 51)

    def test_q_values_shape(self) -> None:
        net = QuantileDuelingDQN("strict_lite", in_frames=4, size=84,
                                 num_quantiles=51)
        x = torch.zeros(3, 4, 84, 84)
        q = net.q_values(x)
        assert q.shape == (3, 5)

    def test_q_values_are_mean(self) -> None:
        """The Q-value used at inference is the mean of the
        quantiles, by construction.  Test this so a future
        refactor cannot quietly change the inference semantics."""
        net = QuantileDuelingDQN("strict_lite", in_frames=4, size=84,
                                 num_quantiles=51)
        # Switch to eval so the NoisyLinear layers don't
        # resample noise between the two forward passes; the
        # contract we pin is "q_values == dist.mean(dim=-1)
        # on the *same* forward pass", which is what the
        # actor/agent actually use.
        net.eval()
        x = torch.randn(2, 4, 84, 84)
        dist = net(x)
        q_mean = net.q_values(x)
        expected = dist.mean(dim=-1)
        assert torch.allclose(q_mean, expected, atol=1e-5)

    def test_deterministic(self) -> None:
        net = QuantileDuelingDQN("strict_lite", in_frames=4, size=84,
                                 num_quantiles=51)
        net.eval()
        x = torch.randn(1, 4, 84, 84)
        with torch.no_grad():
            out1 = net(x)
            out2 = net(x)
        assert torch.allclose(out1, out2, atol=1e-6)


class TestQuantileHuberLoss:
    def test_loss_is_scalar(self) -> None:
        pred = torch.zeros(2, 5, 51)
        target = torch.zeros(2, 5, 51)
        taus = mid_quantiles(51)
        loss = quantile_huber_loss(pred, target, taus)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        # Zero error = zero loss (up to numerical epsilon).
        assert float(loss) < 1e-6

    def test_loss_increases_with_error(self) -> None:
        torch.manual_seed(0)
        pred = torch.zeros(2, 5, 51)
        taus = mid_quantiles(51)
        # Small target offset
        target_small = torch.full((2, 5, 51), 0.1)
        loss_small = quantile_huber_loss(pred, target_small, taus)
        # Large target offset
        target_big = torch.full((2, 5, 51), 5.0)
        loss_big = quantile_huber_loss(pred, target_big, taus)
        assert float(loss_big) > float(loss_small) * 10

    def test_loss_grad_flows(self) -> None:
        pred = torch.zeros(2, 5, 51, requires_grad=True)
        target = torch.full((2, 5, 51), 1.0)
        taus = mid_quantiles(51)
        loss = quantile_huber_loss(pred, target, taus)
        loss.backward()
        assert pred.grad is not None
        assert (pred.grad.abs() > 0).all()


class TestProjectDistribution:
    def test_no_terminal_applies_gamma(self) -> None:
        B, A, N = 2, 5, 51
        next_dist = torch.ones(B, A, N)
        rewards = torch.full((B,), 0.5)
        dones = torch.zeros(B)
        target = project_distribution(next_dist, rewards, dones,
                                       num_quantiles=N, gamma=0.99)
        # target = 0.5 + 0.99 * 1.0 = 1.49
        assert torch.allclose(target, torch.full_like(target, 1.49),
                              atol=1e-5)

    def test_terminal_zeros_future(self) -> None:
        B, A, N = 2, 5, 51
        next_dist = torch.ones(B, A, N)
        rewards = torch.full((B,), 0.5)
        dones = torch.ones(B)
        target = project_distribution(next_dist, rewards, dones,
                                       num_quantiles=N, gamma=0.99)
        # target = 0.5 + 0.99 * (1 - 1) * 1.0 = 0.5
        assert torch.allclose(target, torch.full_like(target, 0.5),
                              atol=1e-5)

    def test_target_has_no_grad(self) -> None:
        B, A, N = 1, 5, 32
        next_dist = torch.ones(B, A, N, requires_grad=True)
        rewards = torch.zeros(B)
        dones = torch.zeros(B)
        target = project_distribution(next_dist, rewards, dones,
                                       num_quantiles=N, gamma=0.99)
        # The projection is a stop-gradient: changing the next
        # distribution must NOT change the target.
        with torch.no_grad():
            next_dist.fill_(10.0)
        target_after = project_distribution(next_dist, rewards, dones,
                                              num_quantiles=N, gamma=0.99)
        # Original target was 9.9 (0 + 0.99 * 10); after fill it
        # would be 9.9 too — but the function returned a detached
        # tensor the first time.  So we can verify .requires_grad
        # is False.
        assert not target.requires_grad


class TestQuantileTrainingStep:
    """End-to-end: a single gradient step on synthetic data
    should reduce the loss."""

    def test_loss_decreases(self) -> None:
        torch.manual_seed(0)
        net = QuantileDuelingDQN("strict_lite", in_frames=4, size=84,
                                 num_quantiles=51)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        taus = mid_quantiles(51)
        # Use *fixed* synthetic data so the network can actually
        # fit something; the per-step target depends on the
        # evolving next_dist, so a random next_obs and a random
        # init will hover at the loss level of "predict mean".
        torch.manual_seed(42)
        obs = torch.randn(8, 4, 84, 84)
        next_obs = torch.randn(8, 4, 84, 84)
        rewards = torch.full((8,), 0.5)
        dones = torch.zeros(8)
        # Initial loss — should be HIGH because the network is
        # untrained.  We compute the target using a *frozen* copy
        # of the next-state distribution so the target does not
        # change as the network trains.
        with torch.no_grad():
            next_dist_fixed = net(next_obs).detach()
        target = project_distribution(
            next_dist_fixed, rewards, dones, num_quantiles=51, gamma=0.99)
        pred = net(obs)
        loss0 = float(quantile_huber_loss(pred, target, taus).detach())
        # 100 gradient steps to actually fit the target.
        for _ in range(100):
            opt.zero_grad()
            pred = net(obs)
            loss = quantile_huber_loss(pred, target, taus)
            loss.backward()
            opt.step()
        # Final loss
        with torch.no_grad():
            pred = net(obs)
        loss1 = float(quantile_huber_loss(pred, target, taus))
        # The loss should drop meaningfully (at least 50 %).
        assert loss1 < loss0 * 0.5, (
            f"loss did not decrease: {loss0:.4f} -> {loss1:.4f}")
