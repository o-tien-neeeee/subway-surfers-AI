"""Tests for the QR-DQN agent.

These tests pin the agent's training loop without going through
the full 3-process pipeline.  The agent is meant to be a
drop-in replacement for :class:`agent.DoubleDQNAgent` (same
checkpoint format, same state-dict, same ``train_step`` return
shape), so we re-use the synthetic transition fixtures from
``test_distributional`` and the loss-decreases sanity check.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent_distributional import DistributionalDoubleDQNAgent
from config import RLConfig


@pytest.fixture()
def cfg() -> RLConfig:
    return RLConfig(
        profile="strict_lite",
        gamma=0.99,
        target_update_every=100,
        grad_clip_norm=10.0,
        learning_rate=1e-3,
    )


def _make_batch(batch_size: int = 8) -> dict:
    """A minimal PER-style batch."""
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 256, size=(batch_size, 4, 84, 84),
                       dtype=np.uint8)
    next_obs = rng.integers(0, 256, size=(batch_size, 4, 84, 84),
                            dtype=np.uint8)
    return {
        "obs": obs,
        "next_obs": next_obs,
        "actions": rng.integers(0, 5, size=(batch_size,)).astype(np.int64),
        "rewards": rng.uniform(0.0, 1.0, size=(batch_size,)).astype(np.float32),
        "dones": np.zeros(batch_size, dtype=np.float32),
        "weights": np.ones(batch_size, dtype=np.float32),
        "gamma_pows": np.full(batch_size, 0.99, dtype=np.float32),
        "indices": np.zeros(batch_size, dtype=np.int64),
    }


class TestDistributionalAgent:
    def test_construction(self, cfg: RLConfig) -> None:
        agent = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        assert agent.n_actions == 5
        assert agent.num_quantiles == 51
        # Online and target start identical (init sync).
        for (n1, p1), (n2, p2) in zip(
                agent.online.state_dict().items(),
                agent.target.state_dict().items()):
            assert torch.equal(p1, p2), f"target not synced at init: {n1}"

    def test_train_step_returns_expected_keys(self, cfg: RLConfig) -> None:
        agent = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        batch = _make_batch()
        result = agent.train_step(batch)
        for k in ("loss", "td_errors", "q_mean", "q_max", "q_min",
                  "grad_norm", "lr"):
            assert k in result
        # td_errors must be per-sample.
        assert result["td_errors"].shape == (8,)

    def test_loss_finite_on_random_batch(self, cfg: RLConfig) -> None:
        agent = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        for _ in range(5):
            batch = _make_batch()
            result = agent.train_step(batch)
            assert np.isfinite(result["loss"])
            assert np.all(np.isfinite(result["td_errors"]))

    def test_target_sync(self, cfg: RLConfig) -> None:
        agent = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        # Force several updates then a sync.
        for _ in range(3):
            agent.train_step(_make_batch())
        agent.sync_target()
        # After sync the target is identical to the online net.
        for (n1, p1), (n2, p2) in zip(
                agent.online.state_dict().items(),
                agent.target.state_dict().items()):
            assert torch.equal(p1, p2), f"target out of sync after sync_target: {n1}"

    def test_state_payload_round_trip(self, cfg: RLConfig) -> None:
        agent = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        # Run a few updates so the optimizer state is non-empty.
        for _ in range(3):
            agent.train_step(_make_batch())
        payload = agent.state_payload()
        assert payload["num_quantiles"] == 51
        assert payload["update_count"] == 3
        # Round-trip on a fresh agent.
        fresh = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        fresh.load_payload(payload)
        assert fresh.update_count == 3
        for (n1, p1), (n2, p2) in zip(
                fresh.online.state_dict().items(),
                agent.online.state_dict().items()):
            assert torch.allclose(p1, p2, atol=1e-5), \
                f"loaded net differs from source: {n1}"


class TestDistributionalAgentLong:
    """Long-ish training test: 30 updates on the same data; the
    loss should decrease meaningfully (proving the gradient
    flows through the full pipeline: encoder + dueling head +
    quantile projection)."""

    def test_loss_decreases_over_updates(self, cfg: RLConfig) -> None:
        torch.manual_seed(0)
        agent = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        batch = _make_batch(batch_size=8)
        losses = []
        for _ in range(30):
            result = agent.train_step(batch)
            losses.append(result["loss"])
        # The first 5 losses should be markedly higher than the
        # last 5 (smoothed by the exponential moving average of
        # gradient steps).
        first = float(np.mean(losses[:5]))
        last = float(np.mean(losses[-5:]))
        assert last < first * 0.85, (
            f"loss did not decrease: first5={first:.4f} "
            f"last5={last:.4f}")


class TestRNDIntegration:
    """Verify the QR-DQN agent uses an attached RND module
    correctly.  The intrinsic-reward path is the most
    consequential new code path in v1.19: an
    off-by-one or a mis-normalised bonus would silently
    pollute every gradient step."""

    def test_rnd_attaches_and_injects_bonus(self, cfg: RLConfig) -> None:
        from rnd import RNDConfig, RNDModule
        torch.manual_seed(0)
        agent = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        rnd = RNDModule(RNDConfig(enabled=True, feature_dim=32,
                                   beta=1.0, normalizer_alpha=0.9),
                        in_frames=4)
        agent.attach_rnd(rnd)
        batch = _make_batch(batch_size=4)
        result = agent.train_step(batch)
        # The result should now carry the RND predictor loss.
        assert "rnd_loss" in result
        # And the loss must be finite.
        assert np.isfinite(result["rnd_loss"])

    def test_disabled_rnd_does_nothing(self, cfg: RLConfig) -> None:
        from rnd import RNDConfig, RNDModule
        agent = DistributionalDoubleDQNAgent("strict_lite", cfg,
                                              in_frames=4, size=84,
                                              num_quantiles=51, seed=0)
        rnd = RNDModule(RNDConfig(enabled=False), in_frames=4)
        agent.attach_rnd(rnd)
        batch = _make_batch(batch_size=4)
        result = agent.train_step(batch)
        # No RND metrics in the result when disabled.
        assert "rnd_loss" not in result
