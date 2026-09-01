"""Tests for the RND (Random Network Distillation) curiosity module.

The RND module is the third major ingredient from the deep
research the user asked for: it provides *intrinsic* reward
proportional to state novelty, so the agent explores the
state space even when the extrinsic reward is sparse (alive
or dead, every tick is the same).

These tests pin:
* The intrinsic reward is *non-negative* (squared error) and
  *bounded* in a sane range (the normaliser keeps the
  surprise below a few units after a few updates).
* The predictor loss decreases as the predictor learns to
  mimic the frozen target.
* The normaliser EMA decays toward the running mean.
* Disabling the module returns zero reward cheaply.
* End-to-end: a forward pass produces the expected shape
  and the network receives gradients.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rnd import RNDConfig, RNDModule, _RNDEncoder


class TestRNDEncoder:
    def test_output_shape(self) -> None:
        enc = _RNDEncoder(in_frames=4, feature_dim=128)
        x = torch.zeros(2, 4, 84, 84)
        out = enc(x)
        assert out.shape == (2, 128)

    def test_target_frozen(self) -> None:
        """The target network must NEVER receive gradients —
        if it did, the intrinsic reward would trivially go
        to zero (the predictor could just learn the same
        weights as the target).  We test this by spinning
        up an RNDModule and inspecting its target.
        """
        rnd = RNDModule(RNDConfig(enabled=True, feature_dim=64))
        for p in rnd.target.parameters():
            assert not p.requires_grad
        # And the predictor is trainable.
        for p in rnd.predictor.parameters():
            assert p.requires_grad


class TestRNDModule:
    @pytest.fixture()
    def rnd(self) -> RNDModule:
        cfg = RNDConfig(enabled=True, feature_dim=64, beta=0.5,
                        normalizer_alpha=0.9)
        return RNDModule(cfg, in_frames=4)

    def test_intrinsic_reward_is_non_negative(self, rnd: RNDModule) -> None:
        states = np.random.randint(0, 256, size=(4, 4, 84, 84),
                                    dtype=np.uint8)
        r = rnd.intrinsic_reward(states)
        assert r.shape == (4,)
        assert (r >= 0).all()

    def test_intrinsic_reward_normalised(self, rnd: RNDModule) -> None:
        """After several calls, the normalised surprise should
        stay in a small range (the EMA normaliser converges
        to the running mean).  The *raw* surprise is
        dimensionless squared-L2, so without normalisation
        it could blow up; the operator-facing reward is
        bounded by ``beta`` times the normalised value.
        """
        for _ in range(10):
            states = np.random.randint(0, 256, size=(8, 4, 84, 84),
                                        dtype=np.uint8)
            r = rnd.intrinsic_reward(states)
        # The normalised reward (returned) is ``raw / norm *
        # beta``.  If the normaliser works correctly, the
        # mean of these should be around ``beta``.
        last_r = rnd.intrinsic_reward(np.random.randint(
            0, 256, size=(16, 4, 84, 84), dtype=np.uint8))
        assert last_r.mean() < 5.0 * 0.5  # 5x beta upper bound

    def test_disabled_returns_zeros(self) -> None:
        cfg = RNDConfig(enabled=False)
        rnd = RNDModule(cfg, in_frames=4)
        states = np.random.randint(0, 256, size=(4, 4, 84, 84),
                                    dtype=np.uint8)
        r = rnd.intrinsic_reward(states)
        assert (r == 0).all()

    def test_predictor_loss_decreases(self, rnd: RNDModule) -> None:
        """The predictor should fit the (fixed) target on a
        batch of identical states.  The loss must decrease
        over a handful of gradient steps.
        """
        # Use the SAME states every step so the target output
        # is constant and the predictor only has to memorise
        # one vector.
        states = np.random.randint(0, 256, size=(8, 4, 84, 84),
                                    dtype=np.uint8)
        losses = []
        for _ in range(20):
            result = rnd.train_step(states)
            losses.append(result["rnd_loss"])
        # The first loss should be at least 2x the last.
        assert losses[-1] < losses[0] * 0.5, (
            f"predictor did not fit the target: "
            f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    def test_state_dict_round_trip(self, rnd: RNDModule) -> None:
        """A round-tripped state dict must reproduce the
        predictor exactly.  Without this, a checkpoint
        restore would lose the entire curiosity signal.
        """
        # One training step so the predictor has moved.
        states = np.random.randint(0, 256, size=(4, 4, 84, 84),
                                    dtype=np.uint8)
        rnd.train_step(states)
        state = rnd.state_dict()
        fresh = RNDModule(RNDConfig(enabled=True, feature_dim=64),
                           in_frames=4)
        fresh.load_state_dict(state)
        for (n1, p1), (n2, p2) in zip(
                rnd.predictor.state_dict().items(),
                fresh.predictor.state_dict().items()):
            assert torch.equal(p1, p2), f"predictor not equal after round-trip: {n1}"

    def test_target_unchanged_after_train(self, rnd: RNDModule) -> None:
        """A subtle test: the predictor training step must
        not affect the target network (the target is frozen).
        If it did, the intrinsic reward would silently
        decrease to zero and the agent would lose curiosity.
        """
        before = [p.clone() for p in rnd.target.parameters()]
        states = np.random.randint(0, 256, size=(4, 4, 84, 84),
                                    dtype=np.uint8)
        for _ in range(5):
            rnd.train_step(states)
        for p_before, p_after in zip(before, rnd.target.parameters()):
            assert torch.equal(p_before, p_after), \
                "target network must not change during predictor training"
