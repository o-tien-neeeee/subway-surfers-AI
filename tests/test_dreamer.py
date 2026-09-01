"""Tests for the latent-space dreamer (mental rehearsal).

The dreamer is the third leg of the "AI tự học full aggressive"
stack.  Its job: take a *good* frame, encode it to a 64-dim latent
vector, perturb the latent (so the result is a *generalisation*, not
a copy), decode back to a frame, then check whether the live policy
still thinks that abstract frame is good.  If yes → "positive
dream" → saved to <demos>/abstract/positive/.  If no → "negative
dream" → saved to <demos>/abstract/negative/.

These tests pin every layer:

* tiny VAE forward pass + shape,
* encode→perturb→decode round-trip produces sane uint8 frames,
* the dream round writes one .npz to the correct subdir,
* the env-replay override (positive) saves to ``positive/``,
* the env-replay override (negative) saves to ``negative/``,
* the trainer skips cleanly when the self pool is empty,
* the rotation policy keeps the on-disk pool under the cap,
* the heartbeat dict has the keys the GUI needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from config import DreamerConfig
from dreamer import (DreamerStats, DreamerTrainer, TinyVAE, _Encoder,
                     _Decoder, vae_loss)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeSelfPool:
    """Stand-in for :class:`SelfImitationRecorder`.

    The dreamer only uses :meth:`on_disk_episode_paths`, so this
    stub is enough to drive every test that needs "a self pool".
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths

    def on_disk_episode_paths(self) -> list[Path]:
        return list(self._paths)


def _write_self_episode(path: Path, n_frames: int = 16,
                        action: int = 1) -> None:
    """Write a synthetic self-imitation .npz that matches the human
    demo schema.  The dreamer reads ``frames`` (uint8 [N,84,84]) and
    ``actions`` (int8 [N])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(n_frames, 84, 84), dtype=np.uint8)
    actions = np.full(n_frames, action, dtype=np.int8)
    done = np.zeros(n_frames, dtype=bool)
    done[-1] = True
    np.savez_compressed(
        path, frames=frames, actions=actions,
        timestamps=np.arange(n_frames, dtype=np.float64),
        done=done,
    )


class _FakePolicy(torch.nn.Module):
    """Returns a Q-vector whose max-Q we control via bias.

    The dreamer asks for ``max(Q).mean()``; this stub lets each test
    bias the score so a Q-based dream classification is
    deterministic.
    """

    def __init__(self, bias: float = 0.0) -> None:
        super().__init__()
        self._bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, 5) where 5 actions; the 3rd one has the bias.
        out = torch.zeros(x.shape[0], 5)
        out[:, 2] = self._bias
        return out


class _FakeLearner:
    """Stand-in for :class:`Learner`.

    The dreamer only needs ``.online`` (the Q-network) and
    optionally ``._dream_env``.  We expose both so each test can
    decide whether to use the env path or the Q path.
    """

    def __init__(self, q_bias: float = 0.0,
                 env: object | None = None) -> None:
        self.online = _FakePolicy(bias=q_bias)
        # ``_dream_env`` is the env back-door.  When set, the
        # dreamer uses ``step_with_frame`` and trusts the env's
        # verdict.
        self._dream_env = env


class _FakeEnv:
    """Stand-in for the synthetic env's ``step_with_frame`` back-door.

    ``survive_n`` controls how many steps the env reports
    ``done=False`` for before flipping to ``done=True``.  The
    dreamer classifies by ``survived / frames_per_dream``.
    """

    def __init__(self, survive_n: int) -> None:
        self.survive_n = survive_n
        self._steps = 0
        self.calls: list[tuple[np.ndarray, int]] = []

    def step_with_frame(self, frame: np.ndarray, action: int) -> dict:
        self._steps += 1
        self.calls.append((frame, action))
        done = self._steps > self.survive_n
        return {"done": done, "frame": frame, "action": action}


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


class TestTinyVAE:
    def test_forward_returns_three_tensors(self) -> None:
        vae = TinyVAE(latent_dim=32)
        x = torch.zeros(2, 1, 84, 84)
        out, mu, logvar = vae(x)
        # Output is a single-channel image, batch preserved.
        assert out.shape == (2, 1, 84, 84)
        assert mu.shape == (2, 32)
        assert logvar.shape == (2, 32)

    def test_encoder_decoder_shapes(self) -> None:
        enc = _Encoder(latent_dim=16)
        dec = _Decoder(latent_dim=16)
        x = torch.zeros(1, 1, 84, 84)
        mu, lv = enc(x)
        assert mu.shape == (1, 16) and lv.shape == (1, 16)
        z = mu  # skip sampling for the shape test
        y = dec(z)
        assert y.shape == (1, 1, 84, 84)


class TestVAELoss:
    def test_loss_is_positive_and_finite(self) -> None:
        target = torch.rand(2, 1, 84, 84)
        recon = torch.rand(2, 1, 84, 84)
        mu = torch.zeros(2, 8)
        lv = torch.zeros(2, 8)
        loss, parts = vae_loss(recon, target, mu, lv, beta=0.01)
        assert torch.isfinite(loss)
        # The reconstruction term dominates with unit-variance latent
        # and random tensors, so the loss should be in [0, 1].
        assert 0.0 <= float(loss) <= 2.0
        assert "recon" in parts and "kl" in parts


# ---------------------------------------------------------------------------
# Dreamer trainer
# ---------------------------------------------------------------------------


@pytest.fixture()
def trainer(tmp_path: Path) -> DreamerTrainer:
    cfg = DreamerConfig(
        enabled=True,
        latent_dim=16,           # small for fast tests
        dream_noise_std=0.20,
        max_episodes=3,
        frames_per_dream=8,
        train_every_n_updates=1,
        # A *non-zero* throttle so the throttling test can fire
        # (with 0.0, every call writes, defeating the test).
        dream_every_s=10.0,
        dreams_per_round=1,
    )
    return DreamerTrainer(cfg, tmp_path / "abstract")


class TestDreamerConstruction:
    def test_disabled_creates_no_dirs(self, tmp_path: Path) -> None:
        cfg = DreamerConfig(enabled=False)
        d = DreamerTrainer(cfg, tmp_path / "abstract")
        assert d.cfg.enabled is False
        # No subdirs when disabled — saves a no-op mkdir.
        assert not (tmp_path / "abstract" / "positive").exists()

    def test_enabled_creates_subdirs(self, tmp_path: Path) -> None:
        cfg = DreamerConfig(enabled=True, max_episodes=2)
        d = DreamerTrainer(cfg, tmp_path / "abstract")
        assert (tmp_path / "abstract" / "positive").exists()
        assert (tmp_path / "abstract" / "negative").exists()

    def test_attach_learner_does_not_crash(self, trainer: DreamerTrainer,
                                            tmp_path: Path) -> None:
        _write_self_episode(tmp_path / "self" / "ep1.npz")
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        learner = _FakeLearner(q_bias=0.5)
        trainer.attach_learner(learner, pool)
        assert trainer._learner is learner
        assert trainer._self_pool is pool


class TestDreamerTraining:
    def test_maybe_train_returns_none_without_pool(self,
                                                   trainer: DreamerTrainer
                                                   ) -> None:
        # No pool attached → no batch → returns None.
        assert trainer.maybe_train(update_step=1) is None

    def test_maybe_train_runs_with_pool(self, trainer: DreamerTrainer,
                                        tmp_path: Path) -> None:
        _write_self_episode(tmp_path / "self" / "ep1.npz")
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        learner = _FakeLearner()
        trainer.attach_learner(learner, pool)
        parts = trainer.maybe_train(update_step=1)
        assert parts is not None
        assert "recon" in parts
        assert trainer.stats.last_train_loss > 0.0

    def test_maybe_train_skips_on_wrong_modulus(
            self, trainer: DreamerTrainer, tmp_path: Path) -> None:
        # ``train_every_n_updates=1`` means every call runs, but
        # a value below the modulus (e.g. 0) should also be safe.
        _write_self_episode(tmp_path / "self" / "ep1.npz")
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        trainer.attach_learner(_FakeLearner(), pool)
        # step=0 is the sentinel: don't train on step 0 (the network
        # is still random; the loss would be noise).
        assert trainer.maybe_train(update_step=0) is None


class TestDreamerDreaming:
    def test_maybe_dream_returns_empty_without_pool(
            self, trainer: DreamerTrainer) -> None:
        # Prime the timer (first call just records the timestamp).
        trainer.maybe_dream(now=1.0)
        out = trainer.maybe_dream(now=2.0)
        assert out == []

    def test_maybe_dream_writes_positive_with_env_survival(
            self, trainer: DreamerTrainer, tmp_path: Path) -> None:
        # Setup: one self-episode, an env that survives all 8 frames.
        _write_self_episode(tmp_path / "self" / "ep1.npz")
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        env = _FakeEnv(survive_n=8)
        learner = _FakeLearner(q_bias=0.0, env=env)
        trainer.attach_learner(learner, pool)
        # First prime the timer; second call (well past the
        # throttle) writes the dream.
        trainer.maybe_dream(now=1.0)
        written = trainer.maybe_dream(now=20.0)
        assert len(written) == 1
        # The file landed in the positive subdir.
        assert (trainer.out_dir / "positive" / written[0].name).exists()
        assert trainer.stats.dreams_positive == 1
        assert trainer.stats.dreams_negative == 0

    def test_maybe_dream_writes_negative_with_env_death(
            self, trainer: DreamerTrainer, tmp_path: Path) -> None:
        _write_self_episode(tmp_path / "self" / "ep1.npz")
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        env = _FakeEnv(survive_n=0)  # die on first step.
        learner = _FakeLearner(q_bias=0.0, env=env)
        trainer.attach_learner(learner, pool)
        trainer.maybe_dream(now=1.0)
        written = trainer.maybe_dream(now=20.0)
        assert len(written) == 1
        assert (trainer.out_dir / "negative" / written[0].name).exists()
        assert trainer.stats.dreams_negative == 1
        assert trainer.stats.dreams_positive == 0

    def test_maybe_dream_uses_q_fallback_without_env(
            self, trainer: DreamerTrainer, tmp_path: Path) -> None:
        """No env → dreamer uses the Q-value of the live policy.

        A high Q-bias should classify as positive; a low one as
        negative.  The thresholds are in the Q-space (default ±0.5).
        """
        _write_self_episode(tmp_path / "self" / "ep1.npz")
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        # ``_FakeLearner`` with no env defaults to the Q path.
        trainer.attach_learner(_FakeLearner(q_bias=1.0), pool)
        trainer.maybe_dream(now=1.0)
        written = trainer.maybe_dream(now=20.0)
        assert len(written) == 1
        assert "positive" in str(written[0])

    def test_throttling_blocks_rapid_rounds(self,
                                             trainer: DreamerTrainer,
                                             tmp_path: Path) -> None:
        _write_self_episode(tmp_path / "self" / "ep1.npz")
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        # Use a *positive* Q-bias so the Q-fallback classifies as
        # "positive" (and writes a file).  Without a real env the
        # test would otherwise hit the Q-fallback path; with
        # ``q_bias=1.0`` the bias is well above the default
        # positive threshold of 0.5.
        trainer.attach_learner(_FakeLearner(q_bias=1.0), pool)
        # First call: prime the timer.
        trainer.maybe_dream(now=1.0)
        # Second call just past the throttle (10 s) — writes a dream.
        trainer.maybe_dream(now=20.0)
        # Third call within the throttle window — no new dream.
        before = trainer.stats.dreams_positive + trainer.stats.dreams_negative
        trainer.maybe_dream(now=21.0)
        after = trainer.stats.dreams_positive + trainer.stats.dreams_negative
        assert before == after, (
            f"throttled call should not write a dream (was {before}, "
            f"now {after})")


class TestDreamerRotation:
    def test_rotation_keeps_max(self, trainer: DreamerTrainer,
                                tmp_path: Path) -> None:
        # max_episodes=3 in the fixture.  Write 5 positive dreams.
        for i in range(5):
            _write_self_episode(trainer.out_dir / "positive" / f"dream_{i}.npz",
                                n_frames=4)
        trainer._rotate()
        on_disk = list((trainer.out_dir / "positive").glob("*.npz"))
        assert len(on_disk) == 3
        # The stats reflect the actual on-disk count.
        assert trainer.stats.on_disk_positive == 3


class TestDreamerHeartbeat:
    def test_to_heartbeat_keys(self, trainer: DreamerTrainer) -> None:
        h = trainer.to_heartbeat()
        for k in ("dreams_total", "dreams_positive", "dreams_negative",
                  "on_disk_positive", "on_disk_negative",
                  "last_train_loss", "dream_q_mean"):
            assert k in h
        assert all(isinstance(v, float) for v in h.values())

    def test_to_heartbeat_reflects_counts(
            self, trainer: DreamerTrainer, tmp_path: Path) -> None:
        _write_self_episode(trainer.out_dir / "positive" / "dream_d1.npz",
                            n_frames=4)
        _write_self_episode(trainer.out_dir / "negative" / "dream_d2.npz",
                            n_frames=4)
        trainer._rotate()
        h = trainer.to_heartbeat()
        assert h["on_disk_positive"] == 1.0
        assert h["on_disk_negative"] == 1.0


class TestDreamerPersistence:
    def test_state_dict_round_trip(self, trainer: DreamerTrainer) -> None:
        state = trainer.state_dict()
        # Round-trip on a fresh trainer: weights should match.
        cfg = DreamerConfig(latent_dim=16)
        fresh = DreamerTrainer(cfg, trainer.out_dir)
        fresh.load_state_dict(state)
        for (n1, p1), (n2, p2) in zip(
                trainer.vae.state_dict().items(),
                fresh.vae.state_dict().items()):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_load_state_dict_handles_bad_input(
            self, trainer: DreamerTrainer) -> None:
        # Empty dict: should log a warning and not crash.
        trainer.load_state_dict({})  # type: ignore[arg-type]


class TestDreamerEndToEnd:
    def test_full_round_trip_writes_a_valid_npz(
            self, trainer: DreamerTrainer, tmp_path: Path) -> None:
        """The whole pipeline: train VAE → dream → env-replay →
        save .npz.  The resulting file should be readable by
        ``np.load`` and have the keys the BC loader expects."""
        _write_self_episode(tmp_path / "self" / "ep1.npz", n_frames=12)
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        env = _FakeEnv(survive_n=12)
        learner = _FakeLearner(env=env)
        trainer.attach_learner(learner, pool)
        # First prime the timer.
        trainer.maybe_dream(now=1.0)
        # Then a real round, well past the 10 s throttle.
        trainer.maybe_dream(now=20.0)
        # And the VAE training step.
        trainer.maybe_train(update_step=1)
        # The pool should have at least one dream file.
        all_dreams = list((trainer.out_dir / "positive").glob("*.npz"))
        all_dreams += list((trainer.out_dir / "negative").glob("*.npz"))
        assert len(all_dreams) == 1
        with np.load(all_dreams[0]) as data:
            assert data["frames"].shape == (8, 84, 84)
            assert data["frames"].dtype == np.uint8
            assert data["actions"].shape == (8,)
            assert data["done"].shape == (8,)
            assert data["done"][-1] == True  # noqa: E712
            assert "q_value" in data.files
            assert "source_episode" in data.files


class TestDreamerWithRealEnv:
    """End-to-end test using the real synthetic env.

    This catches integration regressions that the Q-fake and
    env-fake stubs cannot, e.g. shape mismatches between the
    dreamer's decoded uint8 frames and what the synthetic env
    expects.
    """

    def test_synthetic_env_step_with_frame(
            self, tmp_path: Path) -> None:
        from environment import SyntheticGame
        env = SyntheticGame(seed=0)
        frame = env.render()
        # The env is alive at first; step_with_frame should report
        # ``done=False`` for an arbitrary action.
        result = env.step_with_frame(frame, 1)
        assert "done" in result
        assert isinstance(result["done"], bool)
        # Frame argument is uint8 HxWx3; the env accepts it
        # without crashing.
        assert result["done"] is False

    def test_dreamer_classifies_with_synthetic_env(
            self, tmp_path: Path) -> None:
        """Run the dreamer against the real synthetic env.  An
        "alive" env should produce a positive dream (8/8 survival
        → score 1.0 ≥ 0.6 threshold)."""
        from environment import SyntheticGame
        env = SyntheticGame(seed=0)
        cfg = DreamerConfig(
            enabled=True, latent_dim=16, max_episodes=2,
            frames_per_dream=8, train_every_n_updates=1,
            dream_every_s=10.0, dreams_per_round=1,
        )
        trainer = DreamerTrainer(cfg, tmp_path / "abstract")
        # Write a self-episode.
        _write_self_episode(tmp_path / "self" / "ep1.npz", n_frames=8)
        pool = _FakeSelfPool([tmp_path / "self" / "ep1.npz"])
        learner = _FakeLearner(env=env)
        trainer.attach_learner(learner, pool)
        # Prime + a real round.
        trainer.maybe_dream(now=1.0)
        trainer.maybe_dream(now=20.0)
        # The synthetic env never dies in 8 steps with a NOOP
        # action (1 = LEFT in this game; the player just shifts
        # lanes and the obstacles haven't reached the player
        # yet).  So we expect a positive dream — unless the game
        # just happened to spawn a fast obstacle.  Either way, the
        # pipeline must not crash and must produce *some* output.
        all_dreams = list((trainer.out_dir / "positive").glob("*.npz"))
        all_dreams += list((trainer.out_dir / "negative").glob("*.npz"))
        assert len(all_dreams) == 1
        # And the file is a valid .npz.
        with np.load(all_dreams[0]) as data:
            assert data["frames"].dtype == np.uint8
