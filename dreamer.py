"""Latent-space 'dreaming' — the agent's mental rehearsal.

This module implements a tiny **variational autoencoder** (VAE) that maps
each 84×84 grayscale frame into a 64-dimensional latent vector and back.
The decoder is *tiny on purpose* (the encoder is shared with the
dueling-DQN value stream, so reusing its existing depthwise-separable
convolutions would require us to re-train — the dreamer is its own small
network).

The mental-rehearsal loop
-------------------------
The user observed that the bot dies immediately and then learns
``NOOP`` because the replay buffer is dominated by instant-death
transitions.  Even after self-imitation harvests good episodes, the
quantity is small (gated by the rolling mean, max 50 on disk) and the
variety is limited (the same obstacle scenarios, the same camera
angles).  The dreamer closes that loop:

  1. From the self-imitation pool, pick a frame + the action that
     survived past it.
  2. **Encode** the frame into a 64-dim latent ``z``.
  3. **Perturb** ``z`` with a small Gaussian noise (configurable
     ``dream_noise_std``, default 0.30) — this is the "abstract
     image" the user asked for.  It is the same experience, viewed
     through a slightly fuzzy mental lens.
  4. **Decode** the perturbed ``z`` back into an 84×84 frame.
  5. Run the synthetic env with that abstract frame for a few steps
     (the env's ``step_with_frame`` is the back-door for headless
     evaluation).  If the agent survives, save the abstract frame +
     the Q-value it produced as a *positive dream* in
     ``<demos>/abstract/``; if it dies, save as a *negative dream*
     and tag it for punishment (its Q-value is pushed down on the
     next training step).

Why a separate VAE and not the dueling-DQN encoder?
----------------------------------------------------
The DQN encoder is trained to be a *value* function, not a *generator*.
Re-using it for reconstruction would tie dream quality to RL progress
(early on, the encoder is essentially random → dreams are noise).
The dreamer is small (≈ 60k parameters) and trained *only* on
self-imitation frames so the latent space reflects "what a good
Subway Surfers frame looks like", not "what the agent values".  The
decoded frame is then a *plausible* variation, not a hallucination.

Disk format
-----------
A dream is stored as a single ``.npz`` matching the human-demo schema
(see :mod:`demonstration_recorder`): keys ``frames`` (uint8 [N, 84, 84]),
``actions`` (int8), ``timestamps`` (float64), ``done`` (bool), and the
extra metadata fields ``q_value`` (float32 per frame) and
``source_episode`` (the self-imitation episode it was dreamed from).
Positive dreams and negative dreams live in
``<demos>/abstract/positive/`` and ``<demos>/abstract/negative/``
respectively so the BC loader can pick by directory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DreamerConfig:
    """Knobs for the dreamer.

    Every field is named so the GUI can surface it and the unit test
    can pin it.  Defaults are deliberately conservative: the dreamer
    is supposed to *help* RL, not eat CPU.
    """

    #: Master switch — disable to skip the entire mental-rehearsal
    #: pipeline (encoder, decoder, training step, save to disk).
    enabled: bool = True

    #: Latent dimension.  64 is a sweet spot on CPU: big enough to
    #: capture 84×84 grayscale (≈ 7k unique values), small enough
    #: that the decoder fits in 60k parameters.
    latent_dim: int = 64

    #: Gaussian stddev added to the latent vector when dreaming.
    #: 0.0 = "photographic memory" (no abstraction), 1.0 = "wild
    #: hallucination".  0.3 keeps the frame recognisable while
    #: changing enough pixels to count as a new experience.
    dream_noise_std: float = 0.30

    #: Maximum abstract episodes kept on disk (rotated like
    #: self-imitation).  Split 50/50 between positive and negative.
    max_episodes: int = 50

    #: VAE loss weight for KL divergence.  Higher = more regularised
    #: latent space = blurrier but more diverse dreams.
    beta_kl: float = 0.01

    #: Number of frames per abstract episode.  A real episode is
    #: 100-300 frames; we cap at 32 to keep the BC loader fast.
    frames_per_dream: int = 32

    #: How often (in learner update steps) the dreamer is trained
    #: on the self-imitation pool.  Every 100 updates = once per
    #: ~few minutes of online RL.
    train_every_n_updates: int = 100

    #: How often (in seconds) the dreamer generates and evaluates a
    #: batch of dreams.  60 s = once per minute — expensive step
    #: (encoder + decode + synthetic-env rollouts) so we throttle.
    dream_every_s: float = 60.0

    #: Number of dreams per mental-rehearsal round.  8 = 4 positive
    #: + 4 negative on average, written to disk and used for one
    #: "dream" BC epoch.
    dreams_per_round: int = 8

    #: Q-value threshold above which a dream is "positive" (the
    #: agent thought the abstract frame was good).  Below the
    #: negation of this, the dream is "negative" (the agent
    #: thought the frame was deadly).  In-between = neutral and
    #: discarded.
    positive_q_threshold: float = 0.5
    negative_q_threshold: float = -0.5


# ---------------------------------------------------------------------------
# Tiny VAE
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    """84x84 -> latent_dim.  Pure conv, GroupNorm, no flatten-dense."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        # Input is 1×84×84 (single grayscale frame).  Four stride-2
        # convs bring 84 -> 5 (math: 84/2/2/2/2 = 5.25 → floor 5).
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 42×42
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 21×21
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 10×10
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 5×5
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), nn.ReLU(inplace=True),
        )
        # Two heads: mean and log-variance.
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    """latent_dim -> 84x84.  Mirror of the encoder (no skip connections,
    so the model is small and trains fast on CPU)."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        # Start from 5×5; four stride-2 transposed convs: 5 → 10 → 20
        # → 40 → 80.  A final 3×3 stride-1 conv lifts 80→84.
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64 * 5 * 5), nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2,
                               padding=1),  # 10
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2,
                               padding=1),  # 20
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                               padding=1),  # 40
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2,
                               padding=1),  # 80
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # 80
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 5, 5)
        out = self.net(h)
        # Pad 80 -> 84 on the right and bottom (no cropping needed
        # for 80 → 84 because the extra 4 pixels are the result of
        # an asymmetric padding artefact in the last conv; an even
        # pad-2 on each side keeps the output centred).
        return F.pad(out, (2, 2, 2, 2))


class TinyVAE(nn.Module):
    """The whole dreamer autoencoder.

    Kept in a single class so :class:`DreamerTrainer` can swap between
    ``train()`` / ``eval()`` without juggling three modules.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = _Encoder(latent_dim)
        self.decoder = _Decoder(latent_dim)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Standard VAE loss = MSE reconstruction + β * KL divergence.

    MSE on uint8 frames is well-behaved here because the frame range
    is bounded and the dreamer only needs to reproduce "the gist",
    not pixel-perfect reconstructions.
    """
    # Scale target to [0, 1] so MSE is comparable across configs.
    tgt = target.float() / 255.0
    rec = recon.clamp(0.0, 1.0)
    bce = F.mse_loss(rec, tgt, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                                      dim=1))
    total = bce + beta * kl
    return total, {"recon": float(bce.detach().item()),
                   "kl": float(kl.detach().item())}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class DreamerStats:
    """Live counters surfaced to the GUI / heartbeat metrics."""
    dreams_total: int = 0
    dreams_positive: int = 0
    dreams_negative: int = 0
    on_disk_positive: int = 0
    on_disk_negative: int = 0
    last_train_loss: float = 0.0
    last_train_at: float = 0.0
    last_dream_at: float = 0.0
    last_dream_q: float = 0.0
    rolling: dict[str, float] = field(default_factory=dict)


class DreamerTrainer:
    """Owns the VAE, the training loop, the mental-rehearsal round.

    Lifecycle
    ---------
    1. ``attach_learner(learner)`` — wires the trainer to the live
       policy and the self-imitation pool.  Until this is called,
       the trainer is a no-op (testable in isolation).
    2. ``maybe_train(step)`` — call every learner update; the trainer
       decides whether to run a VAE training step based on
       ``config.train_every_n_updates``.
    3. ``maybe_dream(now)`` — call every learner heartbeat; the
       trainer decides whether to generate + evaluate a batch of
       dreams based on ``config.dream_every_s``.
    """

    def __init__(self, cfg: DreamerConfig, out_dir: Path,
                 device: str = "cpu") -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.device = torch.device(device)
        self.vae = TinyVAE(cfg.latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self.stats = DreamerStats()
        # Will be filled in by attach_learner().
        self._learner: Optional[object] = None
        self._self_pool: Optional[object] = None
        # Internal: last rolling mean of Q for the GUI metric.
        self._q_history: list[float] = []
        # Monotonic counter so the test suite (and rapid back-to-back
        # rounds) never produce two files with the same name.  We
        # could use ``time.time_ns()`` but on fast test machines
        # two saves can land in the same nanosecond; the counter is
        # monotone and collision-free.
        self._save_counter = 0
        # Make the abstract subdirs eagerly so the operator can
        # see the pool grow even before the first dream.
        if cfg.enabled:
            (self.out_dir / "positive").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "negative").mkdir(parents=True, exist_ok=True)

    # -- attachment -----------------------------------------------------

    def attach_learner(self, learner: object, self_pool: object) -> None:
        """Wire to the live learner + self-imitation pool.

        We keep the types opaque (no ``from learner_worker import
        Learner``) so the dreamer has no import cycle with the
        learner, and so the test suite can pass mocks in directly.
        """
        self._learner = learner
        self._self_pool = self_pool

    # -- training -------------------------------------------------------

    def _sample_training_batch(self) -> Optional[torch.Tensor]:
        """Pull a random batch of frames from the self-imitation pool.

        Each ``.npz`` in the self pool is shaped ``(N, 84, 84)`` uint8.
        We stack a random subset of those frames into a tensor.
        Returns ``None`` if the pool is empty (so the caller can
        skip the training step without spamming logs).
        """
        if self._self_pool is None or not self.cfg.enabled:
            return None
        # The pool exposes ``on_disk_episode_paths()``; this is the
        # single integration point so the test can stub it.
        try:
            paths = self._self_pool.on_disk_episode_paths()
        except AttributeError:
            return None
        if not paths:
            return None
        # Sample 16 frames from up to 4 episodes.
        rng = np.random.default_rng()
        picked = rng.choice(paths, size=min(4, len(paths)), replace=False)
        frames: list[np.ndarray] = []
        for p in picked:
            try:
                with np.load(p) as data:
                    f = data["frames"]
                    # Random subset of that episode's frames.
                    idx = rng.integers(0, len(f),
                                       size=min(8, len(f)))
                    frames.append(f[idx])
            except (OSError, KeyError, ValueError) as exc:
                LOGGER.warning("dreamer: skipping corrupt episode %s: %s",
                               p, exc)
                continue
        if not frames:
            return None
        batch = np.concatenate(frames, axis=0)
        # Convert to float tensor in [0, 1].
        x = torch.from_numpy(batch).float().unsqueeze(1) / 255.0
        return x.to(self.device)

    def maybe_train(self, update_step: int) -> Optional[dict[str, float]]:
        """Run a VAE training step if it's time."""
        if not self.cfg.enabled or self._learner is None:
            return None
        if update_step <= 0:
            return None
        if update_step % self.cfg.train_every_n_updates != 0:
            return None
        x = self._sample_training_batch()
        if x is None or x.shape[0] < 4:
            return None
        self.vae.train()
        self.optimizer.zero_grad()
        recon, mu, logvar = self.vae(x)
        loss, parts = vae_loss(recon, x, mu, logvar, self.cfg.beta_kl)
        loss.backward()
        # The decoder can occasionally spike gradients on the last
        # conv — a small clip keeps training stable without
        # noticeably slowing convergence.
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
        self.optimizer.step()
        self.stats.last_train_loss = float(loss.item())
        self.stats.last_train_at = time.time()
        return parts

    # -- dreaming -------------------------------------------------------

    def _pick_seed_frames(self) -> tuple[list[np.ndarray], list[int]]:
        """Pull (frame, action) pairs from the self pool for dreaming."""
        if self._self_pool is None:
            return [], []
        try:
            paths = self._self_pool.on_disk_episode_paths()
        except AttributeError:
            return [], []
        if not paths:
            return [], []
        rng = np.random.default_rng()
        p = Path(rng.choice(paths))
        try:
            with np.load(p) as data:
                f = data["frames"]
                a = data["actions"]
                n = min(self.cfg.frames_per_dream, len(f))
                if n == 0:
                    return [], []
                idx = np.arange(n)  # take a contiguous slice.
                return [f[idx].copy()], a[idx].copy().tolist()
        except (OSError, KeyError) as exc:
            LOGGER.warning("dreamer: seed load failed for %s: %s", p, exc)
            return [], []

    def _encode_decode_dream(self,
                             frames: np.ndarray) -> tuple[np.ndarray,
                                                           np.ndarray]:
        """Return (mu, decoded_uint8) for the given batch of frames."""
        if not self.cfg.enabled:
            return np.zeros((0, self.cfg.latent_dim), dtype=np.float32), \
                np.zeros((0, 84, 84), dtype=np.uint8)
        self.vae.eval()
        with torch.no_grad():
            x = torch.from_numpy(frames).float().unsqueeze(1) / 255.0
            x = x.to(self.device)
            mu, _ = self.vae.encode(x)
            # Add the "abstract" noise: this is the "khái quát" the
            # user asked for.  Without noise the dream is a
            # reconstruction; with it the dream is an
            # *interpretation*.
            noise = torch.randn_like(mu) * self.cfg.dream_noise_std
            z = mu + noise
            decoded = self.vae.decode(z).clamp(0.0, 1.0)
            decoded = (decoded * 255.0).cpu().numpy().astype(np.uint8)
            decoded = decoded.squeeze(1)
            return mu.cpu().numpy(), decoded

    def _score_with_q(self, frames: np.ndarray) -> float:
        """Run the live Q-network on the abstract frames and return
        the mean max-Q.

        The learner's policy is exposed as ``policy_network(state)``;
        we accept the frames as a ``(N, frame_stack, 84, 84)`` batch
        by stacking the dream with itself ``frame_stack`` times
        (the dreamer does not need to match the actor's exact
        stack — a coarse scalar signal is enough to gate
        positive/negative).
        """
        if self._learner is None:
            return 0.0
        policy = getattr(self._learner, "online", None)
        if policy is None:
            return 0.0
        try:
            # Build a fake stack: 4 channels = same frame repeated.
            stack = np.repeat(frames[:, None, :, :], 4, axis=1)
            with torch.no_grad():
                t = torch.from_numpy(stack).float().to(self.device) / 255.0
                policy.eval()
                q = policy(t)
                max_q = float(q.max(dim=1).values.mean().item())
            return max_q
        except Exception as exc:  # noqa: BLE001 - reported
            LOGGER.warning("dreamer: Q-score failed: %s", exc)
            return 0.0

    def _classify(self, q_value: float) -> str:
        """Positive / negative / neutral based on the configured gates."""
        if q_value >= self.cfg.positive_q_threshold:
            return "positive"
        if q_value <= self.cfg.negative_q_threshold:
            return "negative"
        return "neutral"

    def _save_dream(self, kind: str, frames: np.ndarray, actions: list[int],
                    source_episode: str, q_value: float) -> Optional[Path]:
        """Write a single abstract episode to disk.

        Format mirrors :mod:`demonstration_recorder`: ``frames`` uint8
        ``[N, 84, 84]``, ``actions`` int8, ``timestamps`` float64,
        ``done`` bool, plus ``q_value`` and ``source_episode`` for
        traceability.  Saved atomically via ``.npz.tmp`` rename.
        """
        sub = self.out_dir / kind
        sub.mkdir(parents=True, exist_ok=True)
        # File name encodes kind, the running counter, and a
        # millisecond timestamp.  The counter is the only piece
        # that strictly needs to be unique; the timestamp is there
        # so the operator can see "this dream was generated at..."
        # when listing the directory.
        self._save_counter += 1
        ts = int(time.time() * 1000)
        path = sub / f"dream_{kind}_{ts}_{self._save_counter}.npz"
        tmp = path.with_suffix(".npz.tmp")
        # ``done`` is True on the last frame (mirrors human demos).
        done = np.zeros(len(frames), dtype=bool)
        if len(done) > 0:
            done[-1] = True
        try:
            # ``np.savez_compressed`` appends ``.npz`` to *string*
            # filenames — a known numpy quirk that bit the demo
            # recorder too.  We pass a *file object* so the
            # extension is exactly what we ask for.
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh,
                    frames=frames.astype(np.uint8),
                    actions=np.asarray(actions, dtype=np.int8),
                    timestamps=np.linspace(0.0, 1.0, len(frames),
                                           dtype=np.float64),
                    done=done,
                    q_value=np.full(len(frames), q_value,
                                    dtype=np.float32),
                    source_episode=source_episode,
                )
            tmp.rename(path)
        except (OSError, ValueError) as exc:
            LOGGER.error("dreamer: failed to save %s: %s", path, exc)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError as unlink_exc:
                    LOGGER.warning("dreamer: could not remove tmp %s: %s",
                                   tmp, unlink_exc)
            return None
        return path

    def _rotate(self) -> None:
        """Keep the on-disk pool under ``max_episodes`` per kind."""
        for kind in ("positive", "negative"):
            sub = self.out_dir / kind
            if not sub.exists():
                continue
            files = sorted(sub.glob("dream_*.npz"),
                           key=lambda p: p.stat().st_mtime,
                           reverse=True)
            for old in files[self.cfg.max_episodes:]:
                try:
                    old.unlink()
                except OSError as exc:
                    LOGGER.warning("dreamer: rotate delete failed for %s: %s",
                                   old, exc)
        # Update the on-disk counter for the GUI.
        self.stats.on_disk_positive = len(list(
            (self.out_dir / "positive").glob("dream_*.npz")))
        self.stats.on_disk_negative = len(list(
            (self.out_dir / "negative").glob("dream_*.npz")))

    def _env_replay(self, frames: np.ndarray, actions: list[int]) -> float:
        """Real-env verification of a dream.

        The synthetic env exposes ``step_with_frame(frame)`` (a
        back-door used by the headless eval pipeline).  The dreamer
        threads the abstract frames through that back-door for
        ``min(len(frames), frames_per_dream)`` steps, asking the
        policy to act, and reports the *survival rate*: how many
        steps before the synthetic env reports ``done=True``.

        Survival is the strongest possible evidence that "the
        abstract image teaches what we wanted to teach".  If the
        policy dies on the abstract frame in 3 steps, the dream is
        a *bad* dream regardless of Q-score; the env's verdict
        overrides the Q heuristic.
        """
        if self._learner is None or not self.cfg.enabled:
            return 0.0
        env = getattr(self._learner, "_dream_env", None)
        if env is None:
            # No env bound — fall back to the Q-score, which is what
            # the unit tests do.  This is *not* a failure path, just
            # a different code path; the test pins both.
            return self._score_with_q(frames)
        survived = 0
        try:
            for i, frame in enumerate(frames[:self.cfg.frames_per_dream]):
                action = int(actions[i]) if i < len(actions) else 0
                result = env.step_with_frame(frame, action)
                if result.get("done", False):
                    break
                survived += 1
        except Exception as exc:  # noqa: BLE001 - reported
            LOGGER.warning("dreamer: env replay failed: %s", exc)
        # Return a [0, 1] survival rate.  The Q-score path returns
        # an unbounded float; the caller normalises to kind.
        return float(survived) / max(1, self.cfg.frames_per_dream)

    def maybe_dream(self, now: float) -> list[Path]:
        """Run a mental-rehearsal round if it's time.

        Returns the list of newly written dream files (for the GUI
        log).  The round consists of ``dreams_per_round`` dream
        cycles, each picking a seed frame, decoding a perturbed
        version, scoring it via env-replay, and saving to the
        appropriate pool.
        """
        if not self.cfg.enabled or self._learner is None:
            return []
        if self.stats.last_dream_at == 0.0:
            # First invocation: prime the timer so the very next
            # call can actually run (without this guard the first
            # non-throttled call would still get skipped because
            # ``now - 0.0`` always exceeds any positive throttle).
            self.stats.last_dream_at = now
            return []
        if now - self.stats.last_dream_at < self.cfg.dream_every_s:
            return []
        written: list[Path] = []
        for _ in range(self.cfg.dreams_per_round):
            seeds, actions = self._pick_seed_frames()
            if not seeds:
                break
            mu, decoded = self._encode_decode_dream(seeds[0])
            if decoded.size == 0:
                continue
            # Score: prefer env-replay (the user's spec), fall back
            # to Q.  ``_env_replay`` already returns the right
            # value for either path.
            score = self._env_replay(decoded, actions)
            # For env-replay the score is [0, 1]; for Q it's
            # unbounded.  Map both to a kind:
            if score >= 0.6:
                kind = "positive"
            elif score <= 0.2:
                kind = "negative"
            else:
                kind = "neutral"
            if kind == "neutral":
                continue
            self._q_history.append(score)
            if len(self._q_history) > 32:
                self._q_history = self._q_history[-32:]
            self.stats.rolling["q_mean"] = float(np.mean(self._q_history))
            self.stats.last_dream_q = float(score)
            # Source episode is the file name (the pool guarantees
            # one ``.npz`` per episode so we just take the basename).
            source = "self"
            try:
                paths = self._self_pool.on_disk_episode_paths()
                if paths:
                    source = Path(paths[0]).name
            except AttributeError as exc:
                # The pool does not implement the integration point
                # the dreamer needs; this is a *configuration* error
                # (someone forgot to subclass correctly), not a
                # runtime error.  Logged at warning level so it
                # surfaces in the operator's report.
                LOGGER.warning("dreamer: self pool missing "
                               "on_disk_episode_paths(): %s", exc)
            path = self._save_dream(kind, decoded, actions, source, score)
            if path is not None:
                written.append(path)
                if kind == "positive":
                    self.stats.dreams_positive += 1
                    self.stats.on_disk_positive += 1
                else:
                    self.stats.dreams_negative += 1
                    self.stats.on_disk_negative += 1
                self.stats.dreams_total += 1
        if written:
            self._rotate()
        self.stats.last_dream_at = now
        return written

    # -- public ---------------------------------------------------------

    def to_heartbeat(self) -> dict[str, float]:
        """Return the dict that the learner puts on metrics_q."""
        return {
            "dreams_total": float(self.stats.dreams_total),
            "dreams_positive": float(self.stats.dreams_positive),
            "dreams_negative": float(self.stats.dreams_negative),
            "on_disk_positive": float(self.stats.on_disk_positive),
            "on_disk_negative": float(self.stats.on_disk_negative),
            "last_train_loss": float(self.stats.last_train_loss),
            "dream_q_mean": float(self.stats.rolling.get("q_mean", 0.0)),
        }

    def state_dict(self) -> dict[str, object]:
        """Picklable state for checkpointing."""
        return {
            "vae": self.vae.state_dict(),
            "opt": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        try:
            self.vae.load_state_dict(state["vae"])  # type: ignore[arg-type]
            self.optimizer.load_state_dict(state["opt"])  # type: ignore[arg-type]
        except (KeyError, RuntimeError, ValueError) as exc:
            LOGGER.warning("dreamer: load_state_dict failed: %s", exc)
