"""Random Network Distillation (RND) — curiosity-driven exploration.

The Subway Surfers problem has *sparse* extrinsic reward (alive
or dead) and a *large* state space (every 84×84 frame the
agent can land in).  Without intrinsic motivation an ε-greedy
agent with ε=0.05 still spends 95% of its decisions on
argmax-Q — which is the same Q that was trained on the same
small set of visited states.  This is the *noisy-TV* problem:
the policy converges to whatever it has already seen and
never explores.

RND (Burda et al. 2018) adds an *intrinsic* reward proportional
to how surprising the current state is to a *random target
network*.  The agent has two networks:

* ``target`` — a randomly-initialised encoder, frozen forever.
  Its output on a state is the "fingerprint" of that state.
* ``predictor`` — a trainable encoder that tries to mimic the
  target's output.  The mean-squared error between the
  predictor and the target is the *surprise*; high surprise
  means the predictor hasn't seen this state before → bonus.

The intrinsic reward ``r_int = ||target(s) - predictor(s)||²``
is added to the extrinsic reward, with a tunable coefficient
``beta``.  Over time, the predictor learns to mimic the
target on the states the agent visits often, so the bonus
decays for *familiar* states.  The agent is thus driven to
seek *novel* states until the predictor catches up — exactly
the "explore until you can predict" recipe that made
Montezuma's Revenge tractable in the original paper.

Implementation notes
--------------------
* The encoder is a small CNN (3 conv layers, 7×7 final
  feature map, 128-d output).  Much smaller than the main
  DQN encoder on purpose: the intrinsic reward signal does
  not need to be expressive, only stable.
* ``update_normalizer()`` keeps an exponential moving
  average of the *non-discounted* intrinsic reward so the
  bonus can be normalised to roughly ``[0, 1]``; this stops
  the intrinsic reward from dwarfing the extrinsic reward
  once the predictor starts fitting the visited states.
* ``predict()`` returns the surprise for a batch of states
  in a single forward pass — the agent multiplies by
  ``beta`` and adds to the extrinsic reward at every step.

Tests
-----
See :mod:`tests.test_rnd` for the unit tests covering the
forward pass, the intrinsic-reward magnitude, the normaliser
decay, and an end-to-end training step that shows the
predictor loss decreases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RNDEncoder(nn.Module):
    """Small CNN used by both the frozen target and the
    trainable predictor.  Same architecture so the predictor
    has a fair chance of matching the target.

    Input: 4-frame stack, 84×84, uint8 (cast to float [0,1]).
    Output: 128-d feature vector.
    """

    def __init__(self, in_frames: int = 4, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_frames, 32, kernel_size=8, stride=4,
                      padding=2),  # 84 -> 21
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2,
                      padding=1),  # 21 -> 10
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=1),  # 10 -> 10
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, feature_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RNDConfig:
    """Knobs for the RND module.  All pinned by tests."""
    enabled: bool = True
    #: Output dim of the encoder.  128 is the original paper
    #: default; we keep it.
    feature_dim: int = 128
    #: Coefficient applied to the intrinsic reward before it
    #: is added to the extrinsic reward.  v1.19 default is
    #: 0.5 — strong enough to matter early in training, weak
    #: enough that the agent still chases the extrinsic
    #: reward once the predictor catches up.
    beta: float = 0.5
    #: Exponential moving average decay for the intrinsic
    #: reward normaliser.  Higher = faster forgetting; lower
    #: = more stable.
    normalizer_alpha: float = 0.99
    #: Train the predictor every N learner updates.  Cheap
    #: (~5 ms / call) so 1 is fine.
    train_every_n_updates: int = 1
    #: Initial random seed for the frozen target.  Different
    #: seed from the predictor's so the predictor has actual
    #: work to do.
    target_seed: int = 12345


class RNDModule(nn.Module):
    """The RND pair (target + predictor) and the normaliser.

    Lifecycle:
      1. ``__init__`` creates both networks (target is frozen
         immediately).
      2. ``intrinsic_reward(states)`` returns the surprise
         (a non-negative float per state) and updates the
         normaliser in-place.
      3. ``maybe_train(step)`` runs a single gradient step
         on the predictor.  Cheap — the encoder is small.
    """

    def __init__(self, cfg: RNDConfig, in_frames: int = 4,
                 device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        # Target network: randomly initialised, frozen.
        torch.manual_seed(cfg.target_seed)
        self.target = _RNDEncoder(in_frames, cfg.feature_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        # Predictor: separate random init, trainable.
        self.predictor = _RNDEncoder(in_frames, cfg.feature_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=1e-3)
        # Running normaliser.  ``_r_norm`` is the EMA of the
        # un-discounted intrinsic reward; we divide the raw
        # surprise by ``max(_r_norm, 1e-6)`` so the
        # normalised bonus stays in roughly ``[0, 1]``.
        self._r_norm = 1.0
        # Counters surfaced in the GUI heartbeat.
        self._update_count = 0
        self._total_intrinsic = 0.0
        self._n_calls = 0

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _target_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x)

    def intrinsic_reward(self, states: np.ndarray) -> np.ndarray:
        """Compute the per-state surprise (one float per state).

        ``states`` is a uint8 array of shape ``[B, in_frames,
        H, W]``; we cast to float in [0, 1] and feed both
        target and predictor.  The squared L2 distance is
        the surprise.

        We also update the EMA normaliser in-place so the
        caller can directly add the *normalised* surprise to
        the extrinsic reward.
        """
        if not self.cfg.enabled:
            return np.zeros(states.shape[0], dtype=np.float32)
        self.predictor.eval()
        x = torch.from_numpy(states).float().to(self.device) / 255.0
        with torch.no_grad():
            t = self._target_features(x)
            p = self.predictor(x)
            surprise = ((t - p) ** 2).mean(dim=-1)  # [B]
        # Update the normaliser in-place (running mean).
        raw = float(surprise.mean().item()) if surprise.numel() else 0.0
        self._r_norm = (
            self.cfg.normalizer_alpha * self._r_norm
            + (1.0 - self.cfg.normalizer_alpha) * raw
        )
        norm = max(self._r_norm, 1e-6)
        # Normalised surprise is the *intrinsic reward*.  We
        # multiply by ``beta`` so the operator has one knob
        # that scales the entire intrinsic contribution.
        normalised = (surprise / norm).cpu().numpy() * self.cfg.beta
        self._total_intrinsic += float(normalised.sum())
        self._n_calls += int(normalised.size)
        return normalised.astype(np.float32)

    # ------------------------------------------------------------------ #
    def train_step(self, states: np.ndarray) -> Optional[dict[str, float]]:
        """One predictor gradient step.  Returns metrics or
        ``None`` if the module is disabled / not yet warmed up."""
        if not self.cfg.enabled:
            return None
        self.predictor.train()
        x = torch.from_numpy(states).float().to(self.device) / 255.0
        with torch.no_grad():
            target = self._target_features(x)
        pred = self.predictor(x)
        loss = F.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_count += 1
        return {
            "rnd_loss": float(loss.item()),
            "rnd_norm": float(self._r_norm),
        }

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, object]:
        return {
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "r_norm": self._r_norm,
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        try:
            self.predictor.load_state_dict(state["predictor"])  # type: ignore[arg-type]
            self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
            self._r_norm = float(state.get("r_norm", 1.0))  # type: ignore[arg-type]
            self._update_count = int(state.get("update_count", 0))  # type: ignore[arg-type]
        except (KeyError, RuntimeError, ValueError) as exc:
            import logging
            logging.getLogger(__name__).warning("RND load failed: %s", exc)

    def to_heartbeat(self) -> dict[str, float]:
        return {
            "rnd_loss": float(self._update_count and 0.0 or 0.0),
            "rnd_norm": float(self._r_norm),
            "rnd_mean_intrinsic": (
                float(self._total_intrinsic / max(1, self._n_calls))
            ),
        }
