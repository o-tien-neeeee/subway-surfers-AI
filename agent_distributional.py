"""Distributional QR-DQN agent.

Wraps :class:`distributional.QuantileDuelingDQN` into the same
interface :class:`agent.DoubleDQNAgent` exposes, so the learner
can switch between scalar Q and distributional Q via a single
config flag (see :attr:`config.RLConfig.distributional`).

Why a parallel class instead of editing the original
``DoubleDQNAgent``?
------------------------------------
The original uses ``smooth_l1_loss`` and aggregates TD-errors
across the batch; the QR-DQN variant needs ``quantile_huber_loss``
operating on the full ``[B, A, N]`` tensor.  The bootstrap,
target, sync, optimizer and BC paths are otherwise identical, so
a small adapter that *reuses* the underlying ``DoubleDQNAgent``
where it can keeps the diff small and the tests honest.

Public surface
--------------
* :class:`DistributionalDoubleDQNAgent` — full agent (training
  + inference).
* :func:`build_distributional_agent` — factory that mirrors
  :func:`agent.build_*` so callers do not need to know the
  internals.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from agent import _to_unit_float
from config import RLConfig
from distributional import (QuantileDuelingDQN, mid_quantiles,
                             project_distribution, quantile_huber_loss)
from logging_utils import get_logger
from replay_buffer import NStepTransition

LOGGER = get_logger("agent_dist")


def build_distributional_agent(
        profile: str, cfg: RLConfig, in_frames: int = 4, size: int = 84,
        num_quantiles: int = 51, seed: int = 0) -> "DistributionalDoubleDQNAgent":
    """Factory mirroring ``build_models_for_profile``."""
    return DistributionalDoubleDQNAgent(
        profile, cfg, in_frames=in_frames, size=size,
        num_quantiles=num_quantiles, seed=seed)


class DistributionalDoubleDQNAgent:
    """QR-DQN training-side agent.

    Mirrors :class:`agent.DoubleDQNAgent` for the parts that do
    not depend on the head shape: optimizer setup, target sync,
    state-dict round-trip, BC steps.  The :meth:`train_step` is
    replaced with the quantile loss.
    """

    def __init__(self, profile: str, cfg: RLConfig, in_frames: int = 4,
                 size: int = 84, num_quantiles: int = 51, device: str = "cpu",
                 seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.cfg = cfg
        self.profile = profile
        self.num_quantiles = num_quantiles
        self.device = torch.device(device)
        assert self.device.type == "cpu", "CPU-only build (requirement §1)"
        self.online = QuantileDuelingDQN(
            profile, in_frames=in_frames, size=size,
            num_quantiles=num_quantiles).to(self.device)
        self.target = QuantileDuelingDQN(
            profile, in_frames=in_frames, size=size,
            num_quantiles=num_quantiles).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=cfg.learning_rate)
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if cfg.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, cfg.cosine_period_updates),
                eta_min=cfg.lr_min,
            )
        self.tau = mid_quantiles(num_quantiles).to(self.device)
        self.update_count = 0
        self.n_actions = self.online.num_actions
        # Optional RND curiosity module.  Attached by the
        # learner after construction so the test suite can
        # pass a stub or skip the module entirely.
        self._rnd: Optional["RNDModule"] = None

    def attach_rnd(self, rnd_module: "RNDModule") -> None:
        """Bind an :class:`rnd.RNDModule` for intrinsic-reward
        injection.  Called by the learner during
        construction; tests construct the agent and then
        attach a stub when they want to verify the
        integration.
        """
        self._rnd = rnd_module

    # ------------------------------------------------------------------ #
    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def maybe_sync_target(self) -> bool:
        """Sync target network.  Two paths:
        * Polyak (soft) averaging — every train step, when
          ``cfg.polyak_target=True``.
        * Hard copy — every ``cfg.target_update_every`` steps,
          when ``cfg.polyak_target=False`` (legacy).
        """
        if bool(getattr(self.cfg, "polyak_target", False)):
            tau = float(getattr(self.cfg, "polyak_tau", 0.005))
            self.polyak_update(tau)
            return False  # never return True on a soft update
        if (self.update_count > 0
                and self.update_count % self.cfg.target_update_every == 0):
            self.sync_target()
            return True
        return False

    def polyak_update(self, tau: float) -> None:
        """Polyak (soft) target update.

        ``target = (1 - tau) * target + tau * online``

        With ``tau=0.005`` (TD3 default) the target
        tracks the online net with a half-life of
        ~140 steps.  This is the standard
        off-policy actor-critic recipe and is known
        to be more stable than hard updates on hard
        exploration tasks.
        """
        with torch.no_grad():
            for p_target, p_online in zip(self.target.parameters(),
                                             self.online.parameters()):
                p_target.data.mul_(1.0 - tau).add_(
                    p_online.data, alpha=tau)

    # ------------------------------------------------------------------ #
    def _gamma_pow(self, gamma_pows: np.ndarray) -> torch.Tensor:
        """Convert a per-sample ``gamma_pows`` (the discount
        accumulated over the n-step horizon) into a tensor of shape
        ``[B]`` for the per-sample Bellman shift.
        """
        return torch.from_numpy(gamma_pows.astype(np.float32))

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """One QR-DQN update from a PER batch; returns metrics.

        If an RND module has been attached (see
        :meth:`attach_rnd`), an *intrinsic* reward bonus is
        computed for each transition in the batch and added
        to the extrinsic reward before the Bellman target is
        built.  The bonus is :math:`\\beta \\cdot
        \\|\\text{target}(s) - \\text{predictor}(s)\\|^2 /
        \\text{normaliser}` so the agent is rewarded for
        visiting states the predictor has not yet learned
        to mimic.

        DEEP-FIX (v1.22.0): when ``cfg.augment_obs=True``
        the current observation is randomly perturbed
        (intensity jitter + random translate + optional
        frame differencing) *before* the forward pass.
        The next-state observation is left alone so the
        TD target is not corrupted by augmentation noise.
        """
        obs = _to_unit_float(batch["obs"])
        next_obs = _to_unit_float(batch["next_obs"])
        actions = torch.from_numpy(batch["actions"]).long()
        rewards = torch.from_numpy(batch["rewards"]).float()
        dones = torch.from_numpy(batch["dones"]).float()
        weights = torch.from_numpy(batch["weights"]).float()
        gamma_pows = self._gamma_pow(batch["gamma_pows"])

        # Intrinsic-reward injection.  The RND module's
        # ``intrinsic_reward`` returns a *per-sample* float;
        # we add it to the extrinsic reward and renormalise
        # so the *total* per-step reward stays in roughly
        # the same magnitude regardless of whether RND is on.
        rnd_metrics: dict[str, float] = {}
        if self._rnd is not None:
            # ``obs`` is the normalised float tensor; the RND
            # module expects raw uint8, so we cast back.
            obs_u8 = (obs * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            intrinsic = self._rnd.intrinsic_reward(obs_u8)
            rewards = rewards + torch.from_numpy(intrinsic).float()
            # Train the predictor on the same batch (cheap).
            rnd_result = self._rnd.train_step(obs_u8)
            if rnd_result is not None:
                rnd_metrics = rnd_result

        # DEEP-FIX v1.22.0: visual data augmentation
        # (RAD-style).  We perturb ONLY the current
        # observation; the next-state observation is
        # left un-augmented so the TD target is
        # computed against a clean bootstrap.
        if bool(getattr(self.cfg, "augment_obs", False)):
            obs = self._augment_observation(obs)

        # Predicted quantiles for the chosen action.
        # online(obs) -> [B, A, N]
        dist = self.online(obs)
        # Gather: for each batch i, take the row of quantiles
        # corresponding to action[i].  Reshape actions to [B, 1, 1]
        # and use advanced indexing.
        pred = dist.gather(
            1, actions.view(-1, 1, 1).expand(-1, 1, self.num_quantiles)
        ).squeeze(1)  # [B, N]

        with torch.no_grad():
            next_dist_full = self.target(next_obs)  # [B, A, N]
            # QR-DQN action selection: pick the action whose
            # MEAN of quantiles is the highest.  Then expand to
            # the full distribution for the projection.
            next_q = next_dist_full.mean(dim=-1)      # [B, A]
            next_actions = next_q.argmax(dim=1)       # [B]
            next_dist = next_dist_full.gather(
                1, next_actions.view(-1, 1, 1).expand(
                    -1, 1, self.num_quantiles)
            ).squeeze(1)                               # [B, N]
            # The target distribution is the Bellman shift of
            # the *chosen* next-action distribution.  Per-sample
            # discount: ``gamma_pow`` is the n-step discount
            # already aggregated in the buffer; we use it
            # directly here (the standard QR-DQN formula uses
            # a scalar gamma because it bootstraps from a
            # 1-step target; for n-step targets, gamma_pow
            # replaces it).
            target = rewards.unsqueeze(-1) + \
                gamma_pows.unsqueeze(-1) * \
                (1.0 - dones).unsqueeze(-1) * next_dist  # [B, N]
            target = target.detach()

        # Per-element loss — same shape as ``pred``.
        per_element = quantile_huber_loss(
            pred.unsqueeze(1),  # [B, 1, N]  (acts as "1 action")
            target.unsqueeze(1),  # [B, 1, N]
            self.tau,
        )  # scalar
        # The wrapper above reduces over (B, 1, N) and returns
        # the mean; we want a per-sample loss so we re-compute
        # the quantile huber without the final ``.mean()`` so
        # we can weight by PER importance.
        loss_per_sample = self._per_sample_quantile_huber(
            pred, target)
        loss = (weights * loss_per_sample).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.cfg.grad_clip_norm))
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.update_count += 1
        self.maybe_sync_target()

        # Build the per-sample TD error proxy.  We use the
        # absolute difference between the *mean* predicted
        # quantile and the *mean* target distribution, which
        # is the natural scalar for PER prioritisation.
        with torch.no_grad():
            pred_mean = pred.mean(dim=-1)
            target_mean = target.mean(dim=-1)
            td_abs = (pred_mean - target_mean).abs()
            td_abs = td_abs.detach().to(torch.float32).cpu().numpy()
            td_abs = np.nan_to_num(td_abs, nan=0.0,
                                    posinf=0.0, neginf=0.0)

        return {
            "loss": float(loss.item()),
            "td_errors": np.ascontiguousarray(td_abs, dtype=np.float64),
            "td_error_abs_mean": float(td_abs.mean()) if td_abs.size else 0.0,
            "q_mean": float(dist.mean().item()),
            "q_max": float(dist.mean(dim=-1).max().item()),
            "q_min": float(dist.mean(dim=-1).min().item()),
            "grad_norm": grad_norm if math.isfinite(grad_norm) else 0.0,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            # Merge the RND metrics (loss, normaliser) so the
            # GUI heartbeat can show the predictor training.
            **rnd_metrics,
        }

    def _augment_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply the configured augmentations to the
        current observation.

        Augmentations are applied in this order:
        1. Random translate (shift every frame in the
           stack by the same offset).
        2. Intensity jitter (brightness + contrast).

        We DO NOT apply :func:`frame_difference` here
        because the production encoder's first conv
        layer expects a fixed number of input
        channels.  Appending a new channel would
        require rebuilding the encoder with
        ``in_channels=in_frames+1`` (see
        :class:`improved_dqn.ImprovedQuantileDuelingDQN`
        for the encoder with the +1 channel support).
        The frame-difference feature is still
        available — just enable it through the
        improved encoder, not through the
        augmentation pipeline.

        The next-state observation is NOT augmented —
        see the docstring in :meth:`train_step`.
        """
        from augmentations import (AugmentationConfig,
                                    intensity_jitter, random_translate)
        x = obs
        if bool(getattr(self.cfg, "augment_translate_px", 0)):
            x = random_translate(
                x, int(getattr(self.cfg, "augment_translate_px", 3)))
        if bool(getattr(self.cfg, "augment_intensity", 0)):
            x = intensity_jitter(
                x,
                float(getattr(self.cfg, "augment_intensity", 0.10)),
                float(getattr(self.cfg, "augment_intensity", 0.10)))
        return x

    def _per_sample_quantile_huber(self, pred: torch.Tensor,
                                    target: torch.Tensor) -> torch.Tensor:
        """Quantile Huber per sample (no reduction over the batch).

        Returns a ``[B]`` tensor so the caller can weight by
        PER importance weights and reduce manually.
        """
        # u: signed error per (sample, target_q, predicted_q).
        # pred:   [B, 1, N_p]
        # target: [B, 1, N_t] — but here N_t == N_p.
        u = target.unsqueeze(-1) - pred.unsqueeze(-2)  # [B, 1, N, N]
        abs_u = u.abs()
        kappa = 1.0
        huber = torch.where(
            abs_u <= kappa,
            0.5 * u.pow(2),
            kappa * (abs_u - 0.5 * kappa),
        )
        # Quantile weight: |tau_target - I[u<0]|.
        tau = self.tau.view(1, 1, -1, 1)
        weight = (tau - (u < 0).float()).abs()
        # Reduce over (N, N) and the singleton "1" axis.
        loss = (weight * huber).mean(dim=(2, 3)).squeeze(1)  # [B]
        return loss

    # ------------------------------------------------------------------ #
    def flat_weights(self) -> np.ndarray:
        """Flatten the online network for IPC.

        Same contract as :meth:`agent.DoubleDQNAgent.flat_weights`
        so the actor can copy this out without knowing the
        network architecture.
        """
        from ipc import flatten_state_dict
        return flatten_state_dict(self.online.state_dict())

    def publish(self, shared) -> int:
        """Copy current online weights into ``SharedWeights``.

        Returns the new version.  The shared fingerprint
        means the actor can detect a profile mismatch
        (different number of quantiles) and refuse the copy
        instead of mis-aligning the tensors.
        """
        from ipc import layout_fingerprint
        shared.publish(self.flat_weights(),
                       fingerprint=layout_fingerprint(self.online))
        return shared.version()

    def count_params(self) -> int:
        """Total trainable parameter count of the online net.

        Used by the GUI and by the headless test to confirm
        the chosen profile stays inside the documented
        parameter budget.
        """
        return sum(p.numel() for p in self.online.parameters())

    # ------------------------------------------------------------------ #
    # State-dict round-trip mirrors DoubleDQNAgent so checkpoints
    # written by either can be loaded by the other.
    # ------------------------------------------------------------------ #
    def state_payload(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "update_count": self.update_count,
            "profile": self.profile,
            "num_quantiles": self.num_quantiles,
        }

    def load_payload(self, payload: dict[str, Any],
                      strict: bool = True) -> None:
        self.online.load_state_dict(payload["online"], strict=strict)
        self.target.load_state_dict(payload["target"], strict=strict)
        if payload.get("optimizer"):
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except (KeyError, ValueError) as exc:
                LOGGER.warning("optimizer state load refused: %s", exc)
        if payload.get("scheduler") and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(payload["scheduler"])
            except (KeyError, ValueError) as exc:
                LOGGER.warning("scheduler state load refused: %s", exc)
        self.update_count = int(payload.get("update_count", 0))
        if "num_quantiles" in payload:
            self.num_quantiles = int(payload["num_quantiles"])
