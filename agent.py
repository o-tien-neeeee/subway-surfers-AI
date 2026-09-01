"""Double-DQN agent (owned exclusively by the learner process).

* Double DQN action selection: online net picks argmax_a Q_online(s',a),
  target net evaluates it — reduces the classic overestimation bias.
* Huber (smooth L1) loss, gradient-norm clipping, Adam, LR schedule
  (constant or cosine), hard target-network sync every N updates.
* Epsilon is a function of ENV frames (see config docstring), evaluated by
  the actor; the learner only reports metrics.
* The same class exposes a tiny ``InferencePolicy`` used by the actor: it
  holds a local network copy refreshed from shared weights.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from config import RLConfig
from logging_utils import get_logger
from models import DuelingDQN, build_models_for_profile

LOGGER = get_logger("agent")


def _to_unit_float(array: np.ndarray) -> "torch.Tensor":
    """uint8/float [B,k,H,W] -> a NEW float32 tensor scaled to [0,1].

    # DEEP-FIX: the four call sites used ``torch.from_numpy(x).float()
    # .div_(255.0)``.  ``.float()`` is a no-op when the source is already
    # float32, so ``div_`` would divide the *caller's* numpy buffer in place
    # and every later read of that observation would see values 255x too
    # small.  It only stayed latent because every producer happened to emit
    # uint8.  Always materialising an owned tensor removes the aliasing trap.
    """
    import torch

    t = torch.from_numpy(np.ascontiguousarray(array))
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    else:
        t = t.clone()
    return t.div_(255.0)


class InferencePolicy:
    """Actor-side greedy/epsilon policy over a local model copy.

    Weight refresh: the learner publishes flat weights + a version counter in
    shared memory; the actor copies them in only when the version changes
    (a few-hundred-KB memcpy at most — measured, not assumed).
    """

    def __init__(self, model: DuelingDQN, seed: int = 0) -> None:
        self.model = model
        self.model.eval()
        self.rng = np.random.default_rng(seed)
        self._version = -1
        # DEEP-FIX: dedupe the layout-mismatch error so a persistent mismatch
        # cannot flood the log at 30 Hz.
        self._layout_mismatch_logged: str = ""
        self._own_fingerprint: Optional[str] = None

    def refresh_weights(self, shared, version: Optional[int] = None) -> bool:
        """Pull new flat weights from SharedWeights if version changed.

        Returns False when there is nothing new, when the shared buffer is
        contended, or when the published layout does not match this module.
        """
        v = shared.version() if version is None else version
        if v == self._version:
            return False
        from ipc import layout_fingerprint, unflatten_into

        # DEEP-FIX: refuse a layout mismatch instead of copying a meaningless
        # prefix of another profile's vector into this module.  An empty
        # fingerprint means "nothing published yet" (first read) and is
        # accepted so an untrained local model is still usable.  The local
        # fingerprint is cached: this runs on every decision, and a model's
        # parameter shapes never change after construction.
        published = ""
        if hasattr(shared, "fingerprint"):
            published = shared.fingerprint()
        if published:
            if self._own_fingerprint is None:
                self._own_fingerprint = layout_fingerprint(self.model)
            mine = self._own_fingerprint
            if published != mine:
                if self._layout_mismatch_logged != published:
                    self._layout_mismatch_logged = published
                    LOGGER.error(
                        "weight layout mismatch: learner published %s but this "
                        "actor runs %s — keeping the last known-good weights "
                        "(profile switch did not reach the learner?)",
                        published, mine,
                    )
                return False
        flat = shared.copy_out()
        if flat is None:
            # DEEP-FIX: copy_out() now reports contention instead of handing
            # back a torn buffer; retry on the next frame.
            return False
        try:
            unflatten_into(self.model, flat)
        except ValueError as exc:
            if self._layout_mismatch_logged != str(exc):
                self._layout_mismatch_logged = str(exc)
                LOGGER.error("weight refresh refused: %s", exc)
            return False
        self._version = v
        return True

    def q_values(self, stack_u8: np.ndarray) -> np.ndarray:
        # DEEP-FIX: shared, non-aliasing normalisation (see _to_unit_float).
        x = _to_unit_float(stack_u8).unsqueeze(0)
        with torch.inference_mode():
            q = self.model(x)
        return q.squeeze(0).numpy()

    def act(self, stack_u8: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.model.n_actions))
        return int(np.argmax(self.q_values(stack_u8)))


def epsilon_for_frame(frame: int, cfg: RLConfig) -> float:
    """Linear exploration decay over ``cfg.epsilon_decay_frames`` steps.

    MDP-FIX (v1.24.0): the caller must pass the number of *decision* steps
    (actions chosen), not raw captured frames.  An agent step is one
    ActionScheduler decision — the action is then held across the following
    2-4 frames (frame-skip MDP), so decisions are the true time index.
    """
    if frame <= 0:
        return cfg.epsilon_start
    frac = min(1.0, frame / max(1, cfg.epsilon_decay_frames))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def effective_epsilon(frame: int, cfg: RLConfig, bc_pretrained: float) -> float:
    """Exploration rate the actor should use this frame.

    After behaviour cloning produces a policy the actor must actually USE it:
    exploring at epsilon~1.0 ignores a good BC policy and the bot dies randomly
    every ~1s (a real 267-episode run never passed ~1.1s survival with epsilon
    0.99->0.77).  Once ``bc_pretrained`` is set we cap exploration at
    ``cfg.epsilon_after_bc`` so the BC/learned policy drives the bot, while
    still decaying below that cap as the normal schedule progresses.  Before BC
    the normal schedule is used unchanged.

    DEEP-FIX (v1.21.0): audit_bc_then_rl.py proved that even 15%
    ε-greedy *destroys* a BC-pretrained policy — the agent picked
    the wrong lane in 14% of frames and the survival fell from
    30s to 14.6s within 200 episodes.  When
    ``cfg.disable_exploration_after_bc`` is ``True`` (the new
    default) the actor uses pure exploitation (ε=0) after BC.  Set
    to ``False`` to fall back to the legacy ``epsilon_after_bc``
    cap of 0.15.
    """
    eps = epsilon_for_frame(frame, cfg)
    if bc_pretrained > 0:
        if getattr(cfg, "disable_exploration_after_bc", False):
            return 0.0
        eps = min(eps, cfg.epsilon_after_bc)
    return eps


class DoubleDQNAgent:
    """Training-side agent: owns online/target nets, optimizer, losses."""

    def __init__(self, profile: str, cfg: RLConfig, in_frames: int = 4,
                 size: int = 84, device: str = "cpu", seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.cfg = cfg
        self.profile = profile
        self.device = torch.device(device)
        assert self.device.type == "cpu", "CPU-only build (requirement §1)"
        self.online, self.target = build_models_for_profile(profile, in_frames, size)
        self.online.to(self.device)
        self.target.to(self.device)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if cfg.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, cfg.cosine_period_updates),
                eta_min=cfg.lr_min,
            )
        self.update_count = 0
        self.action_space = self.online.n_actions

    # ------------------------------------------------------------------ #
    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def maybe_sync_target(self) -> bool:
        if self.update_count > 0 and self.update_count % self.cfg.target_update_every == 0:
            self.sync_target()
            return True
        return False

    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """One Double-DQN update from a PER batch; returns metrics."""
        obs = _to_unit_float(batch["obs"])
        next_obs = _to_unit_float(batch["next_obs"])
        actions = torch.from_numpy(batch["actions"])
        rewards = torch.from_numpy(batch["rewards"])
        dones = torch.from_numpy(batch["dones"])
        weights = torch.from_numpy(batch["weights"])
        gamma_pows = torch.from_numpy(batch["gamma_pows"])

        q_all = self.online(obs)                                # [B,5]
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        with torch.no_grad():
            next_q_online = self.online(next_obs)               # [B,5]
            next_actions = next_q_online.argmax(dim=1)           # Double DQN
            next_q_target = self.target(next_obs)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + gamma_pows * (1.0 - dones) * next_q

        td_error = q_sa - target
        loss_per = nn.functional.smooth_l1_loss(q_sa, target, reduction="none")
        loss = (weights * loss_per).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.cfg.grad_clip_norm
        ))
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.update_count += 1
        self.maybe_sync_target()

        # DEEP-FIX: the caller needs the PER-SAMPLE error to re-prioritise the
        # buffer.  Only the batch mean used to be returned, and the learner
        # broadcast that one scalar over every sampled index — which erases
        # exactly the ordering PER exists to maintain (verified: after one
        # update all 32 sampled slots held the identical priority 2.1924).
        with torch.no_grad():
            td_abs = td_error.abs().detach().to(torch.float32).cpu().numpy()
        td_abs = np.nan_to_num(td_abs, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "loss": float(loss.item()),
            "td_errors": np.ascontiguousarray(td_abs, dtype=np.float64),
            "td_error_abs_mean": float(td_abs.mean()) if td_abs.size else 0.0,
            "q_mean": float(q_all.mean().item()),
            "q_max": float(q_all.max().item()),
            "q_min": float(q_all.min().item()),
            "grad_norm": grad_norm if math.isfinite(grad_norm) else 0.0,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ------------------------------------------------------------------ #
    # Behaviour cloning (Phase 1) — DQfD-style supervised loss on Q logits
    # ------------------------------------------------------------------ #
    def bc_epoch(
        self,
        obs_u8: np.ndarray,
        actions: np.ndarray,
        sample_weights: np.ndarray,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict[str, float]:
        """Cross-entropy of softmax(advantage logits) vs expert actions."""
        opt = optimizer or self.optimizer
        x = _to_unit_float(obs_u8)
        y = torch.from_numpy(actions)
        w = torch.from_numpy(sample_weights.astype(np.float32))
        logits = self.online(x, return_logits=True)
        loss_per = nn.functional.cross_entropy(logits, y, reduction="none")
        loss = (w * loss_per).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip_norm)
        opt.step()
        acc = float((logits.argmax(dim=1) == y).float().mean().item())
        return {"bc_loss": float(loss.item()), "bc_acc": acc}

    def bc_eval(self, obs_u8: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
        x = _to_unit_float(obs_u8)
        y = torch.from_numpy(actions)
        with torch.inference_mode():
            logits = self.online(x, return_logits=True)
            pred = logits.argmax(dim=1)
        acc = float((pred == y).float().mean().item())
        per_action = {
            int(a): float((pred[y == a] == a).float().mean().item())
            if int((y == a).sum()) > 0 else 0.0
            for a in range(self.action_space)
        }
        return {"bc_acc": acc, "per_action": per_action}

    # ------------------------------------------------------------------ #
    def state_payload(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "update_count": self.update_count,
            "profile": self.profile,
        }

    def load_payload(self, payload: dict[str, Any], strict: bool = True) -> None:
        self.online.load_state_dict(payload["online"], strict=strict)
        self.target.load_state_dict(payload["target"], strict=strict)
        if payload.get("optimizer"):
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except (ValueError, RuntimeError) as exc:
                # DEEP-FIX: torch raises RuntimeError (not ValueError) for a
                # parameter-group size mismatch, so the old handler never
                # caught the case it was written for and the exception
                # escaped as a "corrupt checkpoint" instead of a clean
                # "optimizer state does not fit, keeping a fresh one".
                if strict:
                    raise
                LOGGER.warning(
                    "optimizer state from the checkpoint does not fit this "
                    "profile (%s: %s); continuing with a fresh optimizer",
                    type(exc).__name__, exc,
                )
        if payload.get("scheduler") and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(payload["scheduler"])
            except (ValueError, RuntimeError) as exc:
                # DEEP-FIX: same class of drift; a stale LR schedule must not
                # abort an otherwise valid weight restore.
                LOGGER.warning("scheduler state not restorable (%s); resetting",
                               exc)
        self.update_count = int(payload.get("update_count", 0))
        self.sync_target()

    def flat_weights(self) -> np.ndarray:
        from ipc import flatten_state_dict

        return flatten_state_dict(self.online.state_dict())

    def publish(self, shared) -> int:
        """Copy current online weights into SharedWeights; return version."""
        from ipc import layout_fingerprint

        # DEEP-FIX: tag the vector with this profile's layout so an actor on a
        # different profile refuses the copy instead of mis-aligning tensors.
        shared.publish(self.flat_weights(),
                       fingerprint=layout_fingerprint(self.online))
        return shared.version()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.online.parameters())
