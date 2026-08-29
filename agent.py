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
from models import DuelingDQN, build_models_for_profile


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

    def refresh_weights(self, shared, version: Optional[int] = None) -> bool:
        """Pull new flat weights from SharedWeights if version changed."""
        v = shared.version() if version is None else version
        if v == self._version:
            return False
        from ipc import unflatten_into

        unflatten_into(self.model, shared.copy_out())
        self._version = v
        return True

    def q_values(self, stack_u8: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(stack_u8)).float().div_(255.0)
        x = x.unsqueeze(0)
        with torch.inference_mode():
            q = self.model(x)
        return q.squeeze(0).numpy()

    def act(self, stack_u8: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.model.n_actions))
        return int(np.argmax(self.q_values(stack_u8)))


def epsilon_for_frame(frame: int, cfg: RLConfig) -> float:
    """Linear decay in ENV frames: the only definition of 'step' for eps."""
    if frame <= 0:
        return cfg.epsilon_start
    frac = min(1.0, frame / max(1, cfg.epsilon_decay_frames))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


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
        obs = torch.from_numpy(batch["obs"]).float().div_(255.0)
        next_obs = torch.from_numpy(batch["next_obs"]).float().div_(255.0)
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

        return {
            "loss": float(loss.item()),
            "td_error_abs_mean": float(td_error.abs().mean().item()),
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
        x = torch.from_numpy(obs_u8).float().div_(255.0)
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
        x = torch.from_numpy(obs_u8).float().div_(255.0)
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
            except ValueError as exc:
                if strict:
                    raise
                # profile change: keep fresh optimizer, warn through return
                self.update_count = int(payload.get("update_count", 0))
        if payload.get("scheduler") and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.update_count = int(payload.get("update_count", 0))
        self.sync_target()

    def flat_weights(self) -> np.ndarray:
        from ipc import flatten_state_dict

        return flatten_state_dict(self.online.state_dict())

    def publish(self, shared) -> int:
        """Copy current online weights into SharedWeights; return version."""
        shared.publish(self.flat_weights())
        return shared.version()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.online.parameters())
