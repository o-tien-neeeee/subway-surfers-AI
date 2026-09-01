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
from typing import Any

import numpy as np
import torch
from torch import nn

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

    def refresh_weights(self, shared, version: int | None = None) -> bool:
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
    """Linear decay in ENV frames (legacy; retained for older callers)."""
    decay = getattr(cfg, "epsilon_decay_frames", 0) or 0
    if decay <= 0:
        decay = max(1, getattr(cfg, "epsilon_decay_steps", 300_000))
    if frame <= 0:
        return cfg.epsilon_start
    frac = min(1.0, frame / max(1, decay))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def epsilon_for_step(step: int, cfg: RLConfig) -> float:
    """Linear decay over AGENT DECISION STEPS (the frame-skip clock).

    Deep-research fix (§1.2 #5): epsilon used to finish decaying in ~150k raw
    env frames (~episode 600-800) — before anything could be learned, the
    greedy-blind policy was locked at 5% exploration. Agent steps are the
    coherent MDP clock (one per frame-skip decision), and the schedule spans
    far more of them, so exploration survives long enough for learning.
    """
    decay = max(1, int(getattr(cfg, "epsilon_decay_steps", 300_000)))
    if step <= 0:
        return cfg.epsilon_start
    frac = min(1.0, step / decay)
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
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
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
        """One Double-DQN update from a PER batch; returns metrics.

        DQfD (deep Q-learning from demonstrations): when the batch carries an
        ``expert`` boolean mask, a large-margin supervised loss is added on
        those transitions — it forces the demonstrated action's Q to beat
        every other action by at least ``dqfd_margin``:

            L_margin = mean_i max_a [ Q(s_i, a) + margin * [a != a_E] - Q(s_i, a_E) ]+

        so the policy cannot unlearn the human during RL fine-tuning.
        """
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

        # --- DQfD large-margin loss on demonstrated transitions ---------- #
        margin_loss_val = 0.0
        expert_mask = batch.get("expert")
        if expert_mask is not None:
            em = torch.as_tensor(np.asarray(expert_mask), dtype=torch.bool)
            if bool(em.any()):
                margin = float(getattr(self.cfg, "dqfd_margin", 0.8))
                q_exp = q_all[em]                              # [E,5]
                a_exp = actions[em]                            # [E]
                one_hot = torch.nn.functional.one_hot(
                    a_exp, num_classes=q_exp.shape[1]
                ).float()
                margin_term = q_exp + margin * (1.0 - one_hot)  # +0 on expert action
                margin_target = (q_exp.gather(1, a_exp.unsqueeze(1)).squeeze(1)
                                 )
                l_margin = (margin_term.max(dim=1).values - margin_target).clamp_min(0.0)
                margin_loss_val = float(l_margin.mean().item())
                loss = loss + float(getattr(self.cfg, "dqfd_margin_weight", 0.3)) \
                    * l_margin.mean()

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

        with torch.no_grad():
            td_error_abs = td_error.abs().detach().cpu().numpy().astype(np.float64)
        return {
            "loss": float(loss.item()),
            "td_error_abs": td_error_abs,                          # per-sample for PER
            "td_error_abs_mean": float(td_error_abs.mean()),
            "margin_loss": margin_loss_val,
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
        optimizer: torch.optim.Optimizer | None = None,
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
            except ValueError:
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
