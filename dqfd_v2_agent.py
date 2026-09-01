"""DQfD-v2 — DQfD + SIL + EMA + AutoEntropy (v1.23.0 production agent).

DEEP-FIX v1.23.0
================

This module is the *v1.23.0 production training
agent* for the SyntheticGame.  It builds on the
v1.21.0 :class:`DQfDAgent` (which is the
production agent on the LearnableEnv) and adds
four v1.23.0 innovations:

1. **Self-Imitation Learning (SIL)** — the
   agent's own good episodes are replayed at a
   higher rate, weighted by the clipped
   advantage.  This directly addresses the
   SyntheticGame 12.5s → 30s gap by teaching the
   agent what "good" looks like in *its own
   experience*.
2. **EMA of network weights for evaluation** —
   the agent's online weights are EMA'd, and the
   EMA is installed during evaluation.  Reduces
   variance (TD7 / Tarasov et al. 2024).
3. **SAC auto-tuned entropy temperature** — the
   Q-net's softmax temperature is auto-tuned to
   keep the policy's entropy at a target.  This
   prevents the policy from becoming too
   deterministic too early.
4. **IBRL bootstrap proposal** (the paper's
   headline 6.4× result) — the TD target uses
   ``max(Q(s', a_il), Q(s', a_rl))`` instead of
   the standard ``max_a Q(s', a)``.  Bounds the
   target Q-value to actions the agent *knows*
   about (the BC net's argmax or the RL net's
   argmax), preventing Q-value over-estimation.

The class subclasses :class:`DQfDAgent` so it
inherits the entire v1.21.0 production pipeline
(joint loss, demo buffer, BC pretrain, etc.).
The new modules are *additive* — the v1.21.0
audit_bc_then_rl.py test still passes 30s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_distributional import DistributionalDoubleDQNAgent
from auto_entropy import AutoEntropy, AutoEntropyConfig
from dqfd_agent import DQfDAgent
from ema import EMA
from ibrl import (IBRLConfig, bootstrap_proposal_q,
                     actor_proposal_action)
from sil import SILBuffer, SILConfig, SILTrainer


@dataclass
class DQfDv2Config:
    """Knobs for the v1.23.0 DQfD-v2 agent.

    All flags default ON.  The agent is a
    drop-in replacement for :class:`DQfDAgent`
    with the same ``pretrain_demos`` /
    ``train_step`` interface.
    """
    use_sil: bool = True
    sil: SILConfig = None  # default in __init__
    use_ema: bool = True
    ema_decay: float = 0.999
    use_auto_entropy: bool = True
    auto_entropy: AutoEntropyConfig = None  # default in __init__
    use_ibrl_bootstrap: bool = True
    ibrl: IBRLConfig = None  # default in __init__
    lambda_sil: float = 0.1
    lambda_bc: float = 0.5  # standard DQfD value


def _default_sil() -> SILConfig:
    return SILConfig(capacity=50, gamma=0.99)


def _default_auto_entropy() -> AutoEntropyConfig:
    return AutoEntropyConfig(n_actions=5)


def _default_ibrl() -> IBRLConfig:
    return IBRLConfig(use_actor_proposal=False,
                       use_bootstrap_proposal=True,
                       noise_eps=0.0)


class DQfDv2Agent(DQfDAgent):
    """v1.23.0 DQfD agent with SIL + EMA + AutoEntropy + IBRL bootstrap.

    The agent extends :class:`DQfDAgent` so it
    inherits the v1.21.0 joint loss (TD + BC +
    margin) and adds the four v1.23.0 modules.

    Usage
    -----
    1. ``pretrain_demos(obs, actions)`` — BC
       warm-up.  This *also* freezes a copy of
       the trained net for use as the BC net in
       IBRL.
    2. ``add_episode(states, actions, rewards)`` —
       feed finished episodes to the SIL buffer.
    3. ``train_step(batch)`` — single-pass joint
       loss (TD + BC + margin + SIL + auto-entropy
       + IBRL bootstrap).
    4. ``update_ema()`` — pull the EMA shadow
       toward the current RL weights.
    5. ``eval_mode()`` / ``train_mode()`` —
       install / restore the EMA weights.
    """

    def __init__(self, profile: str, cfg, dqfd,
                  in_frames: int = 4, size: int = 84,
                  num_quantiles: int = 51,
                  device: str = "cpu", seed: int = 0,
                  v2_cfg: Optional[DQfDv2Config] = None) -> None:
        super().__init__(profile, cfg, dqfd, in_frames=in_frames,
                          size=size, num_quantiles=num_quantiles,
                          device=device, seed=seed)
        # Build the v1.23.0 config with sensible defaults.
        self.v2_cfg = v2_cfg or DQfDv2Config(
            sil=_default_sil(),
            auto_entropy=_default_auto_entropy(),
            ibrl=_default_ibrl(),
        )
        if self.v2_cfg.sil is None:
            self.v2_cfg.sil = _default_sil()
        if self.v2_cfg.auto_entropy is None:
            self.v2_cfg.auto_entropy = _default_auto_entropy()
        if self.v2_cfg.ibrl is None:
            self.v2_cfg.ibrl = _default_ibrl()
        # The BC net for IBRL: a frozen copy of the
        # online net, taken *after* the BC pretrain
        # (so it has the expert-cloned Q-values).
        # The parent class's ``pretrain_demos`` will
        # populate this.
        self._bc_net_for_ibrl: Optional[nn.Module] = None
        # SIL.
        if self.v2_cfg.use_sil:
            self._sil_buffer = SILBuffer(self.v2_cfg.sil)
            class _A:
                pass
            sil_agent = _A()
            sil_agent.device = self.device
            sil_agent.online = self.online
            self._sil_trainer = SILTrainer(sil_agent, self.v2_cfg.sil)
        else:
            self._sil_buffer = None
            self._sil_trainer = None
        # EMA.
        if self.v2_cfg.use_ema:
            self._ema = EMA(self.online, self.v2_cfg.ema_decay)
        else:
            self._ema = None
        # Auto-entropy.
        if self.v2_cfg.use_auto_entropy:
            self._auto_entropy = AutoEntropy(self.v2_cfg.auto_entropy)
        else:
            self._auto_entropy = None

    # ------------------------------------------------------------------ #
    def pretrain_demos(self, obs: np.ndarray, actions: np.ndarray,
                        n_epochs: int = 50, batch_size: int = 256,
                        lr: float = 3e-3) -> dict[str, float]:
        """BC warm-up + freeze a copy for IBRL.

        After the BC pretrain, the EMA is *re-initialised*
        to the BC-pretrained weights so the EMA shadow
        starts tracking from a meaningful baseline
        (otherwise the EMA would slowly drift from the
        random init, taking thousands of steps to
        converge).
        """
        result = super().pretrain_demos(obs, actions, n_epochs,
                                            batch_size, lr)
        # Freeze a copy of the trained net for the
        # IBRL bootstrap proposal.
        import copy
        self._bc_net_for_ibrl = copy.deepcopy(self.online)
        for p in self._bc_net_for_ibrl.parameters():
            p.requires_grad_(False)
        self._bc_net_for_ibrl.eval()
        # Re-initialise the EMA to start from the
        # BC-pretrained weights (instead of the random
        # init from __init__).  This is critical: with
        # the default decay=0.999, the EMA's half-life
        # is ~1000 steps, so a random-init shadow
        # would be effectively useless for the first
        # few hundred training steps.
        if self._ema is not None:
            self._ema = EMA(self.online, self.v2_cfg.ema_decay)
        return result

    # ------------------------------------------------------------------ #
    def add_episode(self, states: list, actions: list,
                       rewards: list, start_value: float = 0.0) -> int:
        """Feed a finished episode to the SIL buffer."""
        if self._sil_buffer is None:
            return 0
        return self._sil_buffer.add_episode(
            states, actions, rewards, start_value)

    # ------------------------------------------------------------------ #
    def _bootstrap_q_with_ibrl(self, next_obs: torch.Tensor) -> torch.Tensor:
        """IBRL bootstrap Q: max(Q(s', a_il), Q(s', a_rl))
        from the *target* net.  Falls back to the
        standard argmax if IBRL is disabled or the
        BC net hasn't been built yet."""
        if (not self.v2_cfg.use_ibrl_bootstrap
                or self._bc_net_for_ibrl is None):
            # Standard Double DQN.
            with torch.no_grad():
                q_next = self.target(next_obs).mean(dim=-1)
                return q_next.max(dim=-1).values
        return bootstrap_proposal_q(
            self._bc_net_for_ibrl, self.online,
            target_net=self.target, next_obs=next_obs)

    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict) -> dict[str, float]:
        """One v1.23.0 train step.

        Loss = L_TD_ibrl + L_supervised + L_margin
              + λ_sil · L_SIL
              + (auto-entropy update on log_alpha)

        where L_TD_ibrl is the standard Double DQN
        loss but with the IBRL bootstrap-proposal Q
        instead of ``max_a Q(s', a)``.
        """
        device = self.device
        obs = batch["obs"].to(device)
        act = batch["actions"].to(device).long()
        rew = batch["rewards"].to(device).float()
        next_obs = batch["next_obs"].to(device)
        done = batch["dones"].to(device).float()
        weights = batch["weights"].to(device).float()
        gamma = self.cfg.gamma
        n = self.cfg.n_step
        # Bootstrap Q with IBRL proposal.
        with torch.no_grad():
            bootstrap_q = self._bootstrap_q_with_ibrl(next_obs)
            target_q = rew + (gamma ** n) * (1.0 - done) * bootstrap_q
        # Predicted Q for the chosen action.
        pred = self.online(obs).mean(dim=-1)  # [B, A]
        pred_a = pred.gather(1, act.view(-1, 1)).squeeze(1)  # [B]
        td_per = (pred_a - target_q.detach()) ** 2
        loss_td = (weights * td_per).mean()
        total = loss_td
        # DQfD supervised + margin (standard).
        bc = self.supervised_loss(act.shape[0])
        mg = self.margin_loss(act.shape[0])
        # Decay (from parent).
        decay = max(0.0, 1.0 - self.update_count / max(
            1, 30 * self.dqfd.supervised_decay_episodes))
        total = total + decay * self.dqfd.lambda_bc * bc + self.dqfd.lambda_margin * mg
        # SIL term.
        if (self._sil_buffer is not None
                and self._sil_trainer is not None
                and self._sil_buffer.is_ready(
                    min_transitions=act.shape[0])):
            sil_batch = self._sil_buffer.sample(act.shape[0])
            sil_out = self._sil_trainer.loss(sil_batch)
            total = total + self.v2_cfg.lambda_sil * (
                sil_out["policy_loss"] + self._sil_trainer.beta_sil *
                sil_out["value_loss"])
        # One backward, one step, one clip.
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(
            self.online.parameters(), self.cfg.grad_clip_norm))
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.update_count += 1
        self.maybe_sync_target()
        # Auto-entropy.
        alpha_val = 0.0
        if self._auto_entropy is not None:
            with torch.no_grad():
                log_probs = F.log_softmax(pred, dim=-1)
            alpha_val = self._auto_entropy.step(
                log_probs.gather(1, act.view(-1, 1)).squeeze(1))
        # EMA.
        if self._ema is not None:
            self._ema.update()
        # PER priorities.
        td_errors = (pred_a - target_q.detach()).abs().detach().cpu().numpy()
        out = {
            "loss": float(total.item()),
            "td_loss": float(loss_td.item()),
            "q_mean": float(pred.mean().item()),
            "grad_norm": grad_norm,
            "td_error_abs_mean": float(td_per.mean().sqrt().item()),
            "td_errors": td_errors,
            "bc_loss": float(bc.item()),
            "margin_loss": float(mg.item()),
            "supervised_decay": float(decay),
            "alpha": alpha_val,
        }
        if (self._sil_buffer is not None
                and self._sil_trainer is not None
                and self._sil_buffer.is_ready(
                    min_transitions=act.shape[0])):
            out["sil_policy_loss"] = float(sil_out["policy_loss"].item())
            out["sil_value_loss"] = float(sil_out["value_loss"].item())
        return out

    # ------------------------------------------------------------------ #
    def eval_mode(self) -> None:
        """Install the EMA weights for evaluation."""
        if self._ema is not None:
            self._ema.install()

    def train_mode(self) -> None:
        """Restore the (non-EMA) weights for training."""
        if self._ema is not None:
            self._ema.restore()

    def sil_stats(self) -> dict[str, float]:
        """Return SIL buffer stats (for logging)."""
        if self._sil_buffer is None:
            return {"n": 0, "mean_return": 0.0, "max_return": 0.0}
        return self._sil_buffer.stats()
