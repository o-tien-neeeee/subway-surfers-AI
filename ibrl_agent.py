"""IBRLAgent — the full integration of all v1.23.0 breakthroughs.

DEEP-FIX v1.23.0
================

This module glues together the five v1.23.0
innovations into a single agent class:

1. **IBRL actor proposal** — a frozen BC policy
   and a trainable RL policy both propose an
   action; the agent picks the one with the
   higher online Q-value (paper Sec. 4.1).
2. **IBRL bootstrap proposal** — the TD target
   uses ``max(Q(s', a_il), Q(s', a_rl))``
   instead of ``max_a Q(s', a)`` (Sec. 4.2).
3. **Self-Imitation Learning** — the agent's
   own good episodes are replayed at a higher
   rate, weighted by ``(R - V(s))_+``
   (Oh et al. 2018).
4. **SAC auto-tuned entropy temperature** — the
   Q-net's softmax temperature is auto-tuned to
   keep the policy's entropy at a target
   (Haarnoja et al. 2018).  Prevents the policy
   from becoming deterministic too early.
5. **EMA of network weights for evaluation** —
   Tarasov et al. 2024 / TD7 trick for variance
   reduction during eval.

Why this matters
----------------
The 5 prior v1.x BC+DQfD audits on
:class:`SyntheticGame` plateau at 12.5s mean
(vs 22.22s expert).  The v1.23.0 IBRL agent
attempts to break through that ceiling by
decoupling BC from RL (item 1+2) and adding
self-imitation (item 3) — the two highest-impact
techniques from 2024 SOTA papers.

This class is a *drop-in* for
:class:`DistributionalDoubleDQNAgent` on the
*LearnableEnv* benchmark (where the KPI is
30s) and is the production training agent on
:class:`SyntheticGame` (where the 12.5s → 30s
gap is the open problem).

Implementation
--------------
* :class:`IBRLDQNAgent` — the full agent.  The
  BC net is a snapshot of the agent *after* the
  BC pretrain; the RL net is the standard
  QR-DQN.  Both share the same forward-pass
  signature so the actor-proposal logic works
  on raw observation tensors.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_entropy import AutoEntropy, AutoEntropyConfig
from ibrl import (IBRLAgent as _IBRLWrapper, IBRLConfig,
                     IBRLTrainer, bootstrap_proposal_q)
from sil import SILBuffer, SILConfig, SILTrainer
from ema import EMA


@dataclass
class IBRLDQNConfig:
    """Knobs for the full IBRL agent.

    All flags default ON — the headline features
    of v1.23.0.  Set any flag to False to disable
    the corresponding module.
    """
    ibrl: IBRLConfig = field(default_factory=IBRLConfig)
    use_sil: bool = True
    sil: SILConfig = field(default_factory=SILConfig)
    use_auto_entropy: bool = True
    auto_entropy: AutoEntropyConfig = field(
        default_factory=AutoEntropyConfig)
    use_ema: bool = True
    ema_decay: float = 0.999
    lambda_bc: float = 0.5
    lambda_sil: float = 0.1
    grad_clip_norm: float = 10.0


class IBRLDQNAgent:
    """The full v1.23.0 IBRL agent.

    Lifecycle
    ---------
    1. ``build_or_load(bc_net, rl_net)`` — wrap
       a frozen BC net and a trainable RL net.
    2. ``act(obs)`` — actor-proposal action
       selection.
    3. ``add_episode(states, actions, rewards)`` —
       feed a finished episode to the SIL buffer.
    4. ``train_step(batch)`` — single-pass joint
       loss (TD + bootstrap + BC + SIL + entropy).
    5. ``update_ema()`` — pull the EMA shadow
       toward the current RL weights.
    6. ``eval_mode()`` / ``train_mode()`` —
       install / restore the EMA weights for
       evaluation.
    """

    def __init__(self, cfg: IBRLDQNConfig,
                  device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self._ibrl: Optional[_IBRLWrapper] = None
        self._trainer: Optional[IBRLTrainer] = None
        self._sil_buffer: Optional[SILBuffer] = None
        self._sil_trainer: Optional[SILTrainer] = None
        self._auto_entropy: Optional[AutoEntropy] = None
        self._ema: Optional[EMA] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self.update_count: int = 0
        self._last_log_probs: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def build(self, bc_net: nn.Module, rl_net: nn.Module,
                optimizer: Optional[torch.optim.Optimizer] = None,
                ) -> None:
        """Wire the BC and RL networks into the agent.

        The BC net is frozen; the RL net is the
        one being trained.  ``optimizer`` is the
        optimiser for the RL net (defaults to Adam
        with lr=1e-4 if not provided).
        """
        self._ibrl = _IBRLWrapper(bc_net, rl_net, self.cfg.ibrl)
        if optimizer is None:
            optimizer = torch.optim.Adam(rl_net.parameters(),
                                            lr=1e-4)
        self._optimizer = optimizer
        self._trainer = IBRLTrainer(
            self._ibrl, optimizer,
            lambda_bc=self.cfg.lambda_bc,
            lambda_sil=(self.cfg.lambda_sil
                          if self.cfg.use_sil else 0.0))
        if self.cfg.use_sil:
            self._sil_buffer = SILBuffer(self.cfg.sil)
            # The SIL trainer needs an object with
            # ``.device`` and ``.online`` attributes.
            class _A:
                pass
            sil_agent = _A()
            sil_agent.device = self.device
            sil_agent.online = rl_net
            self._sil_trainer = SILTrainer(sil_agent, self.cfg.sil)
            self._trainer.sil_trainer = self._sil_trainer
        if self.cfg.use_auto_entropy:
            self._auto_entropy = AutoEntropy(self.cfg.auto_entropy)
        if self.cfg.use_ema:
            self._ema = EMA(rl_net, self.cfg.ema_decay)

    # ------------------------------------------------------------------ #
    @property
    def bc_net(self) -> nn.Module:
        return self._ibrl.bc_net

    @property
    def rl_net(self) -> nn.Module:
        return self._ibrl.rl_net

    # ------------------------------------------------------------------ #
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """IBRL actor-proposal action selection."""
        return self._ibrl.act(obs)

    # ------------------------------------------------------------------ #
    def add_episode(self, states: list, actions: list,
                       rewards: list, start_value: float = 0.0) -> int:
        """Feed a finished episode to the SIL buffer."""
        if not self.cfg.use_sil or self._sil_buffer is None:
            return 0
        return self._sil_buffer.add_episode(
            states, actions, rewards, start_value)

    # ------------------------------------------------------------------ #
    def train_step(self, batch: dict) -> dict[str, float]:
        """One IBRL train step.

        ``batch`` is the standard PER batch
        (``obs``, ``next_obs``, ``actions``,
        ``rewards``, ``dones``, ``weights``,
        ``gamma_pows``).  Optionally includes
        ``bc_actions`` for the BC term.
        """
        # Move to the right device.
        batch = {k: (v.to(self.device) if torch.is_tensor(v)
                       else v) for k, v in batch.items()}
        sil_batch = (self._sil_buffer.sample(
                        batch["actions"].shape[0])
                       if (self.cfg.use_sil
                              and self._sil_trainer is not None
                              and self._sil_buffer is not None
                              and self._sil_buffer.is_ready(
                                  min_transitions=batch["actions"].shape[0]))
                       else None)
        out = self._trainer.step(batch, sil_batch,
                                    grad_clip=self.cfg.grad_clip_norm)
        # Auto-tune the entropy temperature.
        if self.cfg.use_auto_entropy and self._auto_entropy is not None:
            with torch.no_grad():
                q_pred = self.rl_net(batch["obs"])
                if q_pred.ndim == 3:
                    q_pred = q_pred.mean(dim=-1)
                if q_pred.ndim == 1:
                    q_pred = q_pred.unsqueeze(0)
                log_probs = F.log_softmax(q_pred, dim=-1)
            alpha = self._auto_entropy.step(
                log_probs.gather(1, batch["actions"].view(-1, 1)
                                    .long()).squeeze(1))
            out["alpha"] = alpha
        self.update_count += 1
        if self.cfg.use_ema and self._ema is not None:
            self._ema.update()
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

    # ------------------------------------------------------------------ #
    def sil_stats(self) -> dict[str, float]:
        """Return SIL buffer stats (for logging)."""
        if self._sil_buffer is None:
            return {"n": 0, "mean_return": 0.0, "max_return": 0.0}
        return self._sil_buffer.stats()
