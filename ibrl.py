"""IBRL — Imitation Bootstrapped Reinforcement Learning.

DEEP-FIX v1.23.0
================

Reference: Hu et al. ICLR 2024 "Imitation Bootstrapped
Reinforcement Learning" (https://arxiv.org/abs/2311.02198).

Key idea
--------
Standard DQfD / RLPD / Pt-Ft all *share* a single
network between the BC policy and the RL policy.
That coupling forces the user to balance BC vs RL
losses (a fragile hyperparameter) and prevents
using different architectures for the two
networks (IBRL found ResNet-18 is best for BC but
a shallow ViT is best for RL — a single network
can't have both).

IBRL decouples them:

  • The **BC policy** (``bc_net``) is trained
    once on the expert demonstrations and frozen.
  • The **RL policy** (``rl_net``, a QR-DQN) is
    trained normally on the agent's own data.
  • At action time both propose an action; the
    agent picks the one with the higher
    online-Q-value (the *actor proposal* in
    Section 4 of the paper).
  • At bootstrap time the TD target uses
    ``max(Q(s', a_il), Q(s', a_rl))`` instead of
    ``max_a Q(s', a)`` (the *bootstrap proposal*).
    This single change is what gives IBRL 6.4× the
    success rate of RLPD on PickPlaceCan.

Why this matters here
---------------------
The BC+DQfD recipe on :class:`SyntheticGame`
plateaus at 12.5s (vs 22.22s expert).  The single
shared network is forced to *simultaneously*
match the expert's BC distribution *and* improve
via TD learning, and these two objectives pull
in opposite directions (the BC loss pins the
policy, the TD loss wants to explore).  IBRL
splits the work: BC stays close to the expert,
RL explores freely, and the agent picks the
better action at runtime.

Implementation
--------------
* :class:`IBRLAgent` — a thin wrapper around a
  *frozen* BC net + a *trainable* RL net.
  ``act(obs)`` implements the actor-proposal
  action selection; ``td_target(s')`` implements
  the bootstrap proposal.
* The :class:`IBRLTrainer` combines the standard
  TD loss with the IBRL bootstrap loss, the
  supervised BC loss (on the BC net, not the RL
  net — this is what makes IBRL modular), and
  optional self-imitation loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IBRLConfig:
    """Knobs for the IBRL wrapper.

    :param use_actor_proposal: True → at action
        time, both BC and RL propose an action and
        the one with the higher online Q is taken.
        The paper's headline result.
    :param use_bootstrap_proposal: True → at
        bootstrap time, the TD target is
        ``max(Q(s', a_il), Q(s', a_rl))`` instead
        of ``max_a Q(s', a)``.
    :param lambda_bc: weight on the BC loss.  The
        paper's default is 1.0 (i.e. equal to the
        RL loss).  We use 0.5 by default to match
        the DQfD recipe.
    :param noise_eps: when ``use_actor_proposal`` is
        True, with probability ``noise_eps`` the
        agent picks a *random* action instead of
        the actor-proposal choice (light
        exploration, like the paper).
    """
    use_actor_proposal: bool = True
    use_bootstrap_proposal: bool = True
    lambda_bc: float = 0.5
    noise_eps: float = 0.05


def _q_of_action(q_values: torch.Tensor,
                  action: int | torch.Tensor) -> torch.Tensor:
    """Return ``Q(s, a)`` for the given action(s).

    ``q_values`` is ``[A]`` or ``[B, A]``;
    ``action`` is an int or a ``[B]`` tensor.
    """
    if q_values.ndim == 1:
        return q_values[action]
    if isinstance(action, int):
        return q_values[:, action]
    return q_values.gather(1, action.view(-1, 1)).squeeze(1)


def actor_proposal_action(bc_net: nn.Module, rl_net: nn.Module,
                            obs: torch.Tensor,
                            noise_eps: float = 0.05) -> tuple[torch.Tensor,
                                                                  torch.Tensor]:
    """Compute the IBRL actor-proposal action.

    Both the BC net and the RL net propose an
    action; the agent picks the one with the
    higher online-Q-value.  Returns a tuple
    ``(action, info)`` where ``action`` is a
    ``[B]`` tensor and ``info`` carries
    diagnostics (``"bc_action"``, ``"rl_action"``,
    ``"proposal_source"``).

    Algorithm (paper Sec. 4.1):
    1. ``a_il = argmax Q_bc(s, ·)``
    2. ``a_rl = argmax Q_rl(s, ·)``
    3. ``if Q_rl(s, a_il) > Q_rl(s, a_rl) → a_il
       else → a_rl``
    4. With probability ``noise_eps``, replace
       the chosen action with a uniform random
       one (light exploration).
    """
    with torch.no_grad():
        # BC net proposes an action (no_grad because
        # the BC net is frozen).
        q_bc = bc_net(obs).mean(dim=-1) if bc_net(obs).ndim == 3 else bc_net(obs)
        a_il = q_bc.argmax(dim=-1)
        # RL net proposes an action.
        q_rl = rl_net(obs)
        if q_rl.ndim == 3:
            q_rl = q_rl.mean(dim=-1)
        a_rl = q_rl.argmax(dim=-1)
        # Pick the action with the higher RL Q.
        q_rl_il = _q_of_action(q_rl, a_il)
        q_rl_rl = _q_of_action(q_rl, a_rl)
        # If the RL net thinks the BC action is
        # better, use it.  Otherwise use the RL
        # action.
        use_il = q_rl_il > q_rl_rl
        chosen = torch.where(use_il, a_il, a_rl)
        # Light exploration.
        if noise_eps > 0:
            n_actions = q_rl.shape[-1]
            random_mask = (torch.rand(chosen.shape,
                                       device=chosen.device)
                            < noise_eps)
            random_action = torch.randint(0, n_actions,
                                            chosen.shape,
                                            device=chosen.device)
            chosen = torch.where(random_mask, random_action, chosen)
        info = {
            "bc_action": a_il,
            "rl_action": a_rl,
            "q_rl_il": q_rl_il,
            "q_rl_rl": q_rl_rl,
            "use_il": use_il.float(),
        }
    return chosen, info


def bootstrap_proposal_q(bc_net: nn.Module, rl_net: nn.Module,
                            target_net: Optional[nn.Module] = None,
                            next_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute the IBRL bootstrap-proposal Q-value.

    Returns ``[B]`` of Q-values used as the
    bootstrap target.  The paper's formula is
    ``Q_target(s, a*)`` where ``a* = argmax_a
    Q_target(s, a)`` from
    ``{a_il, a_rl}``.

    The expected value is
    ``max(Q(s', a_il), Q(s', a_rl))`` — the *max
    over the two candidates*, not the max over
    the whole action space.  This is what the
    paper calls the *bootstrap proposal*; it
    bounds the target Q-value to actions the
    agent *knows* about (i.e. the BC net has
    seen them or the RL net has learned about
    them), which prevents Q-value over-estimation.

    Parameters
    ----------
    bc_net, rl_net : nn.Module
        The BC and online RL networks.
    target_net : nn.Module, optional
        The *target* RL network.  When provided
        (the paper's standard), the bootstrap Q
        is evaluated using the target net (more
        stable).  When ``None`` (the default for
        unit tests), the online net is used.
    next_obs : torch.Tensor
        The next-state observations.  Must be
        provided.
    """
    if next_obs is None:
        raise ValueError("next_obs is required")
    with torch.no_grad():
        # BC net proposes an action.
        q_bc = bc_net(next_obs)
        if q_bc.ndim == 3:
            q_bc = q_bc.mean(dim=-1)
        a_il = q_bc.argmax(dim=-1)
        # RL net (target if available) proposes an action.
        eval_net = target_net if target_net is not None else rl_net
        q_rl = eval_net(next_obs)
        if q_rl.ndim == 3:
            q_rl = q_rl.mean(dim=-1)
        a_rl = q_rl.argmax(dim=-1)
        # The *target* net is the one used for the
        # bootstrap (the caller passes it via the
        # ``eval_net`` argument).  We compute
        # Q_target(s', a_il) and Q_target(s', a_rl)
        # and take the max.
        q_il = _q_of_action(q_rl, a_il)
        q_rl_at_rl = _q_of_action(q_rl, a_rl)
        return torch.maximum(q_il, q_rl_at_rl)


class IBRLAgent:
    """Combines a frozen BC net with a trainable
    RL net using the IBRL actor + bootstrap
    proposal.

    The class does not own a *trainer* — the
    trainer in :class:`IBRLTrainer` below is
    responsible for combining the standard TD
    loss with the BC + SIL losses.
    """

    def __init__(self, bc_net: nn.Module, rl_net: nn.Module,
                  cfg: Optional[IBRLConfig] = None) -> None:
        self.bc_net = bc_net
        self.rl_net = rl_net
        self.cfg = cfg or IBRLConfig()
        # Freeze the BC net.  The paper trains BC
        # once and never updates it again — the
        # RL net's job is to learn *when* to
        # trust BC and when to deviate.
        for p in self.bc_net.parameters():
            p.requires_grad_(False)
        self.bc_net.eval()

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """IBRL actor-proposal action selection.

        Returns ``(action, info)`` as in
        :func:`actor_proposal_action`.  When
        ``cfg.use_actor_proposal`` is False, the
        agent falls back to plain argmax of the
        RL net.
        """
        if self.cfg.use_actor_proposal:
            return actor_proposal_action(
                self.bc_net, self.rl_net, obs, self.cfg.noise_eps)
        with torch.no_grad():
            q_rl = self.rl_net(obs)
            if q_rl.ndim == 3:
                q_rl = q_rl.mean(dim=-1)
            action = q_rl.argmax(dim=-1)
            return action, {"use_il": torch.zeros_like(action,
                                                          dtype=torch.float)}

    def bootstrap_q(self, next_obs: torch.Tensor,
                      target_net: Optional[nn.Module] = None) -> torch.Tensor:
        """The IBRL bootstrap-proposal Q-value.

        Falls back to the standard Double-DQN
        argmax when ``cfg.use_bootstrap_proposal``
        is False.

        Parameters
        ----------
        next_obs : torch.Tensor
            The next-state observations.
        target_net : nn.Module, optional
            The target RL net.  When provided
            (the paper's standard), the bootstrap
            Q is evaluated using the target net
            (more stable).  When ``None``, the
            online net is used (this is what the
            unit tests do).
        """
        if self.cfg.use_bootstrap_proposal:
            return bootstrap_proposal_q(self.bc_net, self.rl_net,
                                           target_net=target_net,
                                           next_obs=next_obs)
        with torch.no_grad():
            eval_net = target_net if target_net is not None else self.rl_net
            q_rl = eval_net(next_obs)
            if q_rl.ndim == 3:
                q_rl = q_rl.mean(dim=-1)
            return q_rl.max(dim=-1).values


class IBRLTrainer:
    """One IBRL train step.

    Loss = L_TD + λ_bc · L_BC + λ_sil · L_SIL
    where L_TD uses the bootstrap-proposal target
    and L_BC is the standard cross-entropy on the
    BC net (with a *stop-gradient* on the BC net —
    the BC loss only updates the RL net, not the
    BC net itself; the BC net is re-trained
    separately from a frozen snapshot of the
    expert data).
    """

    def __init__(self, ibrl: IBRLAgent, optimizer: torch.optim.Optimizer,
                  lambda_bc: float = 0.5,
                  lambda_sil: float = 0.0,
                  sil_trainer: Optional["SILTrainer"] = None) -> None:
        self.ibrl = ibrl
        self.optimizer = optimizer
        self.lambda_bc = lambda_bc
        self.lambda_sil = lambda_sil
        self.sil_trainer = sil_trainer

    def loss(self, batch: dict,
              sil_batch: Optional[dict] = None) -> dict[str, torch.Tensor]:
        """Compute the IBRL joint loss.

        ``batch`` is the standard PER batch
        (``obs``, ``next_obs``, ``actions``,
        ``rewards``, ``dones``, ``weights``,
        ``gamma_pows``).  ``sil_batch`` is an
        optional output of
        :meth:`SILBuffer.sample` for the SIL term.
        """
        rl_net = self.ibrl.rl_net
        device = next(rl_net.parameters()).device
        obs = batch["obs"].to(device)
        next_obs = batch["next_obs"].to(device)
        actions = batch["actions"].to(device).long()
        rewards = batch["rewards"].to(device).float()
        dones = batch["dones"].to(device).float()
        weights = batch["weights"].to(device).float()
        gamma_pow = batch.get("gamma_pows",
                                torch.ones_like(rewards)).to(device)
        # TD loss with IBRL bootstrap proposal.
        # Pass next_obs explicitly (the new
        # bootstrap_proposal_q signature requires it).
        bootstrap_q = self.ibrl.bootstrap_q(next_obs)
        target = rewards + gamma_pow * (1.0 - dones) * bootstrap_q
        # Predicted Q for the chosen action.
        q_pred = rl_net(obs)
        if q_pred.ndim == 3:
            q_pred = q_pred.mean(dim=-1)
        q_a = q_pred.gather(1, actions.view(-1, 1)).squeeze(1)
        td_per = (q_a - target.detach()) ** 2
        loss_td = (weights * td_per).mean()
        loss = loss_td
        out = {"td_loss": loss_td, "loss": loss_td}
        # BC term: cross-entropy of the *RL net* on
        # the expert actions.  The BC loss pulls the
        # RL net's action distribution toward the
        # expert — this is the *BC anchor* that IBRL
        # inherits from DQfD.  The BC net is frozen
        # and the *only* effect of this term is to
        # constrain the RL net's logits.
        if self.lambda_bc > 0 and "bc_actions" in batch:
            bc_actions = batch["bc_actions"].to(device).long()
            # Use the *same* obs as the main loss.
            logits = q_pred
            loss_bc = F.cross_entropy(logits, bc_actions)
            loss = loss + self.lambda_bc * loss_bc
            out["bc_loss"] = loss_bc
        # SIL term: replay good past episodes.
        if (self.lambda_sil > 0 and sil_batch is not None
                and self.sil_trainer is not None):
            sil_out = self.sil_trainer.loss(sil_batch)
            loss = loss + self.lambda_sil * (
                sil_out["policy_loss"] + self.sil_trainer.beta_sil *
                sil_out["value_loss"])
            out["sil_policy_loss"] = sil_out["policy_loss"]
            out["sil_value_loss"] = sil_out["value_loss"]
        out["loss"] = loss
        return out

    def step(self, batch: dict,
              sil_batch: Optional[dict] = None,
              grad_clip: float = 10.0) -> dict[str, float]:
        """Compute the loss, backprop, step, return metrics."""
        out = self.loss(batch, sil_batch)
        self.optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        grad_norm = float(nn.utils.clip_grad_norm_(
            self.ibrl.rl_net.parameters(), grad_clip))
        self.optimizer.step()
        return {k: float(v.detach().item()) if torch.is_tensor(v)
                  else float(v) for k, v in out.items()} | {
            "grad_norm": grad_norm,
        }
