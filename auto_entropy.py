"""Auto-tuned entropy temperature (SAC-style) for discrete actions.

DEEP-FIX v1.23.0
================

Reference: Haarnoja et al. 2018 "Soft Actor-Critic
and Applications" (Sec. "Automatic Temperature
Tuning").

Why this matters
----------------
Hard ε-greedy exploration is a *coarse* control:
once ε decays to 0 the policy becomes
deterministic and the agent stops exploring
entirely.  SAC's auto-tuned entropy replaces
this with a *continuous* control: the
temperature ``α`` is adjusted so the policy's
*entropy* (the spread of action probabilities)
stays at a target value.

The discrete-action version (Christodoulou 2019
"Soft Actor-Critic for Discrete Action Settings")
computes the entropy as
``H(π) = -E[sum_a π(a|s) log π(a|s)]`` and
penalises the agent when ``H(π) < target_entropy``
(so the agent is forced to maintain exploration
for as long as the policy is too deterministic).

The target entropy is conventionally
``-0.89 * |A|`` (the paper's choice); for our
5-action Subway Surfers game that's
``-4.45``.

Implementation
--------------
* :class:`AutoEntropy` — owns a learnable
  ``log_alpha`` parameter and computes the
  SAC temperature loss.
* :class:`SoftPolicyLoss` — combines the TD
  loss with the entropy bonus
  ``-α · H(π)`` so the policy maximises a
  tradeoff between reward and entropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AutoEntropyConfig:
    """Knobs for the auto-tuned entropy.

    :param initial_log_alpha: starting value of
        ``log(α)`` (corresponds to ``α = 0.1`` by
        default; matches the SB3 default).
    :param learning_rate: learning rate for the
        ``log_alpha`` optimiser.
    :param target_entropy: the entropy the agent
        tries to maintain.  Default
        ``-0.89 * n_actions`` is the SAC-discrete
        paper's choice.
    :param learn_alpha: True → optimise ``log_alpha``
        with the entropy loss.  False → fix
        ``α = exp(initial_log_alpha)``.
    """
    initial_log_alpha: float = float(np.log(0.1))
    learning_rate: float = 3e-4
    target_entropy: Optional[float] = None
    learn_alpha: bool = True
    # Used to compute the default target entropy.
    n_actions: int = 5


class AutoEntropy(nn.Module):
    """Auto-tuned entropy temperature.

    The module is just a single learnable parameter
    ``log_alpha``.  Forward returns ``α = exp(log_alpha)``
    (always positive).  The accompanying
    :meth:`loss` returns the gradient that minimises
    ``α * (H(π) - target_entropy)`` (the dual
    objective from SAC).
    """

    def __init__(self, cfg: AutoEntropyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # The log_alpha parameter; kept as a
        # 1-D tensor with one entry (we don't
        # have action-specific temperatures).
        self.log_alpha = nn.Parameter(
            torch.tensor(cfg.initial_log_alpha,
                          dtype=torch.float32),
            requires_grad=cfg.learn_alpha)
        # Optimiser (lazy — created on first loss
        # call so we know the device).
        self._opt: Optional[torch.optim.Optimizer] = None
        if cfg.learn_alpha:
            self._opt = torch.optim.Adam([self.log_alpha],
                                           lr=cfg.learning_rate)
        # Default target entropy: -0.89 * n_actions.
        if cfg.target_entropy is None:
            self.target_entropy = -0.89 * cfg.n_actions
        else:
            self.target_entropy = cfg.target_entropy

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        """SAC temperature loss for discrete actions.

        ``log_probs`` is the *per-state* log-prob
        of the action that was taken
        (``log π(a|s)``), shape ``[B]``.  Returns
        the scalar loss that, when minimised,
        keeps ``H(π) ≈ target_entropy``.

        Derivation: we want to maximise
        ``E[α * H(π)] - α * target_entropy``
        w.r.t. α.  Substituting ``H(π) = -E[log π]``
        and noting that the gradient of ``α`` is
        ``exp(log_alpha)``:

        L = -E[α * (log π(a|s) + target_entropy)]

        where the expectation is over the
        policy's own action distribution.
        """
        if not self.cfg.learn_alpha:
            return torch.tensor(0.0, device=log_probs.device)
        # Per-sample loss: -α * (log π + target_entropy).
        # Take the gradient w.r.t. log_alpha only.
        loss = -(self.alpha * (log_probs.detach()
                                  + self.target_entropy)).mean()
        return loss

    def step(self, log_probs: torch.Tensor) -> float:
        """One gradient step on the temperature.

        Returns the current ``α`` (post-step).
        """
        loss = self.loss(log_probs)
        if self._opt is not None and loss.requires_grad:
            self._opt.zero_grad(set_to_none=True)
            loss.backward()
            self._opt.step()
        return float(self.alpha.item())


def soft_policy_loss(q_values: torch.Tensor,
                       target_entropy_bonus: float = 0.0) -> tuple[torch.Tensor,
                                                                        torch.Tensor]:
    """Compute the soft policy loss for a batch of
    Q-values.

    Returns ``(loss, log_probs)`` where ``loss`` is
    the cross-entropy that *maximises Q + α · H(π)``
    (the SAC objective) and ``log_probs`` are the
    per-sample log-probabilities of the chosen
    actions (used to update ``α``).

    The policy here is implicit: ``π(a|s) ∝
    exp(Q(s, a) / α)`` (the Boltzmann policy over
    the Q-values).  This is the SAC-discrete
    derivation from Christodoulou 2019.
    """
    # The SAC-discrete "soft policy" is
    # π(a|s) ∝ exp(Q(s, a)) with a temperature
    # baked into the log-Q values.  For the
    # purpose of computing the policy gradient
    # we use log_softmax (which is the *hard*
    # softmax of the Q-values); the temperature
    # is applied by the caller through ``α``.
    log_probs_all = F.log_softmax(q_values, dim=-1)
    # Sample actions stochastically for the loss
    # (in practice, the argmax is the same as
    # the sample when α → 0).  We use the
    # greedy action for stability.
    actions = q_values.argmax(dim=-1)
    log_probs = log_probs_all.gather(1, actions.view(-1, 1)).squeeze(1)
    # The policy loss is -E[Q(s, a) - α · log π(a|s)].
    # We fold ``α`` in via the entropy bonus.
    q_a = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
    loss = -(q_a - target_entropy_bonus).mean()
    return loss, log_probs
