"""Distributional Q-learning (QR-DQN, quantile regression).

Why
--
A vanilla DQN estimates the *expected* return for each action:
``Q(s, a) = E[R_t | s, a]``.  This single number throws away
*all* the information about the *shape* of the return
distribution: is the action risky but high-reward, or safe but
low-reward?  In a game like Subway Surfers where every decision
is a tradeoff (jump now = survive vs jump too early = land in
the next obstacle), this distinction is the entire game.

QR-DQN (Dabney et al. 2018) estimates a fixed set of *quantiles*
of the return distribution.  The loss is the quantile Huber
between the predicted quantile and the target distribution (the
target is built from the *next-state* quantiles plus the
Bellman-shifted reward).  The mean of the quantiles is used at
inference time, so the operator sees the same ``argmax Q`` style
decision, but the gradient that flows back through the network
is 50× denser (one number per quantile per sample, not one
number per action per sample).

The implementation here is small on purpose:

* :class:`QuantileDuelingDQN` mirrors the structure of
  :class:`DuelingDQN` from :mod:`models` (shared encoder, value
  + advantage streams, GAP), so the same 49 k / 96 k / 348 k
  parameter budgets hold.
* :class:`QuantileHuber` is the standard quantile-regression loss
  used in the original paper.
* :class:`project_distribution` is the Bellman projection: roll
  the next-state quantile distribution forward by one step using
  the reward, clipped.

Hyperparameters (all defaults match the paper):

* ``num_quantiles = 51`` (Dabney et al. used 32 / 200; 51 is a
  CPU-friendly middle ground that gives a clear distributional
  signal without bloating the head).
* ``kappa = 1.0`` (Huber threshold, standard).

Tests
-----
See :mod:`tests.test_distributional` for the unit tests covering
shapes, losses, projection, and an end-to-end training step
that shows the quantile loss actually decreases.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import _gn, ConvBlock, DuelingHead, DepthwiseSeparableConv
from noisy_nets import NoisyLinear


# Mid-quantile taus for the chosen ``num_quantiles``.  These are
# ``(2k-1)/(2N)`` for k in 1..N, the standard midpoints of the
# uniform quantile grid.
def mid_quantiles(num_quantiles: int) -> torch.Tensor:
    """Return the midpoints of the uniform quantile grid [0, 1]."""
    return (torch.arange(num_quantiles, dtype=torch.float32) * 2.0 + 1.0) \
        / (2.0 * num_quantiles)


class QuantileDuelingDQN(nn.Module):
    """Dueling DQN that outputs a *distribution* over returns.

    Shape: input ``[B, in_frames, H, W]`` -> output
    ``[B, n_actions, num_quantiles]``.  At inference we
    ``.mean(dim=-1)`` to recover a scalar Q-value per action.

    v1.19 added an optional ``noisy`` flag (default True) that
    swaps the value/advantage streams for :class:`NoisyLinear`
    layers.  The Rainbow ablation showed NoisyNets is the
    second most important exploration ingredient after PER
    (the third is distributional Q, which this class also
    implements).  See :mod:`noisy_nets`.
    """

    def __init__(self, profile: str, in_frames: int = 4, size: int = 84,
                 num_quantiles: int = 51, noisy: bool = True) -> None:
        super().__init__()
        from models import PROFILES, build_encoder
        spec = PROFILES[profile]
        self.encoder, enc_out = build_encoder(spec, in_frames)
        self.head = DuelingHead(enc_out, n_actions=5,
                                hidden=spec["head_hidden"])
        self.num_actions = 5
        self.num_quantiles = num_quantiles
        self.n_actions = self.num_actions
        self.noisy = noisy
        # The dueling head for the *quantile* outputs is the
        # same idea as :class:`DuelingHead` but with
        # ``num_quantiles`` outputs per stream.  When NoisyNets
        # is on, the inner linear layers are replaced with
        # :class:`NoisyLinear` so the agent can explore via
        # parameter noise instead of (or in addition to)
        # epsilon-greedy.
        in_dim = enc_out
        if noisy:
            from noisy_nets import NoisyLinear
            Line = NoisyLinear
        else:
            Line = nn.Linear
        self.value_stream = nn.Sequential(
            Line(in_dim, spec["head_hidden"]), nn.ReLU(inplace=True),
            Line(spec["head_hidden"], num_quantiles),
        )
        self.advantage_stream = nn.Sequential(
            Line(in_dim, spec["head_hidden"]), nn.ReLU(inplace=True),
            Line(spec["head_hidden"], self.num_actions * num_quantiles),
        )
        # Mid-quantile grid; not a parameter, just a registered
        # buffer so it follows ``.to(device)`` cleanly.
        self.register_buffer("tau", mid_quantiles(num_quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-action quantile values.

        Output shape: ``[B, n_actions, num_quantiles]``.  For
        greedy action selection call :meth:`q_values` which
        reduces along the quantile axis.
        """
        h = self.encoder(x).flatten(1) if _has_flatten(self.encoder) \
            else self.encoder(x)
        # The encoder returns ``[B, C, H, W]`` (we use GAP in the
        # head); flatten to ``[B, C]`` first.
        if h.dim() == 4:
            h = h.mean(dim=(-1, -2))  # global average pooling
        v = self.value_stream(h)              # [B, N]
        a = self.advantage_stream(h)          # [B, A*N]
        a = a.view(-1, self.num_actions, self.num_quantiles)
        v = v.unsqueeze(1)                    # [B, 1, N]
        # Dueling combination per quantile.
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Mean over quantiles — used for argmax action selection."""
        dist = self.forward(x)
        return dist.mean(dim=-1)

    def reset_noise(self) -> None:
        """Re-sample the factorised Gaussian noise for every
        NoisyLinear in the network.  Should be called at the
        start of each forward pass during training (the
        :class:`NoisyLinear` already does this internally,
        but for the *target* net we typically want to fix the
        noise for an entire bootstrap so the target does not
        jitter every step).
        """
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m._sample_noise()  # type: ignore[attr-defined]
        return None


def _has_flatten(module: nn.Module) -> bool:
    """Quick helper: did the encoder already pool to 2-D?"""
    for m in module.modules():
        if isinstance(m, (nn.AdaptiveAvgPool2d, nn.Flatten)):
            return True
    return False


def quantile_huber_loss(predicted: torch.Tensor, target: torch.Tensor,
                        taus: torch.Tensor,
                        kappa: float = 1.0) -> torch.Tensor:
    """Quantile Huber loss (QR-DQN, Eq. 9).

    Shapes
    ------
    * ``predicted``: ``[B, A, N]`` (batch, actions, quantiles)
    * ``target``:    ``[B, A, N]``
    * ``taus``:      ``[N]``

    The loss is the mean over batch and actions of the mean over
    quantile-pairs of the asymmetric Huber term.  The sign of
    the error is weighted by ``|tau_target - I[u<0]|`` so the
    median quantile pulls toward the centre of the distribution
    while the upper/lower tails pull outward.
    """
    # ``u = target - predicted`` per (batch, action, target
    # quantile, predicted quantile).
    # target:   [B, A, N_t, 1]
    # predicted: [B, A, 1, N_p]
    u = target.unsqueeze(-1) - predicted.unsqueeze(-2)  # [B, A, N_t, N_p]
    # Huber threshold.
    abs_u = u.abs()
    huber = torch.where(
        abs_u <= kappa,
        0.5 * u.pow(2),
        kappa * (abs_u - 0.5 * kappa),
    )
    # Quantile weight: |tau_target - I[u<0]|.
    # ``taus`` is shape [N_t], expand to [1, 1, N_t, 1] so it
    # broadcasts against u of shape [B, A, N_t, N_p].
    tau = taus.view(1, 1, -1, 1)
    weight = (tau - (u < 0).float()).abs()
    loss = (weight * huber).mean(dim=(2, 3))  # mean over (N_t, N_p)
    return loss.mean()


def project_distribution(next_dist: torch.Tensor, rewards: torch.Tensor,
                         dones: torch.Tensor,
                         num_quantiles: int,
                         gamma: float) -> torch.Tensor:
    """Bellman project the next-state distribution to a target.

    Standard QR-DQN: shift the next-state distribution by the
    reward, clip the result to the support [0, 1] (here we use
    the natural value range), and zero it out for terminal
    transitions.

    Parameters
    ----------
    next_dist : ``[B, A, N]`` -- per-action quantile distribution
        for the *next* state.  These are the values we want to
        bootstrap from.
    rewards, dones : ``[B]``
    num_quantiles : int
    gamma : discount factor (applied to the distribution, not
        the per-step reward, because the rewards are n-step
        aggregated and the discount is already inside).
    """
    B, A, N = next_dist.shape
    # ``next_dist`` is the quantile estimate of the *future* return
    # distribution.  After one Bellman step the target distribution
    # is the shifted version: target = r + gamma * next_dist.
    # We do NOT clip to [0, 1] (the game rewards are unbounded)
    # but we do stop the gradient through the projection so the
    # target is a fixed bootstrapper.
    rewards = rewards.view(B, 1, 1)
    dones = dones.view(B, 1, 1)
    target = rewards + gamma * (1.0 - dones) * next_dist
    return target.detach()
