"""NoisyNet layers for parameter-space exploration.

Replaces the standard ``nn.Linear`` in the dueling head with a
*noisy* linear layer whose weights are perturbed by a
factorised Gaussian noise.  During training the network learns
the *mean* of its weights and the *noise scale*; at inference
the noise is zeroed so the action selection is deterministic.

Why NoisyNets beat ε-greedy
----------------------------
In a game like Subway Surfers, where every decision is a
tradeoff (jump NOW vs jump LATER vs lane-shift), an ε-greedy
agent wastes 10-30% of its decisions on completely random
actions — but a randomly-jumping agent looks identical to one
that commits to a strategy and gets unlucky.  NoisyNets
inject noise *into the policy itself*: a "noisy forward pass"
is equivalent to a soft policy mixture where the agent
explores around its current best action.  This is especially
helpful in the first 100 episodes when the value estimates
are still noisy and the agent needs to *commit* to a lane
choice long enough to learn whether it pays off.

The implementation here follows the standard NoisyNet paper
(Fortunato et al. 2018): a single noisy linear is split into
two halves (``mu`` and ``sigma``), and the noise is
factorised (one Gaussian for the input, one for the output)
to keep the per-layer cost down to ``O(in + out)`` random
draws instead of ``O(in * out)``.

The :class:`NoisyDuelingDQN` wrapper replaces the dueling
head's value/advantage streams with noisy versions, so the
rest of the QR-DQN agent is unchanged.  See
:mod:`tests.test_noisy_nets` for the unit tests.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorised Gaussian Noisy linear layer (Fortunato et al.)."""

    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Mean weights and biases — these are the *learnable*
        # part of the layer.  Initialised with the Kaiming
        # uniform so the network starts with a sensible policy
        # (we don't want to depend on the noise at init time).
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), sigma_init / math.sqrt(in_features))
        )
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), sigma_init / math.sqrt(out_features))
        )
        # Buffers for the noise — they are NOT parameters and
        # should not be optimised, but they need to follow
        # ``.to(device)``.  Buffers are perfect for this.
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming uniform init for the mean weights; zero noise
        at start so the layer behaves like a plain linear until
        the optimiser learns the noise scale."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # ``bias_mu`` is already zero; the Kaiming fan-in
        # formula is computed in the constructor.

    def _sample_noise(self) -> None:
        """Sample factorised Gaussian noise.

        Each weight is the product of an input-noise (one per
        in_feature) and an output-noise (one per out_feature).
        This is the "factorised" trick: drawing two vectors
        of size ``in`` and ``out`` and taking the outer
        product is O(in + out), not O(in * out).
        """
        if self.training:
            eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
            eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
            # ``sign * sqrt(abs)`` is the standard factorised
            # Gaussian transform from the paper.
            eps_in = eps_in.sign() * eps_in.abs().sqrt()
            eps_out = eps_out.sign() * eps_out.abs().sqrt()
            self.weight_epsilon = eps_out.outer(eps_in)
            self.bias_epsilon = eps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sampled noise.

        At inference time (``.eval()``), the noise buffers
        are zero so this layer is identical to a plain linear
        with the *mean* weights — the action selection is
        deterministic.
        """
        self._sample_noise()
        # Rescale the noise by the sigma parameters: this is
        # what makes the noise trainable (the network can
        # learn to "turn down" the noise on a specific output
        # if that output is already well-trained).
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}")


class NoisyDuelingHead(nn.Module):
    """Dueling head with noisy value/advantage streams.

    Mirrors the layout of :class:`models.DuelingHead` but
    replaces the inner linear layers with :class:`NoisyLinear`
    so the *only* source of exploration is the network itself
    — no ε-greedy needed.
    """

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.n_actions = n_actions
        # Value stream: state -> scalar value.
        self.value_stream = nn.Sequential(
            NoisyLinear(in_dim, hidden), nn.ReLU(inplace=True),
            NoisyLinear(hidden, 1),
        )
        # Advantage stream: state -> per-action advantage.
        self.advantage_stream = nn.Sequential(
            NoisyLinear(in_dim, hidden), nn.ReLU(inplace=True),
            NoisyLinear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        # Dueling combination (per-element, not per-quantile).
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q
