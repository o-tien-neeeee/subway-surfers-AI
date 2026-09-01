"""AvgL1Norm — the state-action representation normaliser from TD7.

DEEP-FIX v1.23.0
================

Reference: Fujimoto et al. 2023 "For SALE: State-Action
Representation Learning for Deep Reinforcement Learning"
(TD7) — Section 3.2, "AvgL1Norm".

Why this matters
----------------
A recurrent failure mode of TD-based agents is
**scale drift**: the Q-network's output grows
without bound as the policy improves, because
the TD target keeps getting larger and the
optimiser chases it.  This causes the agent to
become increasingly over-confident and ultimately
unstable (a 1000× Q-value blowup is common).

TD7's key insight is that the scale of the
*state embedding* (and the *state-action
embedding*) is the real source of the drift.
If we normalise the embedding **before** the
dueling head, the Q-values stay in a
predictable range, and the agent can learn
without scale-induced instability.

The normalisation is **AvgL1Norm** — divide each
component by the average L1 norm of the vector:

    y_i = x_i / mean_j(|x_j|)

This is cheaper than LayerNorm (no mean
subtraction, no epsilon) and empirically beats
LayerNorm in TD7's ablation (Figure 5 of the
paper).  The intuition is that the mean absolute
value is a single scale parameter, so the
optimisation landscape stays smooth.

Implementation
--------------
* :class:`AvgL1Norm1d` — the normaliser, applied
  along the last dimension of a tensor.
* :class:`AvgL1NormLinear` — a linear layer
  followed by AvgL1Norm, as used in the TD7
  critic (the paper's "Norm 1" layer).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AvgL1Norm(nn.Module):
    """Average L1 normalisation.

    ``y = x / mean(|x|)``

    with a small ``eps`` to avoid division by zero.
    The output has unit mean absolute value along
    the last dimension — a single scale parameter
    that the optimiser does not have to learn.

    Parameters
    ----------
    dim : int
        The dimension to normalise over.  Default
        ``-1`` (the last dim, matching TD7's
        feature-vector convention).
    eps : float
        A small constant added to the denominator
        for numerical stability.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average L1 norm along the chosen dim.
        l1 = x.abs().mean(dim=self.dim, keepdim=True)
        return x / (l1 + self.eps)


class AvgL1NormLinear(nn.Module):
    """Linear -> AvgL1Norm wrapper.

    This is the *Norm 1* layer in TD7's critic
    (the first linear after the state-action
    concatenation).  It is also what we use
    before the dueling head to keep the Q-value
    scale bounded.
    """

    def __init__(self, in_features: int, out_features: int,
                  bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features,
                                  bias=bias)
        self.norm = AvgL1Norm(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


def avg_l1_normalize(x: torch.Tensor, dim: int = -1,
                        eps: float = 1e-6) -> torch.Tensor:
    """Functional interface to :class:`AvgL1Norm`."""
    l1 = x.abs().mean(dim=dim, keepdim=True)
    return x / (l1 + eps)
