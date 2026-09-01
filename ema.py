"""EMA (Exponential Moving Average) of network weights.

DEEP-FIX v1.23.0
================

Reference: TD7 (Fujimoto et al. 2023) — "Policy
Checkpoints" + Tarasov et al. 2024 "Deep REINFORCE
Exceeds Human Performance in Atari" — both use
**EMA of network weights** for evaluation to
reduce variance and improve stability.

Why this matters
----------------
A common observation in deep RL is that the
*current* network's Q-values oscillate as the
optimiser chases the TD target.  At any given
training step the network might be in a
high-variance transient state (the loss is
decreasing but the weights are still moving).
A *smoothed* version of the weights (an
exponential moving average over the last N
updates) is consistently more accurate and
more stable, especially in evaluation.

The recipe (Tarasov et al. 2024):
  1. Every training step, after the optimiser
     update, compute
     ``ema_w = (1 - decay) * ema_w + decay * w``
     for every parameter.
  2. For *evaluation*, swap in the EMA weights,
     evaluate, swap back.
  3. ``decay = 0.999`` is the standard choice
     (about 1000-step half-life).

Implementation
--------------
* :class:`EMA` — a small wrapper that owns a
  copy of the source network's parameters,
  updates the copy after every ``update()``
  call, and provides ``load_into`` /
  ``restore_from`` to swap weights in and out.
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of network
    parameters.

    The EMA is *not* a separate network — it is
    just a dict of parameter tensors that are
    updated after every training step.  This is
    the Tarasov-style "shadow weights" approach,
    which is cheaper than maintaining a second
    network and avoids the memory duplication
    of full checkpoints.
    """

    def __init__(self, source: nn.Module, decay: float = 0.999) -> None:
        self.source = source
        self.decay = float(decay)
        # ``shadow`` is a dict of cloned
        # parameters.  We do NOT register them
        # as buffers (the source module's
        # state_dict would then include them,
        # which is not what we want).
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in source.named_parameters()
        }
        # ``_installed`` is True when the EMA
        # weights are currently loaded into
        # ``source`` (i.e. evaluation mode).
        self._installed = False
        # ``_backup`` holds the original (non-EMA)
        # parameters while the EMA is installed.
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        """Pull ``source``'s parameters into the
        EMA shadow (one step of EMA averaging)."""
        for name, param in self.source.named_parameters():
            if name not in self.shadow:
                # A new parameter appeared (rare;
                # e.g. after adding a layer).  Init
                # the shadow to the current value.
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay)

    def install(self) -> None:
        """Swap the EMA weights into ``source``.

        The original (non-EMA) weights are saved
        in ``self._backup`` so :meth:`restore`
        can put them back.
        """
        if self._installed:
            return
        self._backup = {
            name: param.detach().clone()
            for name, param in self.source.named_parameters()
        }
        with torch.no_grad():
            for name, param in self.source.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name])
        self._installed = True

    def restore(self) -> None:
        """Swap the original (non-EMA) weights
        back into ``source``."""
        if not self._installed:
            return
        with torch.no_grad():
            for name, param in self.source.named_parameters():
                if name in self._backup:
                    param.copy_(self._backup[name])
        self._backup = {}
        self._installed = False

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the EMA's shadow weights as a
        state dict (for serialisation)."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Load EMA weights from a state dict."""
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
            else:
                self.shadow[k] = v.clone()
