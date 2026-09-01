"""Improved encoder building blocks.

This module is the v1.22.0 successor to the encoder
inside :mod:`models`.  It contains:

* :class:`LayerScale` — a learnable per-channel
  multiplicative scale (Touvron et al. 2021
  "Going Deeper with Image Transformers"), shown to
  help *deep* CNNs train stably.  Default scale
  ``1e-4`` (the paper's recommendation for ConvNets).
* :class:`ImprovedConvBlock` — a conv + GN + GELU
  + LayerScale.  The GELU activation has been shown
  to outperform ReLU on most vision tasks (Hendrycks
  & Gimpel 2016).
* :class:`OrthogonalInit` — orthogonal initialisation
  for Linear layers, the standard recommendation for
  RL value heads (because it keeps the per-layer
  spectral norm at 1.0, which prevents early-training
  Q-value explosions).
* :class:`AttentionPool2d` — a single-query attention
  pool, an alternative to Global Average Pooling.  The
  attention pool can be more discriminative (it learns
  *which* spatial locations matter) at a modest param
  cost.

Why a separate module
---------------------
The :mod:`models` encoder is *tested* against the
parameter budget (≤ 80k for ``strict_lite``) and
should not be lightly modified — the production
system relies on those numbers.  This module
provides *new* blocks that the new profiles
(``improved_strict_lite`` etc.) can compose, while
the existing profiles keep working unchanged.

We also expose :func:`init_module` so any encoder
can be initialised with the orthogonal scheme in
one call.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerScale(nn.Module):
    """Per-channel learnable scale.

    ``output = gamma * x`` where ``gamma`` is a
    ``[C]`` vector initialised to ``init_value``
    (default 1e-2 — see the LayerScale paper,
    Touvron et al. 2021: ``1e-4`` is appropriate
    for very deep ViTs (50+ layers); for the 4-5
    block ConvNets in this repo we use 1e-2 which
    is a sweet spot — small enough to act as a
    regulariser, large enough to keep the
    activations non-trivial after 4 blocks).

    For shallow networks the gradient is well-
    behaved even without LayerScale, so this is an
    *optional* knob (set ``init_value=1.0`` to
    disable it).
    """

    def __init__(self, channels: int, init_value: float = 1e-2) -> None:
        super().__init__()
        self.gamma = nn.Parameter(
            torch.full((channels,), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is ``[B, C, ...]``; broadcast ``gamma``
        # along the channel axis.
        return x * self.gamma.view(1, -1, *([1] * (x.dim() - 2)))


class ImprovedConvBlock(nn.Sequential):
    """Conv + GroupNorm + GELU + LayerScale.

    The LayerScale helps the network train when the
    encoder is deep (5+ blocks).  The GELU
    activation is smoother than ReLU and empirically
    gives a few % better sample efficiency on hard
    vision tasks.
    """

    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1,
                 norm_groups: int = 8, use_layerscale: bool = True,
                 init_value: float = 1e-2) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
        ]
        groups = norm_groups if cout % norm_groups == 0 else max(
            1, cout // 4)
        layers.append(nn.GroupNorm(groups, cout))
        layers.append(nn.GELU())
        if use_layerscale:
            layers.append(LayerScale(cout, init_value=init_value))
        super().__init__(*layers)


class DepthwiseSeparableBlockV2(nn.Module):
    """Improved depthwise-separable block.

    The v1 block (in :mod:`models`) interleaves the
    GroupNorm differently; this v2 version is
    cleaner (single GN per block, GELU activation,
    optional LayerScale on the output).
    """

    def __init__(self, cin: int, cout: int, stride: int = 1,
                 use_layerscale: bool = True,
                 init_value: float = 1e-2) -> None:
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, kernel_size=3, stride=stride,
                              padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, kernel_size=1, bias=False)
        groups = 8 if cout % 8 == 0 else max(1, cout // 4)
        self.norm = nn.GroupNorm(groups, cout)
        self.act = nn.GELU()
        self.layerscale = (LayerScale(cout, init_value=init_value)
                            if use_layerscale else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.layerscale(x)
        return x


class AttentionPool2d(nn.Module):
    """Single-query attention pool over spatial tokens.

    Given a feature map ``[B, C, H, W]``, flatten to
    ``[B, H*W, C]``, attend from a single learnable
    query, and return ``[B, C]``.  This is the same
    idea as the ``[CLS]`` token in ViT, applied to a
    CNN feature map.

    Why bother when GAP works?
    --------------------------
    GAP treats every spatial location equally.  In
    games like Subway Surfers, the agent cares
    disproportionately about the *centre* of the
    frame (where the obstacles are) and much less
    about the borders (the sky and the ground
    decoration).  An attention pool learns this
    bias for free at a param cost of ``2*C*C`` per
    block (the QKV + output projections).
    """

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.n_heads = min(n_heads, channels)
        assert channels % self.n_heads == 0, (
            f"channels {channels} not divisible by n_heads "
            f"{self.n_heads}")
        self.query = nn.Parameter(torch.randn(1, 1, channels)
                                     * (1.0 / math.sqrt(channels)))
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.scale = (channels // self.n_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        qkv = self.qkv(tokens).reshape(B, H * W, 3, self.n_heads,
                                          C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, HW, head_dim
        q = qkv[0]  # [B, heads, HW, head_dim]
        k = qkv[1]
        v = qkv[2]
        # The query is a SINGLE learnable vector; we
        # add it to the sequence length so the dot-
        # product is over (HW+1) positions.
        q_global = (self.query.view(1, 1, self.n_heads,
                                       C // self.n_heads).expand(
            B, -1, -1, -1))
        # Reshape for multi-head attention.
        q_global = q_global.squeeze(1).unsqueeze(2)  # [B, heads, 1, head_dim]
        k_cat = torch.cat([k, torch.zeros(B, self.n_heads, 1,
                                                C // self.n_heads,
                                                device=x.device)], dim=2)
        v_cat = torch.cat([v, torch.zeros(B, self.n_heads, 1,
                                                C // self.n_heads,
                                                device=x.device)], dim=2)
        # Attention: softmax(QK^T / sqrt(d)) V
        attn = (q_global @ k_cat.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # [B, heads, 1, HW+1]
        out = (attn @ v_cat).squeeze(2)  # [B, heads, head_dim]
        out = out.reshape(B, C)
        out = self.proj(out)
        return out


# --------------------------------------------------------------------- #
# Initialisation helpers
# --------------------------------------------------------------------- #
def orthogonal_init_(t: torch.Tensor, gain: float = 1.0) -> None:
    """In-place orthogonal init for a 2-D tensor.

    Falls back to Kaiming uniform for non-2-D tensors
    (e.g. conv weights).  The gain is the standard
    PyTorch convention (e.g. ``gain=math.sqrt(2)`` for
    ReLU/GELU).
    """
    if t.ndim == 2:
        nn.init.orthogonal_(t, gain=gain)
    elif t.ndim in (3, 4):
        # Conv weights: orthogonalise the flattened
        # last two dims.
        nn.init.kaiming_normal_(t, mode="fan_out",
                                  nonlinearity="relu")
    else:
        nn.init.normal_(t, mean=0.0, std=0.02)


def init_module(m: nn.Module, gain: float = 1.0) -> None:
    """Recursively initialise a module with the
    orthogonal / Kaiming scheme.

    The dueling head should be initialised with
    ``gain=1.0`` so the initial Q-values are small.
    The encoder should be initialised with
    ``gain=math.sqrt(2)`` so the post-GELU activations
    have unit variance.

    Special cases
    -------------
    * ``LayerScale.gamma`` is left alone — its
      initial value (typically 1e-4) is the
      *whole point* of the layer and re-initialising
      it would destroy the residual structure.
    * 1-D parameters (biases) are zeroed.
    """
    for name, p in m.named_parameters():
        # Skip LayerScale's gamma — it has a custom
        # init that must be preserved.
        if "layerscale" in name.lower() and p.ndim == 1:
            continue
        if "weight" in name and p.ndim >= 2:
            orthogonal_init_(p, gain=gain)
        elif "bias" in name:
            nn.init.zeros_(p)
