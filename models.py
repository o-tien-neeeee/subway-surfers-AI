"""Neural network architectures (CPU-first dueling Double-DQN encoders).

Why NOT the classic "Conv-Conv-Conv-Flatten -> 512-dense" Atari net
-------------------------------------------------------------------
The classic DQN head flattens 7x7x64 = 3136 features into a 512-unit dense
layer: 3136*512 + 512 ~= 1.6M parameters *in the head alone* — dwarfing the
convs and dominating CPU latency and replay RAM.  This repo instead uses:

* **Depthwise-separable convolutions** (StrictLite) — a 3x3 conv with C_in=C_out
  costs C*9*C params; separable costs C*9 + C*C_out, a ~C/9 reduction.
* **Global average pooling** — removes the giant flatten dense entirely.
* **Dueling heads** (value + advantage streams) — better Q estimates for the
  same encoder cost.
* **GroupNorm, not BatchNorm** — see the audit in the README: BatchNorm's
  running statistics go stale when (a) replay batches are tiny relative to
  the buffer distribution, (b) the actor runs in a *different process* doing
  inference while the learner trains (train/eval mode divergence), and
  (c) the online data distribution shifts as the policy improves.  GroupNorm
  is batch-independent, so the actor and learner compute identical features
  for identical weights, and small batches cannot corrupt normalisation.

Profiles
--------
===========  ============  ===========  =====================================
profile      params        design       intended use
===========  ============  ===========  =====================================
strict_lite   ~48.7k        4x DW-sep    <=80k parameter budget (guaranteed)
balanced_cpu  ~87.6k        3x conv      default; must hold 20-30 FPS + p95
quality_cpu  ~348k          4x conv      only if profiling proves headroom
===========  ============  ===========  =====================================

Exact counts are asserted in tests/test_models.py and printed by
``python profiling.py``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

N_ACTIONS = 5


# --------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------- #
def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with at most 8 groups (channels are always divisible)."""
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise 3x3 conv followed by pointwise 1x1 projection."""

    def __init__(self, cin: int, cout: int, stride: int = 1, norm: bool = True) -> None:
        dw = nn.Conv2d(cin, cin, kernel_size=3, stride=stride, padding=1,
                       groups=cin, bias=not norm)
        pw = nn.Conv2d(cin, cout, kernel_size=1, bias=not norm)
        layers: list[nn.Module] = [dw]
        if norm:
            layers.append(_gn(cin))
        layers += [nn.ReLU(inplace=True), pw]
        if norm:
            layers += [_gn(cout)]
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ConvBlock(nn.Sequential):
    """Standard conv + optional GroupNorm + ReLU."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1,
                 norm: bool = True) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, kernel, stride=stride,
                      padding=kernel // 2, bias=not norm)
        ]
        if norm:
            layers.append(_gn(cout))
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class DuelingHead(nn.Module):
    """Two-stream head: V(s) and A(s,a) -> Q(s,a) = V + A - mean(A)."""

    def __init__(self, feature_dim: int, n_actions: int = N_ACTIONS,
                 hidden: int = 128) -> None:
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(features)
        adv = self.advantage_stream(features)
        return value + adv - adv.mean(dim=1, keepdim=True)

    def advantage_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Raw advantage stream — used as logits for behaviour cloning."""
        return self.advantage_stream(features)


class DuelingDQN(nn.Module):
    """Dueling Double-DQN network.  Input: float32 [B, k, 84, 84] in [0,1]."""

    def __init__(self, in_frames: int, encoder: nn.Module, feature_dim: int,
                 n_actions: int = N_ACTIONS, head_hidden: int = 128) -> None:
        super().__init__()
        self.in_frames = in_frames
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = DuelingHead(feature_dim, n_actions, head_hidden)
        self.n_actions = n_actions
        self.feature_dim = feature_dim

    @classmethod
    def from_profile(cls, profile: str, in_frames: int = 4, size: int = 84,
                     n_actions: int = N_ACTIONS) -> DuelingDQN:
        spec = PROFILES[profile]
        encoder, out_ch = build_encoder(spec, in_frames)
        return cls(in_frames, encoder, out_ch, n_actions, spec["head_hidden"])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.pool(z).flatten(1)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        feats = self.features(x)
        if return_logits:
            return self.head.advantage_logits(feats)
        return self.head(feats)


# --------------------------------------------------------------------- #
# Profile specifications
# --------------------------------------------------------------------- #
def build_encoder(spec: dict[str, Any], in_frames: int) -> tuple[nn.Module, int]:
    kind = spec["kind"]
    layers: list[nn.Module] = []
    cin = in_frames
    if kind == "dws":
        # (out, stride) for each depthwise-separable block
        for out, stride in spec["blocks"]:
            layers.append(DepthwiseSeparableConv(cin, out, stride=stride))
            cin = out
    elif kind == "conv":
        for out, kernel, stride in spec["blocks"]:
            layers.append(ConvBlock(cin, out, kernel, stride))
            cin = out
    else:
        raise ValueError(f"unknown encoder kind {kind!r}")
    if spec.get("final_norm", False):
        layers.append(_gn(cin))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers), cin


PROFILES: dict[str, dict[str, Any]] = {
    "strict_lite": {
        "kind": "dws",
        "blocks": [(32, 2), (48, 2), (64, 2), (128, 1)],
        "final_norm": True,
        "head_hidden": 128,
        "param_budget": 80_000,
        "description": "4 depthwise-separable blocks + GAP + dueling head; "
                       "<=80k trainable parameters (hard requirement).",
    },
    "balanced_cpu": {
        "kind": "conv",
        "blocks": [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        "final_norm": False,
        "head_hidden": 128,
        "param_budget": None,
        "description": "3 conv blocks (8x8/4x4/3x3) + GAP + dueling head; "
                       "default profile, must hold 20-30 FPS and p95<100ms.",
    },
    "quality_cpu": {
        "kind": "conv",
        "blocks": [(48, 8, 4), (96, 4, 2), (96, 3, 2), (128, 3, 1)],
        "final_norm": True,
        "head_hidden": 256,
        "param_budget": None,
        "description": "4 conv blocks + GAP + dueling head; enable only when "
                       "profiling shows headroom on the target machine.",
    },
}


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_models_for_profile(profile: str, in_frames: int = 4, size: int = 84
                             ) -> tuple[DuelingDQN, DuelingDQN]:
    """(online, target) networks for a profile; target starts as a copy."""
    online = DuelingDQN.from_profile(profile, in_frames, size)
    target = DuelingDQN.from_profile(profile, in_frames, size)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    return online, target


def weight_size_for_profile(profile: str, in_frames: int = 4) -> int:
    return count_trainable_params(DuelingDQN.from_profile(profile, in_frames))


if __name__ == "__main__":  # pragma: no cover - quick manual check
    for name, spec in PROFILES.items():
        m = DuelingDQN.from_profile(name)
        n = count_trainable_params(m)
        budget = spec.get("param_budget")
        flag = "OK" if budget is None or n <= budget else "OVER BUDGET"
        print(f"{name:<14} params={n:>8,d}  {flag}")
        if budget is not None and n > budget:
            raise SystemExit(f"{name} violates its parameter budget!")
