"""v1.22.0 visual pipeline smoke test.

This audit answers: *does the improved visual
encoder + augmentations actually produce sensible
Q-values from raw pixel observations?*

The audit:
1. Builds an ImprovedQuantileDuelingDQN with
   the "improved_strict_lite" profile.
2. Runs a forward pass on a batch of random
   pixel observations.
3. Checks that the Q-values are finite and
   non-degenerate.
4. Runs a backward pass and checks that the
   gradients are non-zero.
5. Computes the Q-value range (max - min) to
   verify the dueling head is actually producing
   a spread of action preferences (not all the
   same value).
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from augmentations import intensity_jitter, random_translate
from improved_dqn import (IMPROVED_PROFILES, build_improved_agent,
                            mid_quantiles)


def main() -> int:
    print("=== v1.22.0 Improved QR-DQN visual smoke test ===\n")
    # ------------------------------------------------------------------
    # 1. Build the network.
    # ------------------------------------------------------------------
    net = build_improved_agent(
        "improved_strict_lite", num_quantiles=11, seed=0)
    print(f"Network: {net.count_params():,} parameters "
          f"({net.profile})")
    # ------------------------------------------------------------------
    # 2. Random pixel batch.
    # ------------------------------------------------------------------
    x = torch.rand(8, 4, 84, 84)
    print(f"Input shape: {tuple(x.shape)}")
    # ------------------------------------------------------------------
    # 3. Forward pass (no aug).
    # ------------------------------------------------------------------
    net.eval()  # eval mode disables NoisyNets noise
    with torch.no_grad():
        y = net(x)
    print(f"Forward output shape: {tuple(y.shape)} "
          f"(B, A, N)")
    assert y.shape == (8, 5, 11), f"unexpected shape {y.shape}"
    assert torch.isfinite(y).all(), "Q values contain NaN/Inf"
    # ------------------------------------------------------------------
    # 4. Backward pass (with train mode + aug).
    # ------------------------------------------------------------------
    net.train()
    x_aug = random_translate(x, max_pixels=3)
    x_aug = intensity_jitter(x_aug, brightness=0.10,
                               contrast=0.10)
    y = net(x_aug)
    # A fake MSE loss against a target.
    target = torch.zeros_like(y)
    loss = nn.functional.mse_loss(y, target)
    loss.backward()
    # Check at least one weight has a non-zero gradient.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in net.parameters() if p.requires_grad)
    print(f"Loss: {loss.item():.6f}, "
          f"has non-zero grad: {has_grad}")
    assert has_grad, "No gradient flowed to any parameter"
    # ------------------------------------------------------------------
    # 5. Q-value range (action preferences).
    # ------------------------------------------------------------------
    net.eval()
    with torch.no_grad():
        q = net.q_values(x)
    q_range = q.max() - q.min()
    q_std = q.std()
    print(f"Q-value range: {q_range.item():.4f}, "
          f"std: {q_std.item():.4f}")
    # A dueling head with a useful bias should have
    # some spread (the actions get different
    # values).  Strict bounds: std > 1e-4 and
    # range > 1e-3.
    assert q_std > 1e-4, f"Q-values are too uniform: std={q_std}"
    assert q_range > 1e-3, f"Q-value range too small: {q_range}"
    # ------------------------------------------------------------------
    # 6. Mid-quantile buffer shape.
    # ------------------------------------------------------------------
    print(f"Mid-quantile buffer: {net.tau.shape} "
          f"(first 3: {net.tau[:3].tolist()}, "
          f"last 3: {net.tau[-3:].tolist()})")
    # ------------------------------------------------------------------
    # 7. Profile parameter counts.
    # ------------------------------------------------------------------
    print("\nProfile parameter counts:")
    for profile, spec in IMPROVED_PROFILES.items():
        n = build_improved_agent(profile, num_quantiles=51,
                                    seed=0).count_params()
        print(f"  {profile}: {n:,} params "
              f"(blocks: {len(spec['blocks'])}, "
              f"pool: {spec['pool']}, head_hidden: {spec['head_hidden']})")
    print("\n✅ All v1.22.0 visual smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
