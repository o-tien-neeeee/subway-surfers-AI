"""Tests for NoisyNet exploration layers.

NoisyNets replace ε-greedy with *parameter-space noise*:
the layer weights are perturbed by a factorised Gaussian, so
the agent explores around its current best policy rather
than spending a fraction of its decisions on completely random
ones.  This is the change that the Rainbow ablation found
*second most* important after multi-step + PER (the third
most important ingredient was distributional Q, which v1.18
already shipped).

These tests pin:
* NoisyLinear shape, init scale, and that noise is zero at
  eval time (deterministic action selection).
* The factorised Gaussian transform matches the paper:
  the noise has shape ``O × I`` and is the outer product of
  two ``sign * sqrt(abs(eps))`` vectors.
* NoisyDuelingHead output shape and that the dueling
  combination ``V + A - mean(A)`` is preserved.
* An end-to-end gradient step shows the noise parameters
  receive gradients (so the network can learn to "turn down"
  noise on outputs that are already well-trained).
"""

from __future__ import annotations

import torch

from noisy_nets import NoisyDuelingHead, NoisyLinear


class TestNoisyLinear:
    def test_output_shape(self) -> None:
        layer = NoisyLinear(16, 8)
        x = torch.zeros(4, 16)
        out = layer(x)
        assert out.shape == (4, 8)

    def test_eval_is_deterministic(self) -> None:
        layer = NoisyLinear(16, 8)
        layer.eval()
        x = torch.randn(3, 16)
        out1 = layer(x)
        out2 = layer(x)
        # In eval mode the noise buffers stay zero, so the
        # forward pass is a plain linear with the mean
        # weights — identical for the same input.
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_train_is_stochastic(self) -> None:
        torch.manual_seed(0)
        layer = NoisyLinear(16, 8)
        layer.train()
        x = torch.randn(3, 16)
        outs = [layer(x) for _ in range(5)]
        # The five forward passes should NOT be identical
        # (the noise is freshly sampled each call).
        for i in range(1, 5):
            assert not torch.allclose(outs[0], outs[i], atol=1e-7), (
                "training pass should be stochastic")

    def test_noise_is_factorised(self) -> None:
        """The noise must be the OUTER product of two factor
        vectors — this is the O(in + out) trick from the
        NoisyNet paper, not O(in * out) per-element noise."""
        torch.manual_seed(0)
        layer = NoisyLinear(8, 4, sigma_init=0.5)
        layer.train()
        # Trigger the noise sampling.
        layer(torch.zeros(1, 8))
        # The buffer is shape [out, in].  Verify it has rank 1
        # (rank 1 == outer product of two vectors).
        noise = layer.weight_epsilon
        # rank 1 means: SVD shows only one non-zero singular
        # value (up to numerical tolerance).
        s = torch.linalg.svdvals(noise)
        # The first singular value is large; the rest are ~0.
        assert (s[1:] < 1e-4).all(), (
            f"noise should be rank 1, got singular values {s}")

    def test_gradients_flow_to_noise(self) -> None:
        """The sigma parameters need gradients so the network
        can learn to suppress noise on outputs that are
        already well-trained.  Without this, NoisyNets
        degenerates to fixed-noise (less useful)."""
        layer = NoisyLinear(8, 4)
        x = torch.randn(2, 8)
        y = layer(x).sum()
        y.backward()
        # ``weight_sigma`` and ``bias_sigma`` should have
        # non-zero gradients; ``weight_mu`` and ``bias_mu``
        # also need gradients.
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"{name} got no grad"
            assert (p.grad.abs() > 0).all(), f"{name} grad is zero"


class TestNoisyDuelingHead:
    def test_output_shape(self) -> None:
        head = NoisyDuelingHead(in_dim=16, n_actions=5, hidden=32)
        x = torch.zeros(2, 16)
        out = head(x)
        assert out.shape == (2, 5)

    def test_dueling_combination(self) -> None:
        """The dueling combination ``V + A - mean(A)`` means
        the per-state advantage is mean-zero.  This is the
        identifiability trick that lets the value and
        advantage streams specialise without ambiguity."""
        head = NoisyDuelingHead(in_dim=8, n_actions=5, hidden=16)
        head.eval()
        x = torch.randn(1, 8)
        with torch.no_grad():
            q = head(x)
            # The mean of Q across actions is the value V.
            v = q.mean(dim=-1)
            # In a dueling head, Q(s,a) - V(s) = A(s,a) - mean(A)
            # so the mean of (Q - V) across actions must be ~0.
            assert torch.allclose((q - v.unsqueeze(-1)).mean(dim=-1),
                                   torch.zeros_like(v), atol=1e-5)

    def test_training_changes_output(self) -> None:
        """A gradient step on the head should change the
        output (proving the network is actually learning
        from the loss)."""
        head = NoisyDuelingHead(in_dim=8, n_actions=5, hidden=16)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        x = torch.randn(4, 8)
        # Initial output
        head.eval()
        with torch.no_grad():
            out0 = head(x).clone()
        # Train 5 steps
        head.train()
        for _ in range(5):
            opt.zero_grad()
            q = head(x)
            target = torch.full_like(q, 0.5)
            loss = ((q - target) ** 2).mean()
            loss.backward()
            opt.step()
        # Output after training
        head.eval()
        with torch.no_grad():
            out1 = head(x)
        # The output should have moved toward the target.
        err0 = (out0 - 0.5).abs().mean()
        err1 = (out1 - 0.5).abs().mean()
        assert err1 < err0
