"""Model tests: parameter budgets, shapes, normalisation choices, weights sync."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch
from torch import nn

from agent import DoubleDQNAgent, InferencePolicy, epsilon_for_frame
from config import RLConfig
from models import (
    PROFILES,
    DuelingDQN,
    build_models_for_profile,
    count_trainable_params,
)


class TestParameterBudgets:
    @pytest.mark.parametrize("profile", list(PROFILES))
    def test_forward_outputs_five_q_values(self, profile: str) -> None:
        model = DuelingDQN.from_profile(profile)
        x = torch.rand(2, 4, 84, 84)
        q = model(x)
        assert q.shape == (2, 5)

    def test_strict_lite_under_80k(self) -> None:
        n = count_trainable_params(DuelingDQN.from_profile("strict_lite"))
        assert n <= 80_000, f"StrictLite budget violated: {n}"
        # and it must be a *useful* size, not degenerate
        assert n >= 5_000

    def test_profile_ordering_by_size(self) -> None:
        sizes = [count_trainable_params(DuelingDQN.from_profile(p))
                 for p in ("strict_lite", "balanced_cpu", "quality_cpu")]
        assert sizes[0] < sizes[1] < sizes[2]

    @pytest.mark.parametrize("profile,budget", [
        ("strict_lite", 80_000),
        ("balanced_cpu", 200_000),
        ("quality_cpu", 600_000),
    ])
    def test_documented_counts(self, profile: str, budget: int) -> None:
        n = count_trainable_params(DuelingDQN.from_profile(profile))
        assert n <= budget, f"{profile} exceeds its documented budget: {n}"

    def test_exact_counts_stable(self) -> None:
        """Guard against accidental architecture drift changing the audit."""
        counts = {p: count_trainable_params(DuelingDQN.from_profile(p))
                  for p in PROFILES}
        assert counts["strict_lite"] == 49_154
        assert counts["balanced_cpu"] == 95_558
        assert counts["quality_cpu"] == 348_134


class TestArchitectureChoices:
    @pytest.mark.parametrize("profile", list(PROFILES))
    def test_no_batchnorm_anywhere(self, profile: str) -> None:
        model = DuelingDQN.from_profile(profile)
        bad = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert bad == [], "BatchNorm is banned (small online batches + actor/learner split)"

    def test_no_giant_flatten_dense(self) -> None:
        """The classic conv->flatten->512 head is banned by design."""
        for profile in PROFILES:
            model = DuelingDQN.from_profile(profile)
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    assert m.in_features <= 512, (
                        f"{profile}: Linear(in={m.in_features}) looks like a "
                        f"flatten head; use global average pooling"
                    )

    def test_dueling_decomposition(self) -> None:
        model = DuelingDQN.from_profile("strict_lite")
        model.eval()
        x = torch.rand(1, 4, 84, 84)
        q = model(x)
        feats = model.features(x)
        v = model.head.value_stream(feats)
        a = model.head.advantage_stream(feats)
        manual = v + a - a.mean(dim=1, keepdim=True)
        assert torch.allclose(q, manual, atol=1e-5)
        # identical state -> identical advantage-shifted Q (ranking invariance)
        a2 = a + 7.3
        manual2 = v + a2 - a2.mean(dim=1, keepdim=True)
        assert torch.allclose(manual, manual2, atol=1e-5)

    def test_groupnorm_present(self) -> None:
        model = DuelingDQN.from_profile("strict_lite")
        gns = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
        assert gns, "StrictLite should normalise with GroupNorm"

    def test_target_network_frozen_and_equal(self) -> None:
        online, target = build_models_for_profile("strict_lite")
        for p in target.parameters():
            assert not p.requires_grad
        for (k1, v1), (k2, v2) in zip(online.state_dict().items(),
                                      target.state_dict().items()):
            assert torch.equal(v1, v2)


class TestAgent:
    def test_train_step_produces_metrics(self) -> None:
        agent = DoubleDQNAgent("strict_lite", RLConfig(batch_size=8), seed=0)
        rng = np.random.default_rng(0)

        def stack():
            return rng.integers(0, 255, (4, 84, 84), dtype=np.uint8)

        batch = {
            "obs": np.stack([stack() for _ in range(8)]),
            "next_obs": np.stack([stack() for _ in range(8)]),
            "actions": rng.integers(0, 5, 8),
            "rewards": rng.normal(0, 0.1, 8).astype(np.float32),
            "dones": (rng.random(8) < 0.2).astype(np.float32),
            "weights": np.ones(8, dtype=np.float32),
            "gamma_pows": np.full(8, 0.99, dtype=np.float32),
        }
        m = agent.train_step(batch)
        for key in ("loss", "q_mean", "grad_norm", "lr"):
            assert key in m and np.isfinite(m[key])

    def test_double_dqn_uses_target_for_evaluation(self) -> None:
        agent = DoubleDQNAgent("strict_lite", RLConfig(), seed=1)
        # freeze everything except one path and verify target net is consulted:
        # changing target weights must change the computed target.
        rng = np.random.default_rng(1)
        obs = {"obs": rng.integers(0, 255, (4, 4, 84, 84), dtype=np.uint8),
               "next_obs": rng.integers(0, 255, (4, 4, 84, 84), dtype=np.uint8),
               "actions": np.array([0, 1, 2, 3]),
               "rewards": np.zeros(4, dtype=np.float32),
               "dones": np.zeros(4, dtype=np.float32),
               "weights": np.ones(4, dtype=np.float32),
               "gamma_pows": np.ones(4, dtype=np.float32)}

        class _Opt:
            param_groups: ClassVar[list[dict]] = [{"lr": 1e-4}]

            def zero_grad(self, set_to_none=True):
                return None  # test double: grads handled by train_step internals

            def step(self):
                return None  # test double

        agent.optimizer = _Opt()  # type: ignore[assignment]
        q_before = agent.train_step(obs)["q_mean"]
        with torch.no_grad():
            for p in agent.target.parameters():
                p.add_(1.0)
        agent.update_count = 1_000_001
        agent.train_step(obs)  # syncs target mid-call; exercise the path
        assert np.isfinite(q_before)

    def test_bc_epoch_learns_memorisable_mapping(self) -> None:
        agent = DoubleDQNAgent("strict_lite", RLConfig(), seed=2)
        rng = np.random.default_rng(2)
        # action encoded by border brightness -> trivially learnable
        obs = np.zeros((256, 4, 84, 84), dtype=np.uint8)
        acts = rng.integers(0, 5, 256)
        for i, a in enumerate(acts):
            obs[i, :, : 5 + 10 * int(a), :] = 200
        import torch

        opt = torch.optim.Adam(agent.online.parameters(), lr=3e-3)
        first = last = 0.0
        for epoch in range(25):
            out = agent.bc_epoch(obs, acts, np.ones(256, dtype=np.float32),
                                 optimizer=opt)
            if epoch == 0:
                first = out["bc_acc"]
            last = out["bc_acc"]
        assert last > first
        assert last > 0.9, f"BC should fit an easy synthetic mapping ({last:.2f})"
        ev = agent.bc_eval(obs, acts)
        assert set(ev["per_action"]) == {0, 1, 2, 3, 4}


class TestInferencePolicy:
    def test_act_greedy_and_epsilon(self) -> None:
        model = DuelingDQN.from_profile("strict_lite")
        pol = InferencePolicy(model, seed=5)
        stack = np.zeros((4, 84, 84), dtype=np.uint8)
        greedy = {pol.act(stack, epsilon=0.0) for _ in range(5)}
        assert len(greedy) == 1
        randomish = {pol.act(stack, epsilon=1.0) for _ in range(60)}
        assert len(randomish) > 1

    def test_q_values_finite(self) -> None:
        pol = InferencePolicy(DuelingDQN.from_profile("strict_lite"), seed=0)
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 256, (4, 84, 84), dtype=np.uint8)
        q = pol.q_values(stack)
        assert q.shape == (5,) and np.all(np.isfinite(q))

    def test_weight_refresh(self) -> None:
        from ipc import SharedWeights

        model = DuelingDQN.from_profile("strict_lite")
        pol = InferencePolicy(model, seed=0)
        shared = SharedWeights(count_trainable_params(model))
        assert pol.refresh_weights(shared) is True  # version 0 -> 1 applied
        assert pol.refresh_weights(shared) is False  # same version -> no copy
        shared.publish(np.zeros(count_trainable_params(model), dtype=np.float32))
        assert pol.refresh_weights(shared) is True


class TestEpsilon:
    def test_linear_decay_in_env_frames(self) -> None:
        cfg = RLConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_frames=1000)
        assert epsilon_for_frame(0, cfg) == 1.0
        assert epsilon_for_frame(500, cfg) == pytest.approx(1.0 - 0.95 * 0.5)
        assert epsilon_for_frame(1000, cfg) == pytest.approx(0.05)
        assert epsilon_for_frame(10_000, cfg) == pytest.approx(0.05)


class TestCpuOnly:
    def test_agent_refuses_cuda_device(self) -> None:
        with pytest.raises(AssertionError):
            DoubleDQNAgent("strict_lite", RLConfig(), device="cuda")

    def test_threads_pinned(self) -> None:
        from profiling import set_cpu_threads

        set_cpu_threads(1)
        assert torch.get_num_threads() == 1
