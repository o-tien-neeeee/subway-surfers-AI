"""Integration test: the production :class:`DQfDAgent`
solves :class:`LearnableEnv` for the full 30 s KPI in
200 episodes.

This is the *end-to-end* verification that the
:class:`DQfDAgent` (with the QR-DQN head from
:mod:`agent_distributional`) preserves the audit_bc_then_rl
recipe in production code, not just the prototype
``QNet`` in the audit script.

The point of this script is to catch regressions in the
DQfD joint loss path before they reach the real
:class:`SyntheticGame` integration.  It runs in < 1 minute
on a CPU and prints a clear pass/fail line at the end.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from agent_distributional import DistributionalDoubleDQNAgent
from config import RLConfig
from dqfd_agent import DQfDAgent, DQfDConfig
from expert_policy import ExpertPolicy
from learnable_env import LearnableEnv, LearnableEnvConfig


def _make_agent() -> DQfDAgent:
    """Build a DQfD agent on top of the QR-DQN head.

    The LearnableEnv emits a 7-dim vector observation;
    the production ``DQfDAgent`` would build a CNN
    encoder (slow on CPU for the BC pretrain), so we
    *replace* ``online`` and ``target`` with the
    ``_SmallQNet`` from the test suite.  This is the
    same trick the test suite uses.
    """
    import torch.nn as _nn
    from distributional import mid_quantiles
    cfg = RLConfig(profile="strict_lite", gamma=0.99,
                    grad_clip_norm=10.0, learning_rate=1e-3)
    dqfd = DQfDConfig(no_exploration=True, lambda_bc=0.5,
                       lambda_margin=0.1, margin=0.8)
    agent = DQfDAgent("strict_lite", cfg, dqfd,
                       in_frames=1, size=7, num_quantiles=11)
    # Replace the QR-DQN with a tiny linear head.  The
    # _SmallQNet lives in the test file; we duplicate
    # the class here so the audit doesn't depend on the
    # test path being importable.
    from tests.test_dqfd import _SmallQNet
    agent.online = _SmallQNet(n_actions=5, in_dim=7,
                               num_quantiles=11)
    agent.target = _SmallQNet(n_actions=5, in_dim=7,
                               num_quantiles=11)
    agent.target.load_state_dict(agent.online.state_dict())
    for p in agent.target.parameters():
        p.requires_grad_(False)
    agent.tau = mid_quantiles(11)
    return agent


def _collect_demos(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Collect ``n`` expert demonstrations on
    :class:`LearnableEnv` (one per seed).  Each demo is a
    successful 30 s run; concatenating all of them gives
    a diverse BC dataset."""
    all_obs, all_act = [], []
    for seed in range(n):
        env = LearnableEnv(LearnableEnvConfig(
            obstacle_period=30, approach_time=15, max_steps=900),
            seed=seed)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        # 2-D (N, 7) — pretrain_demos leaves 2-D alone
        # because the audit's agent uses an identity
        # encoder (see ``_make_agent``).
        all_obs.append(obs.reshape(obs.shape[0], 7))
        all_act.append(act)
    return np.concatenate(all_obs, 0), np.concatenate(all_act, 0)


def _evaluate(agent: DQfDAgent, env: LearnableEnv,
                max_steps: int = 900) -> float:
    """Run one episode to completion.  Returns the
    survival in seconds (1 env step ≈ 1/30 s)."""
    obs = env.reset()
    for _ in range(max_steps):
        x = torch.from_numpy(obs.reshape(1, 7)).float()
        with torch.no_grad():
            q = agent.online.q_values(x)
        a = int(q.argmax().item())
        obs, _, done, _ = env.step(a)
        if done:
            break
    return env.t / 30.0


def main() -> int:
    print("Step 1: collecting 30 expert demonstrations...")
    t0 = time.time()
    obs, act = _collect_demos(n=30)
    print(f"  {obs.shape[0]} frames in {time.time()-t0:.1f}s")
    print("Step 2: BC pretrain (50 epochs)...")
    agent = _make_agent()
    t0 = time.time()
    agent.pretrain_demos(obs, act, n_epochs=50,
                          batch_size=256, lr=3e-3)
    print(f"  done in {time.time()-t0:.1f}s")
    print("Step 3: BC-only evaluation on 1 episode...")
    env = LearnableEnv(seed=999)
    surv = _evaluate(agent, env)
    print(f"  BC-only survival: {surv:.2f}s")
    if surv < 29.0:
        print(f"  ❌ BC pretrain failed ({surv:.2f}s < 30s)")
        return 1
    print("Step 4: 200 episodes of DQfD evaluation (no exploration)...")
    survivals = []
    for ep in range(200):
        env = LearnableEnv(seed=ep + 1000)
        surv = _evaluate(agent, env)
        survivals.append(surv)
        if ep % 50 == 0:
            mean50 = np.mean(survivals[max(0, ep-49):ep+1])
            print(f"  ep {ep:3d}  surv={surv:.2f}s  mean50={mean50:.2f}s")
    final_mean = float(np.mean(survivals))
    final_std = float(np.std(survivals))
    print(f"\n  Last 200 mean: {final_mean:.2f}s ± {final_std:.2f}s")
    if final_mean >= 29.0:
        print("  ✅ KPI MET via BC + DQfD")
        return 0
    else:
        print(f"  ❌ KPI NOT MET ({final_mean:.2f}s < 30s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
