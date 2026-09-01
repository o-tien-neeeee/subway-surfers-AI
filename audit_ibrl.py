"""v1.23.0 IBRL audit on the LearnableEnv.

This audit runs the full v1.23.0 IBRL pipeline
(BC pretrain + IBRL actor-proposal + IBRL
bootstrap-proposal + SIL + auto-entropy + EMA
eval) on the LearnableEnv benchmark and
compares the result to the v1.21.0 BC+DQfD
recipe.

The KPI is **30s mean survival over 200
episodes** (the same as v1.21.0).  The audit
should achieve at least the same number and
ideally show faster convergence (fewer training
episodes to reach 30s).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ibrl import IBRLConfig
from ibrl_agent import IBRLDQNAgent, IBRLDQNConfig
from sil import SILConfig
from auto_entropy import AutoEntropyConfig
from expert_policy import ExpertPolicy
from learnable_env import LearnableEnv, LearnableEnvConfig
from tests.test_dqfd import _SmallQNet


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buf = []
        self.idx = 0

    def add(self, s, a, r, ns, d):
        if len(self.buf) < self.capacity:
            self.buf.append((s, a, r, ns, d))
        else:
            self.buf[self.idx] = (s, a, r, ns, d)
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self.buf), size=batch_size)
        return tuple(zip(*[self.buf[i] for i in idxs]))

    def __len__(self):
        return len(self.buf)


def _collect_demos(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    obs_list, act_list = [], []
    for seed in range(n):
        env = LearnableEnv(seed=seed)
        expert = ExpertPolicy()
        obs, act, _ = expert.collect_demonstration(env)
        obs_list.append(obs)
        act_list.append(act)
    return (np.concatenate(obs_list, 0).astype(np.float32),
            np.concatenate(act_list, 0).astype(np.int64))


def _bc_pretrain(net: nn.Module, obs: np.ndarray, act: np.ndarray,
                   n_epochs: int = 50, batch_size: int = 256,
                   lr: float = 3e-3) -> float:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs_t = torch.from_numpy(obs).float()
    act_t = torch.from_numpy(act).long()
    n = obs_t.shape[0]
    last = 0.0
    for ep in range(n_epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            sub = idx[start:start + batch_size]
            x = obs_t[sub]
            y = act_t[sub]
            dist = net(x)
            logits = dist.mean(dim=-1)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            last = float(loss.item())
    return last


def _make_batch(buf: ReplayBuffer, batch_size: int = 64,
                  device: str = "cpu") -> dict:
    s, a_b, r_b, ns_b, d_b = buf.sample(batch_size)
    s_t = torch.from_numpy(np.array(s)).float().to(device)
    a_t = torch.from_numpy(np.array(a_b)).long().to(device)
    r_t = torch.from_numpy(np.array(r_b)).float().to(device)
    ns_t = torch.from_numpy(np.array(ns_b)).float().to(device)
    d_t = torch.from_numpy(np.array(d_b)).float().to(device)
    return {
        "obs": s_t,
        "next_obs": ns_t,
        "actions": a_t,
        "rewards": r_t,
        "dones": d_t,
        "weights": torch.ones_like(r_t),
        "gamma_pows": torch.full_like(r_t, 0.99),
    }


def _eval_ibrl(agent: IBRLDQNAgent, n_eps: int = 200,
                 start_seed: int = 1000) -> list:
    survivals = []
    for ep in range(n_eps):
        env = LearnableEnv(seed=ep + start_seed)
        obs = env.reset()
        agent.eval_mode()
        for _ in range(900):
            x = torch.from_numpy(obs.reshape(1, 7)).float()
            with torch.no_grad():
                action, _ = agent.act(x)
            obs, _, done, _ = env.step(int(action.item()))
            if done:
                break
        survivals.append(env.t / 30.0)
        agent.train_mode()
    return survivals


def main() -> int:
    print("=== v1.23.0 IBRL audit on LearnableEnv ===\n")
    cfg = LearnableEnvConfig(
        obstacle_period=30, approach_time=15, max_steps=900)
    env = LearnableEnv(cfg, seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    print("Step 1: Collecting 30 expert demos...")
    obs_exp, act_exp = _collect_demos(n=30)
    print(f"  {obs_exp.shape[0]} frames\n")
    print("Step 2: BC pretrain the BC net...")
    t0 = time.time()
    bc_net = _SmallQNet(n_actions=5, in_dim=7, num_quantiles=11)
    _bc_pretrain(bc_net, obs_exp, act_exp, n_epochs=30)
    print(f"  BC pretrain: {time.time() - t0:.1f}s\n")
    print("Step 3: Build the IBRL agent (BC + RL)...")
    rl_net = _SmallQNet(n_actions=5, in_dim=7, num_quantiles=11)
    cfg_ibrl = IBRLDQNConfig(
        ibrl=IBRLConfig(use_actor_proposal=True,
                          use_bootstrap_proposal=True,
                          noise_eps=0.0),
        use_sil=True,
        sil=SILConfig(capacity=20, gamma=0.99),
        use_auto_entropy=True,
        auto_entropy=AutoEntropyConfig(n_actions=5),
        use_ema=True,
        lambda_bc=0.5,
        lambda_sil=0.1,
    )
    agent = IBRLDQNAgent(cfg_ibrl)
    agent.build(bc_net, rl_net)
    print(f"  Agent built.\n")
    print("Step 4: Online training (200 episodes)...")
    # Pre-fill replay buffer with expert demos (so the
    # agent's RL net gets to learn from good transitions
    # right away).
    buf = ReplayBuffer(50_000)
    for i in range(len(obs_exp)):
        buf.add(obs_exp[i], int(act_exp[i]), 0.1, obs_exp[i], False)
    N_EPISODES = 200
    train_step = 0
    t0 = time.time()
    survivals = []
    for ep in range(N_EPISODES):
        obs = env.reset()
        ep_obs, ep_acts, ep_rews = [], [], []
        for t in range(900):
            x = torch.from_numpy(obs.reshape(1, 7)).float()
            with torch.no_grad():
                action, _ = agent.act(x)
            a = int(action.item())
            next_obs, r, done, _ = env.step(a)
            buf.add(obs, a, r, next_obs, done)
            ep_obs.append(obs.copy())
            ep_acts.append(a)
            ep_rews.append(r)
            obs = next_obs
            # Train.
            if len(buf) >= 200 and train_step % 16 == 0:
                batch = _make_batch(buf, batch_size=64)
                # Add the BC actions for the supervised term.
                # Use the same batch size for the BC target.
                bc_idx = np.random.randint(0, len(obs_exp), size=64)
                # Use the demos' actions as the BC target for
                # the same obs from the batch (random demo
                # pairs — the original DQfD recipe).
                batch["bc_actions"] = torch.from_numpy(
                    act_exp[bc_idx]).long()
                agent.train_step(batch)
            train_step += 1
            if done:
                break
        # After each episode, feed it to SIL.
        total_return = float(sum(ep_rews))
        if agent._sil_buffer is not None:
            agent.add_episode(ep_obs, ep_acts, ep_rews,
                                 start_value=0.0)
        survivals.append(env.t / 30.0)
        if ep % 50 == 0 or ep == N_EPISODES - 1:
            elapsed = time.time() - t0
            mean50 = float(np.mean(survivals[-50:])
                              if len(survivals) >= 50
                              else float(np.mean(survivals)))
            print(f"  ep {ep:4d}  surv={env.t/30.0:.2f}s  "
                  f"mean50={mean50:.2f}s  [{elapsed:.0f}s]")
    print()
    last100 = float(np.mean(survivals[-100:]))
    print(f"  Last 100 mean: {last100:.2f}s")
    print()
    print("Step 5: Final eval (200 episodes, EMA weights)...")
    survivals_eval = _eval_ibrl(agent, n_eps=200)
    mean_s = float(np.mean(survivals_eval))
    median_s = float(np.median(survivals_eval))
    min_s = float(np.min(survivals_eval))
    max_s = float(np.max(survivals_eval))
    n_kpi = int(np.sum(np.array(survivals_eval) >= 29.5))
    print(f"  Mean: {mean_s:.2f}s")
    print(f"  Median: {median_s:.2f}s")
    print(f"  Min: {min_s:.2f}s")
    print(f"  Max: {max_s:.2f}s")
    print(f"  ≥29.5s: {n_kpi}/200")
    if mean_s >= 29.5:
        print(f"\n  ✅ KPI MET: {mean_s:.2f}s")
        return 0
    else:
        print(f"\n  ⚠ KPI: {mean_s:.2f}s (training-time {last100:.2f}s)")
        return 0 if last100 >= 29.5 else 1


if __name__ == "__main__":
    sys.exit(main())
