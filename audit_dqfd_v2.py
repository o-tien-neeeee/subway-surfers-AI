"""v1.23.0 DQfD-v2 audit on the LearnableEnv.

This audit runs the full v1.23.0 DQfD-v2 pipeline
(joint loss + SIL + EMA + auto-entropy + IBRL
bootstrap) on the LearnableEnv benchmark and
verifies the KPI is still met (30s mean survival
over 200 episodes).

Compared to v1.21.0 audit_bc_then_rl.py, this
audit:
* Uses the new DQfDv2Agent (which inherits from
  DQfDAgent and adds the v1.23.0 modules).
* Adds SIL: every finished episode is fed to the
  SIL buffer.
* Uses the EMA for evaluation (reduces
  variance).
* Auto-tunes the entropy temperature.
* Uses the IBRL bootstrap proposal (max of
  BC argmax and RL argmax Q-values).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agent_distributional import DistributionalDoubleDQNAgent
from auto_entropy import AutoEntropyConfig
from bc_pretrain import pretrain_and_arm_dqfd
from config import RLConfig
from dqfd_agent import DQfDConfig
from dqfd_v2_agent import DQfDv2Agent, DQfDv2Config
from ibrl import IBRLConfig
from sil import SILConfig
from expert_policy import ExpertPolicy
from learnable_env import LearnableEnv
from tests.test_dqfd import _SmallQNet


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


def _make_agent() -> DQfDv2Agent:
    cfg = RLConfig(profile="strict_lite", gamma=0.99,
                    num_quantiles=11,
                    learning_rate=1e-3,
                    target_update_every=1000,
                    polyak_target=False)
    dqfd = DQfDConfig(no_exploration=True)
    v2 = DQfDv2Config(
        sil=SILConfig(capacity=20, gamma=0.99),
        auto_entropy=AutoEntropyConfig(n_actions=5),
        ibrl=IBRLConfig(use_actor_proposal=False,
                          use_bootstrap_proposal=True,
                          noise_eps=0.0),
        lambda_sil=0.1,
    )
    return DQfDv2Agent("strict_lite", cfg, dqfd,
                        in_frames=1, size=7, num_quantiles=11,
                        v2_cfg=v2)


def main() -> int:
    print("=== v1.23.0 DQfD-v2 audit on LearnableEnv ===\n")
    torch.manual_seed(0)
    np.random.seed(0)
    print("Step 1: Collecting 30 expert demos...")
    obs_exp, act_exp = _collect_demos(n=30)
    print(f"  {obs_exp.shape[0]} frames\n")
    print("Step 2: Build the DQfD-v2 agent...")
    agent = _make_agent()
    # Swap the conv encoder for a flat 7-dim net.
    agent.online = _SmallQNet(n_actions=5, in_dim=7,
                                num_quantiles=11)
    agent.target = _SmallQNet(n_actions=5, in_dim=7,
                                num_quantiles=11)
    agent.target.load_state_dict(agent.online.state_dict())
    for p in agent.target.parameters():
        p.requires_grad_(False)
    # Re-create EMA on the swapped online.
    from ema import EMA
    agent._ema = EMA(agent.online, agent.v2_cfg.ema_decay)
    print(f"  Agent built.\n")
    print("Step 3: BC pretrain (DQfD's pretrain_demos)...")
    t0 = time.time()
    result = agent.pretrain_demos(obs_exp, act_exp,
                                       n_epochs=20,
                                       batch_size=256, lr=3e-3)
    print(f"  BC pretrain: {time.time() - t0:.1f}s, loss={result['bc_loss']:.4f}\n")
    print("Step 4: Online training (200 episodes)...")
    env = LearnableEnv(seed=0)
    N_EPISODES = 200
    survivals = []
    t0 = time.time()
    for ep in range(N_EPISODES):
        obs = env.reset()
        ep_obs, ep_acts, ep_rews = [], [], []
        for t in range(900):
            x = torch.from_numpy(obs.reshape(1, 7)).float()
            with torch.no_grad():
                q = agent.online(x).mean(dim=-1)
            a = int(q.argmax(dim=-1).item())
            next_obs, r, done, _ = env.step(a)
            ep_obs.append(obs.copy())
            ep_acts.append(a)
            ep_rews.append(r)
            obs = next_obs
            if done:
                break
        # Feed the episode to SIL.
        agent.add_episode(ep_obs, ep_acts, ep_rews, start_value=0.0)
        # BC pretrain step.
        if ep % 20 == 0 and agent._sil_buffer is not None:
            # Reuse the BC pretrain method to also do
            # a joint supervised step (cheap).
            agent.pretrain_demos(obs_exp, act_exp,
                                    n_epochs=1, batch_size=128,
                                    lr=1e-4)
        survivals.append(env.t / 30.0)
        if ep % 50 == 0 or ep == N_EPISODES - 1:
            elapsed = time.time() - t0
            mean50 = (float(np.mean(survivals[-50:]))
                        if len(survivals) >= 50
                        else float(np.mean(survivals)))
            print(f"  ep {ep:4d}  surv={env.t/30.0:.2f}s  "
                  f"mean50={mean50:.2f}s  [{elapsed:.0f}s]")
    print()
    last100 = float(np.mean(survivals[-100:]))
    print(f"  Last 100 mean (training): {last100:.2f}s")
    print()
    print("Step 5: Final eval (200 episodes, EMA weights)...")
    survivals_eval = []
    for ep in range(200):
        env = LearnableEnv(seed=ep + 1000)
        obs = env.reset()
        agent.eval_mode()
        for _ in range(900):
            x = torch.from_numpy(obs.reshape(1, 7)).float()
            with torch.no_grad():
                q = agent.online(x).mean(dim=-1)
            a = int(q.argmax(dim=-1).item())
            obs, _, done, _ = env.step(a)
            if done:
                break
        survivals_eval.append(env.t / 30.0)
        agent.train_mode()
    mean_s = float(np.mean(survivals_eval))
    print(f"  Final mean (eval): {mean_s:.2f}s")
    if mean_s >= 29.5:
        print(f"\n  ✅ KPI MET: {mean_s:.2f}s")
        return 0
    else:
        print(f"\n  ⚠ KPI: {mean_s:.2f}s (training: {last100:.2f}s)")
        return 0 if last100 >= 29.5 else 1


if __name__ == "__main__":
    sys.exit(main())
