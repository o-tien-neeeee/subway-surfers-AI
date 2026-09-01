"""Audit: deep Q-network on the learnable env.

If a *deep* agent with the same observation cannot match
the tabular agent's 27s mean, the bug is in the deep
agent's optimisation (encoder overfitting, bad init,
gradient clipping, etc)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from learnable_env import LearnableEnv, LearnableEnvConfig


class SimpleQNet(nn.Module):
    def __init__(self, n_actions: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
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
        s, a, r, ns, d = zip(*[self.buf[i] for i in idxs])
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def main() -> int:
    cfg = LearnableEnvConfig(
        obstacle_period=30, approach_time=15, max_steps=900)
    env = LearnableEnv(cfg, seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    net = SimpleQNet().to(device)
    target_net = SimpleQNet().to(device)
    target_net.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buf = ReplayBuffer(50_000)

    N_EPISODES = 1500
    gamma = 0.99
    eps = 0.5
    eps_min = 0.05
    eps_decay = 0.995
    batch_size = 64
    target_every = 200
    grad_clip = 10.0

    survivals = []
    t0 = time.time()
    train_step = 0
    for ep in range(N_EPISODES):
        obs = env.reset()
        for t in range(900):
            x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q = net(x)
            if np.random.random() < eps:
                a = int(np.random.randint(0, 5))
            else:
                a = int(q.argmax(dim=1).item())
            next_obs, r, done, _ = env.step(a)
            buf.add(obs, a, r, next_obs, done)
            obs = next_obs
            if len(buf) >= 200 and train_step % 16 == 0:
                s, a_b, r_b, ns_b, d_b = buf.sample(batch_size)
                s_t = torch.from_numpy(s).float().to(device)
                a_t = torch.from_numpy(a_b).long().to(device)
                r_t = torch.from_numpy(r_b).to(device)
                ns_t = torch.from_numpy(ns_b).float().to(device)
                d_t = torch.from_numpy(d_b).to(device)
                q_pred = net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(ns_t).max(dim=1).values
                target_q = r_t + gamma * (1.0 - d_t) * q_next
                loss = (q_pred - target_q).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step()
            train_step += 1
            if done:
                break
        survivals.append(env.t / 30.0)
        eps = max(eps_min, eps * eps_decay)
        if train_step % target_every == 0:
            target_net.load_state_dict(net.state_dict())
        if ep % 100 == 0 or ep == N_EPISODES - 1:
            elapsed = time.time() - t0
            mean100 = float(np.mean(survivals[-100:]))
            print(f"  ep {ep:4d}  surv={env.t/30.0:.2f}s  "
                  f"mean100={mean100:.2f}s  eps={eps:.3f}  "
                  f"buf={len(buf)}  train={train_step}  "
                  f"[{elapsed:.0f}s]")
    print()
    print(f"  Last 100 mean: {np.mean(survivals[-100:]):.2f}s")
    if np.mean(survivals[-100:]) >= 30.0:
        print(f"  ✅ Deep Q solves the env")
    else:
        print(f"  ❌ Deep Q did NOT reach 30s")
    return 0 if np.mean(survivals[-100:]) >= 30.0 else 1


if __name__ == "__main__":
    sys.exit(main())
