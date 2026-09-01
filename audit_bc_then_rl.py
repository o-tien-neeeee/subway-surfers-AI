"""Full audit: BC pretrain from expert → RL on top.

This is the most important test the user asked for: a deep
agent that:
  1. Pre-trains on expert demonstrations (BC).
  2. Then runs online RL on top to refine.

If this passes the 30s KPI, we have *evidence* that the
pipeline works end-to-end and the issue is in the
environment (real game vs synthetic), not in the algorithm.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from expert_policy import ExpertPolicy
from learnable_env import LearnableEnv, LearnableEnvConfig


class QNet(nn.Module):
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


def bc_pretrain(net: QNet, opt, obs: np.ndarray, acts: np.ndarray,
                 n_epochs: int = 50, batch: int = 256) -> float:
    """Behaviour cloning: cross-entropy between the expert
    actions and the network's argmax.

    The dataset is small (one episode ≈ 900 samples) so
    we cycle through it for many epochs.
    """
    n = len(obs)
    last_loss = 0.0
    for ep in range(n_epochs):
        # Shuffle.
        idx = np.random.permutation(n)
        for start in range(0, n, batch):
            sub = idx[start:start + batch]
            x = torch.from_numpy(obs[sub]).float()
            y = torch.from_numpy(acts[sub]).long()
            logits = net(x)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            last_loss = float(loss.item())
    return last_loss


def main() -> int:
    cfg = LearnableEnvConfig(
        obstacle_period=30, approach_time=15, max_steps=900)
    env = LearnableEnv(cfg, seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")

    # Step 1: collect expert demonstrations from MANY episodes
    # (different seeds → different spawn schedules), so the BC
    # dataset covers the full state space.
    print("Step 1: collecting expert demonstrations...")
    expert = ExpertPolicy()
    t0 = time.time()
    all_obs = []
    all_acts = []
    N_DEMOS = 30
    for d in range(N_DEMOS):
        env = LearnableEnv(cfg, seed=d)
        obs_exp, act_exp, _ = expert.collect_demonstration(env)
        all_obs.append(obs_exp)
        all_acts.append(act_exp)
    obs_exp = np.concatenate(all_obs, axis=0)
    act_exp = np.concatenate(all_acts, axis=0)
    print(f"  expert: {obs_exp.shape[0]} frames from {N_DEMOS} demos "
          f"in {time.time()-t0:.1f}s")

    # Step 2: BC pretrain.
    print("Step 2: BC pretrain (50 epochs)...")
    net = QNet().to(device)
    target_net = QNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    t0 = time.time()
    bc_loss = bc_pretrain(net, opt, obs_exp, act_exp, n_epochs=50)
    target_net.load_state_dict(net.state_dict())
    print(f"  BC loss: {bc_loss:.4f}  in {time.time()-t0:.1f}s")

    # Step 3: Verify BC policy.
    print("Step 3: verify BC policy achieves KPI...")
    obs = env.reset()
    for _ in range(900):
        x = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            q = net(x)
        a = int(q.argmax(dim=1).item())
        obs, _, done, _ = env.step(a)
        if done:
            break
    bc_surv = env.t / 30.0
    print(f"  BC-only survival: {bc_surv:.2f}s")

    if bc_surv < 30.0:
        print("  BC did not solve the env — RL after BC won't either.")
        return 1

    # Step 4: DQfD — combine Q-learning loss AND
    # supervised BC loss on every minibatch.  This is the
    # technique from DeepMind's "Deep Q-learning from
    # Demonstrations" (Hester et al. 2018): the supervised
    # term *anchors* the Q-values to the expert actions so
    # online RL cannot drift too far.
    print("Step 4: DQfD (joint Q + supervised loss)...")
    # Fresh optimizer for joint loss.
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    target_net = QNet().to(device)
    target_net.load_state_dict(net.state_dict())
    buf = ReplayBuffer(50_000)
    # Pre-fill the buffer with BC demonstrations.
    for i in range(len(obs_exp)):
        buf.add(obs_exp[i], int(act_exp[i]), 0.1, obs_exp[i], False)
    bc_obs_all = obs_exp
    bc_act_all = act_exp
    N_EPISODES = 200
    gamma = 0.99
    # No exploration: BC is already optimal, so the policy
    # is a fixed-point of itself under the bootstrap.  Any
    # random action injects noise that the supervised loss
    # has to "fight" — and the Q loss wins because the
    # online data has fewer constraints.
    eps = 0.0
    batch_size = 64
    target_every = 100
    grad_clip = 10.0
    survivals = []
    train_step = 0
    t0 = time.time()
    for ep in range(N_EPISODES):
        obs = env.reset()
        for t in range(900):
            x = torch.from_numpy(obs).float().unsqueeze(0)
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
                s_t = torch.from_numpy(np.array(s)).float()
                a_t = torch.from_numpy(np.array(a_b)).long()
                r_t = torch.from_numpy(np.array(r_b)).float()
                ns_t = torch.from_numpy(np.array(ns_b)).float()
                d_t = torch.from_numpy(np.array(d_b)).float()
                q_pred = net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(ns_t).max(dim=1).values
                target_q = r_t + gamma * (1.0 - d_t) * q_next
                q_loss = (q_pred - target_q).pow(2).mean()
                # DQfD supervised term: cross-entropy on a
                # mini-batch of BC demonstrations.  This
                # anchors the policy to the expert and
                # prevents catastrophic forgetting.
                bc_idx = np.random.randint(0, len(bc_obs_all),
                                            size=batch_size // 2)
                bc_s = torch.from_numpy(bc_obs_all[bc_idx]).float()
                bc_a = torch.from_numpy(bc_act_all[bc_idx]).long()
                bc_logits = net(bc_s)
                bc_loss = nn.functional.cross_entropy(bc_logits, bc_a)
                loss = q_loss + 0.5 * bc_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step()
            train_step += 1
            if done:
                break
        survivals.append(env.t / 30.0)
        if train_step % target_every == 0:
            target_net.load_state_dict(net.state_dict())
        if ep % 50 == 0 or ep == N_EPISODES - 1:
            elapsed = time.time() - t0
            mean50 = float(np.mean(survivals[-50:])) if len(survivals) >= 50 else float(np.mean(survivals))
            print(f"  ep {ep:4d}  surv={env.t/30.0:.2f}s  "
                  f"mean50={mean50:.2f}s  "
                  f"[{elapsed:.0f}s]")
    print()
    last100 = float(np.mean(survivals[-100:])) if len(survivals) >= 100 else float(np.mean(survivals))
    print(f"  Last 100 mean: {last100:.2f}s")
    if last100 >= 30.0:
        print(f"  ✅ KPI MET via BC + RL")
    else:
        print(f"  ❌ KPI NOT MET: {last100:.2f}s")
    return 0 if last100 >= 30.0 else 1


if __name__ == "__main__":
    sys.exit(main())
