"""Audit: tabular Q-learning on the learnable env.

If even a *tabular* Q-learner cannot solve this env, the
problem is in the algorithm/reward.  If it CAN solve it,
then the deep agent's failure is in representation /
optimisation, not in the algorithm.

This is the *control* experiment that lets us isolate
"agent code is broken" from "env is too hard".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from learnable_env import LearnableEnv, LearnableEnvConfig


def state_to_idx(obs: np.ndarray, n_lanes: int = 3) -> int:
    """Encode the 7-dim observation to a small int.

    Player one-hot: 3 values.
    Next-obstacle one-hot: 3 values.
    Time-to-impact: bucket into 4 bins (very close, close,
    far, very far).
    Total: 3 * 3 * 4 = 36 states — well within tabular RL.
    """
    p = int(np.argmax(obs[:3]))
    o = int(np.argmax(obs[3:6]))
    tti = float(obs[6])
    tti_bin = min(3, int(tti * 4))
    return p * 12 + o * 4 + tti_bin


def main() -> int:
    cfg = LearnableEnvConfig(
        obstacle_period=30, approach_time=15, max_steps=900)
    env = LearnableEnv(cfg, seed=0)
    n_states = 3 * 3 * 4  # 36
    n_actions = 5
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    visits = np.zeros_like(Q)
    alpha = 0.2
    gamma = 0.99
    epsilon = 0.5
    eps_min = 0.05
    eps_decay = 0.995

    N_EPISODES = 1500
    survivals = []
    t0 = time.time()
    for ep in range(N_EPISODES):
        obs = env.reset()
        s = state_to_idx(obs)
        total_r = 0.0
        for t in range(900):
            if np.random.random() < epsilon:
                a = int(np.random.randint(0, n_actions))
            else:
                a = int(np.argmax(Q[s]))
            next_obs, r, done, _ = env.step(a)
            next_s = state_to_idx(next_obs)
            visits[s, a] += 1
            # Q-learning update.
            Q[s, a] = (1 - alpha) * Q[s, a] + \
                alpha * (r + gamma * (0.0 if done else np.max(Q[next_s])))
            s = next_s
            total_r += r
            if done:
                break
        survivals.append(env.t / 30.0)
        epsilon = max(eps_min, epsilon * eps_decay)
        if ep % 100 == 0 or ep == N_EPISODES - 1:
            elapsed = time.time() - t0
            mean100 = float(np.mean(survivals[-100:]))
            print(f"  ep {ep:4d}  surv={env.t/30.0:.2f}s  "
                  f"mean100={mean100:.2f}s  eps={epsilon:.3f}  "
                  f"[{elapsed:.0f}s]")
    print()
    print(f"  Last 100 episodes mean: {np.mean(survivals[-100:]):.2f}s")
    if np.mean(survivals[-100:]) >= 30.0:
        print(f"  ✅ Tabular Q-learning solves the env in {N_EPISODES} episodes")
    else:
        print(f"  ❌ Tabular Q-learning failed: {np.mean(survivals[-100:]):.2f}s")
    return 0 if np.mean(survivals[-100:]) >= 30.0 else 1


if __name__ == "__main__":
    sys.exit(main())
