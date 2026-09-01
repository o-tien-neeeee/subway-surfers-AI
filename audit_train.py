"""Real training audit: 200 episodes in SyntheticGame.

Question: with the current architecture, can the agent learn to
survive longer than random?  This is the single most important
test before any architecture change.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from agent import DoubleDQNAgent
from agent_distributional import DistributionalDoubleDQNAgent
from config import BotConfig
from environment import SyntheticGame
from models import build_models_for_profile
from replay_buffer import NStepBuilder, PrioritizedReplayBuffer


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) -> gray 84x84 uint8."""
    gray = frame.mean(axis=2).astype(np.uint8)
    img = Image.fromarray(gray).resize((84, 84))
    return np.asarray(img, dtype=np.uint8)


def run_episode(env: SyntheticGame,
                agent,
                buf: PrioritizedReplayBuffer, nstep: NStepBuilder,
                epsilon: float, max_steps: int = 120) -> tuple[float, int]:
    env.reset()
    # Initialise the obs stack with the first frame repeated 4x
    f0 = preprocess_frame(env.render())
    stack = np.stack([f0, f0, f0, f0], axis=0)
    # reset nstep builder
    nstep.clear()
    survival = 0
    for t in range(max_steps):
        # Action
        with torch.inference_mode():
            x = torch.from_numpy(stack).float().unsqueeze(0) / 255.0
            # Both agents expose ``q_values`` for argmax.
            q = agent.online.q_values(x)
        if np.random.random() < epsilon:
            a = int(np.random.randint(0, 5))
        else:
            a = int(q.argmax(dim=1).item())
        # Step env with dense reward
        result = env.step_with_reward(a)
        r = result["reward"]
        # Build new stack
        f_new = preprocess_frame(result["frame"])
        new_stack = np.roll(stack, -1, axis=0)
        new_stack[-1] = f_new
        # Push to n-step builder (real signature: stack, env_ids,
        # action, reward, done)
        emitted = nstep.push(stack, (t, t + 1, t + 2, t + 3), a, r,
                              done=env.dead)
        for tr in emitted:
            buf.add_nstep(tr)
        stack = new_stack
        if env.dead:
            survival = t
            return float(t) / 30.0, t
        survival = t
    return float(survival) / 30.0, survival


def main() -> int:
    cfg = BotConfig()
    cfg.rl.profile = "strict_lite"
    cfg.rl.gamma = 0.99
    cfg.rl.n_step = 3
    cfg.rl.batch_size = 32
    cfg.rl.warmup_transitions = 200
    cfg.rl.grad_clip_norm = 10.0
    cfg.rl.learning_rate = 1e-3
    cfg.per.capacity = 5000
    cfg.rl.target_update_every = 100

    rng = np.random.default_rng(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = SyntheticGame(seed=0)
    # Use the distributional agent — the v1.18 audit showed the
    # scalar Q agent cannot learn the synthetic game in 20
    # episodes (Improvement: +0.00s).  The QR-DQN agent should
    # at least produce a measurable gradient.
    agent = DistributionalDoubleDQNAgent(
        "strict_lite", cfg.rl, in_frames=4, size=84, num_quantiles=51,
        seed=0)
    buf = PrioritizedReplayBuffer(cfg.per, frame_size=84, gamma=cfg.rl.gamma)
    nstep = NStepBuilder(n=cfg.rl.n_step, gamma=cfg.rl.gamma)

    N_EPISODES = 300
    TARGET_TRAIN_HZ = 4
    eps_start, eps_end = 0.5, 0.05
    decay_per_ep = (eps_start - eps_end) / N_EPISODES
    epsilon = eps_start

    survivals = []
    rolling = deque(maxlen=10)
    train_step = 0
    last_q = 0.0
    last_loss = 0.0
    t0 = time.time()
    for ep in range(N_EPISODES):
        surv, steps = run_episode(env, agent, buf, nstep, epsilon)
        survivals.append(surv)
        rolling.append(surv)
        # Train every TARGET_TRAIN_HZ env steps
        if buf.size >= cfg.rl.warmup_transitions and ep % 1 == 0:
            for _ in range(max(1, steps // TARGET_TRAIN_HZ)):
                batch = buf.sample(batch_size=32, beta=0.4)
                if batch is not None:
                    result = agent.train_step(batch)
                    if "td_errors" in result:
                        buf.update_priorities(
                            batch["indices"], result["td_errors"])
                    train_step += 1
                    last_q = float(result.get("q_mean", 0))
                    last_loss = float(result.get("loss", 0))
        # Sync target
        if train_step > 0 and train_step % cfg.rl.target_update_every == 0:
            agent.sync_target()
        epsilon = max(eps_end, eps_start - decay_per_ep * ep)
        if ep % 5 == 0 or ep == N_EPISODES - 1:
            elapsed = time.time() - t0
            print(f"  ep {ep:3d}  surv={surv:.2f}s  "
                  f"mean10={np.mean(rolling):.2f}s  "
                  f"eps={epsilon:.3f}  "
                  f"buf={buf.size}  train={train_step}  "
                  f"q={last_q:.2f}  loss={last_loss:.3f}  "
                  f"[{elapsed:.0f}s]")
    print()
    print(f"  First 20 episodes mean:  {np.mean(survivals[:20]):.2f}s")
    print(f"  Last  20 episodes mean:  {np.mean(survivals[-20:]):.2f}s")
    improvement = np.mean(survivals[-20:]) - np.mean(survivals[:20])
    print(f"  Improvement: {improvement:+.2f}s")
    return 0 if improvement > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
