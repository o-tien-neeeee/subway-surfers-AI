"""SyntheticGame BC pretrain audit.

The LearnableEnv audit (audit_bc_then_rl.py) proved the
DQfD recipe works on a clean benchmark.  The next
question is: does the same recipe work on the *real*
synthetic game?  The SyntheticGame is harder because
its obstacle schedule is random (the agent cannot
memorise a fixed sequence) and the expert itself
survives only ~22s on average (10/20 seeds reach 30s).

This script:
1. Collects N expert demonstrations (the
   ``SyntheticExpert``) on SyntheticGame.
2. Pre-trains a QR-DQN agent on those demos.
3. Evaluates the BC-pretrained policy with ε=0 over
   K=20 episodes.
4. Reports mean/median/min/max survival.

The point is to find out *what the ceiling is* on the
real game with the same recipe.  If the BC-pretrained
agent matches the expert's 22s mean, the recipe
transfers; if it underperforms by a lot, the expert
isn't useful (or the BC dataset is too small).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from bc_pretrain import (build_dqfd_agent, prefill_replay_with_demos,
                            pretrain_and_arm_dqfd)
from config import PERConfig, RLConfig
from dqfd_agent import DQfDConfig
from environment import SyntheticGame
from expert_synthetic import SyntheticExpert
from replay_buffer import PrioritizedReplayBuffer


def _preprocess(frame: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) → 84x84 uint8 grayscale."""
    from PIL import Image
    gray = frame.mean(axis=2).astype(np.uint8)
    img = Image.fromarray(gray).resize((84, 84))
    return np.asarray(img, dtype=np.uint8)


def collect_demos(n_episodes: int = 20,
                    max_steps: int = 300
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect ``n_episodes`` expert demonstrations on
    SyntheticGame.  Each demo is the full 4-frame
    stack per step (so the BC dataset is in the
    production shape ``[N, 4, 84, 84]``)."""
    all_obs: list[np.ndarray] = []
    all_act: list[int] = []
    all_rew: list[float] = []
    for seed in range(n_episodes):
        env = SyntheticGame(seed=seed)
        expert = SyntheticExpert()
        frames, actions, rewards = expert.collect_demonstration(
            env, max_steps=max_steps)
        all_obs.append(frames)
        all_act.append(actions)
        all_rew.append(rewards)
        if (seed + 1) % 5 == 0:
            print(f"  collected {seed + 1}/{n_episodes} demos")
    return (np.concatenate(all_obs, 0).astype(np.float32) / 255.0,
            np.concatenate(all_act, 0).astype(np.int64),
            np.concatenate(all_rew, 0).astype(np.float32))


def main(n_episodes: int = 20, n_epochs: int = 30,
            n_eval: int = 20) -> int:
    # Allow CLI override so the audit can be tuned
    # without editing the source.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=n_episodes)
    p.add_argument("--n-epochs", type=int, default=n_epochs)
    p.add_argument("--n-eval", type=int, default=n_eval)
    p.add_argument("--max-steps", type=int, default=300,
                    help="cap each demo at this many frames "
                         "(keeps the audit under 5 min on CPU)")
    args = p.parse_args()
    return _run(args.n_episodes, args.n_epochs, args.n_eval, args.max_steps)


def _run(n_episodes: int, n_epochs: int, n_eval: int,
            max_steps: int) -> int:
    print(f"Step 1: collect {n_episodes} expert demos on SyntheticGame...")
    t0 = time.time()
    obs, act, rew = collect_demos(n_episodes, max_steps=max_steps)
    print(f"  {obs.shape[0]} frames, action distribution: "
          f"{np.bincount(act, minlength=5).tolist()} "
          f"in {time.time() - t0:.1f}s")
    print(f"  total expert reward: {rew.sum():.2f}")
    print("Step 2: build DQfD agent + BC pretrain...")
    cfg = RLConfig(profile="strict_lite", gamma=0.99,
                    batch_size=32, learning_rate=1e-3,
                    grad_clip_norm=10.0)
    dqfd = DQfDConfig(lambda_bc=0.5, lambda_margin=0.1,
                       margin=0.8, no_exploration=True)
    agent = build_dqfd_agent("strict_lite", cfg, dqfd)
    t0 = time.time()
    result = pretrain_and_arm_dqfd(agent, obs, act, n_epochs=n_epochs,
                                      batch_size=128, lr=3e-3)
    print(f"  BC pretrain: {time.time() - t0:.1f}s, "
          f"final loss {result['bc_loss']:.4f}")
    print("Step 3: evaluate BC-pretrained policy (ε=0)...")
    survivals = []
    for ep in range(n_eval):
        env = SyntheticGame(seed=ep + 1000)
        env.reset()
        # Seed the 4-frame stack.
        f0 = _preprocess(env.render())
        stack = np.stack([f0, f0, f0, f0], axis=0).astype(np.float32) / 255.0
        expert = SyntheticExpert()  # for "what would the expert do?" baseline
        surv = 0
        for _ in range(max_steps):
            x = torch.from_numpy(stack).unsqueeze(0)
            with torch.no_grad():
                q = agent.online(x).mean(dim=-1)
            a = int(q.argmax().item())
            env.step(a)
            # Update the stack with the new frame.
            new_frame = _preprocess(env.render())
            stack = np.concatenate([stack[1:], [new_frame]], axis=0)
            stack = stack.astype(np.float32) / 255.0
            surv += 1
            if env.dead:
                break
        survivals.append(surv / 30.0)
        if ep % 5 == 0:
            print(f"  ep {ep:3d}: {surv / 30.0:.2f}s")
    mean_s = float(np.mean(survivals))
    median_s = float(np.median(survivals))
    min_s = float(np.min(survivals))
    max_s = float(np.max(survivals))
    print(f"\n  SyntheticGame BC-pretrained survival:")
    print(f"    mean:   {mean_s:.2f}s")
    print(f"    median: {median_s:.2f}s")
    print(f"    min:    {min_s:.2f}s")
    print(f"    max:    {max_s:.2f}s")
    print(f"    >= 30s: {sum(1 for s in survivals if s >= 29.9)}/{n_eval}")
    # Compare to the expert baseline.
    print("Step 4: expert baseline on same seeds (for reference)...")
    expert_survivals = []
    for ep in range(n_eval):
        env = SyntheticGame(seed=ep + 1000)
        expert = SyntheticExpert()
        env.reset()
        surv = 0
        for _ in range(max_steps):
            a = expert.act(env.player_lane, env.obstacles)
            env.step(a)
            surv += 1
            if env.dead:
                break
        expert_survivals.append(surv / 30.0)
    print(f"  Expert mean: {np.mean(expert_survivals):.2f}s, "
          f"median: {np.median(expert_survivals):.2f}s")
    if mean_s >= np.mean(expert_survivals) * 0.7:
        print("  ✅ BC-pretrained agent reaches >= 70% of expert performance.")
        return 0
    else:
        print(f"  ⚠️  BC-pretrained agent under-performs expert "
              f"({mean_s:.2f}s vs {np.mean(expert_survivals):.2f}s).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
