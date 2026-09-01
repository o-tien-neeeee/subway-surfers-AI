"""Deep audit: tìm từng bug khiến AI không học được.

Chạy 4 thử nghiệm độc lập:
1. **Observation shape**: encoder có nhận đúng input không?
2. **Action space**: 5 actions có khả thi không? Có bị degenerate?
3. **Reward signal**: reward có thật sự phân biệt giữa sống/chết?
4. **Training**: 1 batch update có thực sự giảm loss không?

Mỗi test in ra verdict + metric. Sau đó tổng hợp → list các fix
cần làm ngay.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from agent import DoubleDQNAgent
from config import BotConfig
from environment import SyntheticGame
from models import DuelingDQN, build_models_for_profile
from replay_buffer import PrioritizedReplayBuffer, NStepBuilder


def audit_observation() -> dict:
    """Observation có đúng shape, đúng dtype, có đủ thông tin?"""
    print("=" * 60)
    print("AUDIT 1: OBSERVATION")
    print("=" * 60)
    issues = []
    env = SyntheticGame(seed=0)
    frame = env.render()
    print(f"  raw frame: {frame.shape} {frame.dtype}  range=[{frame.min()},{frame.max()}]")
    # Convert to gray 84x84
    gray = frame.mean(axis=2).astype(np.uint8)
    from PIL import Image
    img = Image.fromarray(gray).resize((84, 84))
    obs = np.asarray(img, dtype=np.uint8)
    print(f"  obs (single frame): {obs.shape} {obs.dtype}  range=[{obs.min()},{obs.max()}]")

    # Stack 4 frames
    stack = np.stack([obs, obs, obs, obs], axis=0)
    print(f"  stack: {stack.shape}")

    # Encode
    online, _ = build_models_for_profile("strict_lite", 4, 84)
    with torch.no_grad():
        x = torch.from_numpy(stack).float().unsqueeze(0) / 255.0
        out = online(x)
    print(f"  Q-values output: {out.shape}  range=[{out.min():.3f},{out.max():.3f}]")
    # Sanity: random init should give ~uniform Q across actions
    q_std = float(out.std().item())
    q_mean = float(out.mean().item())
    print(f"  Q mean={q_mean:.3f}  std={q_std:.3f}")
    if q_std < 0.01:
        issues.append("Q-values are uniform (degenerate init)")
    if q_mean > 5 or q_mean < -5:
        issues.append(f"Q-values are extreme (mean={q_mean:.2f})")

    # Discrimination test: same stack should give same Q
    with torch.no_grad():
        out2 = online(x)
    if not torch.allclose(out, out2, atol=1e-5):
        issues.append("Q-values are not deterministic on identical input")

    # Frame difference signal: are consecutive frames different enough?
    obs2 = env.step(0)
    gray2 = obs2.mean(axis=2).astype(np.uint8)
    img2 = Image.fromarray(gray2).resize((84, 84))
    obs2_s = np.asarray(img2, dtype=np.uint8)
    diff = np.abs(obs.astype(np.int16) - obs2_s.astype(np.int16)).mean()
    print(f"  consecutive-frame mean abs diff: {diff:.2f}")
    if diff < 1.0:
        issues.append(f"Frames barely change (diff={diff:.1f}); "
                       "the encoder cannot learn dynamics")

    return {"ok": len(issues) == 0, "issues": issues}


def audit_action_space() -> dict:
    """Action space có đúng 5 hành động và 4 actions có ý nghĩa?"""
    print()
    print("=" * 60)
    print("AUDIT 2: ACTION SPACE")
    print("=" * 60)
    issues = []
    env = SyntheticGame(seed=42)

    # Test each action from lane 1 (middle) and see what happens
    lanes_after = {}
    for a in range(5):
        env.reset()
        env.player_lane = 1
        env.step(a)  # apply action
        lanes_after[a] = env.player_lane
    print(f"  lane after each action: {lanes_after}")
    # Expected: 0 (noop)→1, 1 (left)→0, 2 (right)→2, 3 (jump)→1, 4 (slide)→1
    if lanes_after[1] != 0:
        issues.append(f"LEFT (action 1) gives lane {lanes_after[1]}, expected 0")
    if lanes_after[2] != 2:
        issues.append(f"RIGHT (action 2) gives lane {lanes_after[2]}, expected 2")
    if lanes_after[3] != 1:
        issues.append(f"JUMP (action 3) changes lane to {lanes_after[3]}")
    if lanes_after[4] != 1:
        issues.append(f"SLIDE (action 4) changes lane to {lanes_after[4]}")

    # Test determinism: same action should give same outcome
    env.reset()
    env.player_lane = 1
    pre_lane = env.player_lane
    env.step(1)
    post_lane = env.player_lane
    if pre_lane == 1 and post_lane != 0:
        issues.append("LEFT is not deterministic")

    # Obstacle interaction: place obstacle in player's lane, try JUMP/SLIDE/NOOP
    env.reset()
    env.player_lane = 1
    env.obstacles = [{"kind": "low", "lane": 1, "prog": 0.96, "speed": 0.5}]
    # _spawn_timer doesn't matter; obstacles are processed in step()
    env.step(3)  # JUMP - should dodge low
    survived_jump = not env.dead
    print(f"  JUMP against low obstacle: survived={survived_jump}")
    if not survived_jump:
        issues.append("JUMP does not dodge low obstacle in player's lane")

    env.reset()
    env.player_lane = 1
    env.obstacles = [{"kind": "high", "lane": 1, "prog": 0.96, "speed": 0.5}]
    env.step(4)  # SLIDE
    survived_slide = not env.dead
    print(f"  SLIDE against high obstacle: survived={survived_slide}")
    if not survived_slide:
        issues.append("SLIDE does not dodge high obstacle in player's lane")

    return {"ok": len(issues) == 0, "issues": issues}


def audit_reward_signal() -> dict:
    """Reward có thật sự phân biệt giữa sống và chết không? Có đủ gradient?"""
    print()
    print("=" * 60)
    print("AUDIT 3: REWARD SIGNAL")
    print("=" * 60)
    issues = []
    from rewards import SurvivalRewardCalculator
    from horizon_detector import HorizonResult
    cfg = BotConfig()
    calc = SurvivalRewardCalculator(cfg.reward)

    def _h(t: int) -> HorizonResult:
        return HorizonResult(frame_id=t, ts=t / 30.0,
                             change_score=0.0, raw_score=0.0,
                             changed_ratio=0.0, detected=False,
                             confidence=0.0)

    # Episode 1: 5s survival, no hazards
    calc.begin_episode(0.0)
    rewards_alive = []
    for t in range(int(5.0 * 30)):  # 5s @ 30fps
        r = calc.step(ts=t / 30.0, action=0, horizon=_h(t), died=False)
        rewards_alive.append(r.total)
    total_alive = sum(rewards_alive)
    print(f"  5s no-hazard episode: total reward = {total_alive:.3f}")
    print(f"    mean per-step = {np.mean(rewards_alive):.5f}")
    if total_alive <= 0:
        issues.append("Reward for surviving 5s is non-positive")
    if total_alive < 0.5:
        issues.append(f"Reward for surviving 5s is tiny ({total_alive:.2f}); "
                       "gradient signal is invisible vs -10 death")

    # Episode 2: 1s survival, dies
    calc.begin_episode(0.0)
    rewards_die = []
    for t in range(int(1.0 * 30)):
        r = calc.step(ts=t / 30.0, action=0, horizon=_h(t), died=False)
        rewards_die.append(r.total)
    # Death frame
    r = calc.step(ts=1.0, action=0, horizon=_h(30), died=True)
    rewards_die.append(r.total)
    total_die = sum(rewards_die)
    print(f"  1s dying episode: total reward = {total_die:.3f}")
    if total_die > -5:
        issues.append(f"Death penalty is too small ({total_die:.2f}); "
                       "agent doesn't learn to avoid death")

    # Compare: how much better is 5s vs 1s?
    delta = total_alive - total_die
    print(f"  Δ(survive 5s vs 1s) = {delta:.3f}")
    if delta < 1.0:
        issues.append(f"Survival gradient is too small "
                       f"(5s vs 1s = {delta:.2f}); agent has no reason "
                       "to learn to live longer")

    # Curriculum milestones effect
    calc2 = SurvivalRewardCalculator(cfg.reward)
    calc2.begin_episode(0.0)
    milestones_paid = 0
    for t in range(int(20.0 * 30)):  # 20s
        r = calc2.step(ts=t / 30.0, action=0, horizon=_h(t), died=False)
        if r.total > 0:
            milestones_paid += 1
    print(f"  milestones paid in 20s: {milestones_paid} "
          f"(crossed={calc2.milestones_crossed})")

    return {"ok": len(issues) == 0, "issues": issues}


def audit_training_gradient() -> dict:
    """1 batch update có thực sự làm loss giảm không?"""
    print()
    print("=" * 60)
    print("AUDIT 4: TRAINING GRADIENT")
    print("=" * 60)
    issues = []
    cfg = BotConfig()
    cfg.rl.profile = "strict_lite"
    cfg.rl.gamma = 0.99
    cfg.rl.n_step = 3
    cfg.rl.batch_size = 32
    cfg.rl.grad_clip_norm = 10.0
    cfg.rl.learning_rate = 1e-4

    agent = DoubleDQNAgent("strict_lite", cfg.rl, in_frames=4, size=84, seed=0)
    buf = PrioritizedReplayBuffer(cfg.per, frame_size=84, gamma=cfg.rl.gamma)

    # Synthesize 100 transitions: state→reward 0.1, no-op, never dies
    rng = np.random.default_rng(0)
    transitions_added = 0
    for i in range(100):
        obs = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        nxt = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        # Simple NStepTransition
        from replay_buffer import NStepTransition
        # Use add_nstep (the real entry point)
        buf.add_nstep(NStepTransition(
            obs=obs, next_obs=nxt, action=rng.integers(0, 5),
            reward=0.1, done=False, span=3, gamma_pow=0.99 ** 3,
            obs_env_ids=(0, 1, 2, 3), next_env_ids=(1, 2, 3, 4),
        ))
        transitions_added += 1
    print(f"  transitions added: {transitions_added}, buffer size: {buf.size}")

    # Sample a batch and run 10 updates, observe loss
    losses = []
    q_means = []
    for step in range(20):
        batch = buf.sample(batch_size=32, beta=0.4)
        if batch is None:
            break
        result = agent.train_step(batch)
        losses.append(result["loss"])
        q_means.append(result["q_mean"])
    print(f"  first loss: {losses[0]:.4f}")
    print(f"  last loss:  {losses[-1]:.4f}")
    print(f"  Q-mean first: {q_means[0]:.3f}")
    print(f"  Q-mean last:  {q_means[-1]:.3f}")
    if losses[-1] >= losses[0]:
        issues.append(f"Loss did not decrease ({losses[0]:.3f} → {losses[-1]:.3f})")
    if abs(q_means[-1] - q_means[0]) > 5:
        issues.append(f"Q-values diverged (Δ={q_means[-1]-q_means[0]:.2f})")

    # Gradient norm
    print(f"  final grad_norm: {result['grad_norm']:.3f}")
    if result["grad_norm"] < 1e-6:
        issues.append("Gradients are vanishing (grad_norm ≈ 0)")

    return {"ok": len(issues) == 0, "issues": issues}


def main() -> int:
    results = [
        audit_observation(),
        audit_action_space(),
        audit_reward_signal(),
        audit_training_gradient(),
    ]
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    total_issues = sum(len(r["issues"]) for r in results)
    for i, r in enumerate(results):
        tag = "OK" if r["ok"] else "FAIL"
        print(f"  Audit {i+1}: {tag}")
        for issue in r["issues"]:
            print(f"     • {issue}")
    if total_issues == 0:
        print("\nNo fundamental bugs found.  The agent should learn — "
              "the bottleneck is in the recipe (architecture / hyperparams / "
              "exploration), not in the building blocks.")
    else:
        print(f"\n{len(results) - sum(1 for r in results if r['ok'])}/"
              f"{len(results)} audits FAILED with {total_issues} issue(s).")
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
