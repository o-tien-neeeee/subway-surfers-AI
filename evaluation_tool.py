"""Headless evaluation runner: N episodes, fixed epsilon, no learning.

Used by ``python app.py --evaluate N``.  In headless mode it evaluates on the
synthetic game; on the target machine the same code path evaluates the real
game through the actor (see README "Evaluation protocol").  Results are
written to ``runs/evaluation_<timestamp>.json`` (+ ``.md``) and never
overwrite history.
"""

from __future__ import annotations

import time
from typing import Any

from agent import InferencePolicy
from config import BotConfig
from environment import GameEnvironment, SyntheticGame
from evaluation import EpisodeRecord, EvaluationReport
from logging_utils import get_logger, setup_logging
from models import DuelingDQN

LOGGER = get_logger("evaluate")


def load_policy(cfg: BotConfig, which: str = "best") -> InferencePolicy:
    from checkpoint_manager import CheckpointManager

    ckpt = CheckpointManager(cfg.paths.checkpoints_dir, cfg.rl.profile)
    payload = ckpt.load_model(which) or ckpt.load_model("latest")
    model = DuelingDQN.from_profile(cfg.rl.profile, cfg.perception.frame_stack,
                                    cfg.perception.ground_size)
    if payload is not None and payload.get("profile") == cfg.rl.profile:
        model.load_state_dict(payload["agent"]["online"])
        LOGGER.info("evaluation policy loaded from %s checkpoint (%s)",
                    which, ckpt.dir)
    else:
        LOGGER.warning("no checkpoint for profile %s; evaluating an UNTRAINED net",
                       cfg.rl.profile)
    return InferencePolicy(model, seed=cfg.seed)


def run_headless_evaluation(cfg: BotConfig, episodes: int,
                            epsilon: float = 0.05) -> EvaluationReport:
    """Evaluate on the synthetic game with learning disabled."""
    from metrics import FpsMeter, LatencyMeter

    policy = load_policy(cfg)
    report = EvaluationReport()
    for ep in range(episodes):
        # DEEP-FIX: every episode used the SAME cfg.seed for both the
        # synthetic game and the policy's exploration RNG, so all N
        # "independent" evaluation episodes were bit-identical (verified:
        # two SyntheticGame(cfg.seed) runs produce the same 20 frames, and
        # two InferencePolicy(seed=cfg.seed) runs pick the same 10 random
        # actions).  The Mann-Whitney p-value, the bootstrap CI and the
        # "std" in the report were therefore computed on n copies of one
        # sample -- statistics that look rigorous and mean nothing.  Each
        # episode now gets its own derived seed.
        ep_seed = int(cfg.seed) * 1_000_003 + ep
        game = SyntheticGame(seed=ep_seed, fps=cfg.capture.target_fps)
        env = GameEnvironment(cfg, game=game)
        obs = env.reset()
        total = 0.0
        steps = 0
        fps = FpsMeter()
        infer = LatencyMeter()
        t0 = time.monotonic()
        done = False
        while not done:
            t_inf0 = time.perf_counter()
            action = policy.act(obs, epsilon)
            infer.observe_ms((time.perf_counter() - t_inf0) * 1000.0)
            obs, reward, done, _info = env.step(action)
            total += reward
            steps += 1
            fps.tick(time.monotonic())
            if steps > 30 * 180:  # 3-minute cap per episode
                break
        survival = time.monotonic() - t0
        report.add(EpisodeRecord(
            episode_id=ep + 1, survival_s=survival, total_reward=total,
            steps=steps, env_frames=steps, fps=fps.fps(),
            action_latency_p95_ms=infer.snapshot()["p95"],
            inference_p95_ms=infer.snapshot()["p95"], kind="eval",
        ))
        LOGGER.info("eval episode %d: %.1fs reward=%.1f steps=%d",
                    ep + 1, survival, total, steps)
    return report


def run_evaluation(args: Any) -> int:
    setup_logging("evaluate", "logs")
    cfg = BotConfig.load(args.config) if getattr(args, "config", "") else BotConfig()
    if getattr(args, "dry_run", False):
        cfg.input.dry_run = True
    n = max(1, getattr(args, "evaluate", 20))
    report = run_headless_evaluation(cfg, n)
    baseline_path = getattr(args, "compare_baseline", "") or ""
    if baseline_path:
        try:
            imported = report.merge_baseline(baseline_path)
            LOGGER.info("imported %d baseline records from %s",
                        imported, baseline_path)
        except (OSError, ValueError) as exc:
            LOGGER.error("could not load baseline %s: %s", baseline_path, exc)
    out = report.save(f"runs/evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json")
    print(report.to_markdown())
    if baseline_path and report.of_kind("human_baseline"):
        cmp = report.compare()
        print(f"\ncomparison: {cmp.get('verdict', 'n/a')}")
    print(f"\nreport written: {out}")
    print("NOTE: headless evaluation runs on the SYNTHETIC game. Real Poki-game")
    print("evaluation requires the target machine; until then all real-game")
    print("metrics remain 'not yet measured'.")
    return 0


def record_human_baseline_headless(cfg: BotConfig, episodes: int,
                                   policy_fn=None) -> EvaluationReport:
    """Record a 'baseline' of scripted-random play for comparison testing."""
    import numpy as np

    rng = np.random.default_rng(cfg.seed)
    report = EvaluationReport()
    for ep in range(episodes):
        env = GameEnvironment(cfg)
        obs = env.reset()
        total, steps, done = 0.0, 0, False
        t0 = time.monotonic()
        while not done:
            action = int(rng.integers(0, 5)) if policy_fn is None else policy_fn(obs)
            obs, reward, done, _ = env.step(action)
            total += reward
            steps += 1
            if steps > 30 * 120:
                break
        report.add(EpisodeRecord(
            episode_id=ep + 1, survival_s=time.monotonic() - t0,
            total_reward=total, steps=steps, env_frames=steps, fps=30.0,
            action_latency_p95_ms=0.0, inference_p95_ms=0.0,
            kind="human_baseline",
        ))
    return report
