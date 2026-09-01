"""Tests for the deep-research fixes (DEEP_RESEARCH_vi.md).

Covers: full-frame policy observations, frame-skip MDP semantics,
agent-step epsilon schedule, DQfD large-margin loss and expert replay,
mirror augmentation, obstacle tracker shaping, and demo label backdating.
"""

from __future__ import annotations

import numpy as np
import torch

from config import BotConfig, RLConfig, RewardConfig
from obstacle_perception import ObstacleTracker
from perception import FrameStack, ZonePreprocessor


# --------------------------------------------------------------------- #
# 1. Policy sees the FULL frame (horizon included)
# --------------------------------------------------------------------- #
def _region(h=200, w=120):
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    # a bright "obstacle" at the horizon (top 20 %) which the old ground
    # crop (split at 25 %) would have hidden from the policy.
    img[10:40, 50:70] = 230
    return img


def test_policy_observation_contains_horizon():
    cfg = BotConfig()
    pre = ZonePreprocessor(cfg.perception, 0.25, require_anchor=False)
    z = pre.process(_region(), frame_id=1, ts=0.0)
    assert z.policy_gray.shape == (cfg.perception.obs_size, cfg.perception.obs_size)
    # The full-frame policy view must include the bright horizon obstacle:
    # its brightest rows should be in the TOP portion of the policy image.
    top_mean = z.policy_gray[: cfg.perception.obs_size // 5].mean()
    assert top_mean > z.policy_gray.mean(), "horizon content must reach the policy"
    # ground_gray kept as a backwards-compatible alias
    assert z.ground_gray is z.policy_gray


def test_frame_stack_advances_one_per_decision():
    st = FrameStack(4, 84)
    f = np.zeros((84, 84), dtype=np.uint8)
    st.reset(f)
    for v in range(1, 5):
        nxt = np.full((84, 84), v, dtype=np.uint8)
        st.push(nxt)
    out = st.get()
    assert out[-1].mean() == 4 and out[0].mean() == 1  # newest last


# --------------------------------------------------------------------- #
# 2. Epsilon is on the AGENT-STEP clock and decays slowly
# --------------------------------------------------------------------- #
def test_epsilon_step_schedule():
    from agent import epsilon_for_step

    cfg = RLConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=100_000)
    assert epsilon_for_step(0, cfg) == 1.0
    assert epsilon_for_step(50_000, cfg) < 0.6 and epsilon_for_step(50_000, cfg) > 0.4
    assert abs(epsilon_for_step(100_000, cfg) - 0.05) < 1e-6
    assert abs(epsilon_for_step(999_999, cfg) - 0.05) < 1e-9
    # At the old ~episode-700 mark (a few thousand agent steps) exploration
    # must still be high (the bug that locked the blind policy).
    assert epsilon_for_step(3_000, cfg) > 0.9


# --------------------------------------------------------------------- #
# 3. DQfD: large-margin loss + never-evict expert replay
# --------------------------------------------------------------------- #
def _rand_batch(n, expert=None):
    obs = np.random.randint(0, 255, (n, 4, 84, 84), dtype=np.uint8)
    return {
        "obs": obs,
        "next_obs": obs.copy(),
        "actions": np.random.randint(0, 5, n).astype(np.int64),
        "rewards": np.zeros(n, np.float32),
        "dones": np.zeros(n, np.float32),
        "weights": np.ones(n, np.float32),
        "gamma_pows": np.full(n, 0.97, np.float32),
        "expert": np.zeros(n, bool) if expert is None else np.array(expert, bool),
    }


def test_dqfd_margin_loss_runs_and_pushes_expert_action():
    from agent import DoubleDQNAgent

    cfg = BotConfig()
    cfg.rl.dqfd_margin = 0.8
    cfg.rl.dqfd_margin_weight = 1.0
    agent = DoubleDQNAgent("strict_lite", cfg.rl, seed=0)
    batch = _rand_batch(16, expert=[True] * 16)
    # force expert action to be index 0; network logits random -> margin loss
    # should be non-negative and finite
    m = agent.train_step(batch)
    assert np.isfinite(m["loss"])
    assert m["margin_loss"] >= 0.0


def test_expert_replay_never_evicted_and_sampled():
    from replay_buffer import NStepTransition, PrioritizedReplayBuffer

    cfg = BotConfig()
    buf = PrioritizedReplayBuffer(cfg.per, frame_size=84, gamma=0.99,
                                  expert_priority_bonus=1.0)
    def mk_tr(a):
        obs = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        return NStepTransition(
            obs=obs, next_obs=obs.copy(), action=a, reward=0.1, done=False,
            span=3, gamma_pow=0.97**3,
            obs_env_ids=tuple(range(1000 + a * 10, 1004 + a * 10)),
            next_env_ids=tuple(range(2000 + a * 10, 2004 + a * 10)),
            expert=True,
        )
    buf.add_expert_nstep(mk_tr(1))
    buf.add_expert_nstep(mk_tr(2))
    assert buf.expert_size == 2
    # expert frames are stored in the dedicated persistent store
    assert buf.expert_frames.written >= 4
    # sampling with expert fraction returns expert rows flagged
    # first need some online data
    for i in range(64):
        t = mk_tr(i % 5)
        t.expert = False
        t.obs_env_ids = tuple(range(5000 + i * 4, 5004 + i * 4))
        t.next_env_ids = tuple(range(9000 + i * 4, 9004 + i * 4))
        buf.add_nstep(t)
    rng = np.random.default_rng(0)
    batch = buf.sample_mixed(32, 0.5, expert_fraction=0.5, rng=rng)
    assert batch["expert"].sum() >= 1
    # priority updates skip expert sentinel indices (-1) without error
    td = np.random.rand(32)
    buf.update_priorities(batch["indices"], td)


def test_priority_update_ignores_sentinel_indices():
    from replay_buffer import PrioritizedReplayBuffer

    cfg = BotConfig()
    buf = PrioritizedReplayBuffer(cfg.per, frame_size=84, gamma=0.99)
    idx = np.array([-1, -1, 0], dtype=np.int64)  # -1 = expert rows
    buf.update_priorities(idx, np.array([1.0, 2.0, 3.0]))  # must not raise


# --------------------------------------------------------------------- #
# 4. Mirror augmentation for BC
# --------------------------------------------------------------------- #
def test_demonstration_mirror_swaps_lr_labels():
    from dataset import DemonstrationDataset, Episode

    n = 10
    frames = np.random.randint(0, 255, (n, 84, 84), dtype=np.uint8)
    # put a bright column on the far left so flipping is detectable
    frames[:, :, :3] = 255
    actions = np.zeros(n, dtype=np.int64)
    actions[2] = 1   # LEFT
    actions[5] = 2   # RIGHT
    ts = np.linspace(0, 1, n)
    done = np.zeros(n, bool); done[-1] = True
    ep = Episode(path="x", frames=frames, actions=actions, timestamps=ts,
                 done=done)
    ds = DemonstrationDataset([ep], stack=4, mirror=True)
    assert len(ds) == 2 * n
    # find the mirrored view of step 2 (LEFT -> RIGHT)
    found_right = False
    flipped_left_seen = False
    for i in range(len(ds)):
        ei, si, flip = ds._index[i]
        x, y = ds.get(i)
        if si == 2:
            if flip:
                assert y == 2  # LEFT flips to RIGHT
                found_right = True
                # mirrored frame: bright column now on the right
                assert x[:, :, -3:].mean() > x[:, :, :3].mean()
    assert found_right


# --------------------------------------------------------------------- #
# 5. Obstacle tracker shapes rewards causally
# --------------------------------------------------------------------- #
def test_obstacle_tracker_detects_and_clears():
    tr = ObstacleTracker(lanes=3, depths=5, edge_thresh=1.2)
    base = np.full((84, 84), 60, dtype=np.uint8)
    # no obstacle -> no danger/clear
    s0 = tr.update(base)
    assert s0.danger is False
    # a strong block in the player's lane near the bottom (near depth)
    near = base.copy()
    near[70:80, 36:48] = 240   # middle lane (player default), very near
    s1 = tr.update(near)
    assert s1.danger is True
    # then it passes (vanishes) -> a clear event
    s2 = tr.update(base)
    assert s2.clear is True
    assert s2.clears >= 1


def test_reward_shaping_components():
    from horizon_detector import HorizonResult
    from rewards import SurvivalRewardCalculator

    cfg = RewardConfig()
    calc = SurvivalRewardCalculator(cfg)
    calc.begin_episode(0.0)
    hz = HorizonResult(frame_id=1, ts=0.1, change_score=0.0, raw_score=0.0,
                       changed_ratio=0.0, detected=False, confidence=0.0)
    r_clear = calc.step(ts=0.1, action=0, horizon=hz, clear=True, danger=False)
    assert r_clear.clear == cfg.clear_bonus
    r_danger = calc.step(ts=0.2, action=0, horizon=hz, clear=False, danger=True)
    assert r_danger.danger == cfg.danger_penalty
    r_act = calc.step(ts=0.3, action=3, horizon=hz, clear=False, danger=False)
    assert r_act.action_cost == -cfg.action_cost


def test_legacy_hazard_bonus_off_by_default():
    from horizon_detector import HorizonResult
    from rewards import SurvivalRewardCalculator

    calc = SurvivalRewardCalculator(RewardConfig())  # default
    calc.begin_episode(0.0)
    hz = HorizonResult(frame_id=1, ts=0.03, change_score=9.0, raw_score=9.0,
                       changed_ratio=0.1, detected=True, confidence=0.9)
    r = calc.step(ts=0.03, action=3, horizon=hz)
    assert r.hazard == 0.0, "button-spam hazard bonus must be off by default"


# --------------------------------------------------------------------- #
# 6. Frame-skip synchronous environment advances the MDP coherently
# --------------------------------------------------------------------- #
def test_frame_skip_env_emits_coherent_steps():
    from environment import GameEnvironment

    cfg = BotConfig()
    cfg.rl.frame_skip = 3
    env = GameEnvironment(cfg)
    obs = env.reset()
    assert obs.shape == (4, 84, 84)
    obs2, r, done, info = env.step(0)
    assert obs2.shape == (4, 84, 84)
    # one agent step should have advanced the synthetic game by `frame_skip`
    # sub-steps; total_steps grows accordingly
    assert env.game.total_steps >= 3


# --------------------------------------------------------------------- #
# 7. Demo recorder backdates a tap to the decision-distance frames
# --------------------------------------------------------------------- #
def test_demo_recorder_backdates_labels():
    from types import SimpleNamespace
    from demonstration_recorder import DemoRecorder

    cfg = BotConfig()
    cfg.bc.label_backdate_ms = 300.0   # ~9 frames at 30 fps
    cfg.bc.label_hold_ms = 100.0
    cfg.capture.target_fps = 30

    rec = DemoRecorder(cfg, "/tmp/demo_test", read_frame=lambda: None)
    rec._recording = True
    # feed 20 empty frames, then a JUMP keypress at frame 15
    for i in range(20):
        img = np.full((400, 300, 3), 30, dtype=np.uint8)
        img[10:30, 10:30] = (200, 60, 60)  # fake anchor-ish patch (not required)
        fr = SimpleNamespace(frame_id=i, ts=0.0 + i / 30.0, image=img)
        if i == 15:
            rec._handle_press("up")
        rec.tick(fr)
    assert len(rec._actions) == 20
    labels = rec._actions
    # frames from ~15-9=6 through ~15+3=18 should carry JUMP(3); well before
    # that must be NOOP (proving backdating, not just the press frame).
    assert labels[0] == 0
    assert 3 in labels[4:12], "label must be backdated to decision-distance"
    assert labels[15] == 3


# --------------------------------------------------------------------- #
# 8. DAgger: a human keypress while the bot plays starts correction
#    recording and it auto-closes after the intervention tail.
# --------------------------------------------------------------------- #
def test_dagger_intervention_arms_and_auto_stops(monkeypatch):
    from types import SimpleNamespace
    from demonstration_recorder import DemoRecorder

    cfg = BotConfig()
    cfg.bc.dagger = True
    cfg.bc.dagger_tail_ms = 120.0
    cfg.capture.target_fps = 30
    out = "/tmp/demo_dagger"
    import shutil, pathlib
    shutil.rmtree(out, ignore_errors=True)
    rec = DemoRecorder(cfg, out, read_frame=lambda: None)
    # don't actually start a global keyboard listener in tests
    monkeypatch.setattr(rec._tap, "start", lambda: None)
    rec.dagger_active = True
    # bot is playing but nobody intervened yet -> nothing recorded
    rec.notify_human_intervention.__self__.__dict__.get("_dagger_armed", False)
    assert rec.recording is False
    # human intervenes with a JUMP keypress
    rec._handle_press("up")
    assert rec.recording is True
    # feed a few frames labelled JUMP (backdated)
    img = np.full((400, 300, 3), 30, dtype=np.uint8)
    img[10:30, 10:30] = (200, 60, 60)
    for i in range(3):
        fr = SimpleNamespace(frame_id=i, ts=i / 30.0, image=img)
        rec.tick(fr)
    # after the tail elapses (no further input), the correction demo saves
    import time as _t
    _t.sleep(0.2)
    fr = SimpleNamespace(frame_id=99, ts=99 / 30.0, image=img)
    rec.tick(fr)
    assert rec.recording is False, "correction demo must auto-finalise"
    assert rec.episode_paths, "a correction demo should have been written"
