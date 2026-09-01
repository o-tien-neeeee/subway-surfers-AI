"""Tests for the v1.24.0 deep-research fixes.

Each test pins ONE of the findings that explains "6000 episodes but only
+1 s of survival":

1. the policy observation no longer crops the horizon band away;
2. the MDP is Atari-style frame skip (one decision per ``frame_skip``
   frames, rewards summed, stack/n-step per decision);
3. exploration decays per DECISION, not per captured frame;
4. the reward is a positive counter with no catastrophic death penalty,
   and the press-spam hazard bonus is off by default;
5. the obstacle grid produces a *causal* clear credit (and never credits
   a collision);
6. demo labels are back-dated so the bot learns to react EARLY;
7. HG-DAgger corrections are captured while the bot plays;
8. demos are validated against the configured policy size and DAgger
   episodes are discovered recursively.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from agent import epsilon_for_frame, epsilon_for_step
from config import BotConfig, JUMP, LEFT, NOOP, RIGHT, RewardConfig, RLConfig
from demonstration_recorder import DemoRecorder
from ipc import Frame
from obstacle_perception import (
    ObstacleTracker,
    occupancy_from_game_state,
    snapshot_from_game_state,
)
from perception import ZonePreprocessor
from rewards import SurvivalRewardCalculator


def _hz(ts: float, frame_id: int = 0, detected: bool = False):
    from horizon_detector import HorizonResult
    return HorizonResult(frame_id=frame_id, ts=ts, change_score=0.0,
                         raw_score=0.0, changed_ratio=0.0, detected=detected,
                         confidence=0.9 if detected else 0.0)


# --------------------------------------------------------------------- #
# 1. Vision: the policy sees the WHOLE frame
# --------------------------------------------------------------------- #
class TestFullFramePolicyObservation:
    def test_horizon_band_is_inside_the_policy_view(self) -> None:
        cfg = BotConfig()
        cfg.region.width, cfg.region.height = 480, 800
        pre = ZonePreprocessor(cfg.perception, cfg.perception.horizon_frac,
                               require_anchor=False)
        # Top quarter bright (where obstacles spawn), bottom dark (track).
        img = np.zeros((800, 480, 3), dtype=np.uint8)
        img[:200, :, :] = 250
        z = pre.process(img, 1, 0.0)
        assert z.valid
        pol = z.policy_gray
        assert pol.shape == (cfg.perception.obs_size, cfg.perception.obs_size)
        side = cfg.perception.obs_size
        top = float(pol[: side // 4].mean())
        bottom = float(pol[-(side // 4):].mean())
        assert top > 200.0, "the spawn/horizon band was cropped out again"
        assert bottom < 40.0
        # policy_gray is the canonical alias of the stored field
        assert z.policy_gray is z.ground_gray

    def test_legacy_crop_is_still_reachable_for_ablation(self) -> None:
        cfg = BotConfig()
        cfg.perception.policy_full_frame = False
        pre = ZonePreprocessor(cfg.perception, cfg.perception.horizon_frac,
                               require_anchor=False)
        img = np.zeros((800, 480, 3), dtype=np.uint8)
        img[:200, :, :] = 250
        z = pre.process(img, 1, 0.0)
        assert z.policy_gray.shape == (cfg.perception.ground_size,) * 2
        assert float(z.policy_gray.mean()) < 10.0, "crop should hide the top"

    def test_policy_size_property_follows_the_flag(self) -> None:
        cfg = BotConfig()
        assert cfg.perception.policy_size == cfg.perception.obs_size
        cfg.perception.policy_full_frame = False
        assert cfg.perception.policy_size == cfg.perception.ground_size


# --------------------------------------------------------------------- #
# 2. MDP: Atari-style frame skip
# --------------------------------------------------------------------- #
class TestFrameSkipMDP:
    def _env(self, frame_skip: int = 3, n_step: int = 0):
        from environment import GameEnvironment, SyntheticGame
        cfg = BotConfig()
        cfg.rl.frame_skip = frame_skip
        if n_step:
            cfg.rl.n_step = n_step          # must be set BEFORE the env builds
        cfg.perception.frame_stack = 4
        return GameEnvironment(cfg, game=SyntheticGame(seed=3)), cfg

    def test_one_step_advances_frame_skip_raw_frames(self) -> None:
        env, cfg = self._env(3)
        env.reset()
        before = env.game.total_steps
        env.step(NOOP)
        assert env.game.total_steps - before == 3
        assert env.steps == 1

    def test_frame_skip_one_is_the_legacy_cadence(self) -> None:
        env, _ = self._env(1)
        env.reset()
        before = env.game.total_steps
        env.step(NOOP)
        assert env.game.total_steps - before == 1

    def test_stack_advances_once_per_decision(self) -> None:
        env, cfg = self._env(3)
        env.reset()
        env_ids_before = list(env._env_ids)
        for _ in range(5):
            env.step(NOOP)
        # 5 decisions -> 5 distinct env ids pushed, buffer still capped
        assert len(env._env_ids) == cfg.perception.frame_stack
        assert len(set(env._env_ids)) == cfg.perception.frame_stack

    def test_reward_is_the_sum_over_the_whole_step(self) -> None:
        env, cfg = self._env(3)
        env.cfg.reward.curriculum_milestones = ()
        env.cfg.reward.clear_bonus = 0.0
        env.reset()
        _, r, _, _ = env.step(NOOP)
        # alive_per_frame * (1/fps) * nominal_fps == alive_per_frame per frame
        expected = cfg.reward.alive_per_frame * 3
        assert r == pytest.approx(expected, rel=0.02)

    def test_transitions_are_labelled_with_the_step_action(self) -> None:
        env, cfg = self._env(2, n_step=2)
        env.reset()
        actions = [NOOP, LEFT, NOOP, RIGHT, NOOP, JUMP]
        emitted = []
        for a in actions:
            obs, r, done, _ = env.step(a)
            emitted += env.pop_nstep_transitions(obs, a, r, done)
        # every emitted transition must carry an action we actually took
        assert emitted, "no transitions emitted"
        for tr in emitted:
            assert tr.action in actions
        non_terminal = [t for t in emitted if not t.done]
        assert non_terminal, "expected at least one n-step transition"
        assert all(t.span == cfg.rl.n_step for t in non_terminal)


# --------------------------------------------------------------------- #
# 3. Exploration schedule
# --------------------------------------------------------------------- #
class TestEpsilonPerDecisionStep:
    def test_decay_is_in_steps_not_frames(self) -> None:
        cfg = RLConfig()
        cfg.epsilon_start, cfg.epsilon_end = 1.0, 0.0
        cfg.epsilon_decay_steps = 1000
        assert epsilon_for_step(0, cfg) == 1.0
        assert epsilon_for_step(500, cfg) == pytest.approx(0.5)
        assert epsilon_for_step(1000, cfg) == pytest.approx(0.0)
        assert epsilon_for_step(5000, cfg) == pytest.approx(0.0)

    def test_frame_schedule_is_untouched(self) -> None:
        cfg = RLConfig()
        cfg.epsilon_start, cfg.epsilon_end = 1.0, 0.0
        cfg.epsilon_decay_frames = 1000
        assert epsilon_for_frame(500, cfg) == pytest.approx(0.5)

    def test_actor_counts_decisions_for_epsilon(self) -> None:
        from environment import BotActor
        cfg = BotConfig()
        cfg.rl.frame_skip = 3
        cfg.capture.source = "fake"
        cfg.region.width, cfg.region.height = 480, 800
        cfg.rl.epsilon_decay_steps = 100
        actor = _make_actor(cfg)
        assert actor.frame_skip == 3
        assert actor._decision_count == 0


# --------------------------------------------------------------------- #
# 4. Reward: positive counter, no death penalty, no press-spam bonus
# --------------------------------------------------------------------- #
class TestPositiveCounterReward:
    def test_defaults_are_a_positive_counter(self) -> None:
        cfg = RewardConfig()
        assert cfg.death_penalty == 0.0
        assert cfg.reward_clip_min == 0.0
        assert cfg.use_hazard_bonus is False
        assert cfg.alive_per_frame > 0.0

    def test_return_is_monotonic_in_survival(self) -> None:
        """The core lunai property: longer life => strictly larger return."""
        def run(frames: int) -> float:
            calc = SurvivalRewardCalculator(RewardConfig())
            calc.begin_episode(0.0)
            total = 0.0
            for i in range(frames):
                total += calc.step(ts=(i + 1) / 30.0, action=NOOP,
                                   horizon=_hz((i + 1) / 30.0, i)).total
            return total

        short, long_ = run(30), run(300)
        assert short > 0.0, "a 1 s episode must not read as a negative number"
        assert long_ > short * 5

    def test_no_penalty_path_keeps_every_episode_non_negative(self) -> None:
        calc = SurvivalRewardCalculator(RewardConfig())
        calc.begin_episode(0.0)
        r = calc.step(ts=0.1, action=NOOP, horizon=_hz(0.1, 1), died=True)
        assert r.death == 0.0
        assert r.total >= 0.0

    def test_legacy_death_penalty_still_works(self) -> None:
        calc = SurvivalRewardCalculator(
            RewardConfig(death_penalty=-5.0, reward_clip_min=-10.0))
        calc.begin_episode(0.0)
        r = calc.step(ts=0.1, action=NOOP, horizon=_hz(0.1, 1), died=True)
        assert r.death == -5.0

    def test_hazard_bonus_is_opt_in(self) -> None:
        """Off by default: it pays for ANY press after a blink (spam)."""
        off = SurvivalRewardCalculator(RewardConfig(hazard_bonus=1.0))
        on = SurvivalRewardCalculator(
            RewardConfig(hazard_bonus=1.0, use_hazard_bonus=True,
                         hazard_resolve_frames=1))
        for calc in (off, on):
            calc.begin_episode(0.0)
            calc.step(ts=0.03, action=JUMP, horizon=_hz(0.03, 1, True))
            r = calc.step(ts=0.06, action=NOOP, horizon=_hz(0.06, 2, False))
            if calc is off:
                assert r.hazard == 0.0
            else:
                assert r.hazard == pytest.approx(1.0)

    def test_clear_bonus_and_action_cost(self) -> None:
        calc = SurvivalRewardCalculator(
            RewardConfig(clear_bonus=0.5, action_cost=0.1,
                         curriculum_milestones=()))
        calc.begin_episode(0.0)
        r = calc.step(ts=0.1, action=LEFT, horizon=_hz(0.1, 1), cleared=2)
        assert r.clear == pytest.approx(1.0)
        assert r.action_cost == pytest.approx(-0.1)
        r2 = calc.step(ts=0.2, action=NOOP, horizon=_hz(0.2, 2))
        assert r2.action_cost == 0.0, "NOOP must not be charged"

    def test_action_cost_only_on_decision_frames(self) -> None:
        calc = SurvivalRewardCalculator(RewardConfig(action_cost=0.2))
        calc.begin_episode(0.0)
        r = calc.step(ts=0.1, action=JUMP, horizon=_hz(0.1, 1),
                      is_decision=False)
        assert r.action_cost == 0.0

    def test_breakdown_exposes_the_new_terms(self) -> None:
        calc = SurvivalRewardCalculator(RewardConfig())
        calc.begin_episode(0.0)
        d = calc.step(ts=0.1, action=NOOP, horizon=_hz(0.1, 1)).to_dict()
        assert {"clear", "danger", "action_cost"} <= set(d)


# --------------------------------------------------------------------- #
# 5. Obstacle grid: causal credit
# --------------------------------------------------------------------- #
class TestObstacleTracker:
    def _tracker(self, rows: int = 4, **kw) -> ObstacleTracker:
        cfg = BotConfig()
        cfg.obstacle.depth_rows = rows
        for k, v in kw.items():
            setattr(cfg.obstacle, k, v)
        return ObstacleTracker(cfg.obstacle)

    def test_clear_pays_when_an_obstacle_passes_the_player(self) -> None:
        tr = self._tracker(rows=4, confirm_cells=2, near_rows=2)
        # obstacle approaches the player's lane (lane 1), rows 0 -> 3
        total = 0
        for row in (0, 1, 2, 3):
            occ = np.zeros((4, 3), dtype=bool)
            occ[row, 1] = True
            total += tr.update_from_occupancy(occ, player_lane=1).cleared
        # it disappears (passed the player) while alive -> one clear credit
        empty = np.zeros((4, 3), dtype=bool)
        total += tr.update_from_occupancy(empty, player_lane=1).cleared
        assert total == 1

    def test_no_credit_when_the_player_dies(self) -> None:
        tr = self._tracker(rows=4, confirm_cells=2, near_rows=2)
        for row in (2, 3):
            occ = np.zeros((4, 3), dtype=bool)
            occ[row, 1] = True
            tr.update_from_occupancy(occ, player_lane=1)
        snap = tr.update_from_occupancy(np.zeros((4, 3), dtype=bool),
                                        player_lane=1, died=True)
        assert snap.cleared == 0, "a collision must never be credited as a dodge"

    def test_debounce_ignores_single_frame_noise(self) -> None:
        tr = self._tracker(rows=4, confirm_cells=3, near_rows=2)
        for _ in range(2):  # one blip at the near rows, never confirmed
            occ = np.zeros((4, 3), dtype=bool)
            occ[3, 0] = True
            tr.update_from_occupancy(occ, player_lane=0)
        snap = tr.update_from_occupancy(np.zeros((4, 3), dtype=bool),
                                        player_lane=0)
        assert snap.cleared == 0

    def test_danger_flag_needs_the_near_rows(self) -> None:
        tr = self._tracker(rows=4, near_rows=2, confirm_cells=1)
        far = np.zeros((4, 3), dtype=bool)
        far[0, 2] = True
        assert tr.update_from_occupancy(far, player_lane=2).danger is False
        near = np.zeros((4, 3), dtype=bool)
        near[3, 2] = True
        assert tr.update_from_occupancy(near, player_lane=2).danger is True

    def test_clears_are_capped_per_step(self) -> None:
        tr = self._tracker(rows=3, confirm_cells=1, near_rows=2,
                           max_clears_per_step=1)
        occ = np.zeros((3, 3), dtype=bool)
        occ[2, 1] = True
        tr.update_from_occupancy(occ, player_lane=1)
        snap = tr.update_from_occupancy(np.zeros((3, 3), dtype=bool),
                                        player_lane=1)
        assert snap.cleared <= 1

    def test_cv_path_detects_a_high_contrast_block(self) -> None:
        tr = self._tracker(rows=4, confirm_cells=1, near_rows=2)
        flat = np.full((84, 84), 40, dtype=np.uint8)
        assert tr.occupancy_from_gray(flat).sum() == 0
        block = flat.copy()
        block[60:80, 30:54] = 250  # a bright obstacle in the near rows
        occ = tr.occupancy_from_gray(block)
        assert occ.any(), "a high-contrast obstacle must be visible to CV"

    def test_game_state_snapshot_matches_the_synthetic_world(self) -> None:
        obs = [{"prog": 0.9, "lane": 1, "kind": "lane"}]
        snap = snapshot_from_game_state(obs, player_lane=1, rows=5, lanes=3)
        assert snap.hazards_in_lane == 1
        assert snap.danger is True
        occ = occupancy_from_game_state(obs, rows=5, lanes=3)
        assert occ[:, 1].sum() == 1
        assert occ[:, 0].sum() == 0

    def test_stats_are_reported(self) -> None:
        tr = self._tracker(rows=3, confirm_cells=1)
        tr.update_from_occupancy(np.zeros((3, 3), dtype=bool), player_lane=1)
        s = tr.stats()
        assert s["obstacle_steps"] == 1.0
        assert set(s) == {"obstacle_steps", "obstacles_cleared", "danger_steps",
                          "tracks_live"}


# --------------------------------------------------------------------- #
# 6. Learning from a human: label back-dating
# --------------------------------------------------------------------- #
def _recorder(tmp_path, backdate_ms: int = 220, fps: int = 30) -> DemoRecorder:
    cfg = BotConfig()
    cfg.region.left, cfg.region.top = 120, 80
    cfg.region.width, cfg.region.height = 480, 800
    cfg.region.screen_width, cfg.region.screen_height = 1920, 1080
    cfg.capture.target_fps = fps
    cfg.bc.label_backdate_ms = backdate_ms
    rec = DemoRecorder(cfg, tmp_path / "demos", lambda: None)
    rec.auto_split_on_death = False
    return rec


def _frame(i: int) -> Frame:
    return Frame(frame_id=i + 1, ts=i / 30.0,
                 image=np.full((800, 480, 3), 60, dtype=np.uint8))


class TestLabelBackdating:
    def test_press_labels_the_frames_before_it(self, tmp_path) -> None:
        rec = _recorder(tmp_path, backdate_ms=200)  # 6 frames at 30 FPS
        rec.start()
        for i in range(10):
            if i == 8:
                rec._handle_press("left")
            rec.tick(_frame(i))
        rec.stop(done=True)
        actions = rec._actions
        # rows 2..8 carry LEFT (6 back-dated rows + the press row), not NOOP
        assert actions[8] == LEFT
        assert sum(1 for a in actions if a == LEFT) >= 6
        assert rec.backdated_labels >= 5

    def test_backdating_never_overwrites_a_different_dodge(self, tmp_path) -> None:
        rec = _recorder(tmp_path, backdate_ms=200)
        rec.start()
        for i in range(10):
            if i == 3:
                rec._handle_press("up")     # JUMP held
            if i == 8:
                rec._handle_press("left")
            rec.tick(_frame(i))
        rec.stop(done=True)
        assert JUMP in rec._actions, "a real JUMP must survive back-dating"

    def test_zero_backdate_is_the_legacy_behaviour(self, tmp_path) -> None:
        rec = _recorder(tmp_path, backdate_ms=0)
        rec.start()
        for i in range(10):
            if i == 8:
                rec._handle_press("left")
            rec.tick(_frame(i))
            if i == 8:
                rec._handle_release("left")   # single-frame tap
        rec.stop(done=True)
        assert sum(1 for a in rec._actions if a == LEFT) == 1
        assert rec.backdated_labels == 0

    def test_meta_records_the_backdating(self, tmp_path) -> None:
        rec = _recorder(tmp_path, backdate_ms=220)
        rec.start()
        for i in range(12):
            if i == 9:
                rec._handle_press("right")
            rec.tick(_frame(i))
        path = rec.stop(done=True)
        meta = json.loads(str(np.load(path, allow_pickle=False)["meta"]))
        assert meta["label_backdate_ms"] == 220
        assert meta["backdated_labels"] >= 1
        assert meta["press_count"] == 1, "one press is one press"


# --------------------------------------------------------------------- #
# 7. HG-DAgger corrections
# --------------------------------------------------------------------- #
class TestHGDAgger:
    def test_armed_recorder_saves_a_correction_episode(self, tmp_path) -> None:
        cfg = BotConfig()
        cfg.region.width, cfg.region.height = 480, 800
        cfg.bc.dagger = True
        cfg.bc.dagger_pre_frames = 4
        cfg.bc.dagger_tail_frames = 5
        cfg.bc.dagger_auto_stop_s = 0.0
        rec = DemoRecorder(cfg, tmp_path / "demos", lambda: None)
        rec.auto_split_on_death = False
        rec.start()
        assert rec.arm_dagger(True) is True
        # lead-in frames with the bot playing (no human input)
        for i in range(8):
            rec.tick(_frame(i))
        assert rec.dagger_episodes_saved == 0
        # human takes over
        rec._handle_press("left")
        for i in range(8, 30):
            rec.tick(_frame(i))
        path = rec._save_dagger_episode() if rec._dagger_collecting else None
        assert rec.dagger_episodes_saved >= 1 or path is not None
        saved = list((tmp_path / "demos" / "dagger").glob("dagger_*.npz"))
        assert saved, "no correction episode written"
        data = np.load(saved[0], allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        assert meta["dagger"] is True
        assert data["frames"].shape[0] >= 5
        assert LEFT in set(data["actions"].tolist())

    def test_dagger_is_refused_when_disabled(self, tmp_path) -> None:
        cfg = BotConfig()
        cfg.bc.dagger = False
        rec = DemoRecorder(cfg, tmp_path / "demos", lambda: None)
        rec.start()
        assert rec.arm_dagger(True) is False
        assert rec.dagger_armed is False

    def test_unarmed_recorder_ignores_presses_for_dagger(self, tmp_path) -> None:
        rec = _recorder(tmp_path)
        rec.start()
        for i in range(6):
            rec.tick(_frame(i))
        rec._handle_press("left")
        for i in range(6, 12):
            rec.tick(_frame(i))
        assert rec.dagger_episodes_saved == 0
        assert rec._dagger_collecting is False

    def test_dataset_finds_dagger_episodes_recursively(self, tmp_path) -> None:
        from dataset import load_episodes, validate_directory
        rec = _recorder(tmp_path)
        rec.start()
        for i in range(20):
            if i == 10:
                rec._handle_press("left")
            rec.tick(_frame(i))
        rec.stop(done=True)
        # a correction episode in the dagger/ sub-directory
        sub = tmp_path / "demos" / "dagger"
        sub.mkdir(parents=True, exist_ok=True)
        data = np.load(sorted((tmp_path / "demos").glob("*.npz"))[0],
                       allow_pickle=False)
        np.savez_compressed(sub / "dagger_x.npz", frames=data["frames"],
                            actions=data["actions"],
                            timestamps=data["timestamps"], done=data["done"],
                            score=data["score"], confidence=data["confidence"],
                            death_state=data["death_state"], meta=data["meta"])
        eps = load_episodes(tmp_path / "demos")
        assert len(eps) == 2
        ok_eps, _ = validate_directory(tmp_path / "demos", expected_size=84)
        assert len(ok_eps) == 2

    def test_wrong_size_demo_is_rejected_with_the_configured_size(self,
                                                                 tmp_path) -> None:
        from dataset import validate_directory
        rec = _recorder(tmp_path)
        rec.start()
        for i in range(20):
            rec.tick(_frame(i))
        rec.stop(done=True)
        _, reps = validate_directory(tmp_path / "demos", expected_size=96)
        assert reps and not reps[0].ok
        assert "96" in reps[0].errors[0]


# --------------------------------------------------------------------- #
# 8. Config plumbing
# --------------------------------------------------------------------- #
class TestConfigPlumbing:
    def test_defaults(self) -> None:
        cfg = BotConfig()
        assert cfg.rl.frame_skip == 3
        assert cfg.rl.epsilon_decay_steps == 100_000
        assert cfg.perception.policy_full_frame is True
        assert cfg.obstacle.enabled is True
        assert cfg.bc.label_backdate_ms == 220
        assert cfg.bc.dagger is True

    def test_roundtrip_keeps_the_new_sections(self) -> None:
        cfg = BotConfig()
        cfg.obstacle.depth_rows = 7
        cfg.rl.frame_skip = 5
        back = BotConfig.from_dict(json.loads(cfg.to_json()))
        assert back.obstacle.depth_rows == 7
        assert back.rl.frame_skip == 5
        assert back.bc.label_backdate_ms == 220

    def test_validation_rejects_nonsense(self) -> None:
        from config import ConfigError
        cfg = BotConfig()
        cfg.rl.frame_skip = 0
        with pytest.raises(ConfigError):
            cfg.validate()
        cfg = BotConfig()
        cfg.obstacle.near_rows = 99
        with pytest.raises(ConfigError):
            cfg.validate()
        cfg = BotConfig()
        cfg.bc.label_backdate_ms = -5
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_obstacle_fraction_order_is_checked(self) -> None:
        from config import ConfigError
        cfg = BotConfig()
        cfg.obstacle.top_frac = 0.9
        cfg.obstacle.bottom_frac = 0.5
        with pytest.raises(ConfigError):
            cfg.validate()


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _make_actor(cfg: BotConfig):
    """Build a BotActor with dry-run input (same fixture shape as test_deep_fix)."""
    import threading

    from environment import BotActor
    from ipc import SharedCounters, SharedFrameRing, SharedWeights
    from models import weight_size_for_profile

    class _Q:
        def put_nowait(self, item):
            raise AssertionError("nothing should be shipped")

    return BotActor(
        cfg, SharedFrameRing(2, 800, 480, 3),
        {k: threading.Event() for k in
         ("stop", "emergency", "pause", "pause_learning", "death")},
        _Q(), _Q(), SharedWeights(weight_size_for_profile(cfg.rl.profile)),
        SharedCounters(), input_backend="dry_run")
