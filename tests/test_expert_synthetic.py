"""Tests for :class:`expert_synthetic.SyntheticExpert`.

These tests pin:

* The expert picks the right dodge action for each
  obstacle kind (JUMP for low barriers, SLIDE for high
  barriers, LEFT/RIGHT for lane blockers).
* The expert picks NOOP when there is no threat or the
  threat is in another lane.
* The expert stays safe across 20 random seeds long
  enough to provide useful BC data (>= 15s mean).
* The expert's ``collect_demonstration`` returns a
  consistent (frames, actions, rewards) triple that
  ``DemonstrationDataset`` can load.
"""

from __future__ import annotations

import numpy as np
import pytest

from environment import SyntheticGame
from expert_synthetic import NOOP, LEFT, RIGHT, JUMP, SLIDE, SyntheticExpert


@pytest.fixture()
def expert() -> SyntheticExpert:
    return SyntheticExpert()


class TestSyntheticExpertAct:
    def test_noop_with_no_obstacles(self, expert: SyntheticExpert) -> None:
        assert expert.act(player_lane=1, obstacles=[]) == NOOP

    def test_jump_for_low_barrier(self, expert: SyntheticExpert) -> None:
        ob = {"kind": "low", "lane": 1, "prog": 0.5, "speed": 0.5}
        assert expert.act(player_lane=1, obstacles=[ob]) == JUMP

    def test_slide_for_high_barrier(self, expert: SyntheticExpert) -> None:
        ob = {"kind": "high", "lane": 1, "prog": 0.5, "speed": 0.5}
        assert expert.act(player_lane=1, obstacles=[ob]) == SLIDE

    def test_dodge_lane_blocker(self, expert: SyntheticExpert) -> None:
        ob = {"kind": "lane", "lane": 1, "prog": 0.5, "speed": 0.5}
        a = expert.act(player_lane=1, obstacles=[ob])
        assert a in (LEFT, RIGHT)

    def test_noop_when_blocker_in_other_lane(
            self, expert: SyntheticExpert) -> None:
        ob = {"kind": "lane", "lane": 0, "prog": 0.5, "speed": 0.5}
        assert expert.act(player_lane=1, obstacles=[ob]) == NOOP

    def test_noop_when_obstacle_already_passed(
            self, expert: SyntheticExpert) -> None:
        # An obstacle with prog=1.0 has already
        # impacted; we should not act on it.
        ob = {"kind": "lane", "lane": 1, "prog": 1.0, "speed": 0.5}
        assert expert.act(player_lane=1, obstacles=[ob]) == NOOP

    def test_picks_left_when_blocker_on_right_of_player(
            self, expert: SyntheticExpert) -> None:
        # Player in lane 0; blocker in lane 0; safe
        # lanes are 1 and 2.  Closest safe lane is 1,
        # so we should go RIGHT.
        ob = {"kind": "lane", "lane": 0, "prog": 0.5, "speed": 0.5}
        a = expert.act(player_lane=0, obstacles=[ob])
        assert a == RIGHT


class TestSyntheticExpertSurvival:
    def test_expert_survives_reasonable_time(
            self, expert: SyntheticExpert) -> None:
        """The expert must outlive random play by a
        wide margin.  Random play is ~4s on average
        (v1.19 audit).  A 20-seed mean >= 15s proves
        the expert encodes useful structure."""
        survivals = []
        for seed in range(20):
            env = SyntheticGame(seed=seed)
            env.reset()
            for _ in range(900):
                a = expert.act(env.player_lane, env.obstacles)
                env.step(a)
                if env.dead:
                    break
            survivals.append(env.total_steps / 30.0)
        mean_s = float(np.mean(survivals))
        assert mean_s >= 15.0, (
            f"expert mean survival {mean_s:.2f}s is below "
            f"the 15s threshold — the BC pretrain will be "
            f"trained on poor-quality demos")


class TestSyntheticExpertCollectDemonstration:
    def test_returns_correct_shapes(
            self, expert: SyntheticExpert) -> None:
        env = SyntheticGame(seed=0)
        frames, actions, rewards = expert.collect_demonstration(
            env, max_steps=200)
        # 4-frame grayscale stacks of 84x84.
        assert frames.ndim == 4
        assert frames.shape[1] == 4
        assert frames.shape[2] == 84
        assert frames.shape[3] == 84
        assert frames.dtype == np.uint8
        assert actions.ndim == 1
        assert actions.dtype == np.int64
        assert actions.shape[0] == frames.shape[0]
        assert rewards.ndim == 1
        assert rewards.shape[0] == frames.shape[0]
        # All actions must be valid (0..4).
        assert int(actions.max()) <= 4
        assert int(actions.min()) >= 0
