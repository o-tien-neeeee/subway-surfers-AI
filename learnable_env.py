"""Learnable 3-lane synthetic environment.

This is a *deliberately simpler* env than
:class:`environment.SyntheticGame`.  Its purpose is to
*guarantee* that the agent can learn the optimal policy
when the algorithm is correct — so the audit has a clean
"does it learn?" signal.

Rules
-----
* 3 lanes (0, 1, 2).  Player starts in lane 1.
* Every ``obstacle_period`` frames, an obstacle spawns in a
  *known* lane.  The lane is a deterministic function of
  the spawn index, so the optimal policy is "if obstacle
  lane != player lane, NOOP; if obstacle lane == player lane,
  switch to a safe lane".
* The obstacle reaches the player in ``approach_time``
  frames — so the agent has *time* to react, just like
  the real game.
* If the player is in the obstacle lane when it arrives,
  the player dies.  Otherwise, +0.1 alive reward.
* Episode length is capped at ``max_steps`` (default 900
  = 30 s at 30 FPS — matches the user's KPI of "≥30 s").

The state observation is a *minimal* but *information-rich*
representation: a one-hot of the current player lane
(3 dims) + a one-hot of the *next* obstacle lane
(3 dims) + a *time-to-impact* scalar (frames until the
obstacle arrives) — 7 numbers total.  This is *not* the
real game; it's a benchmark that strips away perception
so the algorithm itself is the only variable.

When the agent solves this env (≥30 s average survival
after 500 episodes), the algorithm is verified.  When it
does not, the audit pinpoints which component is broken.

Tests
-----
See :mod:`tests.test_learnable_env` for the determinism
guarantee and the optimal-policy upper bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LearnableEnvConfig:
    n_lanes: int = 3
    start_lane: int = 1
    # An obstacle spawns every ``obstacle_period`` frames.
    # The lane cycles deterministically through 0, 1, 2,
    # 0, 1, 2, ... (the simplest possible "structure" the
    # agent can exploit).
    obstacle_period: int = 30  # 1 second at 30 FPS
    # Obstacles approach at this speed.  A larger value =
    # the agent has *more* time to react.
    approach_time: int = 15
    # Episode length cap.
    max_steps: int = 900
    # Reward shaping.
    alive_reward: float = 0.1
    death_penalty: float = -1.0
    # +0.5 for switching to the safe lane when the current
    # lane is targeted (i.e. *proactive* dodge).
    dodge_bonus: float = 0.5
    # -0.05/frame for being in a lane that *will* be hit
    # (i.e. *anticipatory* danger signal).  This is the
    # dense signal that helps the agent learn *which* lane
    # is safe *before* the obstacle arrives.
    danger_penalty: float = 0.05


class LearnableEnv:
    """The 3-lane learnable env.  See module docstring."""

    def __init__(self, cfg: LearnableEnvConfig | None = None,
                 seed: int = 0) -> None:
        self.cfg = cfg or LearnableEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.player_lane = self.cfg.start_lane
        self.dead = False
        # Spawn schedule: each spawn is an (lane, spawn_step,
        # impact_step) tuple.  Pre-computed so the audit can
        # verify determinism.
        self.spawns: list[tuple[int, int, int]] = []
        # Track which spawns have been resolved (so the
        # reward shaping only fires for upcoming threats).
        self.next_spawn_idx = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        self.t = 0
        self.player_lane = self.cfg.start_lane
        self.dead = False
        self.spawns = []
        # Pre-compute the full spawn schedule.  The lane
        # cycles through 0, 1, 2 deterministically.
        n_spawns = self.cfg.max_steps // self.cfg.obstacle_period + 2
        for i in range(n_spawns):
            spawn_step = i * self.cfg.obstacle_period
            impact_step = spawn_step + self.cfg.approach_time
            lane = i % self.cfg.n_lanes
            self.spawns.append((lane, spawn_step, impact_step))
        self.next_spawn_idx = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        """7-dim state: 3 player one-hot + 3 next-obstacle
        one-hot + 1 time-to-impact (normalised to [0, 1])."""
        obs = np.zeros(3 + 3 + 1, dtype=np.float32)
        obs[self.player_lane] = 1.0
        if self.next_spawn_idx < len(self.spawns):
            lane, spawn_step, impact_step = self.spawns[self.next_spawn_idx]
            obs[3 + lane] = 1.0
            tti = max(0.0, (impact_step - self.t)) / max(1, self.cfg.approach_time)
            obs[6] = float(tti)
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Apply action, advance one frame, return (obs, reward, done, info)."""
        if self.dead:
            return self.observation(), 0.0, True, {}
        info: dict = {"alive": 0.0, "dodge": 0.0, "danger": 0.0,
                      "death": 0.0, "switched": False}
        prev_lane = self.player_lane
        # Action interpretation: 0=NOOP, 1=LEFT, 2=RIGHT, 3/4
        # are NOOPs in this env (no JUMP/SLIDE because the
        # obstacles are 2D).  Keeps the action space the same
        # shape as the real game so the agent code is shared.
        if action == 1:
            self.player_lane = max(0, self.player_lane - 1)
            info["switched"] = prev_lane != self.player_lane
        elif action == 2:
            self.player_lane = min(self.cfg.n_lanes - 1, self.player_lane + 1)
            info["switched"] = prev_lane != self.player_lane
        # Check the *current* spawn (the one we are waiting
        # to impact).  If the obstacle has just reached the
        # player line, it lands and the player dies iff in
        # the obstacle lane.
        if self.next_spawn_idx < len(self.spawns):
            lane, spawn_step, impact_step = self.spawns[self.next_spawn_idx]
            if self.t >= impact_step:
                # Obstacle arrives now.
                if self.player_lane == lane:
                    self.dead = True
                    info["death"] = self.cfg.death_penalty
                # Move on to the next spawn.
                self.next_spawn_idx += 1
        # Reward shaping.
        info["alive"] = self.cfg.alive_reward
        if self.next_spawn_idx < len(self.spawns):
            lane, spawn_step, impact_step = self.spawns[self.next_spawn_idx]
            if (self.player_lane == lane
                    and self.t < impact_step
                    and impact_step - self.t <= self.cfg.approach_time):
                # The player is in the targeted lane — penalise
                # proportional to the proximity (close = bad).
                tti_frac = max(0.0, (impact_step - self.t)) / max(
                    1, self.cfg.approach_time)
                info["danger"] = -self.cfg.danger_penalty * tti_frac
        if info["switched"] and self.next_spawn_idx < len(self.spawns):
            lane, spawn_step, impact_step = self.spawns[self.next_spawn_idx]
            # Switching out of a dangerous lane earns the
            # *dodge* bonus (only if the current lane was
            # about to be hit).
            if prev_lane == lane and self.t < impact_step:
                info["dodge"] = self.cfg.dodge_bonus
        # Advance time.
        self.t += 1
        if self.t >= self.cfg.max_steps:
            self.dead = True
        reward = info["alive"] + info["dodge"] + info["danger"] + info["death"]
        return self.observation(), reward, self.dead, info
