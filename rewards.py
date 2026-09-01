"""Reward computation: survival-first, hack-resistant, fully logged.

Design (requirement §13, cross-checked against the reward-hacking audit):

* Alive reward is proportional to REAL elapsed time (monotonic clock),
  expressed as ``alive_per_frame * dt * nominal_fps`` — adaptive frame
  skipping therefore cannot inflate or deflate it.
* Death penalty is a fixed -10 applied exactly once per episode (guarded by
  the caller's episode boundary, plus an internal once-flag).
* Hazard bonus (+0.1) uses a *pending event* mechanism: an event is opened
  when the horizon detector fires; the action chosen at/near that frame and
  the following ``hazard_resolve_frames`` valid frames decide whether the
  event "resolved".  The bonus is granted only for an already-observed
  resolution, so no future information leaks into the reward.
* Pixel-difference reward exists ONLY behind an ablation switch (default
  OFF), is tightly clipped (+/- pixel_diff_clip) and normalised.  The audit
  concluded it invites reward hacking (UI animations, flashes, camera
  shake) and it must prove itself via the ablation runner before enabling.
* Every component is logged per frame; the total is clipped to
  [reward_clip_min, reward_clip_max].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from config import RewardConfig
from horizon_detector import HorizonResult

#: Actions that count as a "meaningful dodge attempt".
DODGE_ACTIONS = frozenset({1, 2, 3, 4})  # LEFT, RIGHT, JUMP, SLIDE (NOOP excluded)


@dataclass
class RewardBreakdown:
    alive: float = 0.0
    death: float = 0.0
    hazard: float = 0.0
    pixel_diff: float = 0.0
    curriculum: float = 0.0
    #: v1.24.0 causal shaping (see :mod:`obstacle_perception`).
    clear: float = 0.0
    danger: float = 0.0
    action_cost: float = 0.0
    total: float = 0.0
    clipped: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, float]:
        return {
            "alive": self.alive,
            "death": self.death,
            "hazard": self.hazard,
            "pixel_diff": self.pixel_diff,
            "curriculum": self.curriculum,
            "clear": self.clear,
            "danger": self.danger,
            "action_cost": self.action_cost,
            "total": self.total,
            "clipped": 1.0 if self.clipped else 0.0,
        }


@dataclass
class PendingHazard:
    opened_frame_id: int
    #: DEEP-FIX: immutable.  This is the deadline anchor for hazard_expiry_s.
    #: It used to be overwritten on every re-registration, which is the bug.
    opened_ts: float
    #: DEEP-FIX: the "extend the window" semantics now live here, where they
    #: cannot interfere with expiry.
    last_seen_ts: float = 0.0
    action: Optional[int] = None
    action_frame_id: Optional[int] = None
    frames_seen: int = 0
    resolved: bool = False
    expired: bool = False
    granted: bool = False

    def __post_init__(self) -> None:
        if self.last_seen_ts == 0.0:
            self.last_seen_ts = self.opened_ts


class PendingHazardTracker:
    """Implements the pending-event hazard bonus (no future leakage).

    Lifecycle of an event:
      1. ``register`` when the horizon detector fires (frame n).
      2. ``observe_action`` records the action taken for frame n (the dodge
         attempt happens at the moment of danger — not seconds later).
      3. Each subsequent frame updates the event: after
         ``hazard_resolve_frames`` valid frames the event resolves positively
         only if the horizon is quiet again (the danger passed) and the
         recorded action was a dodge action.
      4. Events expire after ``hazard_expiry_s`` seconds without resolution.
    At most one event is open at a time; overlapping hazards extend the
    current event rather than minting new rewards.
    """

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg
        self._current: Optional[PendingHazard] = None
        self.events: list[PendingHazard] = []
        self.bonuses_granted = 0

    def register(self, horizon: HorizonResult) -> None:
        if self._current is not None and not self._current.resolved \
                and not self._current.expired:
            # Extend the existing danger window instead of double-rewarding.
            # DEEP-FIX: this line used to be `self._current.opened_ts =
            # horizon.ts`, overwriting the very field both expiry checks read.
            # `register` is called on *every* frame the horizon detector fires,
            # so a hazard that stayed visible kept pushing its own deadline
            # one frame ahead, forever.  Verified: after 10 s of continuous
            # danger with hazard_expiry_s=1.2, expired=0 and the event was
            # still open.  Worse, when the screen finally went quiet the stale
            # event resolved and paid out a bonus credited to the action
            # recorded on frame 0 -- 300 frames / 10 s of gameplay earlier.
            # A reward for a dodge at an obstacle the agent passed ten seconds
            # ago is exactly the credit-assignment corruption this module
            # exists to prevent.
            self._current.last_seen_ts = horizon.ts
            return
        self._current = PendingHazard(
            opened_frame_id=horizon.frame_id, opened_ts=horizon.ts
        )
        self.events.append(self._current)

    def observe_action(self, action: int, frame_id: int) -> None:
        if self._current is not None and not self._current.resolved \
                and not self._current.expired:
            if self._current.action is None:
                self._current.action = action
                # DEEP-FIX: record *where* the credited action came from so a
                # stale credit is diagnosable instead of silent.
                self._current.action_frame_id = frame_id

    def on_frame(self, horizon: HorizonResult, action: int) -> float:
        """Advance the open event with a new valid frame; return bonus (0/…).

        The frame the event opened on is NOT counted towards resolution —
        only the following ``hazard_resolve_frames`` valid frames are, so the
        bonus always reflects *observed aftermath*, never the trigger itself.
        """
        cur = self._current
        if cur is None or cur.resolved or cur.expired:
            return 0.0
        if horizon.frame_id == cur.opened_frame_id:
            self.observe_action(action, horizon.frame_id)
            return 0.0
        if horizon.ts - cur.opened_ts > self.cfg.hazard_expiry_s:
            cur.expired = True
            return 0.0
        self.observe_action(action, horizon.frame_id)
        cur.frames_seen += 1
        if cur.frames_seen >= self.cfg.hazard_resolve_frames and not horizon.detected:
            cur.resolved = True
            if cur.action in DODGE_ACTIONS:
                cur.granted = True
                self.bonuses_granted += 1
                return self.cfg.hazard_bonus
            return 0.0
        return 0.0

    def expire_old(self, now_ts: float) -> None:
        if (
            self._current is not None
            and not self._current.resolved
            and not self._current.expired
            and now_ts - self._current.opened_ts > self.cfg.hazard_expiry_s
        ):
            self._current.expired = True

    @property
    def has_open_event(self) -> bool:
        return self._current is not None and not self._current.resolved \
            and not self._current.expired

    def stats(self) -> dict[str, int]:
        resolved = sum(1 for e in self.events if e.resolved)
        return {
            "hazards_total": len(self.events),
            "hazards_resolved": resolved,
            "hazards_expired": sum(1 for e in self.events if e.expired),
            "bonuses": self.bonuses_granted,
            # DEEP-FIX: observability for the wedge above.  If this is large
            # the tracker is holding an event open far past hazard_expiry_s.
            "hazards_open_age_frames": (
                0 if self._current is None or not self.has_open_event
                else self._current.frames_seen
            ),
        }


class SurvivalRewardCalculator:
    """Total reward per actor step, from components to clipped total.

    Curriculum milestones
    ---------------------
    ``cfg.curriculum_milestones`` is a sorted tuple of survival times
    (seconds).  The first time the agent's elapsed survival in an
    episode crosses a milestone, a one-shot ``cfg.curriculum_bonus`` is
    paid; subsequent crossings of the SAME milestone are not paid.
    The milestones re-arm at episode start so a long-lived agent gets
    them every life.

    The bonus is small and bounded: the default schedule (2/5/10/20 s
    at +0.5 each) hands out at most +2.0 per episode, dwarfed by the
    -10 death penalty.  The point is not to inflate the reward
    landscape, it is to give the agent a gradient signal in the
    "still alive but not yet surviving 10s" region where the per-frame
    alive reward is too small to drive learning on its own.
    """

    def __init__(self, cfg: RewardConfig, now=time.monotonic) -> None:
        self.cfg = cfg
        self.now = now
        self.hazards = PendingHazardTracker(cfg)
        self._episode_dead = False
        self._last_ts: Optional[float] = None
        self._last_ground_diff = 0.0
        # Curriculum state: the index into curriculum_milestones of the
        # NEXT milestone to cross.  Reset to 0 at episode start.
        self._curriculum_idx = 0
        self._episode_start_ts: Optional[float] = None
        # The number of milestones actually crossed (audit + tests).
        self.milestones_crossed = 0

    # ------------------------------------------------------------------ #
    def begin_episode(self, ts: Optional[float] = None) -> None:
        self._episode_dead = False
        self._last_ts = self.now() if ts is None else ts
        self._last_ground_diff = 0.0
        self._curriculum_idx = 0
        self._episode_start_ts = self._last_ts

    def alive_reward(self, ts: float) -> float:
        """Proportional to real elapsed time since the previous frame."""
        if self._last_ts is None:
            self._last_ts = ts
            return 0.0
        dt = max(0.0, ts - self._last_ts)
        self._last_ts = ts
        frames_equiv = dt * self.cfg.nominal_fps
        return self.cfg.alive_per_frame * frames_equiv

    def curriculum_reward(self, ts: float) -> float:
        """Pay a one-shot bonus for each milestone this episode crosses.

        Returns the bonus granted THIS call (0 or ``cfg.curriculum_bonus``).
        Re-arms only on :meth:`begin_episode`.  Out-of-order milestones
        are accepted (the schedule is sorted at construction time so the
        user can write them in any order) — we keep walking the index
        forward instead of paying the first one we find.
        """
        milestones = self.cfg.curriculum_milestones
        if not milestones or self.cfg.curriculum_bonus <= 0.0:
            return 0.0
        if self._episode_start_ts is None:
            return 0.0
        elapsed = ts - self._episode_start_ts
        bonus = 0.0
        idx = self._curriculum_idx
        # Walk forward as long as the elapsed time has crossed the next
        # milestone; if multiple milestones fall in the same frame we
        # pay them all (rare, but well-defined: an "instant replay"
        # episode that starts at a non-zero time).
        while idx < len(milestones) and elapsed >= float(milestones[idx]):
            bonus += float(self.cfg.curriculum_bonus)
            idx += 1
        if idx > self._curriculum_idx:
            self.milestones_crossed += idx - self._curriculum_idx
            self._curriculum_idx = idx
        return bonus

    def death_reward(self) -> float:
        """-10 exactly once per episode."""
        if self._episode_dead:
            return 0.0
        self._episode_dead = True
        return self.cfg.death_penalty

    def pixel_diff_reward(self, ground_gray, prev_ground_gray) -> float:
        """Tightly-clipped ablation component (default disabled)."""
        if not self.cfg.use_pixel_diff_reward or ground_gray is None:
            return 0.0
        if prev_ground_gray is None or prev_ground_gray.shape != ground_gray.shape:
            return 0.0
        import numpy as np

        raw = float(np.abs(
            ground_gray.astype(np.int16) - prev_ground_gray.astype(np.int16)
        ).mean()) / 255.0
        return max(-self.cfg.pixel_diff_clip, min(self.cfg.pixel_diff_clip, raw))

    # ------------------------------------------------------------------ #
    def step(
        self,
        ts: float,
        action: int,
        horizon: HorizonResult,
        ground_gray=None,
        prev_ground_gray=None,
        died: bool = False,
        cleared: int = 0,
        danger: bool = False,
        is_decision: bool = True,
    ) -> RewardBreakdown:
        alive = self.alive_reward(ts)
        hazard = 0.0
        if horizon.detected:
            self.hazards.register(horizon)
        # DEEP-FIX (v1.24.0): the legacy hazard bonus is opt-in now.  It
        # pays for ANY dodge action held across the frames after a horizon
        # blink — including presses that dodge nothing — which is exactly
        # how a policy ends up "keeping jumping or rolling".  The causal
        # replacement is ``clear`` below.
        if self.cfg.use_hazard_bonus:
            hazard = self.hazards.on_frame(horizon, action)
        self.hazards.expire_old(ts)
        # Causal shaping from the obstacle grid.  ``cleared`` is only ever
        # reported for obstacles that passed the player's lane while the
        # player stayed alive, so it cannot credit a collision, and it is
        # capped per step by ObstacleConfig.max_clears_per_step.
        clear = float(cleared) * float(self.cfg.clear_bonus)
        danger_term = -float(self.cfg.danger_penalty) if danger else 0.0
        cost = 0.0
        if is_decision and self.cfg.action_cost > 0.0 and action in DODGE_ACTIONS:
            cost = -float(self.cfg.action_cost)
        # curriculum is paid BEFORE the death penalty so the
        # terminal-state transition still gets a normal "you just hit
        # 10s for the first time" credit even when the same frame is
        # the death frame.  The bonus is tiny relative to -10 so the
        # net reward is still strongly negative on death.
        curriculum = self.curriculum_reward(ts)
        pd = self.pixel_diff_reward(ground_gray, prev_ground_gray)
        death = self.death_reward() if died else 0.0
        total = alive + hazard + pd + death + curriculum + clear + danger_term + cost
        clipped = False
        if total < self.cfg.reward_clip_min:
            total, clipped = self.cfg.reward_clip_min, True
        elif total > self.cfg.reward_clip_max:
            total, clipped = self.cfg.reward_clip_max, True
        return RewardBreakdown(
            alive=alive,
            death=death,
            hazard=hazard,
            pixel_diff=pd,
            curriculum=curriculum,
            clear=clear,
            danger=danger_term,
            action_cost=cost,
            total=total,
            clipped=clipped,
            reason="death" if died else "step",
        )
