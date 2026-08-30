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
    total: float = 0.0
    clipped: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, float]:
        return {
            "alive": self.alive,
            "death": self.death,
            "hazard": self.hazard,
            "pixel_diff": self.pixel_diff,
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
    """Total reward per actor step, from components to clipped total."""

    def __init__(self, cfg: RewardConfig, now=time.monotonic) -> None:
        self.cfg = cfg
        self.now = now
        self.hazards = PendingHazardTracker(cfg)
        self._episode_dead = False
        self._last_ts: Optional[float] = None
        self._last_ground_diff = 0.0

    # ------------------------------------------------------------------ #
    def begin_episode(self, ts: Optional[float] = None) -> None:
        self._episode_dead = False
        self._last_ts = self.now() if ts is None else ts
        self._last_ground_diff = 0.0

    def alive_reward(self, ts: float) -> float:
        """Proportional to real elapsed time since the previous frame."""
        if self._last_ts is None:
            self._last_ts = ts
            return 0.0
        dt = max(0.0, ts - self._last_ts)
        self._last_ts = ts
        frames_equiv = dt * self.cfg.nominal_fps
        return self.cfg.alive_per_frame * frames_equiv

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
    ) -> RewardBreakdown:
        alive = self.alive_reward(ts)
        hazard = 0.0
        if horizon.detected:
            self.hazards.register(horizon)
        hazard = self.hazards.on_frame(horizon, action)
        self.hazards.expire_old(ts)
        pd = self.pixel_diff_reward(ground_gray, prev_ground_gray)
        death = self.death_reward() if died else 0.0
        total = alive + hazard + pd + death
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
            total=total,
            clipped=clipped,
            reason="death" if died else "step",
        )
