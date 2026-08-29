"""Adaptive action timing: WHEN to decide and WHICH buffered action fires.

Separation of concerns the requirement demands:
* ``env_frame_id`` — capture worker's monotonic frame counter.
* ``action_step``  — one per action actually EXECUTED (owned here).
* ``learner_update_step`` — gradient steps (owned by the learner).
* ``episode_id``   — death->respawn cycles (owned by the environment).

Cadence policy (requirement §9):
* Normal: decide a new action every N captured frames, N in
  [min_decision_frames, max_decision_frames] chosen from CPU load
  (low load -> faster reactions, high load -> 4 frames).
* Danger (horizon detector fires with confidence): decide on the NEXT frame
  and drop any buffered normal-priority action so a fresh decision wins.
* Buffered commands expire (``action_ttl_ms``); duplicates within
  ``cooldown_ms`` are suppressed; everything is measured, not assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from config import NOOP, SchedulerConfig


@dataclass
class PlannedAction:
    action: int
    created_ts: float
    danger: bool = False
    action_step: int = -1


@dataclass
class SchedulerStats:
    decisions: int = 0
    executed: int = 0
    expired: int = 0
    suppressed_duplicates: int = 0
    danger_overrides: int = 0
    interrupts: int = 0


class ActionScheduler:
    def __init__(self, cfg: SchedulerConfig, now: Callable[[], float] = time.monotonic,
                 cooldown_ms: float = 110.0) -> None:
        self.cfg = cfg
        self.now = now
        self.cooldown_ms = cooldown_ms
        self.stats = SchedulerStats()
        self._frames_since_decision = 0
        self._current_cadence = cfg.max_decision_frames
        self._buffer: list[PlannedAction] = []
        self._last_executed: Optional[int] = None
        self._last_executed_ts = -1e9
        self._action_step = 0

    # ------------------------------------------------------------------ #
    # Load adaptation
    # ------------------------------------------------------------------ #
    def set_load(self, load01: float) -> None:
        """Pick cadence from CPU load (0=idle, 1=saturated)."""
        lo, hi = self.cfg.min_decision_frames, self.cfg.max_decision_frames
        if load01 >= self.cfg.slow_load_threshold:
            self._current_cadence = hi
        else:
            self._current_cadence = lo

    @property
    def cadence(self) -> int:
        return self._current_cadence

    # ------------------------------------------------------------------ #
    # Frame-driven decision gates
    # ------------------------------------------------------------------ #
    def on_frame(self, danger: bool) -> bool:
        """Called once per valid captured frame; True = decide a new action."""
        self._frames_since_decision += 1
        if danger:
            self.stats.danger_overrides += 1
            # A high-confidence danger event invalidates buffered normal actions.
            self.interrupt()
            self._frames_since_decision = 0
            return True
        if self._frames_since_decision >= self._current_cadence:
            self._frames_since_decision = 0
            return True
        return False

    # ------------------------------------------------------------------ #
    # Buffering / execution
    # ------------------------------------------------------------------ #
    def submit(self, action: int, danger: bool = False) -> bool:
        """Buffer a chosen action (capacity-bounded, newest kept)."""
        if action is None or not (0 <= action <= 4):
            return False
        if len(self._buffer) >= self.cfg.buffer_size:
            self._buffer.pop(0)  # bounded: drop oldest
        self._buffer.append(PlannedAction(action, self.now(), danger))
        self.stats.decisions += 1
        return True

    def pop_executable(self) -> Optional[PlannedAction]:
        """Next action to execute, applying expiry and duplicate suppression."""
        t = self.now()
        while self._buffer:
            planned = self._buffer.pop(0)
            if (t - planned.created_ts) * 1000.0 > self.ttl_ms:
                # TTL is enforced authoritatively by InputController too; this
                # early drop keeps counters honest when the loop is slow.
                self.stats.expired += 1
                continue
            if (
                planned.action != NOOP
                and planned.action == self._last_executed
                and (t - self._last_executed_ts) * 1000.0 < self.cooldown_ms
            ):
                self.stats.suppressed_duplicates += 1
                continue
            self._last_executed = planned.action
            self._last_executed_ts = t
            self._action_step += 1
            planned.action_step = self._action_step
            self.stats.executed += 1
            return planned
        return None

    #: action TTL in ms (mirrors InputConfig.action_ttl_ms; settable for tests)
    ttl_ms: float = 140.0

    def interrupt(self) -> None:
        """Drop buffered actions (danger/death/focus loss)."""
        if self._buffer:
            self.stats.interrupts += len(self._buffer)
            self._buffer.clear()

    # ------------------------------------------------------------------ #
    @property
    def action_step(self) -> int:
        return self._action_step

    def reset_episode(self) -> None:
        self._buffer.clear()
        self._frames_since_decision = 0
        self._last_executed = None
        self._last_executed_ts = -1e9

    def snapshot(self) -> dict[str, int]:
        s = self.stats
        return {
            "decisions": s.decisions,
            "executed": s.executed,
            "expired": s.expired,
            "suppressed": s.suppressed_duplicates,
            "danger_overrides": s.danger_overrides,
            "interrupts": s.interrupts,
            "cadence": self._current_cadence,
            "action_step": self._action_step,
        }
