"""Lifecycle state machine shared by the GUI, workers, and tests.

The bot is always in exactly one :class:`BotState`.  Only the transitions in
``STATE_TRANSITIONS`` are legal; the GUI owns the current state and every
transition goes through :class:`StateMachine` so lifecycle bugs surface in
unit tests instead of during live gameplay.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional


class BotState(str, Enum):
    """User-visible lifecycle states (requirement: 12 named states)."""

    CALIBRATING = "CALIBRATING"
    READY = "READY"
    RECORDING_DEMO = "RECORDING_DEMO"
    PRETRAINING = "PRETRAINING"
    RUNNING = "RUNNING"
    DANGER = "DANGER"
    DEAD = "DEAD"
    RESPAWNING = "RESPAWNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


#: Legal transitions.  Anything not listed here is rejected by the machine.
STATE_TRANSITIONS: dict[BotState, frozenset[BotState]] = {
    BotState.CALIBRATING: frozenset(
        {BotState.READY, BotState.RECORDING_DEMO, BotState.ERROR, BotState.STOPPING}
    ),
    BotState.READY: frozenset(
        {
            BotState.RECORDING_DEMO,
            BotState.PRETRAINING,
            BotState.RUNNING,
            BotState.CALIBRATING,
            BotState.ERROR,
            BotState.STOPPING,
        }
    ),
    BotState.RECORDING_DEMO: frozenset(
        {BotState.READY, BotState.ERROR, BotState.STOPPING}
    ),
    BotState.PRETRAINING: frozenset(
        {BotState.READY, BotState.RUNNING, BotState.ERROR, BotState.STOPPING}
    ),
    BotState.RUNNING: frozenset(
        {
            BotState.DANGER,
            BotState.DEAD,
            BotState.PAUSED,
            BotState.ERROR,
            BotState.STOPPING,
        }
    ),
    BotState.DANGER: frozenset(
        {BotState.RUNNING, BotState.DEAD, BotState.PAUSED, BotState.ERROR, BotState.STOPPING}
    ),
    BotState.DEAD: frozenset(
        {BotState.RESPAWNING, BotState.PAUSED, BotState.ERROR, BotState.STOPPING}
    ),
    BotState.RESPAWNING: frozenset(
        {BotState.RUNNING, BotState.PAUSED, BotState.ERROR, BotState.STOPPING}
    ),
    BotState.PAUSED: frozenset(
        {BotState.RUNNING, BotState.READY, BotState.ERROR, BotState.STOPPING}
    ),
    BotState.ERROR: frozenset({BotState.PAUSED, BotState.READY, BotState.STOPPING}),
    BotState.STOPPING: frozenset({BotState.STOPPED}),
    BotState.STOPPED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    """Raised when a lifecycle transition is not in STATE_TRANSITIONS."""


class StateMachine:
    """Thread-safe-ish lifecycle tracker (mutated only from the GUI thread)."""

    def __init__(self, initial: BotState = BotState.CALIBRATING) -> None:
        self._state = initial
        self._history: list[tuple[BotState, BotState]] = []

    @property
    def state(self) -> BotState:
        return self._state

    @property
    def history(self) -> Iterable[tuple[BotState, BotState]]:
        return list(self._history)

    def can(self, target: BotState) -> bool:
        return target in STATE_TRANSITIONS[self._state]

    def transition(self, target: BotState) -> BotState:
        """Move to ``target`` or raise :class:`InvalidTransitionError`."""
        if target is self._state:
            return self._state
        if target not in STATE_TRANSITIONS[self._state]:
            raise InvalidTransitionError(
                f"Illegal lifecycle transition {self._state.value} -> {target.value}"
            )
        self._history.append((self._state, target))
        self._state = target
        return self._state

    def try_transition(self, target: BotState) -> Optional[BotState]:
        """Transition if legal; return the new state or ``None`` if refused."""
        if self.can(target):
            return self.transition(target)
        return None

    def reset(self, state: BotState = BotState.CALIBRATING) -> None:
        self._state = state
        self._history.clear()

    def is_terminal(self) -> bool:
        return self._state in (BotState.STOPPING, BotState.STOPPED)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"StateMachine({self._state.value})"
