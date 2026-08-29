"""GUI lifecycle state-machine tests (no Tkinter required)."""

from __future__ import annotations

import pytest

from states import (
    STATE_TRANSITIONS,
    BotState,
    InvalidTransitionError,
    StateMachine,
)


class TestStateMachine:
    def test_initial_state(self) -> None:
        sm = StateMachine()
        assert sm.state is BotState.CALIBRATING

    def test_full_happy_lifecycle(self) -> None:
        sm = StateMachine()
        for target in (BotState.READY, BotState.PRETRAINING, BotState.RUNNING,
                       BotState.PAUSED, BotState.RUNNING, BotState.DANGER,
                       BotState.RUNNING, BotState.DEAD, BotState.RESPAWNING,
                       BotState.RUNNING, BotState.STOPPING, BotState.STOPPED):
            sm.transition(target)
        assert sm.state is BotState.STOPPED

    def test_calibration_reselect_loop(self) -> None:
        sm = StateMachine()
        sm.transition(BotState.READY)
        sm.transition(BotState.CALIBRATING)  # re-select region
        sm.transition(BotState.READY)

    def test_demo_recording_flows(self) -> None:
        sm = StateMachine()
        sm.transition(BotState.READY)
        sm.transition(BotState.RECORDING_DEMO)
        sm.transition(BotState.READY)
        sm.transition(BotState.RUNNING)

    def test_error_recovery_paths(self) -> None:
        sm = StateMachine()
        sm.transition(BotState.READY)
        sm.transition(BotState.ERROR)
        sm.transition(BotState.PAUSED)
        sm.transition(BotState.READY)

    def test_stopped_is_terminal(self) -> None:
        sm = StateMachine()
        sm.transition(BotState.READY)
        sm.transition(BotState.STOPPING)
        sm.transition(BotState.STOPPED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(BotState.READY)
        assert sm.is_terminal()

    def test_illegal_transitions_raise(self) -> None:
        sm = StateMachine()  # CALIBRATING
        for bad in (BotState.RUNNING, BotState.DEAD, BotState.RESPAWNING,
                    BotState.STOPPED, BotState.DANGER):
            with pytest.raises(InvalidTransitionError):
                sm.transition(bad)

    def test_death_requires_running(self) -> None:
        sm = StateMachine()
        sm.transition(BotState.READY)
        with pytest.raises(InvalidTransitionError):
            sm.transition(BotState.DEAD)

    def test_try_transition(self) -> None:
        sm = StateMachine()
        assert sm.try_transition(BotState.READY) is BotState.READY
        assert sm.try_transition(BotState.DEAD) is None

    def test_same_state_is_noop(self) -> None:
        sm = StateMachine(BotState.READY)
        sm.transition(BotState.READY)
        assert sm.state is BotState.READY

    def test_every_state_reaches_stopping_or_is_terminal(self) -> None:
        for state, targets in STATE_TRANSITIONS.items():
            if state is BotState.STOPPED:
                assert not targets
                continue
            assert BotState.STOPPING in targets or state is BotState.STOPPING, (
                f"{state} has no shutdown path"
            )

    def test_history_recorded(self) -> None:
        sm = StateMachine()
        sm.transition(BotState.READY)
        sm.transition(BotState.RUNNING)
        assert sm.history == [(BotState.CALIBRATING, BotState.READY),
                              (BotState.READY, BotState.RUNNING)]

    def test_all_twelve_states_exist(self) -> None:
        expected = {"CALIBRATING", "READY", "RECORDING_DEMO", "PRETRAINING",
                    "RUNNING", "DANGER", "DEAD", "RESPAWNING", "PAUSED",
                    "ERROR", "STOPPING", "STOPPED"}
        assert {s.value for s in BotState} == expected
