"""Input controller tests: dry-run, release guarantees, duplicates, focus.

These tests run WITHOUT a display by forcing the dry-run backend, plus a
fault-injection backend that emulates a failing input stack to prove keys
are released even when the backend raises.
"""

from __future__ import annotations

import time

import pytest

from config import InputConfig, NOOP
import input_controller
from input_controller import InputController, _DryRunBackend


class TestDryRun:
    def test_press_and_scheduled_release(self) -> None:
        cfg = InputConfig(hold_ms=40)
        ctl = InputController(cfg, backend="dry_run")
        try:
            ev = ctl.press_action(3)  # JUMP
            assert ev.pressed and ev.detail == "pressed"
            assert ctl.pressed_count() == 1
            deadline = time.monotonic() + 1.0
            while ctl.pressed_count() > 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ctl.pressed_count() == 0, "key must auto-release"
            assert ctl.stats["releases"] == 1
        finally:
            ctl.dispose()

    def test_noop_never_presses(self) -> None:
        ctl = InputController(InputConfig(), backend="dry_run")
        try:
            ev = ctl.press_action(NOOP)
            assert not ev.pressed and ev.detail == "noop"
            assert ctl.stats["presses"] == 0
        finally:
            ctl.dispose()

    def test_duplicate_press_suppressed(self) -> None:
        ctl = InputController(InputConfig(hold_ms=500), backend="dry_run")
        try:
            ctl.press_action(1)
            ev2 = ctl.press_action(1)  # same key still held
            assert not ev2.pressed
            assert ev2.detail == "duplicate_suppressed"
            assert ctl.stats["suppressed_duplicates"] == 1
            assert ctl.stats["presses"] == 1
        finally:
            ctl.dispose()

    def test_stale_action_expired(self) -> None:
        cfg = InputConfig(action_ttl_ms=50)
        ctl = InputController(cfg, backend="dry_run")
        try:
            ev = ctl.press_action(2, created_ts=time.monotonic() - 1.0)
            assert not ev.pressed and ev.detail == "expired"
            assert ctl.stats["expired_drops"] == 1
            assert ctl.stats["presses"] == 0, "stale command must not press"
        finally:
            ctl.dispose()

    def test_unmapped_action_is_safe(self) -> None:
        cfg = InputConfig(keymap={0: "", 1: "left", 2: "right", 3: "up", 4: "down"})
        cfg.keymap[1] = ""  # effectively unmapped gameplay key
        ctl = InputController(cfg, backend="dry_run")
        try:
            ev = ctl.press_action(1)
            assert not ev.pressed
            assert ev.detail == "unmapped_key"
        finally:
            ctl.dispose()

    def test_guardian_force_releases_stuck_key(self) -> None:
        cfg = InputConfig(hold_ms=10_000, max_hold_ms=200)
        ctl = InputController(cfg, backend="dry_run")
        try:
            ctl.press_action(4)
            deadline = time.monotonic() + 2.0
            while ctl.pressed_count() > 0 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ctl.pressed_count() == 0, "guardian must force-release"
            assert ctl.stats["guardian_releases"] >= 1
        finally:
            ctl.dispose()

    def test_release_all_idempotent(self) -> None:
        ctl = InputController(InputConfig(hold_ms=5_000), backend="dry_run")
        try:
            ctl.press_action(1)
            ctl.press_action(2)
            n = ctl.release_all("test")
            assert n == 2
            assert ctl.release_all("test-again") == 0
        finally:
            ctl.dispose()


class TestBackendFailure:
    def test_press_failure_releases_everything(self) -> None:
        """Backend exceptions must never leave phantom keys or raise out."""
        ctl = InputController(InputConfig(hold_ms=100), backend="dry_run")
        try:
            ctl.press_action(1)  # hold one healthy key
            orig = ctl._backend.press

            def boom(key):
                raise RuntimeError("input stack exploded")

            ctl._backend.press = boom
            ev = ctl.press_action(3)  # would raise -> must be contained
            assert not ev.pressed
            assert ev.detail.startswith("error:")
            assert ctl.pressed_count() == 0, "all keys released on error"
        finally:
            ctl.dispose()


class TestFocusGate:
    def test_gate_blocks_when_unfocused(self) -> None:
        ctl = InputController(InputConfig(), backend="dry_run")
        try:
            ctl.browser_focused = lambda: False  # deterministic stub
            assert ctl.focus_gate() is False
            assert ctl.stats["focus_blocks"] == 1
            ctl.browser_focused = lambda: True
            assert ctl.focus_gate() is True
        finally:
            ctl.dispose()

    def test_gate_allows_when_unknown(self) -> None:
        ctl = InputController(InputConfig(), backend="dry_run")
        try:
            ctl.browser_focused = lambda: None  # platform cannot verify
            assert ctl.focus_gate() is True, "unverifiable focus must not deadlock"
        finally:
            ctl.dispose()

    def test_dry_run_click(self) -> None:
        ctl = InputController(InputConfig(), backend="dry_run")
        try:
            assert ctl.click(100, 200) is True
            assert ctl.backend_name == "dry_run"
        finally:
            ctl.dispose()


class TestAutoFallback:
    """``backend_name`` must report the backend that is actually live.

    This class is the regression guard for a bug that was invisible on Linux
    CI for a whole release: with ``backend="auto"``, ``backend_name`` was
    seeded from the *request* and only overwritten in the degradation branch.
    So on CI (where pynput cannot start) it read "dry_run" and the test
    passed, while on a real Windows box -- where pynput starts fine -- it read
    the literal "auto", a value that is not a backend at all.  The old test
    was green on one platform and red on the other because it never forced
    the success path.  These tests do, by stubbing the backend, so they are
    deterministic everywhere.
    """

    def test_auto_degrades_to_dry_run_on_headless(self) -> None:
        """The original test, kept, but now asserting the real invariant."""
        ctl = InputController(InputConfig(), backend="auto")
        try:
            assert ctl.backend_name in ("dry_run", "pynput"), (
                f"backend_name must name a concrete backend, got "
                f"{ctl.backend_name!r}"
            )
            assert ctl.backend_name != "auto"
        finally:
            ctl.dispose()

    def test_auto_resolves_to_pynput_when_pynput_works(self, monkeypatch) -> None:
        """Force the success path. This is the assertion that used to fail.

        Stubbing ``_PynputBackend`` means no keyboard, display or X server is
        needed, so this exercises the previously-broken branch on every
        platform -- including the Linux CI where the bug hid.
        """
        monkeypatch.setattr(input_controller, "_PynputBackend", _SucceedingBackend)
        ctl = InputController(InputConfig(), backend="auto")
        try:
            assert ctl.backend_name == "pynput"
            assert not ctl.is_dry_run
        finally:
            ctl.dispose()

    def test_auto_degrades_when_pynput_raises(self, monkeypatch) -> None:
        """Force the failure path, so degradation is tested on Windows too."""
        monkeypatch.setattr(input_controller, "_PynputBackend", _FailingBackend)
        ctl = InputController(InputConfig(), backend="auto")
        try:
            assert ctl.backend_name == "dry_run"
            assert ctl.is_dry_run
        finally:
            ctl.dispose()

    def test_explicit_backends_are_reported_verbatim(self) -> None:
        for name in ("dry_run",):
            ctl = InputController(InputConfig(), backend=name)
            try:
                assert ctl.backend_name == name
                assert ctl.requested_backend == name
            finally:
                ctl.dispose()

    def test_explicit_pynput_is_not_silently_degraded(self, monkeypatch) -> None:
        """Asking for pynput must not quietly hand back dry_run."""
        monkeypatch.setattr(input_controller, "_PynputBackend", _FailingBackend)
        with pytest.raises(Exception):
            InputController(InputConfig(), backend="pynput").dispose()

    def test_unknown_backend_is_rejected(self) -> None:
        """A config typo must fail fast, not silently pick another backend.

        Before this, any unrecognised string fell through to the `auto` branch
        and started pressing real keys under a name nobody requested.
        """
        for bad in ("pynpt", "pyautogui", "", "AUTO", "dry-run"):
            with pytest.raises(ValueError):
                InputController(InputConfig(), backend=bad)

    def test_backend_name_is_never_a_non_backend(self) -> None:
        """Invariant across every accepted request value."""
        for name in InputController.KNOWN_BACKENDS:
            if name == "pynput":
                continue                      # needs a working backend; covered above
            ctl = InputController(InputConfig(), backend=name)
            try:
                assert ctl.backend_name in ("dry_run", "pynput")
            finally:
                ctl.dispose()

    def test_dry_run_guards_use_the_resolved_backend(self, monkeypatch) -> None:
        """``is_dry_run`` must track the resolution, not the request."""
        monkeypatch.setattr(input_controller, "_PynputBackend", _SucceedingBackend)
        ctl = InputController(InputConfig(), backend="auto")
        try:
            assert ctl.is_dry_run is False
            # browser_focused() short-circuits to None under dry_run only.
            assert ctl.backend_name == "pynput"
        finally:
            ctl.dispose()


class _SucceedingBackend:
    """Stands in for pynput when it starts successfully."""

    def __init__(self, keymap):
        self.keymap = dict(keymap)
        self.pressed: list[object] = []

    def key_for(self, action: int) -> str:
        return self.keymap.get(action, "noop")

    def press(self, keyobj) -> None:
        self.pressed.append(keyobj)

    def release(self, keyobj) -> None:
        if keyobj in self.pressed:
            self.pressed.remove(keyobj)


class _FailingBackend:
    """Stands in for pynput when it cannot start (no display, missing lib)."""

    def __init__(self, keymap):
        raise RuntimeError("no display / pynput unavailable (test stub)")


class TestDeathInteraction:
    def test_no_new_press_after_death_release(self) -> None:
        """After a death release, only explicitly requested presses happen."""
        ctl = InputController(InputConfig(hold_ms=200), backend="dry_run")
        try:
            ctl.press_action(2)
            released = ctl.release_all("death")
            assert released == 1
            # no scheduled release resurrects the key
            time.sleep(0.05)
            assert ctl.pressed_count() == 0
            assert ctl.pressed_names() == []
        finally:
            ctl.dispose()
