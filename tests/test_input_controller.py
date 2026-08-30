"""Input controller tests: dry-run, release guarantees, duplicates, focus.

These tests run WITHOUT a display by forcing the dry-run backend, plus a
fault-injection backend that emulates a failing input stack to prove keys
are released even when the backend raises.
"""

from __future__ import annotations

import time

from config import NOOP, InputConfig
from input_controller import InputController


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
    def test_auto_degrades_to_dry_run_on_headless(self) -> None:
        # On CI (no X server) the pynput backend cannot start; auto must
        # degrade instead of crashing. On a real Windows box this backend
        # stays live, so only the type contract is checked there.
        ctl = InputController(InputConfig(), backend="auto")
        try:
            assert ctl.backend_name in ("dry_run", "pynput")
        finally:
            ctl.dispose()


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
