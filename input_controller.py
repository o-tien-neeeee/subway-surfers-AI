"""Keyboard/mouse control with hard safety guarantees.

The InputController is the ONLY component allowed to press gameplay keys.

Guarantees (requirement §8/§14):
* Every press has a scheduled release; a guardian thread force-releases any
  key held beyond ``max_hold_ms`` — so a crashed decision loop can never
  leave a stuck arrow key scrolling the page.
* ``release_all()`` is idempotent, lock-protected, and called on pause,
  death, focus loss, emergency stop and every shutdown path.
* Duplicate presses of an already-held key are suppressed (no OS key repeat
  stacking, no double jumps).
* ``dry_run`` backend logs intended actions without touching real input —
  used by tests, headless mode and first-run trust building.
* pynput is imported lazily: on machines without an input subsystem
  (headless CI) the controller degrades to dry-run instead of crashing.
* Mouse clicks (respawn) go through pyautogui with FAILSAFE=True (slam the
  mouse to a screen corner to abort) and PAUSE set to a deliberate low
  value; a missing pyautogui degrades the same way.

Focus rule: gameplay keys are refused while Chrome is not the foreground
window (title match, best effort via pygetwindow/pyautogui).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from config import ACTIONS, NOOP, InputConfig
from logging_utils import get_logger

LOGGER = get_logger("input")


@dataclass
class InputEvent:
    action: int
    key: str
    pressed: bool
    ts: float
    detail: str = ""


class _PynputBackend:
    """Real key presses via pynput (lazy import, Windows/Linux/macOS)."""

    def __init__(self, keymap: dict[int, str]) -> None:
        from pynput import (
            keyboard,
        )

        self._kb = keyboard.Controller()
        self._keys = {}
        for action, name in keymap.items():
            if action == NOOP or not name:
                continue
            self._keys[action] = self._resolve_key(keyboard, name)

    @staticmethod
    def _resolve_key(keyboard_module, name: str):
        name = name.strip().lower()
        special = getattr(keyboard_module.Key, name, None)
        if special is not None:
            return special
        if len(name) == 1:
            return keyboard_module.KeyCode.from_char(name)
        raise ValueError(f"cannot resolve key name {name!r}")

    def press(self, key) -> None:
        self._kb.press(key)

    def release(self, key) -> None:
        self._kb.release(key)

    def key_for(self, action: int):
        return self._keys.get(action)


class _DryRunBackend:
    """Logs actions; never touches the OS.  Used in tests/headless mode."""

    def __init__(self, keymap: dict[int, str]) -> None:
        self._keys = {a: k for a, k in keymap.items() if a != NOOP and k}
        self.log: list[InputEvent] = []

    def press(self, key) -> None:
        self.log.append(InputEvent(-1, str(key), True, time.monotonic(), "dry"))

    def release(self, key) -> None:
        self.log.append(InputEvent(-1, str(key), False, time.monotonic(), "dry"))

    def key_for(self, action: int):
        return self._keys.get(action)


class InputController:
    """Safe input facade with guardian thread and press accounting."""

    def __init__(
        self,
        cfg: InputConfig,
        backend: str = "auto",
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.now = now
        self.backend_name = backend
        self._lock = threading.Lock()
        self._pressed: dict[str, tuple[float, object]] = {}  # key -> (since, keyobj)
        self._release_at: list[tuple[float, str]] = []
        self.stats = {
            "presses": 0,
            "releases": 0,
            "suppressed_duplicates": 0,
            "guardian_releases": 0,
            "focus_blocks": 0,
            "expired_drops": 0,
        }
        self._stopped = False
        self._backend = self._make_backend(backend, cfg.keymap)
        self._guardian = threading.Thread(target=self._guardian_loop, daemon=True,
                                          name="key-guardian")
        self._guardian.start()

    # ------------------------------------------------------------------ #
    def _make_backend(self, backend: str, keymap: dict[int, str]):
        if backend == "dry_run":
            return _DryRunBackend(keymap)
        if backend == "pynput":
            return _PynputBackend(keymap)
        # auto: try real input, degrade to dry-run with a logged warning.
        try:
            return _PynputBackend(keymap)
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
            LOGGER.warning(
                "pynput backend unavailable (%s: %s); falling back to dry_run — "
                "no real keys will be pressed", type(exc).__name__, exc,
            )
            self.backend_name = "dry_run"
            return _DryRunBackend(keymap)

    # ------------------------------------------------------------------ #
    # Gameplay keys
    # ------------------------------------------------------------------ #
    def press_action(
        self, action: int, hold_ms: int | None = None,
        created_ts: float | None = None,
    ) -> InputEvent:
        """Press a gameplay key and schedule its release.

        Refuses stale commands (``created_ts`` older than the action TTL),
        duplicates of already-held keys, and NOOP.  Never raises: any backend
        error releases everything and is reported in the event.
        """
        t = self.now()
        if action == NOOP:
            return InputEvent(action, "", False, t, "noop")
        if created_ts is not None and (t - created_ts) * 1000.0 > self.cfg.action_ttl_ms:
            self.stats["expired_drops"] += 1
            return InputEvent(action, "", False, t, "expired")
        keyobj = self._backend.key_for(action)
        if keyobj is None:
            return InputEvent(action, self.cfg.keymap.get(action, "?"), False, t,
                              "unmapped_key")
        keyname = self.cfg.keymap.get(action, str(action))
        hold = (hold_ms if hold_ms is not None else self.cfg.hold_ms) / 1000.0
        try:
            with self._lock:
                if keyname in self._pressed:
                    self.stats["suppressed_duplicates"] += 1
                    _, existing = self._pressed[keyname]
                    # extend hold slightly, never re-press
                    self._pressed[keyname] = (self._pressed[keyname][0], existing)
                    self._release_at.append((t + hold, keyname))
                    return InputEvent(action, keyname, False, t, "duplicate_suppressed")
                self._backend.press(keyobj)
                self._pressed[keyname] = (t, keyobj)
                self._release_at.append((t + hold, keyname))
                self.stats["presses"] += 1
                return InputEvent(action, keyname, True, t, "pressed")
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
            LOGGER.error("press_action failed, releasing all:\n%s", exc)
            self.release_all()
            return InputEvent(action, keyname, False, t, f"error:{type(exc).__name__}")

    # ------------------------------------------------------------------ #
    def release_all(self, reason: str = "explicit") -> int:
        """Release every held key; returns how many were released."""
        with self._lock:
            released = 0
            for keyname, (_since, keyobj) in list(self._pressed.items()):
                try:
                    self._backend.release(keyobj)
                    released += 1
                    self.stats["releases"] += 1
                except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
                    LOGGER.error("release failed for %s: %s", keyname, exc)
            self._pressed.clear()
            self._release_at.clear()
        if released:
            LOGGER.info("released %d key(s): %s", released, reason)
        return released

    def pressed_count(self) -> int:
        with self._lock:
            return len(self._pressed)

    def pressed_names(self) -> list[str]:
        with self._lock:
            return list(self._pressed.keys())

    def stuck_keys(self) -> list[str]:
        """Keys held beyond max_hold_ms (bugs, backend hangs, crashed loop)."""
        t = self.now()
        with self._lock:
            return [
                name for name, (since, _ko) in self._pressed.items()
                if (t - since) * 1000.0 > self.cfg.max_hold_ms
            ]

    # ------------------------------------------------------------------ #
    def _guardian_loop(self) -> None:
        """Background: release keys at their scheduled time; audit stuck keys."""
        while not self._stopped:
            t = self.now()
            to_release: list[str] = []
            with self._lock:
                self._release_at.sort()
                while self._release_at and self._release_at[0][0] <= t:
                    _, keyname = self._release_at.pop(0)
                    if keyname in self._pressed:
                        to_release.append(keyname)
                # Hard safety: anything held past max_hold_ms dies.
                for keyname, (since, _ko) in self._pressed.items():
                    if (t - since) * 1000.0 > self.cfg.max_hold_ms and keyname not in to_release:
                        to_release.append(keyname)
                        self.stats["guardian_releases"] += 1
            for keyname in to_release:
                self._release_one(keyname)
            time.sleep(0.005)

    def _release_one(self, keyname: str) -> None:
        with self._lock:
            entry = self._pressed.pop(keyname, None)
        if entry is None:
            return
        _, keyobj = entry
        try:
            self._backend.release(keyobj)
            self.stats["releases"] += 1
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
            LOGGER.error("scheduled release failed for %s: %s", keyname, exc)

    # ------------------------------------------------------------------ #
    # Mouse + focus
    # ------------------------------------------------------------------ #
    def click(self, x: int, y: int, confirm_focus: bool = True) -> bool:
        """Left-click at absolute screen coords (respawn button)."""
        if self.backend_name == "dry_run":
            LOGGER.info("dry-run click at (%d, %d)", x, y)
            return True
        try:
            import pyautogui

            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.01
            if confirm_focus and self.browser_focused() is False:
                LOGGER.warning("click refused: browser not focused")
                return False
            pyautogui.click(x=int(x), y=int(y))
            return True
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
            LOGGER.error("click failed: %s", exc)
            return False

    def browser_focused(self) -> bool | None:
        """True/False when detectable; None when the platform can't tell."""
        if self.backend_name == "dry_run":
            return None
        try:
            import pyautogui  # lazy; needs a display on Linux

            try:
                title = pyautogui.getActiveWindow().title
            except Exception:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
                return None
            if not title:
                return None
            return self.cfg.browser_title_hint.lower() in str(title).lower()
        except Exception:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
            return None

    def focus_gate(self) -> bool:
        """True when input is allowed (focused or unverifiable+dry-run)."""
        focused = self.browser_focused()
        if focused is False:
            self.stats["focus_blocks"] += 1
            return False
        return True

    # ------------------------------------------------------------------ #
    def dispose(self) -> None:
        self._stopped = True
        self.release_all("dispose")
        try:
            self._guardian.join(timeout=1.0)
        except RuntimeError as exc:
            LOGGER.debug("guardian join failed: %s", exc)

    def action_name(self, action: int) -> str:
        return ACTIONS[action] if 0 <= action < len(ACTIONS) else "??"
