"""Safety watchdog: independent supervision with priority over the learner.

Two independent layers (requirement E):
* ``SafetyWatchdog`` — a daemon thread inside the ACTOR process.  It watches
  the emergency event, browser focus, capture stalls, stuck keys and the
  death flag.  It can pause gameplay, force key release, and trigger a
  global shutdown.  It shares NO code path with the learner, so a learner
  crash leaves the watchdog fully operational.
* ``EmergencyHotkey`` — a global F8 listener in the GUI process that sets
  the shared emergency event.  It is explicitly stopped (hooks
  unregistered) on shutdown.

Priority rule: any watchdog intervention releases keys FIRST, then pauses
training/actions, then reports.  Nothing here can be overridden by the
learner.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from logging_utils import format_exception, get_logger, put_bounded

LOGGER = get_logger("watchdog")


class SafetyWatchdog(threading.Thread):
    """Periodic safety checks with authority over input and pause events."""

    def __init__(
        self,
        events: dict,
        input_controller: Any,
        counters: Any,
        metrics_q: Any,
        ring: Any = None,
        interval_s: float = 0.25,
        stall_timeout_s: float = 2.0,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(daemon=True, name="safety-watchdog")
        self.events = events
        self.input = input_controller
        self.counters = counters
        self.metrics_q = metrics_q
        self.ring = ring
        self.interval_s = interval_s
        self.stall_timeout_s = stall_timeout_s
        self.on_status = on_status
        self._stop = threading.Event()
        self._last_frame_id = -1
        self._last_progress_ts = time.monotonic()
        self._focus_loss_reported = False

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        LOGGER.info("watchdog running (interval=%.2fs)", self.interval_s)
        while not self._stop.is_set():
            try:
                self._check_once()
            except Exception as exc:
                LOGGER.error("watchdog check failed:\n%s", format_exception(exc))
            self._stop.wait(self.interval_s)
        LOGGER.info("watchdog stopped")

    def _check_once(self) -> None:
        # 1. Emergency stop has the highest priority.
        if self.events["emergency"].is_set():
            self._intervene("emergency", "Emergency stop detected — releasing keys "
                                        "and shutting down", shutdown=True)
            return
        # 2. Global stop: just make sure keys are up.
        if self.events["stop"].is_set():
            self.input.release_all("stop")
            return
        # 3. Death flag: keys must never be held while dead.
        if int(self.counters.death_flag.value) == 1:
            self.input.release_all("death")
        # 4. Pause: keep keys released while paused.
        if self.events["pause"].is_set():
            self.input.release_all("paused")
            return
        # 5. Capture stall: no new frames while running -> pause + release.
        if self.ring is not None:
            fid = self.ring.latest_frame_id()
            if fid != self._last_frame_id:
                self._last_frame_id = fid
                self._last_progress_ts = time.monotonic()
            elif time.monotonic() - self._last_progress_ts > self.stall_timeout_s:
                self._intervene(
                    "capture_stall",
                    f"No new frames for {self.stall_timeout_s:.0f}s — pausing",
                )
                self._last_progress_ts = time.monotonic()
                return
        # 6. Browser focus (best effort; None = cannot verify).
        focused = self.input.browser_focused()
        if focused is False:
            if not self._focus_loss_reported:
                self._intervene("focus", "Browser focus lost — pausing gameplay")
                self._focus_loss_reported = True
        else:
            self._focus_loss_reported = False
        # 7. Stuck keys: only keys held beyond the safety limit are touched —
        #    short legitimate holds (60ms) must never be interrupted.
        stuck = self.input.stuck_keys()
        if stuck:
            self._intervene("stuck_keys", f"keys held too long: {stuck}")

    def _intervene(self, tag: str, message: str, shutdown: bool = False) -> None:
        released = self.input.release_all(tag)
        self.events["pause"].set()
        if shutdown:
            self.events["stop"].set()
        LOGGER.warning("watchdog intervention [%s]: %s (released %d keys)",
                       tag, message, released)
        put_bounded(self.metrics_q, {
            "type": "watchdog", "tag": tag, "msg": message,
            "released": released, "shutdown": shutdown,
        })
        if self.on_status is not None:
            try:
                self.on_status(tag, message)
            except Exception as exc:
                LOGGER.error("on_status callback failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()


class EmergencyHotkey:
    """Global emergency hotkey (default F8) using pynput; degrades cleanly."""

    def __init__(self, hotkey_name: str, events: dict) -> None:
        self.events = events
        self.hotkey_name = hotkey_name.strip().lower()
        self._listener = None
        self.available = False
        try:
            from pynput import keyboard

            target = getattr(keyboard.Key, self.hotkey_name, None)
            if target is None:
                LOGGER.error("emergency hotkey %r not recognised", hotkey_name)
                return

            def on_press(key):
                if key == target:
                    LOGGER.warning("EMERGENCY HOTKEY %s PRESSED", self.hotkey_name)
                    self.events["emergency"].set()

            self._listener = keyboard.Listener(on_press=on_press, on_release=None)
            self._listener.start()
            self.available = True
            LOGGER.info("emergency hotkey '%s' armed", self.hotkey_name)
        except Exception as exc:
            LOGGER.warning("hotkey listener unavailable (%s: %s) — use the "
                           "GUI STOP button or Ctrl+C", type(exc).__name__, exc)

    def stop(self) -> None:
        """Unregister hooks (requirement §14: no leftover listeners)."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as exc:
                LOGGER.warning("hotkey stop failed: %s", exc)
            self._listener = None
