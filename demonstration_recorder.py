"""Human demonstration recorder (Phase-1 data source).

Records what the HUMAN does while the bot only captures:
* preprocessed 84x84 ground-zone frames (uint8, no duplicated stacks),
* the action mapped from the pressed arrow key,
* capture timestamps, done flag, optional score/death/confidence,
* browser geometry + config metadata for reproducibility.

The recorder NEVER presses keys itself; a pynput listener (when available)
observes the human's keys.  Recording stops on F9, death detection, or a
frame gap (browser stall), and the episode is written atomically as
``demos/episode_YYYYmmdd_HHMMSS.npz``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from config import N_ACTIONS, NOOP, BotConfig
from logging_utils import get_logger
from perception import ZonePreprocessor

LOGGER = get_logger("demo_recorder")

#: arrow-key name -> action id (mirrors the configurable keymap at record time)
DEFAULT_KEY_TO_ACTION = {
    "left": 1, "right": 2, "up": 3, "down": 4,
    "\x1b[d": 1, "\x1b[c": 2, "\x1b[a": 3, "\x1b[b": 4,  # xterm escape forms
}


class KeyboardTap:
    """Cross-platform key observer with graceful headless degradation."""

    def __init__(self, on_press: Callable[[str], None],
                 on_release: Callable[[str], None]) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self._listener = None
        self.available = False
        try:
            from pynput import keyboard

            def _press(key):
                name = self._key_name(key)
                if name:
                    self.on_press(name)

            def _release(key):
                name = self._key_name(key)
                if name:
                    self.on_release(name)

            self._listener = keyboard.Listener(on_press=_press, on_release=_release)
            self._keyboard = keyboard
            self.available = True
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
            LOGGER.warning("keyboard listener unavailable: %s (%s) — "
                           "demos will record NOOP only",
                           type(exc).__name__, exc)

    @staticmethod
    def _key_name(key) -> str | None:
        if hasattr(key, "char") and key.char:
            return str(key.char).lower()
        if hasattr(key, "name") and key.name:
            return str(key.name).lower()
        return None

    def start(self) -> None:
        if self._listener is not None:
            self._listener.start()

    def stop(self) -> None:
        """Unregister hooks (requirement: no leftover listeners)."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
                LOGGER.warning("listener stop failed: %s", exc)
            self._listener = None


class DemoRecorder:
    """Subscribes to the frame ring and the human's keys to build episodes."""

    def __init__(self, cfg: BotConfig, out_dir: str | Path,
                 read_frame: Callable[[], object | None]) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.read_frame = read_frame  # returns Frame or None (latest-wins)
        # Record the FULL-FRAME policy view (obstacles spawn at the horizon;
        # the old lower-crop hid them from BC).
        self.pre = ZonePreprocessor(cfg.perception, cfg.perception.horizon_frac,
                                    require_anchor=False,
                                    policy_size=cfg.perception.obs_size)
        self._frames: list[np.ndarray] = []
        self._actions: list[int] = []
        self._ts: list[float] = []
        self._conf: list[float] = []
        self._death: list[str] = []
        self._score: list[float] = []
        self._recording = False
        self._lock = threading.Lock()
        self._current_action = NOOP
        # ACTION EVENTS: a keydown taps an action; the label is BACKDATED so
        # the policy learns to commit while the obstacle is still far away
        # (human reactions land ~200 ms before impact). ``_label_until_idx``
        # marks frames that should inherit the tap label (event + hold window).
        self._events: list[tuple[float, int]] = []   # (monotonic ts, action)
        self._label_until_idx: int = -1
        self._label_action: int = NOOP
        # DAgger correction recording state
        self._dagger_armed: bool = False
        self._dagger_last_ev: float = -1e9
        self.dagger_active: bool = False
        self._tap = KeyboardTap(self._handle_press, self._handle_release)
        self._stop_requested = threading.Event()
        self.episode_paths: list[str] = []

    # ------------------------------------------------------------------ #
    # DAgger / HG-DAgger: human-gated intervention while the BOT plays.
    # When the human seizes control (any gameplay key), start recording a
    # correction episode and keep recording for ``dagger_tail_ms`` after the
    # last keypress so the recovery trajectory is captured. These are the
    # states the policy actually visits and fails in — the covariate-shift
    # fix for behavioural cloning.
    # ------------------------------------------------------------------ #
    def notify_human_intervention(self) -> None:
        if not self.cfg.bc.dagger:
            return
        with self._lock:
            self._dagger_armed = True
            self._dagger_last_ev = time.monotonic()
            if not self._recording:
                self.start()

    def _dagger_tick(self) -> bool:
        """Return True while a DAgger correction episode should be recorded."""
        if not self.cfg.bc.dagger:
            return False
        with self._lock:
            if not self._dagger_armed:
                return False
            tail = self.cfg.bc.dagger_tail_ms / 1000.0
            if time.monotonic() - self._dagger_last_ev > tail:
                self._dagger_armed = False
                return False
            return True

    def _handle_press(self, key_name: str) -> None:
        action = DEFAULT_KEY_TO_ACTION.get(key_name)
        # honour the configured keymap too (custom bindings)
        for a, name in self.cfg.input.keymap.items():
            if a != NOOP and name and name.lower() == key_name:
                action = a
        if action is not None:
            with self._lock:
                self._current_action = action
                # record the decision EVENT at the keypress timestamp
                self._events.append((time.monotonic(), int(action)))
                # any human keypress during bot play (re)starts the DAgger tail
                if self.cfg.bc.dagger:
                    self._dagger_armed = True
                    self._dagger_last_ev = time.monotonic()
                    if not self._recording:
                        self._recording = True
                        self._tap.start()
                        LOGGER.info("DAgger intervention — correction recording started")

    def _handle_release(self, key_name: str) -> None:
        action = DEFAULT_KEY_TO_ACTION.get(key_name)
        for a, name in self.cfg.input.keymap.items():
            if a != NOOP and name and name.lower() == key_name:
                action = a
        if action is not None:
            with self._lock:
                if self._current_action == action:
                    self._current_action = NOOP

    # ------------------------------------------------------------------ #
    @property
    def recording(self) -> bool:
        return self._recording

    def start(self) -> bool:
        if self._recording:
            return False
        self._frames.clear()
        self._actions.clear()
        self._ts.clear()
        self._conf.clear()
        self._death.clear()
        self._score.clear()
        self._current_action = NOOP
        self._events.clear()
        self._label_until_idx = -1
        self._label_action = NOOP
        self._recording = True
        self._stop_requested.clear()
        self._tap.start()
        LOGGER.info("demo recording started")
        return True

    def stop(self, done: bool = True) -> str | None:
        """Finalise and atomically write the episode; returns its path."""
        if not self._recording:
            return None
        self._recording = False
        self._tap.stop()
        n = len(self._frames)
        if n == 0:
            LOGGER.warning("demo recording stopped with 0 frames; nothing saved")
            return None
        done_flags = np.zeros(n, dtype=bool)
        if done:
            done_flags[-1] = True
        meta = {
            "fps_target": self.cfg.capture.target_fps,
            "region": {
                "left": self.cfg.region.left, "top": self.cfg.region.top,
                "width": self.cfg.region.width, "height": self.cfg.region.height,
                "screen_width": self.cfg.region.screen_width,
                "screen_height": self.cfg.region.screen_height,
                "dpi_scale": self.cfg.region.dpi_scale,
            },
            "horizon_frac": self.cfg.perception.horizon_frac,
            "profile": self.cfg.rl.profile,
            "keymap": {str(a): k for a, k in self.cfg.input.keymap.items()},
            "anchor_calibrated": self.cfg.death.anchor_set(),
            "respawn_calibrated": self.cfg.input.respawn_set(),
            "recorded_at": time.time(),
        }
        path = self.out_dir / (
            "episode_" + time.strftime("%Y%m%d_%H%M%S") + ".npz"
        )
        tmp = path.with_suffix(".npz.tmp")
        # NOTE: pass an open file object — numpy silently appends ".npz" to a
        # str filename, which would leave the atomic tmp file never created.
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                frames=np.stack(self._frames).astype(np.uint8),
                actions=np.asarray(self._actions, dtype=np.int64),
                timestamps=np.asarray(self._ts, dtype=np.float64),
                done=done_flags,
                score=np.asarray(self._score, dtype=np.float32),
                confidence=np.asarray(self._conf, dtype=np.float32),
                death_state=np.asarray(self._death, dtype="U16"),
                meta=np.str_(json.dumps(meta)),
            )
        tmp.replace(path)
        self.episode_paths.append(str(path))
        LOGGER.info("demo episode saved: %s (%d steps)", path, n)
        return str(path)

    # ------------------------------------------------------------------ #
    def tick(self, frame, death_state: str = "ALIVE", confidence: float = 0.0,
             score: float = 0.0) -> None:
        """Consume one frame while recording (called from the capture pump).

        Label semantics (deep-research fix §1.2 #7): Subway Surfers latches a
        tap at keydown; the human commits ~``label_backdate_ms`` BEFORE impact.
        We therefore label frames from (keypress − backdate) through
        (keypress + hold) with the tapped action, so the frame that NEEDS the
        decision (obstacle still far) carries the label instead of NOOP.
        """
        if not self._recording or frame is None:
            return
        # DAgger mode: only keep the recorder open while interventions are
        # recent; when the tail elapses, auto-finalise the correction demo.
        dagger_mode = getattr(self, "dagger_active", False)
        if dagger_mode and not self._dagger_tick():
            path = self.stop(done=False)
            if path:
                LOGGER.info("DAgger correction demo saved: %s", path)
            return
        z = self.pre.process(frame.image, frame.frame_id, frame.ts)
        if not z.valid or z.policy_gray is None:
            return
        idx = len(self._frames)
        with self._lock:
            held_action = self._current_action
            # consume tap events: open a backdated labelling window.
            while self._events:
                ev_ts, ev_action = self._events[0]
                age = time.monotonic() - ev_ts
                backdate = self.cfg.bc.label_backdate_ms / 1000.0
                hold = self.cfg.bc.label_hold_ms / 1000.0
                # frames older than the tap get the label for the window
                # [tap-backdate, tap+hold]; expressed as indices:
                back_frames = int(round(backdate * self.cfg.capture.target_fps))
                hold_frames = max(1, int(round(hold * self.cfg.capture.target_fps)))
                start = max(0, idx - back_frames)
                end = idx + hold_frames
                # apply backdated label to already-stored frames
                for j in range(start, min(idx, len(self._actions))):
                    self._actions[j] = int(ev_action)
                # upcoming frames inherit until end
                self._label_until_idx = max(self._label_until_idx, end)
                self._label_action = int(ev_action)
                # remove the event only once it is older than backdate
                self._events.pop(0)
                _ = age  # (age kept for clarity; window uses frame counts)
            # decide THIS frame's label
            if idx <= self._label_until_idx:
                action = self._label_action
            else:
                action = held_action
        self._frames.append(z.policy_gray)
        self._actions.append(int(action))
        self._ts.append(float(frame.ts))
        self._death.append(str(death_state))
        self._conf.append(float(confidence))
        self._score.append(float(score))

    def dispose(self) -> None:
        self._tap.stop()


def action_is_valid(a: int) -> bool:
    return 0 <= int(a) < N_ACTIONS
