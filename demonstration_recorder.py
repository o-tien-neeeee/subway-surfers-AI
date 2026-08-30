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
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config import BotConfig, N_ACTIONS, NOOP
from logging_utils import get_logger
from perception import ZonePreprocessor
from states import BotState

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
        self._keyboard = None
        self._build_listener()

    def _build_listener(self) -> bool:
        """Create a fresh pynput listener; False when no input system exists.

        # DEEP-FIX: the listener used to be built once in __init__ and
        # ``stop()`` set ``self._listener = None``.  ``start()`` then did
        # nothing for every later episode, so episode 1 recorded the human's
        # keys and episodes 2..N silently recorded NOOP for every frame --
        # verified with a stub listener (start() -> no-op, _listener stays
        # None).  A dataset of all-NOOP demos trains a policy that never
        # dodges, and nothing downstream reports it.  pynput listeners are
        # single-use, so a stopped one must be rebuilt.
        """
        if self._keyboard is None:
            try:
                from pynput import keyboard

                self._keyboard = keyboard
            except Exception as exc:
                LOGGER.warning("keyboard listener unavailable: %s (%s) — "
                               "demos will record NOOP only",
                               type(exc).__name__, exc)
                self.available = False
                return False

        def _press(key):
            name = self._key_name(key)
            if name:
                self.on_press(name)

        def _release(key):
            name = self._key_name(key)
            if name:
                self.on_release(name)

        try:
            self._listener = self._keyboard.Listener(
                on_press=_press, on_release=_release)
            self.available = True
            return True
        except Exception as exc:
            LOGGER.warning("could not create a keyboard listener: %s (%s)",
                           type(exc).__name__, exc)
            self._listener = None
            self.available = False
            return False

    @staticmethod
    def _key_name(key) -> Optional[str]:
        if hasattr(key, "char") and key.char:
            return str(key.char).lower()
        if hasattr(key, "name") and key.name:
            return str(key.name).lower()
        return None

    def start(self) -> bool:
        """(Re)create and start the listener; True when keys will be seen."""
        if self._listener is None:
            self._build_listener()
        if self._listener is not None:
            self._listener.start()
            return True
        return False

    def stop(self) -> None:
        """Unregister hooks (requirement: no leftover listeners)."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as exc:
                LOGGER.warning("listener stop failed: %s", exc)
            self._listener = None


class DemoRecorder:
    """Subscribes to the frame ring and the human's keys to build episodes."""

    def __init__(self, cfg: BotConfig, out_dir: str | Path,
                 read_frame: Callable[[], Optional[object]]) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.read_frame = read_frame  # returns Frame or None (latest-wins)
        self.pre = ZonePreprocessor(cfg.perception, cfg.perception.horizon_frac,
                                    require_anchor=False)
        self._frames: list[np.ndarray] = []
        self._actions: list[int] = []
        self._ts: list[float] = []
        self._conf: list[float] = []
        self._death: list[str] = []
        self._score: list[float] = []
        self._recording = False
        self._lock = threading.Lock()
        self._current_action = NOOP
        self._tap = KeyboardTap(self._handle_press, self._handle_release)
        self._stop_requested = threading.Event()
        self.episode_paths: list[str] = []

    # ------------------------------------------------------------------ #
    def _handle_press(self, key_name: str) -> None:
        action = DEFAULT_KEY_TO_ACTION.get(key_name)
        # honour the configured keymap too (custom bindings)
        for a, name in self.cfg.input.keymap.items():
            if a != NOOP and name and name.lower() == key_name:
                action = a
        if action is not None:
            with self._lock:
                self._current_action = action

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

    def frame_count(self) -> int:
        """Rows recorded so far in the open episode (thread-safe)."""
        with self._lock:
            return len(self._frames)

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
        self._recording = True
        self._stop_requested.clear()
        # DEEP-FIX: surface a missing keyboard hook loudly.  Previously a
        # headless machine logged one warning at construction and then
        # happily recorded an all-NOOP episode that looked perfectly valid.
        keys_ok = self._tap.start()
        LOGGER.info("demo recording started (keyboard hook %s)",
                    "active" if keys_ok else "UNAVAILABLE — NOOP only")
        if not keys_ok:
            LOGGER.warning("no keyboard hook: this episode will contain NOOP "
                           "actions only and is not useful for BC")
        return True

    def stop(self, done: bool = True) -> Optional[str]:
        """Finalise and atomically write the episode; returns its path."""
        if not self._recording:
            return None
        self._recording = False
        self._tap.stop()
        # DEEP-FIX: tick() appends to these lists from the capture pump while
        # stop() reads them from the GUI thread; _recording=False is not a
        # barrier for a tick already in flight, so np.stack() could observe a
        # different length than the action/timestamp arrays.  Snapshot
        # everything under the same lock and verify the columns agree.
        with self._lock:
            frames = list(self._frames)
            actions = list(self._actions)
            ts = list(self._ts)
            conf = list(self._conf)
            death = list(self._death)
            score = list(self._score)
        lengths = {len(frames), len(actions), len(ts), len(conf),
                   len(death), len(score)}
        if len(lengths) != 1:
            LOGGER.error(
                "demo episode columns disagree (%s); truncating to the "
                "shortest so the saved file stays internally consistent",
                sorted(lengths))
            n_min = min(lengths)
            frames, actions, ts = frames[:n_min], actions[:n_min], ts[:n_min]
            conf, death, score = conf[:n_min], death[:n_min], score[:n_min]
        n = len(frames)
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
        # DEEP-FIX: %Y%m%d_%H%M%S has one-second resolution, so two episodes
        # finished in the same second overwrote each other.  Add a monotonic
        # counter (and keep scanning for a free name) so history is never lost.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        seq = 0
        while True:
            name = f"episode_{stamp}" + (f"_{seq:02d}" if seq else "") + ".npz"
            path = self.out_dir / name
            if not path.exists():
                break
            seq += 1
            if seq > 999:  # pragma: no cover - defensive
                path = self.out_dir / f"episode_{stamp}_{time.time_ns()}.npz"
                break
        tmp = path.with_suffix(".npz.tmp")
        # NOTE: pass an open file object — numpy silently appends ".npz" to a
        # str filename, which would leave the atomic tmp file never created.
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                frames=np.stack(frames).astype(np.uint8),
                actions=np.asarray(actions, dtype=np.int64),
                timestamps=np.asarray(ts, dtype=np.float64),
                done=done_flags,
                score=np.asarray(score, dtype=np.float32),
                confidence=np.asarray(conf, dtype=np.float32),
                death_state=np.asarray(death, dtype="U16"),
                meta=np.str_(json.dumps(meta)),
            )
        tmp.replace(path)
        self.episode_paths.append(str(path))
        LOGGER.info("demo episode saved: %s (%d steps)", path, n)
        return str(path)

    # ------------------------------------------------------------------ #
    def tick(self, frame, death_state: str = "ALIVE", confidence: float = 0.0,
             score: float = 0.0) -> bool:
        """Consume one frame while recording; True when a row was appended."""
        if not self._recording or frame is None:
            return False
        z = self.pre.process(frame.image, frame.frame_id, frame.ts)
        if not z.valid or z.ground_gray is None:
            return False
        # DEEP-FIX: every column is appended under the same lock so a
        # concurrent stop() can never see a ragged episode.
        with self._lock:
            if not self._recording:
                return False
            self._frames.append(np.ascontiguousarray(z.ground_gray))
            self._actions.append(int(self._current_action))
            self._ts.append(float(frame.ts))
            self._death.append(str(death_state))
            self._conf.append(float(confidence))
            self._score.append(float(score))
        return True

    def pump(self, max_frames: int = 4) -> int:
        """Pull up to ``max_frames`` from ``read_frame`` and record them.

        # DEEP-FIX: ``read_frame`` was stored by __init__ and never called by
        # anything in the repository -- grep shows ``.tick(`` only in
        # environment.py (FpsMeter), evaluation_tool.py and the recorder's own
        # unit tests.  The GUI wired a ring reader into the constructor and
        # nothing ever pumped it, so every recorded episode had 0 frames and
        # stop() always reported "nothing saved": behaviour cloning had no
        # data source at all.  The GUI now calls this from its Tk polling
        # loop, which is the only thread allowed to touch widgets.
        """
        if not self._recording or self.read_frame is None:
            return 0
        count = 0
        while count < max_frames:
            try:
                frame = self.read_frame()
            except Exception as exc:
                LOGGER.warning("demo frame read failed: %s (%s)",
                               type(exc).__name__, exc)
                return count
            if frame is None:
                break
            if self.tick(frame):
                count += 1
        return count

    def dispose(self) -> None:
        self._tap.stop()


def action_is_valid(a: int) -> bool:
    return 0 <= int(a) < N_ACTIONS
