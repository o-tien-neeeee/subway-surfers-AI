"""Death detection and survival sentinel.

Primary signal: a calibrated stable-UI colour anchor (5x5 RGB patch).
For every frame we compute the Euclidean RGB distance between the patch
median and the calibrated ALIVE baseline, apply a configurable threshold
(default 25) and a temporal debounce (default 3 consecutive confirmations)
before declaring DEAD_CONFIRMED.

Debounce state ladder (requirement §7):

    ALIVE -> POSSIBLE_EVENT (1 off-baseline frame)
          -> DEAD_CANDIDATE (2 .. confirm-1 frames)
          -> DEAD_CONFIRMED  (>= confirm frames)

A single changed pixel/flash therefore cannot trigger a respawn, and every
decision logs its raw distance + reason.

Secondary (optional) fallbacks: stagnation timeout, and template matching of
a saved game-over/respawn screenshot when the user provides one.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from config import DeathConfig
from perception import patch_median_rgb, patch_rgb_distance


class DeathState(enum.Enum):
    ALIVE = "ALIVE"
    POSSIBLE_EVENT = "POSSIBLE_EVENT"
    DEAD_CANDIDATE = "DEAD_CANDIDATE"
    DEAD_CONFIRMED = "DEAD_CONFIRMED"
    UNKNOWN = "UNKNOWN"


@dataclass
class DeathResult:
    state: DeathState
    distance: float
    reason: str
    frame_id: int = -1
    ts: float = 0.0
    stable_count: int = 0


class ColorAnchorDeathDetector:
    """Debounced colour-anchor death detector."""

    def __init__(self, cfg: DeathConfig) -> None:
        self.cfg = cfg
        self._off_frames = 0
        self._on_frames = 0
        self._state = DeathState.ALIVE
        self.last_result: Optional[DeathResult] = None
        self.last_distance = 0.0
        if not cfg.anchor_set():
            raise ValueError(
                "ColorAnchorDeathDetector requires a calibrated anchor "
                "(death.anchor_baseline_rgb / anchor_fx / anchor_fy)"
            )

    # ------------------------------------------------------------------ #
    def update(self, patch: Optional[np.ndarray], frame_id: int, ts: float) -> DeathResult:
        if patch is None:
            res = DeathResult(DeathState.UNKNOWN, -1.0, "invalid_or_missing_patch",
                              frame_id, ts, self._on_frames)
            self._state = DeathState.UNKNOWN
            self.last_result = res
            return res

        dist = patch_rgb_distance(patch, self.cfg.anchor_baseline_rgb)
        self.last_distance = dist
        threshold = self.cfg.threshold

        if dist < threshold:
            self._on_frames += 1
            self._off_frames = 0
            if self._state in (DeathState.DEAD_CONFIRMED, DeathState.DEAD_CANDIDATE):
                # Real death states need a stability window before ALIVE.
                if self._on_frames >= self.cfg.stable_frames:
                    new_state = DeathState.ALIVE
                    reason = f"stable_{self._on_frames}_frames"
                else:
                    new_state = self._state  # recovering, not yet stable
                    reason = f"recovering_{self._on_frames}/{self.cfg.stable_frames}"
            else:
                # POSSIBLE_EVENT (a single-frame flash) reverts immediately:
                # one frame of animation must not demand a stability window.
                new_state = DeathState.ALIVE
                reason = "on_baseline"
        else:
            self._on_frames = 0
            self._off_frames += 1
            if self._off_frames >= self.cfg.confirm_frames:
                new_state = DeathState.DEAD_CONFIRMED
                reason = f"off_baseline_{self._off_frames}_frames"
            elif self._off_frames == 1:
                new_state = DeathState.POSSIBLE_EVENT
                reason = "first_off_baseline"
            else:
                new_state = DeathState.DEAD_CANDIDATE
                reason = f"off_baseline_{self._off_frames}_frames"

        self._state = new_state
        res = DeathResult(new_state, dist, reason, frame_id, ts, self._on_frames)
        self.last_result = res
        return res

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> DeathState:
        return self._state

    def is_dead(self) -> bool:
        return self._state is DeathState.DEAD_CONFIRMED

    def is_alive_stable(self) -> bool:
        return self._state is DeathState.ALIVE and self._on_frames >= self.cfg.stable_frames


class StagnationDetector:
    """Fallback: flags long visual stagnation (possible hidden game-over)."""

    def __init__(self, timeout_s: float, change_threshold: float = 1.0) -> None:
        self.timeout_s = timeout_s
        self.change_threshold = change_threshold
        self._last_change_ts: Optional[float] = None
        self._last_frame: Optional[np.ndarray] = None

    def update(self, ground_gray: Optional[np.ndarray], ts: float) -> bool:
        """Returns True when stagnation exceeded the timeout."""
        if ground_gray is None:
            return False
        if self._last_frame is None or self._last_frame.shape != ground_gray.shape:
            self._last_frame = ground_gray.copy()
            self._last_change_ts = ts
            return False
        diff = np.abs(ground_gray.astype(np.int16) - self._last_frame.astype(np.int16))
        if float(diff.mean()) > self.change_threshold:
            self._last_change_ts = ts
        self._last_frame = ground_gray.copy()
        if self._last_change_ts is None:
            return False
        return (ts - self._last_change_ts) > self.timeout_s

    def reset(self) -> None:
        self._last_change_ts = None
        self._last_frame = None


class TemplateDetector:
    """Optional secondary: normalised cross-correlation against a saved png."""

    def __init__(self, template_path: str, threshold: float = 0.8) -> None:
        import cv2  # local import: optional path only

        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError(f"template not readable: {template_path}")
        self.threshold = threshold

    def match(self, region_gray: np.ndarray) -> tuple[bool, float]:
        import cv2

        if region_gray.shape[0] < self.template.shape[0] or \
           region_gray.shape[1] < self.template.shape[1]:
            return False, 0.0
        res = cv2.matchTemplate(region_gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, _ = cv2.minMaxLoc(res)
        return bool(max_v >= self.threshold), float(max_v)


@dataclass
class RespawnStatus:
    action: str  # "CLICK" | "WAIT" | "RECOVERED" | "FAILED"
    detail: str
    ts: float


class RespawnController:
    """Death -> respawn state machine (unit-tested with synthetic frames).

    Behaviour on DEAD_CONFIRMED (requirement §7):
      * click the calibrated respawn point every ``interval_s`` seconds,
      * recheck the death anchor between clicks,
      * give up after ``timeout_s`` (caller moves to ERROR/PAUSED — never
        clicks forever),
      * declare RECOVERED only after ``stable_frames`` alive frames.
    """

    def __init__(
        self,
        click_fn: Callable[[float], bool],
        cfg: DeathConfig,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.click_fn = click_fn
        self.cfg = cfg
        self.now = now
        self._active = False
        self._start_ts = 0.0
        self._last_click_ts = -1e9
        self._stable = 0
        self.clicks = 0

    def start(self) -> None:
        self._active = True
        self._start_ts = self.now()
        self._last_click_ts = -1e9
        self._stable = 0
        self.clicks = 0

    @property
    def active(self) -> bool:
        return self._active

    def update(self, death_state: DeathState, ts: Optional[float] = None) -> RespawnStatus:
        """Feed the current anchor state; returns the next respawn action."""
        t = ts if ts is not None else self.now()
        if not self._active:
            return RespawnStatus("WAIT", "inactive", t)

        if t - self._start_ts > self.cfg.respawn_timeout_s:
            self._active = False
            return RespawnStatus("FAILED", "timeout", t)

        if death_state is DeathState.ALIVE:
            self._stable += 1
            if self._stable >= self.cfg.stable_frames:
                self._active = False
                return RespawnStatus("RECOVERED", f"stable_{self._stable}", t)
            return RespawnStatus("WAIT", f"stabilising_{self._stable}", t)

        self._stable = 0
        due = (t - self._last_click_ts) >= self.cfg.respawn_interval_s
        if due:
            ok = bool(self.click_fn(t))
            self._last_click_ts = t
            self.clicks += 1
            return RespawnStatus("CLICK" if ok else "WAIT", "click_sent", t)
        return RespawnStatus("WAIT", "cooldown", t)


def synthetic_patch(rgb: tuple[int, int, int], noise: float = 0.0,
                    rng: Optional[np.random.Generator] = None,
                    size: int = 5) -> np.ndarray:
    """Build a synthetic NxNx3 anchor patch (used by tests and dry runs)."""
    rng = rng or np.random.default_rng(0)
    patch = np.tile(np.array(rgb, dtype=np.float32), (size, size, 1))
    if noise > 0:
        patch += rng.normal(0.0, noise, patch.shape)
    return np.clip(patch, 0, 255).astype(np.uint8)
