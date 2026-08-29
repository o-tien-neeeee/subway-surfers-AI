"""Horizon-zone change detector (cheap hazard proxy).

Signals (per frame): change score (smoothed mean abs diff 0..255), binary
detection with temporal debounce, confidence in [0,1], timestamp, frame id.

The detector is intentionally simple: a smoothed frame-difference over the
40x40 horizon band.  It is a *danger heuristic* that tightens action timing
and feeds the pending-hazard reward — it is NOT claimed to be an object
detector, and real-game precision/recall is pending calibration (see README).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from config import HorizonConfig


@dataclass
class HorizonResult:
    frame_id: int
    ts: float
    change_score: float      # smoothed mean |diff| in 0..255
    raw_score: float         # unsmoothed, for logging
    changed_ratio: float     # fraction of cells above cell threshold
    detected: bool
    confidence: float        # 0..1
    reason: str = "ok"


class HorizonDetector:
    def __init__(self, cfg: HorizonConfig, size: int = 40) -> None:
        self.cfg = cfg
        self.size = size
        self._prev: np.ndarray | None = None
        self._ewma = 0.0
        self._window: deque[int] = deque(maxlen=max(1, cfg.debounce_window))
        self._no_data_count = 0

    def reset(self) -> None:
        self._prev = None
        self._ewma = 0.0
        self._window.clear()

    def update(self, horizon_gray: np.ndarray | None, frame_id: int, ts: float | None = None
               ) -> HorizonResult:
        ts = ts if ts is not None else time.monotonic()
        if horizon_gray is None:
            self._no_data_count += 1
            self._prev = None
            return HorizonResult(
                frame_id, ts, 0.0, 0.0, 0.0, False, 0.0, reason="no_input"
            )
        frame = np.asarray(horizon_gray, dtype=np.uint8)
        if self._prev is None or self._prev.shape != frame.shape:
            self._prev = frame.copy()
            return HorizonResult(frame_id, ts, 0.0, 0.0, 0.0, False, 0.0, reason="first_frame")

        # Raw mean absolute difference and per-cell change map.
        diff = cv2_absdiff(frame, self._prev)
        raw = float(diff.mean())
        changed = (diff >= self.cfg.cell_threshold).astype(np.float32)
        changed_ratio = float(changed.mean())

        # Smoothed score (EWMA) + debounced binary detection.
        a = self.cfg.ewma_alpha
        self._ewma = a * raw + (1.0 - a) * self._ewma
        hit = 1 if self._ewma >= self.cfg.diff_threshold else 0
        self._window.append(hit)
        hits = sum(self._window)
        big_object = changed_ratio >= self.cfg.min_changed_cell_ratio
        detected = bool(hits >= self.cfg.debounce_hits and big_object)
        if self._ewma >= self.cfg.diff_threshold and not big_object:
            reason = "diff_small_area"
        elif detected:
            reason = "debounced_hit"
        else:
            reason = "quiet"
        confidence = min(1.0, max(0.0, (self._ewma - self.cfg.diff_threshold)
                                  / max(1e-6, self.cfg.confidence_scale)))
        self._prev = frame.copy()
        return HorizonResult(
            frame_id=frame_id,
            ts=ts,
            change_score=float(self._ewma),
            raw_score=raw,
            changed_ratio=changed_ratio,
            detected=detected,
            confidence=confidence if detected else 0.0,
            reason=reason,
        )

    @property
    def no_data_count(self) -> int:
        return self._no_data_count


def cv2_absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """cv2-free abs diff (numpy is fast enough at 40x40 and keeps CI light)."""
    d = a.astype(np.int16) - b.astype(np.int16)
    return np.abs(d).astype(np.uint8)
