"""Zone preprocessing and temporal frame stacking.

Pipeline per frame (all numpy/cv2, no torch on this path):

  captured RGB region
    ├─ horizon band  -> resize 40x40, grayscale  -> HorizonDetector
    └─ ground band   -> resize 84x84, grayscale  -> uint8 FrameStack

* Observations stay uint8 until they enter the network (normalise inside the
  forward pass) — 4x less RAM than float32 and zero-copy views for stacking.
* The stack is a ring of 4 buffers; ``get()`` builds the [4,84,84] uint8
  array once per inference (28 KB copy — negligible).
* Handles black frames, duplicates and invalid frames explicitly: they never
  enter the stack, they are counted, and they never block the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import PerceptionConfig


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """uint8 RGB -> uint8 grayscale (ITU-R BT.601 luma, as in Atari papers)."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def is_black_frame(image: np.ndarray, threshold: float) -> bool:
    """True when the whole region is essentially black (tab switch, load)."""
    return float(image.mean()) < threshold


def anchor_patch(image: np.ndarray, cx: int, cy: int, half: int = 2) -> Optional[np.ndarray]:
    """Extract a (2*half+1)^2 RGB patch centred on (cx, cy), None if clipped."""
    h, w = image.shape[:2]
    x0, x1 = cx - half, cx + half + 1
    y0, y1 = cy - half, cy + half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return image[y0:y1, x0:x1]


def patch_median_rgb(patch: np.ndarray) -> tuple[int, int, int]:
    """Robust baseline: per-channel median over the patch (trimmed mean also OK)."""
    med = np.median(patch.reshape(-1, 3), axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def patch_rgb_distance(patch: np.ndarray, baseline: tuple[int, int, int]) -> float:
    """Euclidean distance between the patch median and the baseline RGB."""
    r, g, b = patch_median_rgb(patch)
    br, bg, bb = baseline
    return float(np.sqrt((r - br) ** 2 + (g - bg) ** 2 + (b - bb) ** 2))


def patch_stability(patch_samples: list[np.ndarray]) -> tuple[float, tuple[int, int, int]]:
    """Max per-channel std across samples — the anchor 'stability score'."""
    if not patch_samples:
        return float("inf"), (-1, -1, -1)
    arr = np.stack([p.reshape(-1, 3).mean(axis=0) for p in patch_samples])
    stds = arr.std(axis=0)
    med = np.median(np.stack([p.reshape(-1, 3) for p in patch_samples]).reshape(-1, 3), axis=0)
    return float(stds.max()), (int(med[0]), int(med[1]), int(med[2]))


@dataclass
class ZoneResult:
    """Preprocessed zones + validity flags for one captured frame."""

    frame_id: int
    ts: float
    horizon_gray: Optional[np.ndarray]  # 40x40 uint8
    ground_gray: Optional[np.ndarray]  # 84x84 uint8
    anchor_patch: Optional[np.ndarray]  # 5x5x3 uint8
    valid: bool
    reason: str = "ok"


class ZonePreprocessor:
    """Splits the region into horizon/ground zones and samples the anchor."""

    def __init__(
        self,
        cfg: PerceptionConfig,
        horizon_frac: float,
        anchor_xy: Optional[tuple[int, int]] = None,
        require_anchor: bool = True,
    ) -> None:
        self.cfg = cfg
        self.horizon_frac = horizon_frac
        self.anchor_xy = anchor_xy  # absolute pixels inside region
        # When False, frames stay valid without a death-anchor patch — used by
        # the demo recorder (BC data does not need the anchor) and by the
        # actor when the user has not calibrated step 3 yet.
        self.require_anchor = require_anchor
        self.horizon_h_cache: dict[int, int] = {}

    def set_anchor(self, xy: Optional[tuple[int, int]]) -> None:
        self.anchor_xy = xy

    def split_line(self, region_h: int) -> int:
        """Pixel row separating horizon band from ground band."""
        return max(1, int(round(region_h * self.horizon_frac)))

    def process(self, image: np.ndarray, frame_id: int, ts: float) -> ZoneResult:
        h, w = image.shape[:2]
        if h < 8 or w < 8:
            return ZoneResult(frame_id, ts, None, None, None, False, "tiny_region")
        if is_black_frame(image, self.cfg.black_mean_threshold):
            return ZoneResult(frame_id, ts, None, None, None, False, "black")
        split = self.split_line(h)

        horizon = image[:split, :, :]
        ground = image[split:, :, :]

        horizon_gray = cv2.resize(
            rgb_to_gray(horizon),
            (self.cfg.horizon_size, self.cfg.horizon_size),
            interpolation=cv2.INTER_AREA,
        )
        ground_gray = cv2.resize(
            rgb_to_gray(ground),
            (self.cfg.ground_size, self.cfg.ground_size),
            interpolation=cv2.INTER_AREA,
        )
        patch = anchor_patch(image, *self.anchor_xy) if self.anchor_xy else None
        valid = patch is not None or not self.require_anchor
        return ZoneResult(
            frame_id=frame_id,
            ts=ts,
            horizon_gray=horizon_gray,
            ground_gray=ground_gray,
            anchor_patch=patch,
            valid=valid,
            reason="ok" if valid else "anchor_missing",
        )


class FrameStack:
    """Ring buffer of ``k`` uint8 ground frames -> [k,H,W] observation.

    ``reset`` fills the stack with the first frame (standard Atari practice)
    so episode starts have a well-defined observation without leaking the
    previous episode's pixels.
    """

    def __init__(self, k: int, size: int) -> None:
        if k < 1:
            raise ValueError("frame stack size must be >= 1")
        self.k = k
        self.size = size
        self._buf = np.zeros((k, size, size), dtype=np.uint8)
        self._filled = 0
        self._idx = 0

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        frame = np.ascontiguousarray(first_frame, dtype=np.uint8)
        for i in range(self.k):
            self._buf[i] = frame
        self._idx = 0
        self._filled = self.k
        return self.get()

    def push(self, frame: np.ndarray) -> None:
        self._buf[self._idx] = frame
        self._idx = (self._idx + 1) % self.k
        self._filled = min(self._filled + 1, self.k)

    def ready(self) -> bool:
        return self._filled >= self.k

    def get(self) -> np.ndarray:
        """Return [k,H,W] uint8 with the newest frame last."""
        if self._filled < self.k:
            raise RuntimeError("FrameStack not ready — call reset() first")
        out = np.empty_like(self._buf)
        newest = self._idx - 1  # last written slot
        for i in range(self.k):
            src = (newest - i) % self.k  # newest, newest-1, ...
            out[self.k - 1 - i] = self._buf[src]
        return out


def normalize_obs(stack_u8: np.ndarray) -> "torch.Tensor":  # noqa: F821
    """uint8 [k,H,W] -> float32 tensor [1,k,H,W] in [0,1] (for the network)."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(stack_u8)).float().div_(255.0)
    return t.unsqueeze(0)
