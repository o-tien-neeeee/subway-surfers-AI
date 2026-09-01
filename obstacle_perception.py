"""Structured obstacle perception for causal reward shaping.

Why this module exists (deep-research finding, v1.24.0)
------------------------------------------------------
The legacy reward could only say two things: "you are still alive"
(+alive_per_frame) and "the horizon detector blinked, and you pressed a
key afterwards" (``hazard_bonus``).  The second one is the classic
reward-hacking trap: it pays for *any* press near a blink, so the policy
that maximises it is button-mashing — precisely the "agent learnt to keep
jumping or rolling" failure mode reported by every published Subway
Surfers DQN write-up.

This module gives the reward a *causal* fact instead:

    an obstacle that was in the player's lane is now behind the player,
    and the player is still alive  =>  that dodge worked.

and its negative counterpart:

    an obstacle is sitting in the player's lane in the nearest rows
    =>  the player is in danger right now.

Two producers, one consumer
---------------------------
* :class:`ObstacleTracker` — pure CV (Sobel edge density per lane/depth
  cell) for the LIVE game, where the only ground truth is pixels.  Runs
  in well under a millisecond on an 84x84 frame, numpy/cv2 only, no
  torch on this path.
* :func:`snapshot_from_game_state` — the same snapshot computed from
  :class:`environment.SyntheticGame`'s authoritative obstacle list, for
  headless training/evaluation where pixel CV would be a lossy proxy for
  data we already have exactly.

Both return an :class:`ObstacleSnapshot`, which is what
:meth:`rewards.SurvivalRewardCalculator.step` consumes, so the reward
code does not care where the fact came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from config import ObstacleConfig


@dataclass
class ObstacleSnapshot:
    """One step's structured view of the track in front of the player.

    ``cleared`` is the number of obstacles that left the player's lane
    *while the player stayed alive* — the causal success credit.
    ``danger`` is True when an obstacle occupies the player's lane in one
    of the nearest depth rows — the "you are about to be hit" signal.
    ``hazards_in_lane`` counts obstacles currently inside the player's
    lane at any depth (useful for logging/telemetry).
    """

    occupancy: np.ndarray  # [depth_rows, lanes] bool
    player_lane: int
    cleared: int = 0
    danger: bool = False
    hazards_in_lane: int = 0
    ts: float = 0.0
    frame_id: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "cleared": float(self.cleared),
            "danger": 1.0 if self.danger else 0.0,
            "hazards_in_lane": float(self.hazards_in_lane),
            "occupancy": float(self.occupancy.sum()),
        }


class ObstacleTracker:
    """Pixel-domain lane/depth occupancy tracker (live game path).

    The captured region is projected onto a ``depth_rows x lanes`` grid:

    * columns are the lanes (the region is split into ``lanes`` equal
      vertical strips, with ``lane_margin_frac`` of each strip ignored so
      the painted lane dividers do not read as obstacles);
    * rows are depth, from ``top_frac`` (far, near the horizon) to
      ``bottom_frac`` (the player's row).

    A cell is *occupied* when its Sobel edge density exceeds
    ``grad_threshold`` — obstacles in Subway Surfers (trains, barriers,
    ramps) are large high-contrast objects, while the scrolling track is
    low-contrast.  Occupancy is then debounced: a cell must stay occupied
    for ``confirm_cells`` consecutive updates before it becomes a
    a latched "this lane had a confirmed obstacle" flag, which is what
    keeps single-frame noise from paying out a dodge bonus.

    A lane pays ``cleared`` exactly once, on the update where it goes
    empty after having latched a confirmed near-row obstacle — i.e. the
    obstacle passed the player.
    """

    def __init__(self, cfg: ObstacleConfig) -> None:
        self.cfg = cfg
        self.rows = max(1, int(cfg.depth_rows))
        self.lanes = max(1, int(cfg.lanes))
        # Per-lane state (see update_from_occupancy): consecutive near-row
        # hits, and the latch set once an obstacle is confirmed as imminent.
        self._lane_near_hits: list[int] = [0] * self.lanes
        self._lane_saw_near: list[bool] = [False] * self.lanes
        self.cleared_total = 0
        self.danger_steps = 0
        self.steps = 0
        self._last_occupancy: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._lane_near_hits = [0] * self.lanes
        self._lane_saw_near = [False] * self.lanes
        self._last_occupancy = None
        self.cleared_total = 0
        self.danger_steps = 0
        self.steps = 0

    # ------------------------------------------------------------------ #
    def _cells(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Pixel rectangles (y0, y1, x0, x1) for each (row, lane) cell."""
        h, w = gray.shape[:2]
        y0 = int(round(h * self.cfg.top_frac))
        y1 = max(y0 + 1, int(round(h * self.cfg.bottom_frac)))
        margin = self.cfg.lane_margin_frac
        lane_w = w / self.lanes
        cells: list[tuple[int, int, int, int]] = []
        row_h = (y1 - y0) / self.rows
        for r in range(self.rows):
            ry0 = int(round(y0 + r * row_h))
            ry1 = max(ry0 + 1, int(round(y0 + (r + 1) * row_h)))
            for c in range(self.lanes):
                cx0 = c * lane_w
                cx1 = (c + 1) * lane_w
                pad = (cx1 - cx0) * margin
                x0 = max(0, int(round(cx0 + pad)))
                x1 = min(w, max(x0 + 1, int(round(cx1 - pad))))
                cells.append((ry0, ry1, x0, x1))
        return cells

    def occupancy_from_gray(self, gray: np.ndarray) -> np.ndarray:
        """[depth_rows, lanes] bool occupancy map for one grayscale frame."""
        if gray is None or gray.size == 0:
            return np.zeros((self.rows, self.lanes), dtype=bool)
        g = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        strong = mag > float(self.cfg.edge_pixel_threshold)
        out = np.zeros((self.rows, self.lanes), dtype=bool)
        for i, (ry0, ry1, x0, x1) in enumerate(self._cells(g)):
            cell = strong[ry0:ry1, x0:x1]
            if cell.size == 0:
                continue
            frac = float(cell.mean())
            r, c = divmod(i, self.lanes)
            out[r, c] = (frac >= self.cfg.min_cell_frac
                         and float(mag[ry0:ry1, x0:x1].mean())
                         >= self.cfg.grad_threshold)
        return out

    # ------------------------------------------------------------------ #
    def update(self, gray: np.ndarray, player_lane: int, ts: float = 0.0,
               frame_id: int = 0, died: bool = False) -> ObstacleSnapshot:
        """Advance one step; return the causal snapshot for this step."""
        occ = self.occupancy_from_gray(gray)
        return self.update_from_occupancy(occ, player_lane, ts, frame_id, died)

    def update_from_occupancy(
        self, occupancy: np.ndarray, player_lane: int, ts: float = 0.0,
        frame_id: int = 0, died: bool = False,
    ) -> ObstacleSnapshot:
        """Core state machine — shared by the CV and game-state paths.

        An obstacle MOVES down the grid, so a per-cell "seen N times in a
        row" rule never confirms anything (each cell is occupied for one or
        two updates only).  The state is therefore per LANE:

        * ``_lane_near_hits[c]`` counts consecutive updates with an
          obstacle in lane ``c``'s near rows — the debounce that keeps a
          single noisy frame from paying a dodge bonus;
        * ``_lane_saw_near[c]`` latches once that count reaches
          ``confirm_cells``;
        * when lane ``c`` goes completely empty after having latched, the
          obstacle has passed the player.  That is the ``cleared`` event,
          paid only while the player is alive.
        """
        self.steps += 1
        near_from = max(0, self.rows - max(1, self.cfg.near_rows))
        lane = int(max(0, min(self.lanes - 1, player_lane)))
        cleared = 0

        for c in range(self.lanes):
            in_lane = int(occupancy[:, c].sum())
            near_hits = int(occupancy[near_from:, c].sum()) > 0
            if near_hits:
                self._lane_near_hits[c] += 1
                if self._lane_near_hits[c] >= self.cfg.confirm_cells:
                    self._lane_saw_near[c] = True
            elif in_lane == 0:
                # Lane is empty again: anything that had reached the near
                # rows is now behind the player.
                if self._lane_saw_near[c]:
                    if c == lane and not died:
                        cleared += 1
                    self._lane_saw_near[c] = False
                self._lane_near_hits[c] = 0
            # else: an obstacle is still travelling through the far rows —
            # keep the latch and the counter as they are.

        cleared = int(min(cleared, self.cfg.max_clears_per_step))
        self.cleared_total += cleared
        self._last_occupancy = occupancy

        in_player_lane = occupancy[:, lane]
        hazards = int(in_player_lane.sum())
        danger = bool(hazards > 0
                      and int(occupancy[near_from:, lane].sum()) > 0)
        if danger:
            self.danger_steps += 1
        return ObstacleSnapshot(
            occupancy=occupancy, player_lane=lane, cleared=cleared,
            danger=danger, hazards_in_lane=hazards, ts=ts, frame_id=frame_id,
        )

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, float]:
        return {
            "obstacle_steps": float(self.steps),
            "obstacles_cleared": float(self.cleared_total),
            "danger_steps": float(self.danger_steps),
            "tracks_live": float(sum(1 for s in self._lane_saw_near if s)),
        }


# --------------------------------------------------------------------- #
# Synthetic / headless path — exact occupancy from the game's own state
# --------------------------------------------------------------------- #
def occupancy_from_game_state(
    obstacles: Iterable[dict], rows: int, lanes: int,
    top_prog: float = 0.0, bottom_prog: float = 0.97,
) -> np.ndarray:
    """[rows, lanes] bool map from a ``SyntheticGame.obstacles`` list.

    Each obstacle carries ``prog`` (0 = horizon, ~0.97 = collision row)
    and ``lane``; row 0 is the farthest band and row ``rows-1`` the
    player's row, mirroring the CV tracker's geometry.
    """
    out = np.zeros((max(1, rows), max(1, lanes)), dtype=bool)
    span = max(1e-6, bottom_prog - top_prog)
    for ob in obstacles:
        prog = float(ob.get("prog", 0.0))
        lane = int(ob.get("lane", 0))
        if not (0 <= lane < out.shape[1]):
            continue
        frac = (prog - top_prog) / span
        r = int(min(out.shape[0] - 1, max(0, round(frac * (out.shape[0] - 1)))))
        out[r, lane] = True
    return out


def snapshot_from_game_state(
    obstacles: Sequence[dict], player_lane: int, rows: int = 5, lanes: int = 3,
) -> ObstacleSnapshot:
    """Build an :class:`ObstacleSnapshot` without any pixel work."""
    occ = occupancy_from_game_state(obstacles, rows, lanes)
    lane = int(max(0, min(lanes - 1, player_lane)))
    return ObstacleSnapshot(
        occupancy=occ, player_lane=lane,
        hazards_in_lane=int(occ[:, lane].sum()),
        danger=bool(occ[max(0, rows - 2):, lane].sum() > 0),
    )
