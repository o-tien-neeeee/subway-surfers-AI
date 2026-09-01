"""Structured obstacle perception for reward shaping and early danger.

Deep-research fix (DEEP_RESEARCH_vi.md §3.3/§4.3): the old reward only knew
"alive time" and a crude horizon frame-diff. It could never tell *why* a run
died or reward a *successful dodge*. This module turns the same grayscale
policy observation into a small, interpretable occupancy estimate:

    3 lanes  x  N depth cells  (far -> near, i.e. horizon -> player)

In Subway Surfers obstacles spawn at the horizon (small, near the top of the
track) and rush toward the camera (grow toward the bottom). A depth cell is
"occupied" when its edge energy against the track background exceeds a
threshold relative to that row's own noise floor. From one frame to the next
we track occupied cells and derive two CAUSAL, no-future-leak signals:

  * ``clear``   — an obstacle that was near (depth >= 1) in a previous frame
                  is no longer present AND the player is alive: it passed the
                  player harmlessly -> reward a successful dodge.
  * ``danger``  — an obstacle occupies the player's OWN lane in the nearest
                  depth cell (imminent collision) -> small early penalty so
                  the credit for dying starts ~1 s before impact.

This is deliberately classical CV (numpy only, no torch): it costs a fraction
of a millisecond per decision on the i5-7200U target, is debuggable (the
occupancy grid can be logged/drawn), and gives RL dense grounding that pure
survival time cannot. The learned policy still consumes the full image; this
module only shapes the reward and the scheduler's danger flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OccupancyState:
    lanes: int
    depths: int
    grid: np.ndarray | None = None          # [depths, lanes] float 0..1
    danger: bool = False
    danger_lane: int = 1                      # best-effort player lane (0..2)
    clear: bool = False
    clears: int = 0
    dangers: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "danger": 1.0 if self.danger else 0.0,
            "clear": 1.0 if self.clear else 0.0,
            "clears": float(self.clears),
            "dangers": float(self.dangers),
        }


class ObstacleTracker:
    """Lane/depth occupancy from the grayscale policy frame.

    Parameters
    ----------
    lanes, depths: grid resolution (3 lanes x ``depths`` depth cells).
    top_frac, bottom_frac: track strip in observation fractions (the track
        occupies roughly the lower 2/3; the UI/sky is ignored).
    lane_margin_frac: side margin so the outer lane centres sit on the rails.
    edge_thresh: occupancy threshold expressed as a multiplier of the per-row
        edge median (robust to theme/brightness changes).
    near_depth: depth index (0=far) at/above which an obstacle is "near" and
        counts toward danger/clear tracking.
    """

    def __init__(self, lanes: int = 3, depths: int = 5, top_frac: float = 0.30,
                 bottom_frac: float = 0.92, lane_margin_frac: float = 0.16,
                 edge_thresh: float = 1.8, near_depth: int | None = None,
                 player_lane: int = 1) -> None:
        self.lanes = lanes
        self.depths = depths
        self.top_frac = top_frac
        self.bottom_frac = bottom_frac
        self.lane_margin_frac = lane_margin_frac
        self.edge_thresh = edge_thresh
        self.near_depth = max(1, depths - 1) if near_depth is None else near_depth
        self.player_lane = player_lane
        self._prev_near: set[tuple[int, int]] = set()
        self.clears = 0
        self.dangers = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._prev_near = set()
        self.clears = 0
        self.dangers = 0

    def _lane_centres(self, w: int) -> np.ndarray:
        x0 = int(w * self.lane_margin_frac)
        x1 = int(w * (1.0 - self.lane_margin_frac))
        return np.linspace(x0, x1, self.lanes)

    def occupancy(self, gray: np.ndarray) -> np.ndarray:
        """Return a [depths, lanes] float occupancy grid in [0, 1]."""
        h, w = gray.shape
        y0 = int(h * self.top_frac)
        y1 = int(h * self.bottom_frac)
        track = gray[y0:y1, :].astype(np.float32)
        # Edge energy: strong local intensity change = obstacle silhouette
        # against the relatively smooth track/ground.
        gx = np.abs(np.diff(track, axis=1))
        gy = np.abs(np.diff(track, axis=0))
        edge = np.zeros_like(track)
        edge[:, :-1] += gx
        edge[:-1, :] += gy
        th, tw = track.shape
        lanes_x = self._lane_centres(tw)
        lane_w = max(4, int((lanes_x[1] - lanes_x[0]) * 0.62))
        cell_h = max(4, th // self.depths)
        grid = np.zeros((self.depths, self.lanes), dtype=np.float32)
        for d in range(self.depths):
            cy0 = d * cell_h
            cy1 = min(th, cy0 + cell_h)
            row = edge[cy0:cy1, :]
            if row.size == 0:
                continue
            floor = np.median(row) + 1e-3
            for li, cx in enumerate(lanes_x):
                cx0 = max(0, int(cx - lane_w / 2))
                cx1 = min(tw, int(cx + lane_w / 2))
                block = row[:, cx0:cx1]
                energy = float(np.mean(block))
                # Depth-normalised: far obstacles are smaller -> scale the
                # near rows a touch stricter? We use per-row floor for
                # robustness and a soft occupancy in [0,1].
                grid[d, li] = np.clip(energy / (self.edge_thresh * floor), 0.0, 1.0)
        return grid

    def update(self, gray: np.ndarray, player_lane: int | None = None) -> OccupancyState:
        if gray is None:
            return OccupancyState(self.lanes, self.depths)
        if player_lane is not None:
            self.player_lane = int(np.clip(player_lane, 0, self.lanes - 1))
        grid = self.occupancy(gray)
        occ = grid > 0.5
        # DANGER: occupied cell in the player's lane at a near depth.
        near_cells = {
            (d, li)
            for d in range(self.near_depth, self.depths)
            for li in range(self.lanes)
            if occ[d, li]
        }
        danger = any(li == self.player_lane for _d, li in near_cells)
        # CLEAR: an obstacle that WAS near last frame(s) has vanished (passed
        # the camera) without the episode terminating (death is handled by the
        # caller, which only calls update on live frames).
        clear = False
        if self._prev_near and not near_cells:
            # Something near before, nothing near now => it went past us.
            clear = True
            self.clears += 1
        elif self._prev_near:
            # Cells can disappear when an obstacle is dodged off-lane and
            # exits near row: count a clear when a near cell in a DIFFERENT
            # lane vanishes (it passed beside/over us).
            still = self._prev_near & near_cells
            vanished = self._prev_near - near_cells
            if vanished and not any(li == self.player_lane for _d, li in still):
                clear = True
                self.clears += 1
        if danger:
            self.dangers += 1
        # Keep a light memory: persist near cells that are still visible so a
        # single noisy frame cannot mint a clear; update prev set.
        self._prev_near = near_cells if near_cells else self._prev_near
        if clear:
            # Consume the memory so one obstacle can't pay twice.
            self._prev_near = near_cells
        return OccupancyState(
            lanes=self.lanes, depths=self.depths, grid=grid,
            danger=danger, danger_lane=self.player_lane,
            clear=clear, clears=self.clears, dangers=self.dangers,
        )
