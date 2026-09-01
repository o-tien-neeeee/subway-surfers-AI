"""Tests for :mod:`obstacle_perception` (v1.24.0 causal shaping module).

The tracker is pure-numpy structured vision that turns the full-frame policy
observation into a 3-lane x N-depth occupancy grid and emits a one-shot
``danger`` (something occupying the player's lane up close) and ``clear`` (a
near obstacle has passed) signal.  It must never throw on plain frames and
must be cheap enough to run per frame on CPU.
"""

import time

import numpy as np

from obstacle_perception import ObstacleTracker, OccupancyState


SIZE = 84


def _blank(lum: int = 60) -> np.ndarray:
    return np.full((SIZE, SIZE), lum, dtype=np.uint8)


def _with_block(frame: np.ndarray, row: int, col: int, h: int = 12,
                w: int = 14, lum: int = 240) -> np.ndarray:
    frame = frame.copy()
    frame[row:row + h, col:col + w] = lum
    return frame


def test_blank_frame_is_idle() -> None:
    tr = ObstacleTracker()
    st = tr.update(_blank())
    assert isinstance(st, OccupancyState)
    assert st.danger is False
    assert st.clear is False


def test_near_obstacle_in_player_lane_raises_danger() -> None:
    tr = ObstacleTracker(lanes=3, depths=5)
    # Player lane is the centre lane; put a bright block at the bottom (near)
    # of that lane.
    near = _with_block(_blank(), row=SIZE - 18, col=SIZE // 2 - 7)
    st = tr.update(near)
    assert st.danger is True
    assert st.dangers == 1


def test_obstacle_clears_after_passing() -> None:
    tr = ObstacleTracker(lanes=3, depths=5)
    near = _with_block(_blank(), row=SIZE - 18, col=SIZE // 2 - 7)
    assert tr.update(near).danger is True
    # Obstacle gone on the next frames -> once it was near and now isn't, the
    # tracker reports a clear (dodge/pass credit).
    cleared = False
    for _ in range(3):
        cleared = cleared or tr.update(_blank()).clear
    assert cleared is True
    assert tr.clears >= 1


def test_distant_obstacle_does_not_yet_danger() -> None:
    tr = ObstacleTracker(lanes=3, depths=5)
    # Near the horizon (top) the obstacle is far: not yet a near-lane danger.
    far = _with_block(_blank(), row=4, col=SIZE // 2 - 7)
    st = tr.update(far)
    # Far blocks populate the grid but should not trigger the *near* danger.
    assert st.danger is False


def test_runs_fast_enough_per_frame() -> None:
    tr = ObstacleTracker()
    frame = _blank()
    # warmup
    tr.update(frame)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        tr.update(frame)
    dt = (time.perf_counter() - t0) / n * 1000.0
    # Budget is generous on purpose: the tracker must stay well under a frame
    # on weak CPUs (< 2 ms per call).
    assert dt < 2.0, f"tracker too slow: {dt:.3f} ms/frame"
