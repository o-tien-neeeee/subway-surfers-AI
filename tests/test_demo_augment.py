"""Tests for demo augmentation (mirror + keypress-window).

These tests do NOT touch the recorder — they verify the rulebook the
dataset uses at training time and the recorder could use to count
augmentations live.
"""

from __future__ import annotations

import numpy as np

from config import NOOP, LEFT, RIGHT, JUMP, SLIDE
from demo_augment import (
    DemoAugmentor,
    keypress_windows,
    mirror_action,
    mirror_action_counts,
    mirror_frame,
)


# --------------------------------------------------------------------- #
# Pure numpy utilities
# --------------------------------------------------------------------- #
class TestMirrorFrame:
    def test_2d_horizontal_flip(self) -> None:
        a = np.array([[1, 2, 3, 4]], dtype=np.uint8)
        m = mirror_frame(a)
        assert m.shape == (1, 4)
        assert m.tolist() == [[4, 3, 2, 1]]

    def test_3d_horizontal_flip_keeps_k_axis(self) -> None:
        # 3 frames of width 4 — flipping W must keep the 3 frame slots
        # in the same order (newest = last, that is the convention).
        a = np.array([
            [[1, 2, 3, 4]],
            [[5, 6, 7, 8]],
            [[9, 10, 11, 12]],
        ], dtype=np.uint8)
        m = mirror_frame(a)
        assert m.shape == (3, 1, 4)
        assert m[0].tolist() == [[4, 3, 2, 1]]
        assert m[-1].tolist() == [[12, 11, 10, 9]]

    def test_does_not_alias_input(self) -> None:
        a = np.array([[1, 2, 3]], dtype=np.uint8)
        m = mirror_frame(a)
        m[0, 0] = 99
        # If the input buffer were aliased this would also have changed.
        assert a[0, 0] == 1

    def test_rejects_other_dims(self) -> None:
        try:
            mirror_frame(np.zeros((2, 2, 2, 2), dtype=np.uint8))
        except ValueError:
            return
        raise AssertionError("expected ValueError for unsupported ndim")


class TestMirrorAction:
    def test_lane_swap(self) -> None:
        assert mirror_action(LEFT) == RIGHT
        assert mirror_action(RIGHT) == LEFT

    def test_jump_slide_noop_are_symmetric(self) -> None:
        assert mirror_action(JUMP) == JUMP
        assert mirror_action(SLIDE) == SLIDE
        assert mirror_action(NOOP) == NOOP

    def test_unknown_action_raises(self) -> None:
        try:
            mirror_action(7)
        except ValueError:
            return
        raise AssertionError("expected ValueError for unknown action")


class TestKeypressWindows:
    def test_no_presses_returns_empty(self) -> None:
        a = np.array([NOOP, NOOP, NOOP], dtype=np.int64)
        assert keypress_windows(a) == []

    def test_single_press_window(self) -> None:
        a = np.array([NOOP, NOOP, LEFT, NOOP, NOOP], dtype=np.int64)
        out = keypress_windows(a, window_pre=2, window_post=2)
        assert out == [(0, 4)]  # spans the whole episode (0..4)

    def test_window_does_not_extend_beyond_bounds(self) -> None:
        a = np.array([LEFT, NOOP, NOOP], dtype=np.int64)
        out = keypress_windows(a, window_pre=5, window_post=5)
        # lo clipped to 0, hi clipped to 2.
        assert out == [(0, 2)]

    def test_overlapping_windows_merge(self) -> None:
        # Two presses 2 frames apart with window_pre=2, window_post=2
        # produce windows [0,3] and [1,4] -> they merge to [0,4].
        a = np.array([NOOP, LEFT, NOOP, RIGHT, NOOP], dtype=np.int64)
        out = keypress_windows(a, window_pre=2, window_post=2)
        assert out == [(0, 4)]

    def test_far_apart_windows_stay_separate(self) -> None:
        a = np.zeros(20, dtype=np.int64)
        a[2] = LEFT
        a[17] = RIGHT
        out = keypress_windows(a, window_pre=2, window_post=2)
        assert len(out) == 2
        assert out[0][0] == 0
        assert out[1][1] == 19


class TestMirrorActionCounts:
    def test_left_produces_right(self) -> None:
        counts = mirror_action_counts([LEFT, LEFT, LEFT])
        assert counts[RIGHT] == 3
        assert counts[LEFT] == 0

    def test_right_produces_left(self) -> None:
        counts = mirror_action_counts([RIGHT, RIGHT])
        assert counts[LEFT] == 2
        assert counts[RIGHT] == 0

    def test_jump_slide_stay_self(self) -> None:
        counts = mirror_action_counts([JUMP, SLIDE])
        assert counts[JUMP] == 1
        assert counts[SLIDE] == 1


# --------------------------------------------------------------------- #
# High-level rulebook
# --------------------------------------------------------------------- #
def _stack(frames_2d: np.ndarray, k: int = 4) -> np.ndarray:
    """Replicate a single 84x84 frame across the k stack slots.

    Helper for tests that only care about indexing logic, not the
    stacked history.
    """
    n = frames_2d.shape[0]
    out = np.zeros((n, k, frames_2d.shape[-2], frames_2d.shape[-1]),
                   dtype=np.uint8)
    for i in range(n):
        for j in range(k):
            out[i, j] = frames_2d[i]
    return out


class TestDemoAugmentor:
    def test_disable_keypress_window_keeps_everything(self) -> None:
        f = np.arange(10, dtype=np.uint8).reshape(10, 1) * np.ones((1, 1), np.uint8)
        a = np.array([NOOP] * 10, dtype=np.int64)
        aug = DemoAugmentor(keypress_window=False, mirror_horizontal=False)
        idx = aug.select_indices(a)
        assert idx.tolist() == list(range(10))

    def test_disable_mirror_does_not_double(self) -> None:
        f = _stack(np.full((6, 4, 4), 5, dtype=np.uint8), k=2)
        a = np.array([NOOP, LEFT, NOOP, RIGHT, JUMP, NOOP], dtype=np.int64)
        aug = DemoAugmentor(keypress_window=False, mirror_horizontal=False)
        fo, ao = aug.expand(f, a)
        assert fo.shape[0] == 6
        assert (ao == a).all()

    def test_keypress_window_drops_noop_only_stretches(self) -> None:
        # 30 frames: only frames 10..14 contain a press; the rest are
        # NOOP-only.  With window_pre=2 and window_post=2, the windows
        # from frames 10 and 12 are [8,14] — that is what should be kept.
        a = np.zeros(30, dtype=np.int64)
        a[10] = LEFT
        a[12] = RIGHT
        f = _stack(np.arange(30, dtype=np.uint8).reshape(30, 1, 1), k=2)
        aug = DemoAugmentor(keypress_window=True, keypress_pre=2,
                            keypress_post=2, mirror_horizontal=False)
        fo, ao = aug.expand(f, a)
        # frames 10 and 12 with pre=2, post=2 -> windows [8,12] and [10,14],
        # merged to [8,14] -> 7 frames: 8, 9, 10, 11, 12, 13, 14.
        assert fo.shape[0] == 7
        assert ao.tolist() == [NOOP, NOOP, LEFT, NOOP, RIGHT, NOOP, NOOP]

    def test_mirror_doubles_count_and_swaps_lanes(self) -> None:
        a = np.array([NOOP, LEFT, RIGHT, JUMP, SLIDE], dtype=np.int64)
        f = _stack(np.full((5, 4, 4), 7, dtype=np.uint8), k=2)
        aug = DemoAugmentor(keypress_window=False, mirror_horizontal=True,
                            stack_mirror=True)
        fo, ao = aug.expand(f, a)
        assert fo.shape[0] == 10
        # Original first, then mirror: 5 originals + 5 mirrors.
        assert (ao[:5] == a).all()
        # Mirror must swap LEFT<->RIGHT and keep JUMP/SLIDE/NOOP.
        assert (ao[5:10] == np.array([NOOP, RIGHT, LEFT, JUMP, SLIDE],
                                     dtype=np.int64)).all()

    def test_stack_mirror_flips_all_frames(self) -> None:
        a = np.array([LEFT], dtype=np.int64)
        # Newest frame has 1s on the right (LEFT is the answer because
        # an obstacle on the LEFT made the human press LEFT — wait, the
        # action=LEFT here is just a label; we test the PIXELS flip).
        f = np.zeros((1, 1, 4, 4), dtype=np.uint8)
        f[0, 0, :, 3] = 1  # column 3 of the OLDEST frame
        # Tag the entire column 0 of the newest frame with a non-1
        # value so the test can detect both the axis-flip AND the
        # value being preserved (a 4-cell column of 9s stays 9s).
        f[0, -1, :, 0] = 9
        aug = DemoAugmentor(keypress_window=False, mirror_horizontal=True,
                            stack_mirror=True)
        fo, _ = aug.expand(f, a)
        mirror = fo[1]
        # Oldest frame: column 3 -> column 0 in the mirror.
        assert int(mirror[0, :, 0].sum()) == 4
        # Newest frame: column 0 -> column 3 in the mirror.
        assert int(mirror[-1, :, 3].sum()) == 36  # 4 cells * 9 each

    def test_mirror_disabled_keeps_action_layout(self) -> None:
        a = np.array([LEFT, RIGHT, JUMP], dtype=np.int64)
        f = _stack(np.full((3, 4, 4), 3, dtype=np.uint8), k=2)
        aug = DemoAugmentor(keypress_window=False, mirror_horizontal=False,
                            stack_mirror=True)
        fo, ao = aug.expand(f, a)
        assert (ao == a).all()
        assert fo.shape[0] == 3

    def test_empty_input_returns_empty(self) -> None:
        f = np.zeros((0, 2, 4, 4), dtype=np.uint8)
        a = np.zeros((0,), dtype=np.int64)
        aug = DemoAugmentor()
        fo, ao = aug.expand(f, a)
        assert fo.shape[0] == 0
        assert ao.shape[0] == 0

    def test_ragged_input_raises(self) -> None:
        f = np.zeros((3, 2, 4, 4), dtype=np.uint8)
        a = np.zeros((2,), dtype=np.int64)  # 2 != 3
        aug = DemoAugmentor()
        try:
            aug.expand(f, a)
        except ValueError:
            return
        raise AssertionError("expected ValueError on ragged input")
