"""Demo augmentation: keypress-driven windows + horizontal mirror.

DESIGN
------
The Subway Surfers playfield is left↔right symmetric: a LEFT dodge and a
RIGHT dodge of an identical obstacle look like horizontal mirrors of each
other, and the obstacle sits on the OPPOSITE side of the screen.  This
module implements the augmentation rules the user asked for:

  1. *Keypress-window recording*: when the human presses a key, the
     recorder no longer dumps every 30-fps frame (overwhelmingly NOOP).
     It keeps a ±N frame window around each key press — that is the moment
     the human SAW an obstacle and decided what to do, so the network
     learns the "see → press" mapping instead of "see → do nothing".
     The BC dataset still accepts the legacy "every frame" episodes so old
     recordings keep working.

  2. *Horizontal mirror*: for every kept frame, a horizontally flipped
     copy is added with the mirrored action:
        LEFT  (1)  →  mirror frame + RIGHT  (2)
        RIGHT (2)  →  mirror frame + LEFT   (1)
        JUMP  (3)  →  mirror frame + JUMP   (3)   (symmetric)
        SLIDE (4)  →  mirror frame + SLIDE  (4)   (symmetric)
        NOOP  (0)  →  mirror frame + NOOP   (0)   (symmetric)
     Subway Surfers is left/right symmetric; the mirror is a free x2
     data augmentation that teaches the policy "an obstacle on the
     left is the same problem as one on the right".

  3. *Configurable*: every knob is a field on ``BotConfig`` so the user
     can disable either augmentation (e.g. for a non-symmetric game).

The augmentation lives HERE (not in the recorder) so:

  * The on-disk format stays compact and reproducible (one frame = one
    action; the augmented copy never hits disk).
  * Old recordings keep working unchanged.
  * It is unit-testable without the heavy pynput / mss stack.
  * The same module is used by both the recorder (for live preview
    counts) and the dataset (for the actual training data).

ONLINE VS OFFLINE
-----------------
Two execution paths exist on purpose:

  * :func:`mirror_frame` is a pure numpy utility used both at training
    time (online) and to make preview thumbnails for the GUI.
  * :class:`DemoAugmentor` is the high-level rulebook: it takes a
    stream of (frame, action) pairs and decides what the training set
    should look like, including the keypress-window extraction when
    the recorder saved every frame.

Horizon- and death-aware frames: the recorder's anchor already
trims the death-stumble tail, so we do not re-filter for liveness
here — every frame the recorder accepted is treated as a live frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import NOOP, LEFT, RIGHT, JUMP, SLIDE, N_ACTIONS

#: canonical action id for each keep-side (LEFT=1, RIGHT=2) so other
#: modules can iterate the symmetric pair without hard-coding numbers.
LANE_ACTIONS: Tuple[int, int] = (LEFT, RIGHT)
DODGE_ACTIONS: Tuple[int, ...] = (LEFT, RIGHT, JUMP, SLIDE)
ALL_ACTIONS: Tuple[int, ...] = (NOOP, LEFT, RIGHT, JUMP, SLIDE)


def mirror_frame(frame_u8: np.ndarray) -> np.ndarray:
    """Horizontally flip a [H, W] or [k, H, W] uint8 frame.

    The implementation is a view + copy (np.fliplr already returns a new
    array), so callers can hand in a shared buffer without aliasing it.
    The action mapping is applied by :func:`mirror_action`; this
    function only mirrors pixels.
    """
    arr = np.asarray(frame_u8)
    if arr.ndim == 2:
        return np.ascontiguousarray(np.fliplr(arr))
    if arr.ndim == 3:
        # Could be [k, H, W] (a frame stack) — flip the W axis (axis 2).
        return np.ascontiguousarray(arr[:, :, ::-1])
    raise ValueError(
        f"mirror_frame: expected 2-D or 3-D array, got shape {arr.shape}"
    )


def mirror_action(action: int) -> int:
    """Map an action under horizontal reflection.

    Subway Surfers' lanes are symmetric: a LEFT from the camera's POV
    after a horizontal flip is a RIGHT, and vice-versa.  JUMP/SLIDE/NOOP
    are the same action in both halves because they are not
    left/right-specific (you jump over or slide under the same kind of
    obstacle regardless of which lane it spawns in).
    """
    if action == LEFT:
        return RIGHT
    if action == RIGHT:
        return LEFT
    if action in (NOOP, JUMP, SLIDE):
        return action
    raise ValueError(f"mirror_action: unknown action id {action}")


def keypress_windows(actions: np.ndarray,
                     window_pre: int = 5,
                     window_post: int = 5,
                     max_span_s: float = 0.0) -> List[Tuple[int, int]]:
    """Indices of the frame ranges to KEEP for BC under keypress-window mode.

    ``actions`` is a 1-D array of action ids recorded by the demo
    recorder.  The function returns a list of ``(lo, hi)`` index pairs
    (both inclusive) such that every key press ``i`` with
    ``actions[i] != NOOP`` is covered by at least one window, and
    consecutive presses overlap.  Frames that no key press "explains"
    (NOOP-only stretches far from any press) are dropped — that is
    the whole point of this mode.

    The window grows to the left by ``window_pre`` and to the right by
    ``window_post`` from the press, then clipped to the array bounds
    and merged with neighbours that overlap.

    When ``max_span_s > 0`` the window is also clipped in time by the
    neighbouring press (so two presses 2s apart do not stretch a
    1.5s+1.5s=3s window) — pass 0 to disable (default).  Used by
    ``DemonstrationDataset`` when the recorder saved timestamps.
    """
    n = len(actions)
    if n == 0:
        return []
    presses = np.where(actions != NOOP)[0]
    if presses.size == 0:
        return []  # all-NOOP demo — nothing to learn from
    spans: List[Tuple[int, int]] = []
    pre, post = int(window_pre), int(window_post)
    for p in presses.tolist():
        lo = max(0, p - pre)
        hi = min(n - 1, p + post)
        spans.append((lo, hi))
    # Merge overlapping / adjacent windows so a chain of presses does
    # not produce fragmented ranges that BC would later pad with
    # duplicate copies of the edge frame.
    spans.sort()
    merged: List[List[int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


@dataclass
class DemoAugmentor:
    """Rulebook for turning a recorded (frame, action) episode into BC data.

    Three orthogonal switches (all on by default; disable by setting the
    field to ``False`` or 0):

    * ``keypress_window``: crop the episode to a frame window around
      every key press (the recorder saves a contiguous array; this
      filter discards the stretches the human spent doing nothing).
    * ``mirror_horizontal``: append a flipped copy of every kept frame
      with the mirrored action.
    * ``stack_mirror`` (only meaningful when the frame stack contains
      a horizontal dimension): mirror the WHOLE stack (the 4 most
      recent frames become the 4 flipped frames in the same order).
    """

    #: Keep ±5 frames around each key press (≈ 167 ms before + after
    #: at 30 FPS — the obstacle that triggered the press is usually
    #: 2-4 frames away and the recovery / next press is 3-5 frames in).
    keypress_window: bool = True
    keypress_pre: int = 5
    keypress_post: int = 5

    #: Append a horizontally flipped copy of every kept frame.
    mirror_horizontal: bool = True

    #: When True, the flipped copy's *frame stack* is also flipped as a
    #: whole (the k newest frames in the original become the k newest
    #: flipped frames in the same order).  Set False to flip only the
    #: newest frame and leave history unchanged — the BC network can
    #: confuse the two if the stack is asymmetric, so whole-stack
    #: flipping is the default.
    stack_mirror: bool = True

    # ------------------------------------------------------------------ #
    def select_indices(self, actions: np.ndarray) -> np.ndarray:
        """Return the indices to KEEP from a full episode.

        With keypress_window=False the answer is ``arange(N)``; with it
        on, the answer is the union of the per-press windows, sorted
        and deduplicated.  Empty selection is reported as an empty
        array (the caller decides what that means — usually "skip the
        episode").
        """
        n = len(actions)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if not self.keypress_window:
            return np.arange(n, dtype=np.int64)
        windows = keypress_windows(actions,
                                   window_pre=self.keypress_pre,
                                   window_post=self.keypress_post)
        if not windows:
            return np.zeros(0, dtype=np.int64)
        # Concatenate the windows; np.unique keeps them in order.
        idx = np.concatenate([np.arange(lo, hi + 1) for lo, hi in windows])
        return np.unique(idx).astype(np.int64)

    # ------------------------------------------------------------------ #
    def expand(self, frames: np.ndarray, actions: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply keypress-window selection + horizontal mirror.

        ``frames`` has shape ``[N, k, H, W]`` or ``[N, H, W]`` uint8.
        ``actions`` is ``[N]`` int.  Returns ``(frames_out, actions_out)``
        with the same dtype.  Empty selection -> empty arrays.
        """
        if frames.shape[0] != actions.shape[0]:
            raise ValueError(
                f"expand: frames/rows mismatch ({frames.shape[0]} vs "
                f"{actions.shape[0]}) — corrupt demo file"
            )
        idx = self.select_indices(actions)
        if idx.size == 0:
            return frames[:0], actions[:0]
        sel_frames = frames[idx]
        sel_actions = actions[idx]
        if not self.mirror_horizontal:
            return sel_frames, sel_actions
        # Build the mirror copy in the same shape.
        mirror = self._mirror_stack(sel_frames) if self.stack_mirror \
            else self._mirror_newest_only(sel_frames)
        out_frames = np.concatenate([sel_frames, mirror], axis=0)
        out_actions = np.concatenate(
            [sel_actions, np.array([mirror_action(int(a))
                                    for a in sel_actions], dtype=actions.dtype)],
            axis=0,
        )
        return out_frames, out_actions

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mirror_stack(frames: np.ndarray) -> np.ndarray:
        """Flip the whole stack: shape [N,k,H,W] -> [N,k,H,W]."""
        if frames.ndim == 3:
            return mirror_frame(frames)
        if frames.ndim == 4:
            return np.ascontiguousarray(frames[:, :, :, ::-1])
        raise ValueError(
            f"_mirror_stack: unsupported frame shape {frames.shape}"
        )

    @staticmethod
    def _mirror_newest_only(frames: np.ndarray) -> np.ndarray:
        """Flip only the newest frame in each stack; copy the rest."""
        if frames.ndim == 3:
            return mirror_frame(frames)
        if frames.ndim == 4:
            out = frames.copy()
            out[:, -1] = mirror_frame(frames[:, -1])
            return out
        raise ValueError(
            f"_mirror_newest_only: unsupported frame shape {frames.shape}"
        )


# --------------------------------------------------------------------- #
# Convenience helpers used by both recorder (live preview counts) and
# dataset (training-time statistics)
# --------------------------------------------------------------------- #
def mirror_action_counts(actions: Iterable[int]) -> dict[int, int]:
    """Return how many mirror copies each action would produce.

    Used to show the operator "left=12 -> mirror adds 12 right" in the
    demo / BC summary, so an obviously one-sided demo is diagnosed
    without running the full pipeline.
    """
    counts: dict[int, int] = {a: 0 for a in ALL_ACTIONS}
    for a in actions:
        ma = mirror_action(int(a))
        counts[ma] += 1
    return counts


def action_name(a: int) -> str:
    """Human-readable action name (Vietnamese labels match the GUI)."""
    return {
        NOOP: "NOOP",
        LEFT: "trái",
        RIGHT: "phải",
        JUMP: "nhảy",
        SLIDE: "trượt",
    }.get(int(a), "?")
