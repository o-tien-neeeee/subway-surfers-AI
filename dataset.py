"""Demonstration dataset loading, validation and episode-based splitting.

On-disk format (one ``.npz`` per episode, written by demonstration_recorder):

  frames      [N,84,84] uint8   preprocessed ground-zone grayscale frames
  actions     [N] int64          0..4
  timestamps  [N] float64        monotonic capture timestamps
  done        [N] bool           True on the final step of the episode
  score       [N] float32        optional (0 when unavailable)
  death_state [N] <U16           optional per-frame death state
  confidence  [N] float32        optional detector confidence
  meta        json str           browser geometry, config snapshot, fps

Validation (requirement §10) checks:
* missing frames (timestamp gaps vs the target cadence),
* invalid actions (outside 0..4),
* timestamp ordering (strictly non-decreasing),
* episode boundaries (exactly one done, at the end),
* class imbalance (per-action distribution report).

Splitting is BY EPISODE — never random frames — so near-duplicate stacked
observations cannot leak between train and validation.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import NOOP


@dataclass
class Episode:
    path: str
    frames: np.ndarray      # [N,84,84] uint8
    actions: np.ndarray     # [N]
    timestamps: np.ndarray  # [N]
    done: np.ndarray        # [N]
    score: Optional[np.ndarray] = None
    death_state: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    @property
    def fps(self) -> float:
        if len(self) < 2:
            return 0.0
        span = float(self.timestamps[-1] - self.timestamps[0])
        return (len(self) - 1) / span if span > 0 else 0.0

    def action_counts(self) -> dict[int, int]:
        counts = {a: 0 for a in range(5)}
        for a in self.actions:
            a = int(a)
            if a in counts:
                counts[a] += 1
        return counts


@dataclass
class ValidationReport:
    path: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_steps: int = 0
    n_missing_frames: int = 0
    action_counts: dict[int, int] = field(default_factory=dict)

    def __str__(self) -> str:
        head = f"{self.path}: {'OK' if self.ok else 'INVALID'} ({self.n_steps} steps)"
        lines = [head]
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  warning {w}")
        counts = ", ".join(f"{k}:{v}" for k, v in sorted(self.action_counts.items()))
        lines.append(f"  actions {counts}")
        return "\n".join(lines)


class EpisodeLoadError(RuntimeError):
    """A demo file exists but cannot be turned into an :class:`Episode`."""


def load_episode(path: str | Path) -> Episode:
    """Load one ``.npz`` episode, raising :class:`EpisodeLoadError` if unusable.

    # DEEP-FIX: a missing key raised a bare KeyError straight out of
    # ``validate_directory``, so ONE malformed file in ``demos/`` aborted the
    # entire behaviour-cloning run (``Learner.pretrain`` has no per-file
    # isolation).  Every failure is now a typed error the caller can turn
    # into an "invalid episode" report row.
    """
    p = Path(path)
    try:
        data = np.load(str(p), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise EpisodeLoadError(f"{p}: not a readable .npz ({exc})") from exc
    missing = [k for k in ("frames", "actions", "timestamps", "done")
               if k not in data]
    if missing:
        raise EpisodeLoadError(f"{p}: missing required arrays {missing}")
    try:
        frames = np.asarray(data["frames"], dtype=np.uint8)
        actions = np.asarray(data["actions"], dtype=np.int64)
        ts = np.asarray(data["timestamps"], dtype=np.float64)
        done = np.asarray(data["done"], dtype=bool)
    except (ValueError, TypeError) as exc:
        raise EpisodeLoadError(f"{p}: unreadable array dtype ({exc})") from exc
    n = int(frames.shape[0]) if frames.ndim >= 1 else 0
    # DEEP-FIX: ragged columns used to survive loading and were only noticed
    # (if at all) by the validator comparing against frames' length.
    for name, arr in (("actions", actions), ("timestamps", ts), ("done", done)):
        if arr.shape[0] != n:
            raise EpisodeLoadError(
                f"{p}: {name} has {arr.shape[0]} rows but frames has {n}")
    meta = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"]))
        except (ValueError, TypeError):
            meta = {}

    def optional(key: str, dtype):
        if key not in data:
            return None
        arr = np.asarray(data[key], dtype=dtype)
        return arr if arr.shape[0] == n else None

    return Episode(
        path=str(p),
        frames=frames,
        actions=actions,
        timestamps=ts,
        done=done,
        score=optional("score", np.float32),
        death_state=(np.asarray(data["death_state"]).astype(str)
                     if "death_state" in data
                     and np.asarray(data["death_state"]).shape[0] == n else None),
        confidence=optional("confidence", np.float32),
        meta=meta,
    )


def load_episodes(directory: str | Path) -> list[Episode]:
    """Load every ``*.npz`` in a directory.

    # DEEP-FIX: an unreadable file no longer aborts the whole load; it is
    # reported through :func:`validate_directory` as an invalid episode so
    # the operator sees WHICH file is broken and BC continues with the rest.
    """
    root = Path(directory)
    out: list[Episode] = []
    for path in sorted(root.glob("*.npz")):
        try:
            out.append(load_episode(path))
        except EpisodeLoadError as exc:
            _LOAD_ERRORS[str(path)] = str(exc)
    return out


#: path -> reason, for files that could not be parsed at all.
_LOAD_ERRORS: dict[str, str] = {}


def take_load_errors() -> dict[str, str]:
    """Return and clear the accumulated per-file load failures."""
    errs = dict(_LOAD_ERRORS)
    _LOAD_ERRORS.clear()
    return errs


def validate_episode(ep: Episode, target_fps: float = 30.0,
                     gap_factor: float = 3.0) -> ValidationReport:
    rep = ValidationReport(path=ep.path, ok=True)
    rep.n_steps = len(ep)
    n = len(ep)
    if n == 0:
        rep.ok = False
        rep.errors.append("empty episode")
        return rep
    if ep.frames.ndim != 3 or ep.frames.shape[1:] != (84, 84):
        rep.ok = False
        rep.errors.append(f"frames shape {ep.frames.shape} != [N,84,84]")
    if not np.all((ep.actions >= 0) & (ep.actions <= 4)):
        bad = int(np.sum((ep.actions < 0) | (ep.actions > 4)))
        rep.ok = False
        rep.errors.append(f"{bad} invalid action ids")
    if np.any(np.diff(ep.timestamps) < 0):
        rep.ok = False
        rep.errors.append("timestamps not monotonic")
    done_idx = np.where(ep.done)[0]
    if len(done_idx) != 1 or int(done_idx[0]) != n - 1:
        rep.ok = False
        rep.errors.append(
            f"episode boundary invalid: done at {done_idx.tolist()} of {n - 1}"
        )
    if n >= 2:
        dts = np.diff(ep.timestamps)
        expected = 1.0 / max(1.0, target_fps)
        missing = int(np.sum(dts > gap_factor * expected))
        rep.n_missing_frames = missing
        if missing > max(2, int(0.05 * n)):
            rep.warnings.append(
                f"{missing} timestamp gaps > {gap_factor}x target cadence"
            )
        fps = ep.fps
        if fps < target_fps * 0.5:
            rep.warnings.append(f"low effective capture fps {fps:.1f}")
    rep.action_counts = ep.action_counts()
    nonzero = {a: c for a, c in rep.action_counts.items() if a != 0}
    if nonzero:
        max_c = max(nonzero.values())
        min_c = min(nonzero.values())
        if min_c == 0:
            rep.warnings.append("an action class never appears")
        elif max_c / max(1, min_c) > 20:
            rep.warnings.append(f"action imbalance ratio {max_c / max(1, min_c):.0f}x")
    return rep


def validate_directory(directory: str | Path, target_fps: float = 30.0
                       ) -> tuple[list[Episode], list[ValidationReport]]:
    _LOAD_ERRORS.clear()
    eps = load_episodes(directory)
    reps = [validate_episode(e, target_fps) for e in eps]
    # DEEP-FIX: files that could not even be parsed are surfaced as invalid
    # reports instead of vanishing (or crashing the caller).
    for path, reason in take_load_errors().items():
        reps.append(ValidationReport(path=path, ok=False,
                                     errors=[reason], n_steps=0))
    return eps, reps


# --------------------------------------------------------------------- #
# Torch dataset for behaviour cloning (built lazily, uint8 on disk)
# --------------------------------------------------------------------- #
class DemonstrationDataset:
    """Frame-stack BC dataset over validated episodes (memory-light).

    Augmentation hook
    -----------------
    Pass an :class:`demo_augment.DemoAugmentor` to ``augment=`` to enable
    the keypress-window filter and the horizontal mirror (left↔right
    symmetric playfield).  Both behaviours are on by default in the
    Augmentor and can be turned off there independently of this class.

    The augmentation is applied at *index build time* (per-episode, once
    per process) and is folded into ``self._index`` with an extra
    "mirror?" flag, so :meth:`get` and :meth:`batch` stay simple and the
    BC loop pays no per-batch cost for the rulebook.

    The mirror flag is important: without it, two passes over the SAME
    episode would be mirror of each other, but the per-action press
    counts would double-count in unpredictable ways.  Tracking the flag
    keeps :meth:`dodge_press_counts` honest — every press the human made
    is counted exactly once, and the mirror copies are reported
    separately in :meth:`augmentation_summary`.
    """

    def __init__(self, episodes: list[Episode], stack: int = 4,
                 deterministic: bool = True, dodge_oversample: int = 1,
                 augment: Any = None) -> None:
        # Lazy import so importing this module does not pull in numpy /
        # the augment rulebook when only the validator is wanted (e.g.
        # for the headless validator script).
        if augment is None:
            from demo_augment import DemoAugmentor
            augment = DemoAugmentor()
        self.stack = stack
        self.deterministic = deterministic
        self.dodge_oversample = max(1, int(dodge_oversample))
        self.augment = augment
        self._episodes = episodes
        # The index carries (episode_idx, step, mirror_flag) — mirror_flag
        # is 0 for an original frame and 1 for a flipped copy.  The frame
        # is produced on demand in :meth:`get` from the original episode
        # so the dataset does not allocate a 2x copy of every frame.
        self._index: list[tuple[int, int, int]] = []
        # DEEP-FIX ("why did the human press?"): dodge actions are rare but
        # decide life-or-death, so plain BC drowns them in NOOP frames and never
        # learns to dodge — val_acc looks high only because NOOP dominates.  We
        # count every key press and oversample the dodge frames so BC actually
        # learns the trigger->dodge mapping.  The frame-stack at a dodge frame
        # already contains the approaching obstacle, i.e. the "why".
        self._dodge_presses: dict[int, int] = {a: 0 for a in range(5)}
        # Mirror copy counts — surfaced for the GUI summary so an
        # all-LEFT demo that has no RIGHT frames is obvious.
        self._mirror_added: dict[int, int] = {a: 0 for a in range(5)}
        # ---- build the index once, episode by episode ------------------ #
        for ei, ep in enumerate(episodes):
            self._add_episode(ei, ep)

    # ------------------------------------------------------------------ #
    def _add_episode(self, ei: int, ep: Episode) -> None:
        """Index one episode under the augmentor's rules.

        The function does three things in order:

        1. Ask the augmentor which frame indices to KEEP (the
           keypress-window filter).
        2. For every kept index, append the (original, mirror=False)
           entry.  If the human press-counting is enabled the same press
           is recorded exactly once on the original side.  Dodge frames
           (action != NOOP) are repeated ``dodge_oversample`` times so
           the rare, life-critical presses actually drive the BC
           gradient (the original "why did the human press?" fix).
        3. If mirror is enabled, append the (original_index, mirror=True)
           entry for every kept frame.  The mirrored action is what the
           dataset hands to BC for those indices, so LEFT becomes RIGHT
           in the training target — that is the whole point of the
           augmentation.  Mirror copies of dodge frames are also
           oversampled, so the augmented dataset's per-action counts
           stay balanced.

        The frame-stacks are produced on demand in :meth:`_get_stack` so
        the memory cost is the same as before, regardless of whether
        mirror doubles the index length.
        """
        keep = self.augment.select_indices(ep.actions)
        if keep.size == 0:
            return
        # --- (1) original frames + dodge press oversample -------------- #
        actions = ep.actions
        for si in keep.tolist():
            a = int(actions[si])
            self._index.append((ei, si, 0))
            if a != NOOP and (si == 0 or int(actions[si - 1]) != a):
                # Count every distinct press, not every kept frame of it
                # (the keypress-window keeps ~pre+post+1 frames around
                # each press; we want one count, not eleven).
                self._dodge_presses[a] = self._dodge_presses.get(a, 0) + 1
                # Oversample only the actual press frames so the rare,
                # life-critical dodges aren't drowned in NOOP samples.
                # (Legacy behaviour from round 19; preserved here so the
                # old "TestWhyPressOversampling" tests still pass.)
                for _ in range(self.dodge_oversample - 1):
                    self._index.append((ei, si, 0))
        # --- (2) horizontal mirror ------------------------------------ #
        if not getattr(self.augment, "mirror_horizontal", True):
            return
        from demo_augment import mirror_action
        for si in keep.tolist():
            a = int(actions[si])
            mirror_a = mirror_action(a)
            self._index.append((ei, si, 1))
            if mirror_a != NOOP:
                # Mirror copy of a dodge frame is a different action
                # (LEFT <-> RIGHT, JUMP/SLIDE stay self) — count it once
                # in the per-action mirror stats AND oversample it so
                # the symmetry is balanced on the mirror side too.
                self._mirror_added[mirror_a] += 1
                for _ in range(self.dodge_oversample - 1):
                    self._index.append((ei, si, 1))

    def __len__(self) -> int:
        return len(self._index)

    def dodge_press_counts(self) -> dict[int, int]:
        """Per-action count of dodge initiations (key presses) in the demos.

        These are the HUMAN presses only — mirror copies are reported
        separately in :meth:`augmentation_summary` so a one-sided demo
        is diagnosed honestly.
        """
        return dict(self._dodge_presses)

    def mirror_added_counts(self) -> dict[int, int]:
        """How many mirror copies each action gained during augmentation."""
        return dict(self._mirror_added)

    def augmentation_summary(self) -> dict[str, int]:
        """Compact stats the learner / GUI log to show what the dataset became."""
        n_orig = sum(1 for _ei, _si, m in self._index if m == 0)
        n_mirror = sum(1 for _ei, _si, m in self._index if m == 1)
        return {
            "kept_original": n_orig,
            "kept_mirror": n_mirror,
            "n_episodes": len(self._episodes),
            "mirror_added_by_action": dict(self._mirror_added),
        }

    def class_counts(self) -> dict[int, int]:
        """Per-action counts on the AUGMENTED index (what BC actually sees)."""
        counts = {a: 0 for a in range(5)}
        for ei, si, m in self._index:
            a = int(self._episodes[ei].actions[si])
            if m == 1:
                from demo_augment import mirror_action
                a = mirror_action(a)
            if a in counts:
                counts[a] += 1
        return counts

    def class_weights(self, mode: str = "inverse_sqrt") -> np.ndarray:
        counts = np.array([self.class_counts()[a] for a in range(5)], dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        if mode == "inverse":
            w = 1.0 / counts
        else:
            w = 1.0 / np.sqrt(counts)
        return (w / w.max()).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _get_stack(self, ei: int, si: int, mirror: int) -> np.ndarray:
        """Produce the [k, H, W] uint8 stack, with mirror applied if asked.

        The augmentation is applied to the WHOLE stack (newest frame +
        history) when ``augment.stack_mirror`` is true (the default),
        because flipping only the newest frame would mix the obstacle
        position with stale pixels and the policy would learn the
        wrong mapping.  The flag is read on the instance so the user
        can change it from a single place.
        """
        ep = self._episodes[ei]
        idxs = [max(0, si - k) for k in range(self.stack - 1, -1, -1)]
        stack = np.stack([ep.frames[j] for j in idxs], axis=0)
        if mirror and getattr(self.augment, "stack_mirror", True):
            from demo_augment import mirror_frame
            return mirror_frame(stack)
        if mirror:
            from demo_augment import mirror_frame
            out = stack.copy()
            out[-1] = mirror_frame(stack[-1])
            return out
        return stack

    def get(self, i: int) -> tuple[np.ndarray, int]:
        """Return ([stack,84,84] uint8, action); stack slides within episode."""
        ei, si, mirror = self._index[i]
        ep = self._episodes[ei]
        stack = self._get_stack(ei, si, mirror)
        a = int(ep.actions[si])
        if mirror:
            from demo_augment import mirror_action
            a = mirror_action(a)
        return stack, a

    def batch(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = zip(*(self.get(i) for i in indices))
        return np.stack(xs), np.asarray(ys, dtype=np.int64)

    def episode_done_mask(self, indices: list[int]) -> np.ndarray:
        """Return a boolean mask of length ``len(indices)``
        marking the LAST frame of every episode in the
        index list.  Used by the DQfD demo pre-fill to
        add a +1 terminal reward to those frames so
        the n-step builder knows where to stop.
        """
        if not indices:
            return np.zeros(0, dtype=bool)
        out = np.zeros(len(indices), dtype=bool)
        for pos, idx in enumerate(indices):
            # _index entries are (ei, si, mirror_flag).
            ei = self._index[idx][0]
            # The "next" entry is a different episode
            # (or we are at the end of the list).
            if pos == len(indices) - 1:
                out[pos] = True
            else:
                next_ei = self._index[indices[pos + 1]][0]
                if next_ei != ei:
                    out[pos] = True
        return out

    def split_by_episode(self, val_fraction: float, seed: int = 0
                         ) -> tuple[list[int], list[int]]:
        """Episode-level split -> (train_indices, val_indices) into self._index.

        Mirror copies stay together with their original index in the
        train/val split — a single ``(ei, si, 1)`` entry is treated the
        same as its sibling ``(ei, si, 0)``.  The split is therefore
        still by EPISODE (the original and its mirror can never end up
        in different folds), so val accuracy still measures
        generalisation, not memorisation of a flipped lookup table.
        """
        rng = np.random.default_rng(seed)
        n_eps = len(self._episodes)
        order = rng.permutation(n_eps)
        n_val = max(1, int(round(n_eps * val_fraction))) if n_eps > 1 else 0
        val_eps = set(int(e) for e in order[:n_val])
        train_idx, val_idx = [], []
        for pos, (ei, _si, _m) in enumerate(self._index):
            (val_idx if int(ei) in val_eps else train_idx).append(pos)
        return train_idx, val_idx


def summarize_reports(reports: list[ValidationReport]) -> str:
    if not reports:
        return "No demonstration episodes found."
    lines = []
    total_missing = 0
    for r in reports:
        lines.append(str(r))
        total_missing += r.n_missing_frames
    ok = sum(1 for r in reports if r.ok)
    lines.append(f"--- {ok}/{len(reports)} episodes valid; "
                 f"{total_missing} missing-frame gaps total")
    return "\n".join(lines)
