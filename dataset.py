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
from typing import Optional

import numpy as np


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
    """Frame-stack BC dataset over validated episodes (memory-light)."""

    def __init__(self, episodes: list[Episode], stack: int = 4,
                 deterministic: bool = True) -> None:
        self.stack = stack
        self.deterministic = deterministic
        self._episodes = episodes
        self._index: list[tuple[int, int]] = []  # (episode_idx, step)
        for ei, ep in enumerate(episodes):
            for si in range(len(ep)):
                self._index.append((ei, si))

    def __len__(self) -> int:
        return len(self._index)

    def class_counts(self) -> dict[int, int]:
        counts = {a: 0 for a in range(5)}
        for ei, si in self._index:
            a = int(self._episodes[ei].actions[si])
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

    def get(self, i: int) -> tuple[np.ndarray, int]:
        """Return ([stack,84,84] uint8, action); stack slides within episode."""
        ei, si = self._index[i]
        ep = self._episodes[ei]
        idxs = [max(0, si - k) for k in range(self.stack - 1, -1, -1)]
        stack = np.stack([ep.frames[j] for j in idxs], axis=0)
        return stack, int(ep.actions[si])

    def batch(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = zip(*(self.get(i) for i in indices))
        return np.stack(xs), np.asarray(ys, dtype=np.int64)

    def split_by_episode(self, val_fraction: float, seed: int = 0
                         ) -> tuple[list[int], list[int]]:
        """Episode-level split -> (train_indices, val_indices) into self._index."""
        rng = np.random.default_rng(seed)
        n_eps = len(self._episodes)
        order = rng.permutation(n_eps)
        n_val = max(1, int(round(n_eps * val_fraction))) if n_eps > 1 else 0
        val_eps = set(int(e) for e in order[:n_val])
        train_idx, val_idx = [], []
        for pos, (ei, _si) in enumerate(self._index):
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
