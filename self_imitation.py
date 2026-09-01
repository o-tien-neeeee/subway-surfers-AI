"""Self-imitation storage: keep the agent's GOOD episodes for re-training.

The user asked "tìm cách nào đó để AI nó tự học được đi?" — the answer
this module provides is DQfD-style self-imitation: every time the
agent finishes an episode whose survival is meaningfully better than
its own rolling mean, the action/frame stream of that episode is
serialised as a ``.npz`` demo and added to the self-imitation pool.
The learner periodically re-runs behaviour cloning on the union of
human demos and the self-imitation pool, so good plays are
re-experienced instead of being forgotten after the next bad run.

DESIGN
------
* The on-disk format is **identical** to the human-demo format
  (frames, actions, timestamps, done, meta) so a single
  :class:`dataset.DemonstrationDataset` (with its keypress-window +
  horizontal-mirror augmentations) trains on both pools without any
  special-casing.
* Episodes are gated on ``episode_survival_s >= self_imitation_factor *
  rolling_mean_survival_s`` — the rolling mean is maintained by the
  learner from the same episode results the best-model gate already
  uses, so self-imitation and best-model gating cannot disagree on
  "is this a good run?".
* Old episodes are rotated out when the pool exceeds
  ``self_imitation_max`` so disk growth is bounded.

WHY NOT JUST STORE THE REPLAY-BUFFER SLICES?
--------------------------------------------
A high-TD transition from the replay buffer is exactly the kind of
"lucky break" the network was already biased to repeat; the
self-imitation data set we want here is the agent's whole "good
life", not a handful of high-TD transitions.  Storing full episodes
also gives us the context frames the network needs to actually
recover the mapping at BC time, and lets the same keypress-window +
mirror pipeline train on it.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from logging_utils import get_logger

LOGGER = get_logger("self_imitation")

#: Sub-directory under the demo path that holds agent-originated episodes.
#: Kept separate from human demos so the user can wipe one pool
#: without touching the other.
SELF_DIR_NAME = "self"


@dataclass
class SelfImitationConfig:
    """The knobs of :class:`SelfImitationRecorder`.

    All fields are plain primitives so they can live on
    :class:`config.BotConfig` without dragging numpy into the config
    tree.
    """

    #: When True, the recorder is on; when False, every other knob is
    #: ignored and no self-imitation files are written.
    enabled: bool = True
    #: Episodes with survival >= factor * rolling_mean are saved.
    #: 1.0 means "at or above the average", 1.2 means "20% better",
    #: etc.  0.0 disables the gate (every episode would be saved,
    #: which is too noisy to be useful).
    factor: float = 1.2
    #: Maximum number of self-imitation episodes kept on disk; the
    #: oldest ones are rotated out when this is exceeded.
    max_episodes: int = 50
    #: The minimum number of finished episodes the learner must have
    #: SEEN before any self-imitation gate can open.  Without this
    #: the first episode ever would always be "1.0x the rolling mean"
    #: and end up in the pool, polluting the BC dataset with one
    #: lucky life.
    min_episodes_before_save: int = 3
    #: How often the self-imitation pool is retrained into the model.
    #: ``0`` disables automatic retraining; the GUI surfaces a button
    #: so the user can trigger BC from the pool on demand.
    bc_every_n_episodes: int = 0


class SelfImitationRecorder:
    """Decide which finished episodes are worth saving for re-BC.

    The recorder is the *policy* side of self-imitation: it sees the
    rolling survival mean and the just-finished episode and decides
    whether to keep the latter.  The :class:`Learner
    <learner_worker.Learner>` is the *mechanism* side: it owns the
    episode's frames/actions and, on a positive gate, writes them
    out via :meth:`save_episode`.
    """

    def __init__(self, cfg: SelfImitationConfig, out_dir: str | Path) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir) / SELF_DIR_NAME
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Rolling mean of finished-episode survival.  Updated from the
        # same SharedCounters signal the best-model gate uses, so a
        # "good" episode for self-imitation is the same notion as a
        # "good" episode for the best_model.pth gate.
        self._survival_window: deque[float] = deque(maxlen=20)
        self._episodes_seen: int = 0
        self._saved: int = 0
        # The per-episode callback is registered once and the
        # :class:`Learner` invokes it with (episode_id, survival_s)
        # for every finished episode — that is the only input the
        # recorder needs to update its state.
        self._on_episode_end: Optional[Callable[[int, float], None]] = None

    # ------------------------------------------------------------------ #
    def attach_learner(self, learner: Any) -> None:
        """Register the learner's episode-end callback.

        Wiring is one-way: the learner calls back into
        :meth:`note_episode`; the recorder does not poke the learner
        in return.  Decoupling matters because the recorder has to
        keep working in tests without a real Learner.
        """
        # Lazy callback wrapper so the learner's polling code does not
        # need to know about the recorder's existence.
        def _on_end(stats: dict[str, float]) -> None:
            surv = float(stats.get("survival_s", 0.0))
            eid = int(stats.get("episode_id", 0))
            self.note_episode(eid, surv)
        learner._on_episode_end_cb = _on_end  # type: ignore[attr-defined]

    def note_episode(self, episode_id: int, survival_s: float) -> dict[str, Any]:
        """Receive the just-finished episode and return a decision.

        Returns a small dict the caller can log to the GUI ("saved" /
        "rejected" / "disabled" / "warming-up") — the recorder does
        not own the data, only the policy.
        """
        self._episodes_seen += 1
        if not self.cfg.enabled:
            return {"saved": False, "reason": "disabled"}
        if survival_s <= 0 or not np.isfinite(survival_s):
            return {"saved": False, "reason": "non_positive_survival"}
        self._survival_window.append(survival_s)
        if self._episodes_seen < int(self.cfg.min_episodes_before_save):
            return {"saved": False,
                    "reason": f"warming_up({self._episodes_seen}/"
                              f"{self.cfg.min_episodes_before_save})"}
        if self.cfg.factor <= 0.0:
            return {"saved": False, "reason": "factor_zero"}
        threshold = float(self.cfg.factor) * float(np.mean(self._survival_window))
        if survival_s < threshold:
            return {"saved": False, "reason": "below_threshold",
                    "survival_s": survival_s, "threshold": threshold}
        return {"saved": True,
                "survival_s": survival_s, "threshold": threshold,
                "rolling_mean": float(np.mean(self._survival_window))}

    # ------------------------------------------------------------------ #
    def save_episode(self, episode_id: int, survival_s: float,
                     frames: np.ndarray, actions: np.ndarray,
                     timestamps: np.ndarray, done: np.ndarray,
                     meta: Optional[dict[str, Any]] = None) -> Optional[Path]:
        """Atomically write one self-imitation episode to disk.

        Returns the path on success, ``None`` on failure.  Rotates
        older files out so the pool stays bounded at
        ``cfg.max_episodes``.  The on-disk format is identical to a
        human demo so :class:`dataset.DemonstrationDataset` can load
        both pools interchangeably.
        """
        n = int(frames.shape[0])
        if n == 0:
            return None
        # Build the destination filename with a monotonic suffix so
        # two episodes in the same second do not collide (mirrors the
        # recorder's filename scheme).
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"self_{stamp}_{int(time.time_ns() % 10**9)}.npz"
        meta_out = dict(meta or {})
        meta_out["self_imitation"] = {
            "episode_id": int(episode_id),
            "survival_s": float(survival_s),
            "saved_at": time.time(),
            "rolling_mean": float(np.mean(self._survival_window))
            if self._survival_window else 0.0,
        }
        tmp = path.with_suffix(".npz.tmp")
        try:
            # np.savez_compressed appends ".npz" to a string filename;
            # pass an open file object so the atomic temp is honoured.
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh,
                    frames=np.asarray(frames, dtype=np.uint8),
                    actions=np.asarray(actions, dtype=np.int64),
                    timestamps=np.asarray(timestamps, dtype=np.float64),
                    done=np.asarray(done, dtype=bool),
                    meta=np.str_(json.dumps(meta_out)),
                )
            tmp.replace(path)
        except (OSError, ValueError) as exc:
            LOGGER.error("self-imitation save failed for %s: %s", path, exc)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError as unlink_exc:
                    LOGGER.warning("could not remove %s after failed save: %s",
                                   tmp, unlink_exc)
            return None
        self._saved += 1
        self._rotate()
        return path

    def _rotate(self) -> int:
        """Delete the oldest episodes past the configured cap.

        Returns the number of files removed.  Best-effort: a delete
        failure is logged but does not abort the save.
        """
        try:
            files = sorted(self.out_dir.glob("*.npz"))
        except OSError as exc:
            LOGGER.warning("self-imitation rotate failed to list %s: %s",
                           self.out_dir, exc)
            return 0
        overflow = len(files) - int(self.cfg.max_episodes)
        removed = 0
        for old in files[:max(0, overflow)]:
            try:
                old.unlink()
                removed += 1
            except OSError as exc:
                LOGGER.warning("could not remove %s: %s", old, exc)
                continue
        return removed

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, int]:
        """Snapshot for the GUI / metrics heartbeat."""
        try:
            on_disk = sum(1 for _ in self.out_dir.glob("*.npz"))
        except OSError:
            on_disk = 0
        return {
            "self_episodes_seen": self._episodes_seen,
            "self_episodes_saved": self._saved,
            "self_on_disk": on_disk,
            "self_rolling_window": len(self._survival_window),
        }

    def on_disk_episode_paths(self) -> list[Path]:
        """List every saved self-episode, newest first.

        The dreamer uses this to seed its mental-rehearsal round —
        we keep the method tiny and dependency-free so the dreamer
        can be unit-tested with a mock that just returns paths.
        """
        if not self.out_dir.exists():
            return []
        files = sorted(self.out_dir.glob("*.npz"),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True)
        return files
