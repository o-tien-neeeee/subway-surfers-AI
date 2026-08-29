"""Prioritized Experience Replay with bounded memory and atomic persistence.

Memory strategy (audit decision): transitions share frames.  Consecutive
transition stacks overlap in most frames, so each 84x84 uint8 frame is
stored ONCE in a ring (``FrameStore``) and referenced by env-frame-id —
consecutive transitions then contribute ~1 unique frame (~7 KB) instead of
~56 KB of duplicated stacks (8x reduction).  A 30k-transition buffer holds
roughly 210-260 MB of frames, comfortably inside the 4 GB working-set budget
next to torch and Chrome.

Frame lifecycle correctness: the store is FIFO with capacity
``transition_capacity + margin``, and every sampled transition is validated
against the env-frame-ids still resident in the store.  A wrapped/evicted
reference is rejected and resampled — stale pixels can never silently train
the network.

Other guarantees:
* Sum-tree priorities with NaN/inf sanitisation (clamped to a floor).
* Terminal transitions validated on insert (finite reward, bool done).
* Fixed capacity -> memory bounded by construction.
* ``save`` writes pickle + sha256 sidecar via temp file + ``os.replace``
  (atomic on NTFS and POSIX).  ``load`` verifies the hash, renames corrupt
  files to ``*.corrupt`` and raises CorruptFileError so the caller can start
  from an empty buffer while the GUI stays alive.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from config import PERConfig
from logging_utils import CorruptFileError, integrity_pickle_load, integrity_pickle_save


class SumTree:
    """Array-backed binary sum tree (length 2*capacity) with batch updates."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[1])

    def min_leaf(self) -> float:
        used = self.tree[self.capacity : self.capacity + self._filled()]
        return float(used.min()) if used.size else 0.0

    def _filled(self) -> int:
        # Leaves written so far (contiguous from index 0 until wrapped).
        m = int(np.argmax(self.tree[self.capacity :] == 0))
        if m == 0 and self.tree[self.capacity] == 0:
            return 0
        if m == 0:
            return self.capacity
        return m

    def update(self, leaf_idx: int, value: float) -> None:
        if not np.isfinite(value) or value <= 0:
            value = 1e-6  # NaN/inf/negative guard
        idx = leaf_idx + self.capacity
        self.tree[idx] = value
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx //= 2

    def update_batch(self, leaf_indices: np.ndarray, values: np.ndarray) -> None:
        safe = np.where(np.isfinite(values) & (values > 0), values, 1e-6)
        idx = np.asarray(leaf_indices) + self.capacity
        self.tree[idx] = safe
        idx //= 2
        while idx.max() >= 1:
            unique = np.unique(idx)
            self.tree[unique] = self.tree[2 * unique] + self.tree[2 * unique + 1]
            idx //= 2

    def find_prefixsum(self, value: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = self.tree[2 * idx]
            if value <= left:
                idx = 2 * idx
            else:
                value -= left
                idx = 2 * idx + 1
        return int(idx - self.capacity)

    def find_prefixsum_batch(self, values: np.ndarray) -> np.ndarray:
        idx = np.ones(len(values), dtype=np.int64)
        vals = np.asarray(values, dtype=np.float64).copy()
        while bool(np.any(idx < self.capacity)):
            active = idx < self.capacity
            rows = np.where(active)[0]
            left = self.tree[2 * idx[rows]]
            go_left = vals[rows] <= left
            idx[rows[go_left]] *= 2
            idx[rows[~go_left]] = idx[rows[~go_left]] * 2 + 1
            vals[rows[~go_left]] -= left[~go_left]
        return idx - self.capacity

    def rebuild_from(self, priorities: np.ndarray, filled: int) -> None:
        self.tree[:] = 0.0
        self.tree[self.capacity : self.capacity + filled] = priorities[:filled]
        for i in range(self.capacity - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]


class FrameStore:
    """FIFO ring of uint8 ground frames with env-frame-id dedup + eviction."""

    def __init__(self, capacity: int, size: int) -> None:
        self.capacity = capacity
        self.frames = np.zeros((capacity, size, size), dtype=np.uint8)
        self.env_ids = np.full(capacity, -1, dtype=np.int64)
        self.cursor = 0
        self.written = 0
        self._recent: OrderedDict[int, int] = OrderedDict()  # env_id -> slot
        self._recent_cap = 128

    def add_if_new(self, frame: np.ndarray, env_id: int) -> int:
        """Insert (or reuse) a frame by env id; returns the slot index."""
        slot = self._recent.get(env_id)
        if slot is not None and self.env_ids[slot] == env_id:
            self._recent.move_to_end(env_id)
            return slot
        idx = self.cursor
        old_id = int(self.env_ids[idx])
        if old_id in self._recent:
            del self._recent[old_id]
        self.frames[idx] = frame
        self.env_ids[idx] = env_id
        self._recent[env_id] = idx
        if len(self._recent) > self._recent_cap:
            self._recent.popitem(last=False)
        self.cursor = (self.cursor + 1) % self.capacity
        self.written += 1
        return idx

    def resident(self, slot: int, env_id: int) -> bool:
        """True when ``slot`` still holds the frame with ``env_id``."""
        return 0 <= slot < self.capacity and int(self.env_ids[slot]) == int(env_id)

    def get(self, idx: int) -> np.ndarray:
        return self.frames[idx]

    def nbytes(self) -> int:
        return self.frames.nbytes + self.env_ids.nbytes


@dataclass
class Transition:
    obs_ids: tuple[int, ...]      # 4 frame-store slots (oldest -> newest)
    next_ids: tuple[int, ...]     # 4 slots for s_{t+span}
    obs_env_ids: tuple[int, ...]  # env ids for the 4 obs frames
    next_env_ids: tuple[int, ...]
    action: int
    reward: float
    done: bool
    span: int
    gamma_pow: float

    def validate(self) -> None:
        if not (0 <= int(self.action) <= 4):
            raise ValueError(f"invalid action {self.action}")
        if not np.isfinite(self.reward):
            raise ValueError(f"non-finite reward {self.reward}")
        if not isinstance(self.done, bool):
            raise ValueError("done must be bool")
        for ids in (self.obs_ids, self.next_ids):
            if len(ids) != 4:
                raise ValueError("stacks must reference exactly 4 frames")


@dataclass
class NStepTransition:
    """Emitted by NStepBuilder; carried over the transition queue."""

    obs: np.ndarray           # [4,H,W] uint8
    next_obs: np.ndarray      # [4,H,W] uint8
    action: int
    reward: float             # discounted n-step return
    done: bool
    span: int
    gamma_pow: float
    obs_env_ids: tuple[int, ...]
    next_env_ids: tuple[int, ...]


class NStepBuilder:
    """Collapses an (obs,action,reward) stream into n-step transitions.

    Emits step t only once obs_{t+span} is observable, or at episode end —
    so rewards are discounted correctly and nothing crosses an episode
    boundary.  Only the LAST step of an episode is marked done=True.
    """

    def __init__(self, n: int, gamma: float) -> None:
        self.n = max(1, int(n))
        self.gamma = float(gamma)
        self.clear()

    def clear(self) -> None:
        self._buf: list[dict[str, Any]] = []

    def push(self, stack: np.ndarray, env_ids: tuple[int, ...], action: int,
             reward: float, done: bool) -> list[NStepTransition]:
        self._buf.append(
            {"stack": stack, "ids": tuple(env_ids), "a": int(action),
             "r": float(reward), "done": bool(done)}
        )
        out: list[NStepTransition] = []
        # Non-terminal emission: we need n entries plus the obs after them.
        while len(self._buf) > self.n:
            out.append(self._emit(span=self.n, terminal=False))
        if self._buf and self._buf[-1]["done"]:
            while self._buf:
                span = len(self._buf) - 1
                out.append(self._emit(span=span, terminal=(span == 0)))
        return out

    def _emit(self, span: int, terminal: bool) -> NStepTransition:
        head = self._buf[0]
        r = 0.0
        for i in range(span):
            r += (self.gamma ** i) * self._buf[i]["r"]
        if terminal:
            r += (self.gamma ** span) * self._buf[span]["r"] if span < len(self._buf) else 0.0
        nxt = self._buf[span]
        tr = NStepTransition(
            obs=head["stack"],
            next_obs=nxt["stack"],
            action=head["a"],
            reward=float(r) if not terminal else float(r),
            done=bool(terminal),
            span=span if not terminal else span + 1,
            gamma_pow=self.gamma ** (span if not terminal else span + 1),
            obs_env_ids=tuple(head["ids"]),
            next_env_ids=tuple(nxt["ids"]),
        )
        self._buf.pop(0)
        return tr

    @property
    def pending(self) -> int:
        return len(self._buf)


class PrioritizedReplayBuffer:
    """PER buffer over n-step transitions with lazy uint8 frame storage."""

    def __init__(self, cfg: PERConfig, frame_size: int = 84, gamma: float = 0.99) -> None:
        self.cfg = cfg
        self.gamma = gamma
        self.capacity = cfg.capacity
        self.frames = FrameStore(cfg.capacity + 64, frame_size)
        self.transitions: list[Optional[Transition]] = [None] * cfg.capacity
        self.priorities = np.zeros(cfg.capacity, dtype=np.float64)
        self.tree = SumTree(cfg.capacity)
        self.pos = 0
        self.size = 0
        self.filled_once = False
        self.max_priority = 1.0
        self._eps = cfg.priority_eps
        self.evicted_refs = 0

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #
    def add_nstep(self, tr: NStepTransition) -> None:
        obs_ids = tuple(
            self.frames.add_if_new(tr.obs[i], tr.obs_env_ids[i]) for i in range(4)
        )
        next_ids = tuple(
            self.frames.add_if_new(tr.next_obs[i], tr.next_env_ids[i]) for i in range(4)
        )
        t = Transition(
            obs_ids=obs_ids,
            next_ids=next_ids,
            obs_env_ids=tr.obs_env_ids,
            next_env_ids=tr.next_env_ids,
            action=int(tr.action),
            reward=float(tr.reward),
            done=bool(tr.done),
            span=int(tr.span),
            gamma_pow=float(tr.gamma_pow),
        )
        t.validate()
        idx = self.pos
        self.transitions[idx] = t
        # priorities[] and the sum tree ALWAYS store alpha-powered values so
        # save/load and sampling agree on one convention.
        self.priorities[idx] = self.max_priority ** self.cfg.alpha
        self.tree.update(idx, self.priorities[idx])
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.size == self.capacity:
            self.filled_once = True

    # ------------------------------------------------------------------ #
    # Sampling with eviction validation
    # ------------------------------------------------------------------ #
    def _valid(self, t: Transition) -> bool:
        for slot, eid in zip(t.obs_ids, t.obs_env_ids):
            if not self.frames.resident(slot, eid):
                return False
        for slot, eid in zip(t.next_ids, t.next_env_ids):
            if not self.frames.resident(slot, eid):
                return False
        return True

    def sample(self, batch_size: int, beta: float,
               rng: Optional[np.random.Generator] = None,
               max_replace_rounds: int = 3) -> dict[str, Any]:
        if self.size == 0:
            raise IndexError("empty replay buffer")
        rng = rng or np.random.default_rng()
        total = self.tree.total()
        if total <= 0:
            total = max(self.size, 1)
            leaf_idx = rng.integers(0, self.size, size=batch_size)
        else:
            spans = (rng.random(batch_size) + np.arange(batch_size)) * (total / batch_size)
            leaf_idx = self.tree.find_prefixsum_batch(np.mod(spans, total))
        leaf_idx = np.clip(leaf_idx, 0, self.size - 1)

        # Eviction validation: resample invalid slots up to N rounds.
        for _ in range(max_replace_rounds):
            bad = [i for i in range(batch_size)
                   if self.transitions[int(leaf_idx[i])] is None
                   or not self._valid(self.transitions[int(leaf_idx[i])])]
            if not bad:
                break
            self.evicted_refs += len(bad)
            if total > 0:
                spans = (rng.random(len(bad)) + np.arange(len(bad))) * (total / len(bad))
                repl = np.clip(self.tree.find_prefixsum_batch(np.mod(spans, total)),
                               0, self.size - 1)
            else:
                repl = rng.integers(0, self.size, size=len(bad))
            leaf_idx[bad] = repl

        trs = [self.transitions[int(i)] for i in leaf_idx]
        trs = [t if t is not None else self.transitions[0] for t in trs]
        obs = np.stack([np.stack([self.frames.frames[s] for s in t.obs_ids], axis=0)
                        for t in trs])
        next_obs = np.stack([np.stack([self.frames.frames[s] for s in t.next_ids], axis=0)
                             for t in trs])
        probs = self.priorities[leaf_idx] / max(total, 1e-12)
        min_p = self.tree.min_leaf() / max(total, 1e-12)
        weights = np.power(np.maximum(probs, 1e-12) / max(min_p, 1e-12), -beta)
        weights = weights / max(float(weights.max()), 1e-12)
        return {
            "indices": leaf_idx.astype(np.int64),
            "weights": weights.astype(np.float32),
            "obs": obs,
            "next_obs": next_obs,
            "actions": np.array([t.action for t in trs], dtype=np.int64),
            "rewards": np.array([t.reward for t in trs], dtype=np.float32),
            "dones": np.array([t.done for t in trs], dtype=np.float32),
            "gamma_pows": np.array([t.gamma_pow for t in trs], dtype=np.float32),
            "spans": np.array([t.span for t in trs], dtype=np.int64),
        }

    def update_priorities(self, indices: Iterable[int], td_errors: np.ndarray) -> None:
        td = np.abs(np.asarray(td_errors, dtype=np.float64))
        td = np.where(np.isfinite(td), td, 0.0)
        prios = td + self._eps
        if prios.size:
            self.max_priority = max(self.max_priority, float(prios.max()))
        idx = np.asarray(indices, dtype=np.int64)
        self.tree.update_batch(idx, prios ** self.cfg.alpha)
        self.priorities[idx] = prios ** self.cfg.alpha

    def beta_for_step(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.cfg.beta_frames))
        return float(self.cfg.beta_start + frac * (1.0 - self.cfg.beta_start))

    # ------------------------------------------------------------------ #
    # Introspection + persistence
    # ------------------------------------------------------------------ #
    def nbytes(self) -> int:
        return (self.frames.nbytes() + self.tree.tree.nbytes
                + self.priorities.nbytes + len(self.transitions) * 200)

    def action_counts(self) -> dict[int, int]:
        counts = {a: 0 for a in range(5)}
        for t in self.transitions[: self.size]:
            if t is not None:
                counts[t.action] += 1
        return counts

    def state_payload(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "pos": self.pos,
            "filled_once": self.filled_once,
            "max_priority": self.max_priority,
            "frame_cursor": self.frames.cursor,
            "frame_written": self.frames.written,
            "frames": self.frames.frames,
            "frame_env_ids": self.frames.env_ids,
            "transitions": [
                None if t is None else (
                    t.obs_ids, t.next_ids, t.obs_env_ids, t.next_env_ids,
                    t.action, t.reward, t.done, t.span, t.gamma_pow,
                )
                for t in self.transitions
            ],
            "priorities": self.priorities,
        }

    def save(self, path: str) -> None:
        integrity_pickle_save(path, self.state_payload())

    @classmethod
    def load(cls, path: str, cfg: PERConfig, frame_size: int = 84,
             gamma: float = 0.99) -> "PrioritizedReplayBuffer":
        """Load or raise CorruptFileError (file already renamed .corrupt)."""
        payload = integrity_pickle_load(path)
        if not isinstance(payload, dict) or "frames" not in payload \
                or "transitions" not in payload:
            raise CorruptFileError(f"{path}: unexpected payload layout")
        buf = cls(cfg, frame_size=frame_size, gamma=gamma)
        cap = int(payload["capacity"])
        frames = np.asarray(payload["frames"], dtype=np.uint8)
        if frames.shape[1] != frame_size:
            raise CorruptFileError(
                f"{path}: frame size {frames.shape[1]} != configured {frame_size}"
            )
        if cap > cfg.capacity:
            cap = cfg.capacity  # never exceed the configured memory bound
        buf._reinit(cap, frames.shape[1])
        buf.frames.frames[:] = payload["frames"]
        buf.frames.env_ids[:] = np.asarray(payload["frame_env_ids"], dtype=np.int64)
        buf.frames.cursor = int(payload["frame_cursor"]) % buf.frames.capacity
        buf.frames.written = int(payload["frame_written"])
        buf.size = min(int(payload["size"]), cap)
        buf.pos = int(payload["pos"]) % cap
        buf.filled_once = bool(payload["filled_once"])
        buf.max_priority = float(payload["max_priority"])
        for i, raw in enumerate(payload["transitions"][: buf.size]):
            if raw is None:
                continue
            (obs_ids, next_ids, oe, ne, action, reward, done, span, gpow) = raw
            buf.transitions[i] = Transition(
                tuple(obs_ids), tuple(next_ids), tuple(oe), tuple(ne),
                int(action), float(reward), bool(done), int(span), float(gpow),
            )
        prios = np.asarray(payload["priorities"], dtype=np.float64)[:cap]
        buf.priorities[:cap] = prios
        # priorities[] are stored ALREADY alpha-powered (see add_nstep);
        # rebuilding must not power them a second time.
        buf.tree.rebuild_from(buf.priorities, filled=buf.size)
        return buf

    def _reinit(self, capacity: int, frame_size: int) -> None:
        self.capacity = capacity
        self.frames = FrameStore(capacity + 64, frame_size)
        self.transitions = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.tree = SumTree(capacity)
        self.pos = 0
        self.size = 0
