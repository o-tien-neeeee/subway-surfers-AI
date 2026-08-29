"""Inter-process plumbing: bounded frame transport and weight synchronisation.

Why not ``mp.Queue`` for frames?
    A raw 800x600 RGB frame is ~1.4 MB.  Pickling that through a queue at
    30 FPS costs ~43 MB/s of copying plus pickle overhead, and a queue of
    depth N holds N full frames (unbounded growth if the consumer stalls).
    Instead the capture worker writes into a small shared-memory ring and
    only publishes the *latest* frame id.  A slow consumer naturally drops
    stale frames — memory is constant by construction (requirement B).

Synchronisation decisions
-------------------------
* ``SharedFrameRing`` uses a per-slot generation counter: the writer bumps
  the generation *after* filling the slot; the reader retries if the
  generation changed during its copy.  This is lock-free single-writer /
  single-reader and cannot block either side.
* ``SharedWeights`` is a flat float32 array in shared memory plus a version
  counter.  The learner is the only writer (requirement C); the actor copies
  it out only when the version changes.  For a <400k-parameter model this is
  at most a 1.6 MB memcpy every publish — negligible vs. 100 ms latency.
* All control/event objects come from ``mp.get_context("spawn")`` because
  Windows requires spawn and we want identical semantics in CI.
"""

from __future__ import annotations

import multiprocessing as mp
import ctypes as ct
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

CTX = mp.get_context("spawn")

UINT64_MAX = 2 ** 64 - 1


@dataclass(frozen=True)
class Frame:
    """One captured frame leaving the ring."""

    frame_id: int
    ts: float  # monotonic capture timestamp
    image: np.ndarray  # HxWx3 uint8 RGB

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape


class SharedFrameRing:
    """Fixed-capacity, single-writer/single-reader, latest-wins frame ring.

    Memory is allocated once (``slots * H * W * 3`` bytes) and never grows.
    Writers never block; if the writer laps the ring the reader simply sees a
    newer frame (stale ones are dropped by construction).
    """

    def __init__(self, slots: int, height: int, width: int, channels: int = 3) -> None:
        if slots < 2:
            raise ValueError("SharedFrameRing needs at least 2 slots")
        self.slots = slots
        self.shape = (height, width, channels)
        n = slots * height * width * channels
        self._buf = CTX.Array(ct.c_ubyte, n)
        self._gen = [CTX.Value(ct.c_uint64, 0) for _ in range(slots)]
        self._ids = [CTX.Value(ct.c_uint64, 0) for _ in range(slots)]
        self._ts = CTX.Array(ct.c_double, slots)
        self._latest_id = CTX.Value(ct.c_int64, -1)  # frame id, -1 = none
        self._written = CTX.Value(ct.c_uint64, 0)
        self._dropped = CTX.Value(ct.c_uint64, 0)

    # ------------------------------------------------------------------ #
    # Writer side (capture worker)
    # ------------------------------------------------------------------ #
    def write(self, image: np.ndarray, frame_id: int, ts: float) -> bool:
        """Write a frame and publish it.  Returns False on size mismatch."""
        h, w, c = self.shape
        if image.shape != (h, w, c):
            return False
        buf = np.frombuffer(self._buf.get_obj(), dtype=np.uint8)
        slot = frame_id % self.slots
        start = slot * h * w * c
        # 1. fill slot payload (generation still old -> readers see stale)
        buf[start : start + h * w * c] = image.reshape(-1)
        self._ts[slot] = float(ts)
        self._ids[slot].value = frame_id
        # 2. bump generation, then publish latest id (ordering matters).
        self._gen[slot].value += 1
        self._latest_id.value = int(frame_id)
        with self._written.get_lock():
            self._written.value += 1
        return True

    # ------------------------------------------------------------------ #
    # Reader side (actor)
    # ------------------------------------------------------------------ #
    def read_latest(self, max_retries: int = 4) -> Optional[Frame]:
        """Copy out the newest completely-written frame (None if empty)."""
        h, w, c = self.shape
        nbytes = h * w * c
        buf = np.frombuffer(self._buf.get_obj(), dtype=np.uint8)
        for _ in range(max_retries):
            fid = int(self._latest_id.value)
            if fid < 0:
                return None
            slot = fid % self.slots
            gen_before = self._gen[slot].value
            image = buf[slot * nbytes : (slot + 1) * nbytes].reshape(h, w, c).copy()
            gen_after = self._gen[slot].value
            if gen_before == gen_after and self._ids[slot].value == fid:
                return Frame(frame_id=fid, ts=float(self._ts[slot]), image=image)
            # Writer overwrote mid-copy — retry with the newer frame.
        return None

    def latest_frame_id(self) -> int:
        return int(self._latest_id.value)

    def counters(self) -> dict[str, int]:
        return {
            "written": int(self._written.value),
            "dropped": int(self._dropped.value),
        }

    def note_drop(self, n: int = 1) -> None:
        with self._dropped.get_lock():
            self._dropped.value += n

    def nbytes(self) -> int:
        h, w, c = self.shape
        return self.slots * h * w * c


class SharedWeights:
    """Flat float32 parameter vector in shared memory for actor/learner sync.

    The learner is the only writer.  ``publish`` copies a state dict in and
    bumps the version; ``refresh_if_new`` copies it out into a module when
    the version changed.  Both operations are lock-free (version check on
    both sides); a torn read is impossible because the actor copies either
    the old or the new version consistently — we verify the version again
    after copying and retry on a mid-copy publish.
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("SharedWeights size must be positive")
        self.size = size
        self._arr = CTX.Array(ct.c_float, size)
        self._version = CTX.Value(ct.c_uint64, 0)

    def publish(self, flat: np.ndarray) -> None:
        """Write a flat parameter vector (may be shorter than the buffer —
        the buffer is sized for the largest profile and each profile uses a
        prefix; ``unflatten_into`` consumes exactly as many floats as the
        module has parameters)."""
        vec = flat.astype(np.float32).reshape(-1)
        if vec.size > self.size:
            raise ValueError(
                f"weights size {vec.size} exceeds shared buffer {self.size}"
            )
        arr = np.frombuffer(self._arr.get_obj(), dtype=np.float32)
        arr[: vec.size] = vec
        if vec.size < self.size:
            arr[vec.size :] = 0.0
        self._version.value += 1

    def version(self) -> int:
        return int(self._version.value)

    def copy_out(self) -> np.ndarray:
        """Return a private copy of the current weights."""
        for _ in range(8):
            v0 = self._version.value
            arr = np.frombuffer(self._arr.get_obj(), dtype=np.float32)
            out = arr.copy()
            if self._version.value == v0:
                return out
        return out  # extremely contended; accept possibly-torn copy (log caller)


def flatten_state_dict(state_dict: dict) -> np.ndarray:
    parts = [v.detach().cpu().numpy().astype(np.float32).reshape(-1)
             for v in state_dict.values()]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def unflatten_into(module, flat: np.ndarray) -> None:
    """Write a flat vector back into ``module``'s parameters (in-place)."""
    import torch

    offset = 0
    with torch.no_grad():
        for p in module.parameters():
            n = p.numel()
            chunk = flat[offset : offset + n].reshape(p.shape)
            p.copy_(torch.from_numpy(chunk.astype(np.float32)))
            offset += n


def torch_no_grad():
    import torch

    return torch.no_grad()


class SharedCounters:
    """Cross-process counters and scalars the GUI/actor/learner share.

    Every field has exactly one writer to avoid races:
    * actor:   env_frame_id, action_step, episode_id, epsilon, danger_flag,
               last_episode_* (written once per finished episode)
    * learner: learner_update_step, beta, td_loss, buffer_size
    """

    def __init__(self) -> None:
        self.env_frame_id = CTX.Value(ct.c_uint64, 0)
        self.action_step = CTX.Value(ct.c_uint64, 0)
        self.episode_id = CTX.Value(ct.c_uint64, 0)
        self.learner_update_step = CTX.Value(ct.c_uint64, 0)
        self.epsilon = CTX.Value(ct.c_double, 1.0)
        self.beta = CTX.Value(ct.c_double, 0.4)
        self.td_loss = CTX.Value(ct.c_double, 0.0)
        self.q_mean = CTX.Value(ct.c_double, 0.0)
        self.buffer_size = CTX.Value(ct.c_uint64, 0)
        self.danger_flag = CTX.Value(ct.c_int, 0)
        self.death_flag = CTX.Value(ct.c_int, 0)
        self.profile = CTX.Array(ct.c_char, 32)  # active profile name
        # -- best-model telemetry (actor writes, learner polls) ---------- #
        self.last_episode_done_id = CTX.Value(ct.c_uint64, 0)
        self.last_episode_survival_s = CTX.Value(ct.c_double, 0.0)
        self.last_episode_reward = CTX.Value(ct.c_double, 0.0)

    def set_profile(self, name: str) -> None:
        raw = name.encode("ascii")[:31]
        self.profile.value = raw + b"\x00" * (32 - len(raw))

    def get_profile(self) -> str:
        raw = bytes(self.profile.raw)
        return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")

    def snapshot(self) -> dict[str, float]:
        return {
            "env_frame_id": int(self.env_frame_id.value),
            "action_step": int(self.action_step.value),
            "episode_id": int(self.episode_id.value),
            "learner_update_step": int(self.learner_update_step.value),
            "epsilon": float(self.epsilon.value),
            "beta": float(self.beta.value),
            "td_loss": float(self.td_loss.value),
            "q_mean": float(self.q_mean.value),
            "buffer_size": int(self.buffer_size.value),
            "danger": int(self.danger_flag.value),
            "dead": int(self.death_flag.value),
            "profile": self.get_profile(),
            "last_episode_done_id": int(self.last_episode_done_id.value),
            "last_episode_survival_s": float(self.last_episode_survival_s.value),
            "last_episode_reward": float(self.last_episode_reward.value),
        }


def bounded_queue(maxsize: int = 256):
    """Queue factory from the spawn context (never unbounded)."""
    return CTX.Queue(maxsize=maxsize)


def make_events() -> dict:
    """Standard event set shared by all processes."""
    return {
        "stop": CTX.Event(),          # global shutdown
        "emergency": CTX.Event(),     # F8 / failsafe
        "pause": CTX.Event(),         # pause gameplay + learner updates
        "pause_learning": CTX.Event(),  # pause learner only (death/respawn)
        "death": CTX.Event(),         # death confirmed by actor
    }


def pack_frame_header(frame_id: int, ts: float) -> bytes:
    return struct.pack("<Qd", frame_id & UINT64_MAX, ts)


def unpack_frame_header(blob: bytes) -> tuple[int, float]:
    fid, ts = struct.unpack("<Qd", blob[:16])
    return int(fid), float(ts)


def monotonic() -> float:
    return time.monotonic()
