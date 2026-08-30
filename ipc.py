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
from dataclasses import dataclass
from typing import Optional

import numpy as np

CTX = mp.get_context("spawn")


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


#: Bytes reserved for the published weight-layout fingerprint.
_FINGERPRINT_BYTES = 64


def layout_fingerprint(module) -> str:
    """Stable identity of a module's parameter *layout* (shapes, in order).

    # DEEP-FIX: the flat weight vector carries no schema.  Before this, the
    # actor blindly consumed the first ``sum(p.numel())`` floats of whatever
    # the learner published, so an actor running profile A against a learner
    # running profile B silently loaded a meaningless prefix of B's vector
    # (verified: quality_cpu -> strict_lite overwrote a (4,1,3,3) conv with
    # the leading bytes of a (48,4,8,8) conv, with no error anywhere).  The
    # fingerprint makes that mismatch detectable instead of silent.
    """
    import hashlib

    desc = "|".join(",".join(str(int(d)) for d in tuple(p.shape))
                    for p in module.parameters())
    return hashlib.sha256(desc.encode("ascii")).hexdigest()[:32]


class SharedWeights:
    """Flat float32 parameter vector in shared memory for actor/learner sync.

    The learner is the only writer.  ``publish`` copies a state dict in and
    bumps the version; ``refresh_if_new`` copies it out into a module when
    the version changed.  Both operations are lock-free (version check on
    both sides); a torn read is impossible because the actor copies either
    the old or the new version consistently — we verify the version again
    after copying and retry on a mid-copy publish.

    A layout fingerprint travels with the vector so a reader whose module
    shape does not match the published layout refuses the copy instead of
    silently mis-aligning every tensor (see :func:`layout_fingerprint`).
    """

    def __init__(self, size: int, max_retries: int = 64) -> None:
        if size <= 0:
            raise ValueError("SharedWeights size must be positive")
        self.size = size
        self.max_retries = max_retries
        self._arr = CTX.Array(ct.c_float, size)
        self._version = CTX.Value(ct.c_uint64, 0)
        # DEEP-FIX: layout identity of the last successful publish ("" = none).
        self._layout = CTX.Array(ct.c_char, _FINGERPRINT_BYTES)
        self._torn_reads = CTX.Value(ct.c_uint64, 0)

    def publish(self, flat: np.ndarray, fingerprint: str = "") -> None:
        """Write a flat parameter vector plus its layout fingerprint.

        The buffer is sized for the largest profile, so a smaller profile
        legitimately publishes a prefix; the tail is zeroed and the
        fingerprint records exactly which layout the prefix represents.
        """
        vec = flat.astype(np.float32).reshape(-1)
        if vec.size > self.size:
            raise ValueError(
                f"weights size {vec.size} exceeds shared buffer {self.size}"
            )
        if not np.all(np.isfinite(vec)):
            # DEEP-FIX: a NaN/inf blow-up in the learner used to be published
            # straight into the actor, which then pressed keys forever with a
            # poisoned policy.  Refuse to publish non-finite weights.
            raise ValueError("refusing to publish non-finite weights")
        arr = np.frombuffer(self._arr.get_obj(), dtype=np.float32)
        arr[: vec.size] = vec
        if vec.size < self.size:
            arr[vec.size :] = 0.0
        # DEEP-FIX: write the layout BEFORE the version bump so a reader that
        # observes the new version always observes the matching fingerprint.
        raw = str(fingerprint).encode("ascii")[: _FINGERPRINT_BYTES - 1]
        self._layout.value = raw + b"\x00" * (_FINGERPRINT_BYTES - len(raw))
        self._version.value += 1

    def fingerprint(self) -> str:
        raw = bytes(self._layout.raw)
        return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")

    def version(self) -> int:
        return int(self._version.value)

    def torn_reads(self) -> int:
        return int(self._torn_reads.value)

    def copy_out(self) -> Optional[np.ndarray]:
        """Return a private copy of the current weights (None if contended).

        # DEEP-FIX: the old implementation returned a *possibly torn* copy
        # after 8 failed attempts with a comment telling the caller to log
        # it — and no caller did.  Returning None makes contention explicit
        # and lets the actor keep its last known-good weights.
        """
        for _ in range(self.max_retries):
            v0 = self._version.value
            arr = np.frombuffer(self._arr.get_obj(), dtype=np.float32)
            out = arr.copy()
            if self._version.value == v0:
                return out
        with self._torn_reads.get_lock():
            self._torn_reads.value += 1
        return None


def flatten_state_dict(state_dict: dict) -> np.ndarray:
    parts = [v.detach().cpu().numpy().astype(np.float32).reshape(-1)
             for v in state_dict.values()]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def module_param_count(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def unflatten_into(module, flat: np.ndarray) -> int:
    """Write a flat vector back into ``module``'s parameters (in-place).

    # DEEP-FIX: this used to consume a blind prefix of ``flat`` with no
    # validation, which is exactly how a profile mismatch between the actor
    # and the learner turned into silent garbage weights.  The length is now
    # checked against the module's own parameter count.
    """
    import torch

    if flat is None:
        raise ValueError("no weight vector to unflatten")
    vec = np.asarray(flat).reshape(-1)
    need = module_param_count(module)
    if vec.size < need:
        raise ValueError(
            f"weight vector too short for this module: have {vec.size}, "
            f"need {need} (profile mismatch?)"
        )
    offset = 0
    with torch.no_grad():
        for p in module.parameters():
            n = p.numel()
            chunk = vec[offset : offset + n].reshape(p.shape)
            p.copy_(torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)))
            offset += n
    return offset


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

    # -- best-model telemetry (single writer: the actor) ---------------- #
    def publish_episode_result(self, episode_id: int, survival_s: float,
                               total_reward: float) -> None:
        """Publish one finished episode as a three-phase seqlock write.

        # DEEP-FIX: the fields used to be written id-FIRST.  The learner also
        # read the id first, so it could observe a new id and then read the
        # *previous* episode's survival/reward — three independent shared
        # values are not an atomic record.
        #
        # Writing the payload first and the id last is NOT enough on its own:
        # a reader that sampled the id *before* this publish, then read the
        # new payload, would re-check the id and still find the old one, so
        # its "consistency" check passed on a torn pair (reproduced: id=2217
        # paired with survival=2218.0).  A real seqlock invalidates *before*
        # touching the payload, so any reader whose window overlaps the write
        # sees the invalid marker on one of its two id reads and backs off.
        """
        # 1. invalidate -- 0 is the "no episode / in progress" marker that
        #    read_episode_result() already treats as "nothing to report".
        self.last_episode_done_id.value = 0
        # 2. payload
        self.last_episode_survival_s.value = float(survival_s)
        self.last_episode_reward.value = float(total_reward)
        # 3. commit -- publishing the id last makes the record visible.
        self.last_episode_done_id.value = int(episode_id)

    def read_episode_result(self, last_seen_id: int) -> Optional[dict[str, float]]:
        """Read the newest episode newer than ``last_seen_id`` (None if none).

        Returns ``{"episode_id", "survival_s", "total_reward"}`` only when the
        id is stable across the payload read (seqlock-style consistency).
        """
        done_id = int(self.last_episode_done_id.value)
        if done_id <= 0 or done_id == last_seen_id:
            return None
        survival = float(self.last_episode_survival_s.value)
        reward = float(self.last_episode_reward.value)
        if int(self.last_episode_done_id.value) != done_id:
            return None  # writer moved on mid-read; the next poll catches it
        return {"episode_id": done_id, "survival_s": survival,
                "total_reward": reward}

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
