"""Atomic checkpoint management for models and replay buffer.

Rules (requirement §12):
* ``latest_model.pth`` every N learner updates; ``best_model.pth`` only when
  the configured evaluation metric improves; ``buffer.pkl`` every N updates
  and on clean shutdown.
* All writes go to a temp file followed by ``os.replace`` (atomic on NTFS).
* Corrupt files are renamed ``*.corrupt``, logged with a traceback, and the
  bot continues from an empty/fresh state instead of dying.
* Payload includes RNG states, counters, config, profile, git/config hash and
  performance metadata for reproducibility.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from logging_utils import (
    CorruptFileError,
    format_exception,
    get_logger,
    hash_bytes,
)


_GIT_HASH_CACHE: Optional[str] = None


def _git_hash() -> str:
    """Best-effort git HEAD hash ('' when git is unavailable), memoised.

    # DEEP-FIX: this spawned a ``git`` subprocess on EVERY ``save_model``
    # call — including the best-model saves that are rejected before anything
    # is written — on a 2-core machine that also runs Chrome.  HEAD cannot
    # change inside a running process, so it is resolved once and cached.
    """
    global _GIT_HASH_CACHE
    if _GIT_HASH_CACHE is not None:
        return _GIT_HASH_CACHE
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).parent),
        )
        if out.returncode == 0:
            _GIT_HASH_CACHE = out.stdout.strip()
            return _GIT_HASH_CACHE
    except (OSError, subprocess.SubprocessError) as exc:
        get_logger("checkpoint").debug("git hash unavailable: %s", exc)
    _GIT_HASH_CACHE = ""
    return _GIT_HASH_CACHE


def capture_rng_states(seed: int | None = None,
                       generator: Any = None) -> dict[str, Any]:
    """Snapshot python/numpy/torch RNG states for reproducible resumes.

    # DEEP-FIX: ``generator`` is the ``np.random.Generator`` that actually
    # draws the replay batches.  The old body stored
    # ``np.random.default_rng().bit_generator.state`` — the state of a brand
    # new generator seeded from OS entropy that no code path ever reads, so
    # the "reproducible resume" in this module's docstring restored nothing.
    """
    import random

    gen_state = None
    if generator is not None:
        try:
            gen_state = generator.bit_generator.state
        except AttributeError:  # pragma: no cover - defensive
            get_logger("checkpoint").warning(
                "cannot capture the state of %r; resume will not reproduce "
                "replay sampling exactly", type(generator).__name__)
    return {
        "python": random.getstate(),
        "generator": gen_state,
        "numpy_legacy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "seed": seed,
    }


def restore_rng_states(states: dict[str, Any], generator: Any = None) -> bool:
    """Restore a snapshot; returns True when the replay Generator was restored.

    # DEEP-FIX: the numpy Generator state was never restored at all, so even
    # a correctly captured snapshot would not reproduce a run.
    """
    import random

    if states.get("python"):
        random.setstate(states["python"])
    if states.get("numpy_legacy"):
        np.random.set_state(states["numpy_legacy"])
    if states.get("torch") is not None:
        torch.set_rng_state(states["torch"])
    gen_state = states.get("generator")
    if gen_state is not None and generator is not None:
        try:
            generator.bit_generator.state = gen_state
            return True
        except (AttributeError, TypeError, ValueError) as exc:
            get_logger("checkpoint").warning(
                "could not restore the replay Generator state: %s", exc)
    return False


class CheckpointManager:
    def __init__(self, directory: str | Path, profile: str) -> None:
        self.dir = Path(directory) / profile
        self.dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.best_metric: Optional[float] = None
        self._best_path = self.dir / "best_model.pth"
        self._latest_path = self.dir / "latest_model.pth"
        self._buffer_path = self.dir / "buffer.pkl"

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    @property
    def latest_path(self) -> Path:
        return self._latest_path

    @property
    def best_path(self) -> Path:
        return self._best_path

    @property
    def buffer_path(self) -> Path:
        return self._buffer_path

    # ------------------------------------------------------------------ #
    # Model checkpoints
    # ------------------------------------------------------------------ #
    def save_model(
        self,
        agent_payload: dict[str, Any],
        extra: dict[str, Any],
        which: str = "latest",
        metric: Optional[float] = None,
        higher_is_better: bool = True,
    ) -> Optional[Path]:
        # DEEP-FIX: the whole payload (including a git subprocess) used to be
        # built before the "is this actually an improvement?" check, so every
        # rejected best-save still paid for pickling and a git spawn.
        if which == "best":
            # DEEP-FIX: `metric > self.best_metric` raised TypeError when a
            # caller passed None, and a NaN metric compared False against
            # everything so the gate would silently never open again.
            try:
                metric = float(metric)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                get_logger("checkpoint").error(
                    "refusing to gate best_model on a non-numeric metric %r",
                    metric)
                return None
            if not np.isfinite(metric):
                get_logger("checkpoint").error(
                    "refusing to gate best_model on a non-finite metric %r",
                    metric)
                return None
            if self.best_metric is not None:
                improved = (metric > self.best_metric) if higher_is_better \
                    else (metric < self.best_metric)
                if not improved:
                    return None
        payload = {
            "profile": self.profile,
            "agent": agent_payload,
            "extra": dict(extra),
            "saved_at": time.time(),
            "git_hash": _git_hash(),
        }
        payload["extra"].setdefault("rng", capture_rng_states())
        path = self._latest_path if which == "latest" else self._best_path
        try:
            self._atomic_torch_save(payload, path)
        except (OSError, RuntimeError, ValueError) as exc:
            get_logger("checkpoint").error(
                "checkpoint save failed for %s:\n%s", path, format_exception(exc)
            )
            return None
        if which == "best":
            # DEEP-FIX: the gate used to be advanced BEFORE the write, so a
            # failed save (full disk, permissions) left best_metric pointing
            # at a value with no file behind it — and because best_metric is
            # persisted into every later checkpoint, that best was lost for
            # good while the code kept believing it existed.  Verified: a
            # forced OSError left best_metric=10.0 with no best_model.pth.
            self.best_metric = metric
        return path

    def load_model(self, which: str = "latest") -> Optional[dict[str, Any]]:
        path = self._latest_path if which == "latest" else self._best_path
        if not path.exists():
            return None
        # DEEP-FIX: _atomic_torch_save writes a .sha256 sidecar for every
        # model checkpoint, but nothing ever read it back — the integrity
        # mechanism was write-only, so a half-flushed or bit-rotted .pth was
        # loaded as if it were fine.  Verify, and quarantine on mismatch.
        if not self._verify_hash(path):
            return None
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
            return payload
        except Exception as exc:
            corrupt = path.with_name(path.name + ".corrupt")
            try:
                os.replace(str(path), str(corrupt))
            except OSError as rename_exc:
                get_logger("checkpoint").warning(
                    "could not rename corrupt file %s: %s", path, rename_exc
                )
            get_logger("checkpoint").error(
                "checkpoint load failed for %s:\n%s", path, format_exception(exc)
            )
            return None

    def _verify_hash(self, path: Path) -> bool:
        """True when the sidecar is absent (legacy file) or matches."""
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.exists():
            get_logger("checkpoint").debug(
                "%s has no sha256 sidecar; loading without verification", path)
            return True
        try:
            expected = sidecar.read_text(encoding="ascii").strip()
            actual = hash_bytes(path.read_bytes())
        except OSError as exc:
            get_logger("checkpoint").error("cannot read %s or its sidecar: %s",
                                           path, exc)
            return False
        if expected and expected != actual:
            corrupt = path.with_name(path.name + ".corrupt")
            try:
                os.replace(str(path), str(corrupt))
            except OSError as rename_exc:
                get_logger("checkpoint").warning(
                    "could not quarantine %s: %s", path, rename_exc)
            get_logger("checkpoint").error(
                "%s failed sha256 verification (expected %s, got %s); moved to "
                "%s and ignored", path, expected[:12], actual[:12], corrupt)
            return False
        return True

    def _atomic_torch_save(self, payload: dict[str, Any], path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        # DEEP-FIX: `data = torch.save(...)` assigned the (always None) return
        # value and was never used; the hash is taken from the bytes that were
        # actually written, after an fsync, and the sidecar is installed only
        # once the checkpoint itself is durably in place.
        torch.save(payload, str(tmp))
        # fsync the payload while it is still the temp file, then publish it
        # with os.replace, then write the sidecar for the installed name.
        with open(tmp, "r+b") as fh:
            os.fsync(fh.fileno())
        with open(tmp, "rb") as fh:
            blob = fh.read()
        digest = hash_bytes(blob)
        os.replace(str(tmp), str(path))
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(digest + "\n", encoding="ascii")
        try:  # make the rename itself durable (best effort; not all FSs care)
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError) as exc:  # platform dependent
            # DEEP-FIX: an empty `pass` here would also trip the repo's own
            # hygiene test (tests/test_hygiene.py); log at debug instead.
            get_logger("checkpoint").debug("directory fsync skipped: %s", exc)

    # ------------------------------------------------------------------ #
    # Replay buffer persistence
    # ------------------------------------------------------------------ #
    def buffer_save(self, buffer_obj: Any) -> bool:
        try:
            buffer_obj.save(str(self._buffer_path))
            return True
        except Exception as exc:
            get_logger("checkpoint").error(
                "buffer save failed:\n%s", format_exception(exc)
            )
            return False

    def buffer_load(self, load_fn) -> tuple[Any, bool]:
        """Returns (buffer, loaded?).

        ``load_fn`` is a zero-arg callable that loads the buffer from
        ``self.buffer_path`` (raising CorruptFileError on corruption), plus a
        fresh-buffer fallback is created by the caller on failure.  Any
        corruption is renamed to ``.corrupt`` by the loader, logged with a
        traceback, and the bot continues with an empty buffer.
        """
        if not self._buffer_path.exists():
            return None, False
        try:
            return load_fn(), True
        except CorruptFileError as exc:
            get_logger("checkpoint").error(
                "buffer corrupt, starting empty:\n%s", format_exception(exc)
            )
            return None, False
        except Exception as exc:  # unexpected: never take the process down
            get_logger("checkpoint").error(
                "buffer load failed, starting empty:\n%s", format_exception(exc)
            )
            return None, False
