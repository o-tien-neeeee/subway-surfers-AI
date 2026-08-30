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

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from logging_utils import (
    CorruptFileError,
    format_exception,
    get_logger,
    hash_bytes,
)


def _git_hash() -> str:
    """Best-effort git HEAD hash ('' when git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).parent), check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        get_logger("checkpoint").debug("git hash unavailable: %s", exc)
    return ""


def capture_rng_states(seed: int | None = None) -> dict[str, Any]:
    """Snapshot python/numpy/torch RNG states for reproducible resumes."""
    import random

    return {
        "python": random.getstate(),
        "numpy": np.random.default_rng().bit_generator.state,
        "numpy_legacy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "seed": seed,
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    import random

    if states.get("python"):
        random.setstate(states["python"])
    if states.get("numpy_legacy"):
        np.random.set_state(states["numpy_legacy"])
    if states.get("torch") is not None:
        torch.set_rng_state(states["torch"])


class CheckpointManager:
    def __init__(self, directory: str | Path, profile: str) -> None:
        self.dir = Path(directory) / profile
        self.dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.best_metric: float | None = None
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
        metric: float | None = None,
        higher_is_better: bool = True,
    ) -> Path | None:
        payload = {
            "profile": self.profile,
            "agent": agent_payload,
            "extra": dict(extra),
            "saved_at": time.time(),
            "git_hash": _git_hash(),
        }
        payload["extra"].setdefault("rng", capture_rng_states())
        if which == "best":
            if self.best_metric is not None:
                improved = (metric > self.best_metric) if higher_is_better \
                    else (metric < self.best_metric)
                if not improved:
                    return None
            self.best_metric = metric
        path = self._latest_path if which == "latest" else self._best_path
        try:
            self._atomic_torch_save(payload, path)
        except OSError as exc:
            get_logger("checkpoint").error(
                "checkpoint save failed for %s:\n%s", path, format_exception(exc)
            )
            return None
        return path

    def load_model(self, which: str = "latest") -> dict[str, Any] | None:
        path = self._latest_path if which == "latest" else self._best_path
        if not path.exists():
            return None
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
            return payload
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
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

    def _atomic_torch_save(self, payload: dict[str, Any], path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, str(tmp))
        # torch.save returns None; read back for hashing.
        with open(tmp, "rb") as fh:
            digest = hash_bytes(fh.read())
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(digest + "\n", encoding="ascii")
        os.replace(str(tmp), str(path))

    # ------------------------------------------------------------------ #
    # Replay buffer persistence
    # ------------------------------------------------------------------ #
    def buffer_save(self, buffer_obj: Any) -> bool:
        try:
            buffer_obj.save(str(self._buffer_path))
            return True
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
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
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
            get_logger("checkpoint").error(
                "buffer load failed, starting empty:\n%s", format_exception(exc)
            )
            return None, False
