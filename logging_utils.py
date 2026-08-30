"""Logging helpers shared by every process.

Rules enforced here:
* No bare ``except:`` and no ``except Exception: pass`` anywhere in the codebase
  (grep-able pattern in tests).
* Worker processes report tracebacks through a bounded queue so the GUI stays
  alive and informed when a worker crashes.
* A rotating file handler captures everything even when the console is closed.
"""

from __future__ import annotations

import logging
import logging.handlers
import pickle
import sys
import traceback
import types
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s | %(processName)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str = "ssbot",
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    filename: str = "bot.log",
) -> logging.Logger:
    """Configure the shared ``ssbot`` logger tree with console + file output.

    ``name`` only chooses which child logger is returned; handlers always
    attach to ``ssbot`` so every ``get_logger(...)`` child in the process
    inherits them (worker processes call this first thing on entry).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    parent = logging.getLogger("ssbot")
    parent.setLevel(level)
    parent.propagate = False
    if not any(getattr(h, "_ssbot_marker", False) for h in parent.handlers):
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=_DATEFMT))
        console._ssbot_marker = True  # type: ignore[attr-defined]
        fileh = logging.handlers.RotatingFileHandler(
            log_path / filename, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        fileh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=_DATEFMT))
        fileh._ssbot_marker = True  # type: ignore[attr-defined]
        parent.addHandler(console)
        parent.addHandler(fileh)
    return logging.getLogger(f"ssbot.{name}" if name != "ssbot" else "ssbot")


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ssbot namespace (inherits handlers)."""
    return logging.getLogger(f"ssbot.{name}")


def format_exception(exc: BaseException) -> str:
    """Full traceback text for an exception (never swallowed)."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def report_exception(
    logger: logging.Logger,
    exc: BaseException,
    context: str,
    queue: Any | None = None,
) -> None:
    """Log a traceback and forward it to the GUI error queue if provided."""
    tb = format_exception(exc)
    logger.error("Exception in %s:\n%s", context, tb)
    if queue is not None:
        put_bounded(
            queue,
            {
                "type": "error",
                "src": context,
                "error": f"{type(exc).__name__}: {exc}",
                "tb": tb,
            },
        )


# --------------------------------------------------------------------- #
# Bounded queue helpers — workers must never block the GUI or the loop
# --------------------------------------------------------------------- #
def put_bounded(queue: Any, item: Any, name: str = "queue") -> bool:
    """Try to put ``item``; drop it (with a warning) if the queue is full."""
    import queue as _queue

    try:
        queue.put_nowait(item)
        return True
    except _queue.Full:
        kind = item.get("type", "?") if isinstance(item, dict) else type(item).__name__
        logging.getLogger("ssbot.ipc").debug("%s full; dropping %s message", name, kind)
        return False
    except (EOFError, BrokenPipeError, ConnectionError):
        # Consumer died (e.g. during shutdown) — nothing to do but note it.
        return False


def drain(queue: Any, limit: int = 256) -> list[Any]:
    """Non-blocking drain of up to ``limit`` items from a queue."""
    import queue as _queue

    items: list[Any] = []
    while len(items) < limit:
        try:
            items.append(queue.get_nowait())
        except _queue.Empty:
            break
        except (EOFError, BrokenPipeError, ConnectionError):
            break
    return items


# --------------------------------------------------------------------- #
# Integrity-checked pickle I/O for the replay buffer
# --------------------------------------------------------------------- #
class CorruptFileError(RuntimeError):
    """Raised when a persisted artifact fails integrity verification."""


def hash_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes, suffix: str = ".tmp") -> None:
    """Write ``data`` to a temp file then ``os.replace`` it into place."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + suffix)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        import os as _os

        _os.fsync(fh.fileno())
    _os.replace(tmp, p)


def integrity_pickle_save(path: str | Path, obj: Any) -> None:
    """Pickle + sha256 sidecar, written atomically."""
    data = pickle.dumps(obj, protocol=4)
    atomic_write_bytes(path, data)
    atomic_write_bytes(str(path) + ".sha256", (hash_bytes(data) + "\n").encode("ascii"))


def integrity_pickle_load(path: str | Path) -> Any:
    """Load a pickle produced by :func:`integrity_pickle_save`.

    Verifies the sha256 sidecar when present, renames corrupt files to
    ``*.corrupt`` and raises :class:`CorruptFileError` so callers can start
    from scratch while keeping the GUI alive.  Only ever point this at files
    the bot itself wrote (a pickle is executed-equivalent untrusted input).
    """
    import shutil

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = p.read_bytes()
    sidecar = Path(str(p) + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="ascii").strip()
        if expected and expected != hash_bytes(data):
            corrupt = p.with_name(p.name + ".corrupt")
            shutil.move(str(p), str(corrupt))
            raise CorruptFileError(
                f"{p} failed sha256 verification; moved to {corrupt}"
            )
    try:
        return pickle.loads(data)
    except Exception as exc:  # narrow handling, then re-raise typed
        corrupt = p.with_name(p.name + ".corrupt")
        try:
            shutil.move(str(p), str(corrupt))
        except OSError as move_exc:
            logging.getLogger("ssbot.io").warning(
                "could not quarantine corrupt file %s: %s", p, move_exc
            )
        raise CorruptFileError(f"{p} is unreadable ({exc}); moved to {corrupt}") from exc


def safe_repr(value: Any, limit: int = 200) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def module_has_bare_except(module: types.ModuleType) -> bool:
    """Static check used by the hygiene test: no ``except:`` / ``except Exception: pass``.

    Parses the module source with ``ast`` instead of grepping, so comments and
    strings do not trigger false positives.
    """
    import ast

    src = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                return True
            if (
                isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
                and all(isinstance(b, ast.Pass) for b in node.body)
            ):
                return True
    return False
