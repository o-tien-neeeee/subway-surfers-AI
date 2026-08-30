"""Shared pytest configuration: repo root on sys.path + spawn-safe env."""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows-consistent process semantics in CI.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError as _exc:  # already set by an earlier import — keep spawn
    print(f"conftest: start method already set ({_exc}); continuing with spawn")
