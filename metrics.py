"""Metrics primitives: latency percentiles, counters, and system usage.

All measurement windows are bounded ring buffers — memory cannot grow with
runtime.  Percentiles are computed with ``statistics``-free numpy-free code so
the module imports instantly (it sits on the hot GUI update path).
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Iterable, Optional


class Ring:
    """Bounded FIFO of floats (oldest evicted automatically)."""

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self._data: deque[float] = deque(maxlen=capacity)

    def add(self, value: float) -> None:
        self._data.append(float(value))

    def extend(self, values: Iterable[float]) -> None:
        for v in values:
            self.add(v)

    def values(self) -> list[float]:
        return list(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in [0, 100]); 0.0 when empty."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    xs = sorted(values)
    if pct <= 0:
        return float(xs[0])
    if pct >= 100:
        return float(xs[-1])
    k = (len(xs) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def stats(values: list[float]) -> dict[str, float]:
    """Mean/std/min/max/p50/p95/p99 summary."""
    if not values:
        return {k: 0.0 for k in ("n", "mean", "std", "min", "max", "p50", "p95", "p99")}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "n": float(n),
        "mean": mean,
        "std": math.sqrt(var),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


class LatencyMeter:
    """Ring of latency samples in milliseconds with cached percentiles."""

    def __init__(self, capacity: int = 512, name: str = "latency") -> None:
        self.name = name
        self.ring = Ring(capacity)

    def observe_ms(self, ms: float) -> None:
        if ms is None or not math.isfinite(ms) or ms < 0:
            return
        self.ring.add(ms)

    def snapshot(self) -> dict[str, float]:
        return stats(self.ring.values())


class Counter:
    """Monotonic counter with a resettable window (for rates)."""

    def __init__(self) -> None:
        self.total = 0
        self._window_start = time.monotonic()
        self._window_count = 0

    def inc(self, n: int = 1) -> None:
        self.total += n
        self._window_count += n

    def rate_per_s(self, now: Optional[float] = None) -> float:
        t = now if now is not None else time.monotonic()
        span = t - self._window_start
        if span <= 0:
            return 0.0
        return self._window_count / span

    def reset_window(self, now: Optional[float] = None) -> None:
        self._window_start = now if now is not None else time.monotonic()
        self._window_count = 0


class FpsMeter:
    """Effective FPS from processed-frame timestamps (unique frames only)."""

    def __init__(self, window_s: float = 5.0) -> None:
        self.window_s = window_s
        self._ts: deque[float] = deque()

    def tick(self, t: Optional[float] = None) -> None:
        t = t if t is not None else time.monotonic()
        self._ts.append(t)
        while self._ts and self._ts[0] < t - self.window_s:
            self._ts.popleft()

    def fps(self, now: Optional[float] = None) -> float:
        now = now if now is not None else (self._ts[-1] if self._ts else 0.0)
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        if span <= 0:
            return 0.0
        return (len(self._ts) - 1) / span


# --------------------------------------------------------------------- #
# System usage (psutil; lazy-imported so headless CI still works)
# --------------------------------------------------------------------- #
def system_usage() -> dict[str, float]:
    """CPU percent (process + system) and RAM (process + system) in GB.

    Returns zeros (plus ``"available": 0.0``) when psutil is missing or the
    platform refuses a query, so callers can always index the result.
    """
    out: dict[str, float] = {
        "cpu_process": 0.0,
        "cpu_system": 0.0,
        "ram_process_gb": 0.0,
        "ram_system_gb": 0.0,
        "ram_system_total_gb": 0.0,
        "available": 1.0,
    }
    try:
        import psutil

        proc = psutil.Process()
        out["cpu_process"] = proc.cpu_percent(interval=None)
        out["ram_process_gb"] = proc.memory_info().rss / (1024 ** 3)
        vm = psutil.virtual_memory()
        out["ram_system_gb"] = vm.used / (1024 ** 3)
        out["ram_system_total_gb"] = vm.total / (1024 ** 3)
        out["cpu_system"] = psutil.cpu_percent(interval=None)
    except Exception:
        out["available"] = 0.0
    return out


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def metrics_message(src: str, data: dict[str, Any]) -> dict[str, Any]:
    """Uniform envelope for worker -> GUI metrics messages."""
    return {"type": "metrics", "src": src, "t": time.time(), "data": data}


def summarize_learning_progress(history: list[float],
                                epsilon: Optional[float] = None) -> Optional[str]:
    """Plain-language learning trend, emitted every 10 finished episodes.

    # DEEP-FIX: the user repeatedly reported "AI không tiến triển" but had no
    # way to SEE whether it was learning.  Turn the episode-survival history
    # into a visible trend (rising = learning, flat = still exploring or stuck,
    # falling = regression) so progress is observable instead of guessed.
    Returns None until 10 episodes have accumulated.
    """
    n = len(history)
    if n < 10 or n % 10 != 0:
        return None
    recent = sum(history[-10:]) / 10.0
    if n >= 20:
        prev = sum(history[-20:-10]) / 10.0
        delta = recent - prev
        if delta > 0.5:
            arrow = "↑ TIẾN BỘ"
        elif delta < -0.5:
            arrow = "↓ thụt lùi (kiểm tra calibration / vùng chọn)"
        else:
            arrow = "→ chững (có thể vẫn đang khám phá — epsilon còn cao)"
        trend = f" (10 episode trước: {prev:.1f}s) {arrow}"
    else:
        trend = ""
    eps = "" if epsilon is None else f", epsilon={epsilon:.2f}"
    return (f"📈 TIẾN TRÌNH HỌC: {n} episode | survival TB 10 episode gần nhất "
            f"= {recent:.1f}s{trend}{eps}")
