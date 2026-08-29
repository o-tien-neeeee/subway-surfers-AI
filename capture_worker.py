"""Capture worker: mss grabs into the shared frame ring at the target FPS.

Runs as its own PROCESS (spawned on Windows) so screen-grab hiccups can never
block the actor or the GUI.  Bounded transport: the ring has a fixed slot
count; the newest frame wins; stale frames are dropped by construction and
counted (capture-side writes + actor-side gap counts).

Two sources:
* ``mss``     — real screen capture of the calibrated region (default).
* ``fake``    — SyntheticGame rendered in this process; used by headless mode
  and CI so the whole pipeline runs without a display or Chrome.  The fake
  source also consumes the actor's action queue so the synthetic game
  actually reacts to the bot (and honours respawn clicks).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from config import BotConfig
from ipc import SharedFrameRing
from logging_utils import format_exception, get_logger, put_bounded, setup_logging

LOGGER = get_logger("capture")


def _mss_grab_factory(monitor: dict[str, int]):
    import mss

    sct = mss.mss()

    def grab() -> Optional[np.ndarray]:
        raw = sct.grab(monitor)
        arr = np.asarray(raw)[:, :, :3]  # BGRA -> BGR
        rgb = arr[:, :, ::-1]            # -> RGB
        return np.ascontiguousarray(rgb)

    return grab


def _fake_grab_factory(cfg: BotConfig, action_q: Any):
    from environment import SyntheticGame

    game = SyntheticGame(
        width=max(240, min(cfg.region.width or 480, 960)),
        height=max(320, min(cfg.region.height or 800, 1200)),
        fps=cfg.capture.target_fps,
        seed=cfg.seed,
    )
    last_action = {"a": 0}

    def drain_actions() -> None:
        if action_q is None:
            return
        import queue as _q

        drained = 0
        while drained < 8:
            try:
                msg = action_q.get_nowait()
            except _q.Empty:
                break
            except (EOFError, ConnectionError):
                break
            drained += 1
            if msg.get("type") == "action":
                last_action["a"] = int(msg.get("action", 0))
            elif msg.get("type") == "respawn":
                game.respawn()

    def grab() -> np.ndarray:
        drain_actions()
        return game.step(last_action["a"], dt=1.0 / cfg.capture.target_fps)

    return grab


def capture_main(
    stop_event: Any,
    ring: SharedFrameRing,
    cfg_dict: dict,
    metrics_q: Any,
    action_q: Any = None,
    log_dir: str = "logs",
) -> None:
    """Entry point of the capture process."""
    setup_logging("capture", log_dir)
    cfg = BotConfig.from_dict(cfg_dict)
    interval = cfg.capture.frame_interval()
    source = cfg.capture.source
    if source == "auto":
        source = "mss"
    try:
        if source == "mss":
            grab = _mss_grab_factory(cfg.region.to_monitor())
        elif source == "fake":
            grab = _fake_grab_factory(cfg, action_q)
        else:
            raise ValueError(f"unknown capture source {source!r}")
    except Exception as exc:
        put_bounded(metrics_q, {"type": "error", "src": "capture",
                                "error": f"{type(exc).__name__}: {exc}",
                                "tb": format_exception(exc)})
        return

    frame_id = 0
    written_total = 0
    window_written = 0
    window_start = time.monotonic()
    failures = 0
    last_report = time.monotonic()
    report_interval = max(1.0, cfg.perf.report_interval_s)
    LOGGER.info("capture start: %s %dx%d+%d+%d @ %dfps", source,
                cfg.region.width, cfg.region.height, cfg.region.left,
                cfg.region.top, cfg.capture.target_fps)

    try:
        next_t = time.monotonic()
        while not stop_event.is_set():
            t0 = time.monotonic()
            try:
                image = grab()
            except Exception as exc:
                failures += 1
                LOGGER.error("grab failed (#%d): %s", failures, exc)
                if failures >= 5:
                    put_bounded(metrics_q, {
                        "type": "error", "src": "capture",
                        "error": f"capture failing repeatedly: {exc}",
                        "tb": format_exception(exc),
                    })
                    return
                time.sleep(0.1)
                next_t = time.monotonic()
                continue
            failures = 0
            ts = time.monotonic()
            if image is not None and ring.write(image, frame_id, ts):
                written_total += 1
                window_written += 1
                frame_id += 1
            else:
                ring.note_drop()
            # pace to target fps
            next_t += interval
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # running late: don't accumulate debt
            now = time.monotonic()
            if now - last_report >= report_interval:
                span = now - window_start
                put_bounded(metrics_q, {
                    "type": "metrics", "src": "capture", "t": time.time(),
                    "data": {
                        "capture_fps": window_written / max(1e-9, span),
                        "written": written_total,
                        "frame_id": frame_id,
                        "failures": failures,
                        "last_grab_ms": (time.monotonic() - t0) * 1000.0,
                    },
                })
                window_written = 0
                window_start = now
                last_report = now
    except Exception as exc:
        put_bounded(metrics_q, {"type": "error", "src": "capture",
                                "error": f"{type(exc).__name__}: {exc}",
                                "tb": format_exception(exc)})
    finally:
        LOGGER.info("capture stop (frames=%d)", frame_id)
