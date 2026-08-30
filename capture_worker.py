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
from typing import Any

import numpy as np

from config import BotConfig
from ipc import SharedFrameRing
from logging_utils import format_exception, get_logger, put_bounded, setup_logging

LOGGER = get_logger("capture")


def _mss_grab_factory(monitor: dict[str, int]):
    import mss

    sct = mss.mss()

    def grab() -> np.ndarray | None:
        raw = sct.grab(monitor)
        arr = np.asarray(raw)[:, :, :3]  # BGRA -> BGR
        rgb = arr[:, :, ::-1]            # -> RGB
        return np.ascontiguousarray(rgb)

    return grab


def virtual_screen_geometry() -> tuple[int, int] | None:
    """(width, height) of the virtual screen, or None when unavailable."""
    try:
        import mss

        with mss.mss() as sct:
            mon0 = sct.monitors[0]
            return int(mon0["width"]), int(mon0["height"])
    except Exception:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
        return None


def geometry_matches(region: Any, actual_w: int | None,
                     actual_h: int | None) -> bool | None:
    """Compare the current virtual screen against the calibrated one.

    Returns True (same), False (changed — recalibration advised), or None
    (cannot verify: no stored reference or no display).  Pure function so it
    is unit-testable without a display.
    """
    if actual_w is None or actual_h is None:
        return None
    ref_w = int(getattr(region, "screen_width", 0) or 0)
    ref_h = int(getattr(region, "screen_height", 0) or 0)
    if ref_w <= 0 or ref_h <= 0:
        return None
    return actual_w == ref_w and actual_h == ref_h


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
    except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
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
    # geometry-drift detection (window move / resolution / DPI change):
    # compare the virtual screen against the calibrated reference every 5 s
    # and warn exactly once per change so the user can re-calibrate.
    geometry_check_interval = 5.0
    next_geometry_check = time.monotonic() + geometry_check_interval
    last_geometry_state: bool | None = None
    LOGGER.info("capture start: %s %dx%d+%d+%d @ %dfps", source,
                cfg.region.width, cfg.region.height, cfg.region.left,
                cfg.region.top, cfg.capture.target_fps)

    try:
        next_t = time.monotonic()
        while not stop_event.is_set():
            t0 = time.monotonic()
            try:
                image = grab()
            except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
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
            if source != "fake" and now >= next_geometry_check:
                next_geometry_check = now + geometry_check_interval
                geo = virtual_screen_geometry()
                state = geometry_matches(cfg.region, geo[0] if geo else None,
                                         geo[1] if geo else None)
                if state is not None and state != last_geometry_state:
                    last_geometry_state = state
                    if state is False:
                        put_bounded(metrics_q, {
                            "type": "log", "level": "warning", "src": "capture",
                            "msg": (f"screen geometry changed: calibrated "
                                    f"{cfg.region.screen_width}x"
                                    f"{cfg.region.screen_height}, now "
                                    f"{geo[0]}x{geo[1]} — re-run calibration "
                                    f"(Step 1) or the region may be wrong"),
                        })
                    else:
                        put_bounded(metrics_q, {
                            "type": "log", "level": "info", "src": "capture",
                            "msg": "screen geometry matches calibration again",
                        })
    except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
        put_bounded(metrics_q, {"type": "error", "src": "capture",
                                "error": f"{type(exc).__name__}: {exc}",
                                "tb": format_exception(exc)})
    finally:
        LOGGER.info("capture stop (frames=%d)", frame_id)
