"""Application entry point and multi-process orchestrator.

Process layout (Windows spawn):

    GUI/main process  (Tkinter, calibration, state machine, metrics display)
      ├─ capture process   (mss or SyntheticGame -> SharedFrameRing)
      ├─ actor process     (BotActor: perception, inference, input, rewards)
      └─ learner process   (Double-DQN + PER + checkpoints)

Headless mode runs the SAME three processes with the fake source and a
dry-run input backend — no Chrome, no display, no real keys — for CI smoke
tests and profiling.

CLI examples:
    python app.py                       # GUI
    python app.py --headless --steps 600
    python app.py --headless --profile-models
    python app.py --validate-demos demos
    python app.py --record-demo         # GUI-based, 3 episodes
    python app.py --evaluate 20 --headless
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Optional

from config import PROFILE_ORDER, BotConfig
from ipc import (
    CTX,
    SharedCounters,
    SharedFrameRing,
    SharedWeights,
    bounded_queue,
    make_events,
)
from logging_utils import drain, format_exception, get_logger, setup_logging
from models import PROFILES, weight_size_for_profile

LOGGER = get_logger("app")


class BotApplication:
    """Owns worker processes and shared state (used by GUI and headless CLI)."""

    def __init__(self, cfg: BotConfig, input_backend: str = "auto",
                 capture_source: Optional[str] = None, log_dir: str = "logs") -> None:
        self.cfg = cfg
        self.input_backend = input_backend
        if capture_source is not None:
            self.cfg.capture.source = capture_source
        self.log_dir = log_dir
        self.paths.ensure_dirs()

        self.events = make_events()
        self.counters = SharedCounters()
        self.counters.set_profile(cfg.rl.profile)
        max_w = weight_size_for_profile(PROFILE_ORDER[-1],
                                        cfg.perception.frame_stack)
        self.shared_weights = SharedWeights(max_w)

        self.ring = self._make_ring()
        self.metrics_q = bounded_queue(512)
        self.transition_q = bounded_queue(128)
        self.cmd_q = bounded_queue(32)
        self.action_q = bounded_queue(64)  # actor -> fake game

        self.capture_proc: Optional[mp.process.BaseProcess] = None
        self.actor_proc: Optional[mp.process.BaseProcess] = None
        self.learner_proc: Optional[mp.process.BaseProcess] = None
        self._started = False

    # ------------------------------------------------------------------ #
    @property
    def paths(self) -> Any:
        """The PathsConfig (``self.cfg.paths``), for directory bootstrap."""
        return self.cfg.paths

    def _make_ring(self) -> SharedFrameRing:
        r = self.cfg.region
        w = max(64, int(r.width or 480))
        h = max(64, int(r.height or 800))
        if w * h > self.cfg.capture.max_region_pixels:
            raise ValueError(
                f"region {w}x{h} exceeds max_region_pixels "
                f"{self.cfg.capture.max_region_pixels}"
            )
        return SharedFrameRing(self.cfg.capture.ring_slots, h, w, 3)

    # ------------------------------------------------------------------ #
    def start(self, with_learner: bool = True, with_capture: bool = True) -> None:
        """Spawn capture, actor and (optionally) learner processes."""
        if self._started:
            return
        self._started = True
        cfg_dict = self.cfg.to_dict()
        if with_capture:
            self.capture_proc = CTX.Process(
                target=self._capture_entry, name="capture-worker",
                args=(self.events["stop"], self.ring, cfg_dict, self.metrics_q,
                      self.action_q, str(self.cfg.paths.logs_dir)),
            )
            self.capture_proc.start()
        self.actor_proc = CTX.Process(
            target=self._actor_entry, name="actor-worker",
            args=(self.cfg, self.ring, self.events, self.transition_q,
                  self.metrics_q, self.shared_weights, self.counters,
                  self.input_backend, self.action_q, self.cfg.seed,
                  str(self.cfg.paths.logs_dir)),
        )
        self.actor_proc.start()
        if with_learner:
            self.learner_proc = CTX.Process(
                target=self._learner_entry, name="learner-worker",
                args=(self.events["stop"], self.events["pause_learning"],
                      self.cmd_q, self.transition_q, self.metrics_q,
                      self.shared_weights, self.counters, cfg_dict,
                      str(self.cfg.paths.checkpoints_dir),
                      str(self.cfg.paths.logs_dir)),
            )
            self.learner_proc.start()
        LOGGER.info("workers started (capture=%s actor=%s learner=%s)",
                    self.capture_proc is not None, True, self.learner_proc is not None)

    # static entry wrappers keep spawn-pickling trivial ------------------ #
    @staticmethod
    def _capture_entry(*args) -> None:
        from capture_worker import capture_main

        capture_main(*args)

    @staticmethod
    def _actor_entry(cfg, ring, events, transition_q, metrics_q, shared_weights,
                     counters, input_backend, action_q, seed, log_dir) -> None:
        from environment import BotActor
        from safety_watchdog import SafetyWatchdog

        setup_logging("actor", log_dir)
        from profiling import set_cpu_threads

        set_cpu_threads(1)
        actor = BotActor(cfg, ring, events, transition_q, metrics_q,
                         shared_weights, counters, input_backend, action_q, seed)
        # Independent safety layer inside the actor process: emergency stop,
        # focus loss, capture stalls and stuck keys — all with authority over
        # the learner's pause event.
        watchdog = SafetyWatchdog(
            events=events,
            input_controller=actor.input,
            counters=counters,
            metrics_q=metrics_q,
            ring=ring,
            interval_s=cfg.perf.watchdog_interval_s,
        )
        watchdog.start()
        try:
            actor.run()
        finally:
            watchdog.stop()
            watchdog.join(timeout=1.0)

    @staticmethod
    def _learner_entry(*args) -> None:
        from learner_worker import learner_main

        learner_main(*args)

    # ------------------------------------------------------------------ #
    def command(self, cmd: str, **kw) -> None:
        from logging_utils import put_bounded

        put_bounded(self.cmd_q, {"cmd": cmd, **kw})

    def pause(self) -> None:
        self.events["pause"].set()

    def resume(self) -> None:
        self.events["pause"].clear()

    def emergency_stop(self) -> None:
        self.events["emergency"].set()
        self.events["stop"].set()

    # ------------------------------------------------------------------ #
    def workers_alive(self) -> dict[str, bool]:
        return {
            "capture": bool(self.capture_proc and self.capture_proc.is_alive()),
            "actor": bool(self.actor_proc and self.actor_proc.is_alive()),
            "learner": bool(self.learner_proc and self.learner_proc.is_alive()),
        }

    def drain_metrics(self, limit: int = 256) -> list[dict]:
        return drain(self.metrics_q, limit)

    # ------------------------------------------------------------------ #
    def shutdown(self, timeout_s: float = 12.0) -> None:
        """Ordered shutdown: stop -> release keys -> save -> join -> exit.

        While joining, every shared queue is drained continuously: a child
        that has queued items cannot exit until its feeder thread flushes
        them, so a full, undrained queue would deadlock the join.
        """
        LOGGER.info("shutdown sequence start")
        self.events["stop"].set()
        deadline = time.monotonic() + timeout_s
        procs = (("capture", self.capture_proc),
                 ("actor", self.actor_proc),
                 ("learner", self.learner_proc))
        for name, proc in procs:
            if proc is None:
                continue
            while proc.is_alive() and time.monotonic() < deadline:
                self._drain_all_queues()
                proc.join(timeout=0.1)
            if proc.is_alive():
                LOGGER.error("%s did not exit in time; terminating", name)
                proc.terminate()
                proc.join(timeout=2.0)
        # final drain so buffered metrics/logs are not lost
        self._drain_all_queues()
        self._started = False
        LOGGER.info("shutdown complete")

    def _drain_all_queues(self) -> None:
        for q in (self.metrics_q, self.transition_q, self.cmd_q, self.action_q):
            drain(q, limit=256)

    # ------------------------------------------------------------------ #
    # Headless run (CI smoke test / profiling)
    # ------------------------------------------------------------------ #
    def headless_run(self, max_env_frames: int = 600, learn: bool = True,
                     report_path: Optional[str] = None) -> dict[str, Any]:
        """Run the full pipeline against the synthetic game; return summary."""
        from evaluation import EpisodeRecord, EvaluationReport

        self.cfg.rl.warmup_transitions = min(self.cfg.rl.warmup_transitions, 120)
        self.start(with_learner=learn)
        report = EvaluationReport()
        t0 = time.monotonic()
        last_frame = 0
        stall_ticks = 0
        errors: list[str] = []
        episodes: list[dict] = []
        try:
            while time.monotonic() - t0 < min(600, max(60, max_env_frames / 5.0)):
                for msg in self.drain_metrics(256):
                    if msg.get("type") == "error":
                        errors.append(f"{msg.get('src')}: {msg.get('error')}")
                    elif msg.get("type") == "episode_end":
                        episodes.append(msg["data"])
                        report.add(EpisodeRecord(kind="train", **msg["data"]))
                    elif msg.get("type") == "pretrain_done":
                        LOGGER.info("pretrain: %s", msg.get("result", {}).get("status"))
                fid = self.ring.latest_frame_id()
                if learn and self.counters.learner_update_step.value > 0 and \
                        fid >= max_env_frames:
                    break
                if not learn and fid >= max_env_frames:
                    break
                if fid == last_frame:
                    stall_ticks += 1
                    if stall_ticks > 300:  # ~60s without any frame
                        errors.append("capture stalled in headless run")
                        break
                else:
                    stall_ticks = 0
                    last_frame = fid
                time.sleep(0.2)
        finally:
            self.shutdown()
        summary = {
            "env_frames": int(self.counters.env_frame_id.value),
            "learner_updates": int(self.counters.learner_update_step.value),
            "episodes": episodes,
            "errors": errors,
            "workers_exited_cleanly": all(not v for v in self.workers_alive().values()),
            "held_keys_after_shutdown": [],
        }
        if report_path:
            report.save(report_path)
        return summary


# --------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------- #
def cmd_validate_demos(demos_dir: str) -> int:
    from dataset import summarize_reports, validate_directory

    eps, reps = validate_directory(demos_dir)
    print(summarize_reports(reps))
    ok = sum(1 for r in reps if r.ok)
    print(f"\n{ok}/{len(reps)} episodes valid for behaviour cloning")
    return 0 if ok == len(reps) and reps else 1


def synthetic_calibration(cfg: BotConfig) -> BotConfig:
    """Point the config at the synthetic game's built-in calibration.

    Headless mode has no user to run the wizard, so region/anchor/respawn are
    set to the SyntheticGame's known geometry (documented in environment.py).
    """
    from environment import SyntheticGame

    w, h = 480, 800
    cfg.region.left, cfg.region.top = 0, 0
    cfg.region.width, cfg.region.height = w, h
    cfg.death.anchor_fx = 30 / w
    cfg.death.anchor_fy = 30 / h
    cfg.death.anchor_baseline_rgb = SyntheticGame.ALIVE_ANCHOR
    cfg.death.anchor_baseline_std = 2.0
    cfg.input.respawn_fx, cfg.input.respawn_fy = 0.5, 0.6
    return cfg


def cmd_headless(args: argparse.Namespace) -> int:
    cfg = BotConfig.load(args.config) if args.config else BotConfig()
    if args.dry_run:
        cfg.input.dry_run = True
    cfg.capture.source = "fake"  # headless never touches a real display
    synthetic_calibration(cfg)
    cfg.rl.warmup_transitions = min(cfg.rl.warmup_transitions, 150)
    app = BotApplication(cfg, input_backend="dry_run", log_dir=args.log_dir)
    t0 = time.monotonic()
    summary = app.headless_run(max_env_frames=args.steps or 600,
                               learn=not args.no_learn,
                               report_path=args.report)
    dt = time.monotonic() - t0
    print(json_dumps(summary))
    print(f"wall time: {dt:.1f}s")
    ok = summary["workers_exited_cleanly"] and not summary["errors"]
    print("HEADLESS SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def json_dumps(obj: Any) -> str:
    import json

    def default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    return json.dumps(obj, indent=2, default=default)


def cmd_pretrain_headless(args: argparse.Namespace) -> int:
    """BC pretraining directly in-process (no game needed)."""
    from learner_worker import Learner

    cfg = BotConfig.load(args.config) if args.config else BotConfig()
    Path(cfg.paths.root if cfg.paths.root else ".").mkdir(exist_ok=True)
    cfg.paths.ensure_dirs()
    max_w = weight_size_for_profile(PROFILE_ORDER[-1], cfg.perception.frame_stack)
    weights = SharedWeights(max_w)
    counters = SharedCounters()
    metrics_q = bounded_queue(256)
    learner = Learner(cfg, weights, counters, metrics_q, str(cfg.paths.checkpoints_dir))
    result = learner.pretrain(args.pretrain)
    print(json_dumps(result))
    return 0 if result.get("status") == "ok" else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.py", description="Subway Surfers research bot (screen-capture only)"
    )
    parser.add_argument("--config", default="", help="path to config.json")
    parser.add_argument("--headless", action="store_true",
                        help="no GUI/Chrome/keys: synthetic game end-to-end")
    parser.add_argument("--steps", type=int, default=600,
                        help="headless env-frame budget")
    parser.add_argument("--no-learn", action="store_true",
                        help="headless: run pipeline without learner training")
    parser.add_argument("--dry-run", action="store_true",
                        help="force dry-run input backend (no real keys)")
    parser.add_argument("--source", default="", help="capture source override")
    parser.add_argument("--profile-models", action="store_true",
                        help="print model profile measurements and exit")
    parser.add_argument("--validate-demos", default="",
                        help="validate a demos directory and exit")
    parser.add_argument("--pretrain", default="",
                        help="run behaviour cloning on a demos dir (headless)")
    parser.add_argument("--evaluate", type=int, default=0,
                        help="run N headless evaluation episodes")
    parser.add_argument("--compare-baseline", default="",
                        help="with --evaluate: import a saved report's records "
                             "as the human-baseline comparison set")
    parser.add_argument("--record-demo", action="store_true",
                        help="start GUI in demo-recording mode")
    parser.add_argument("--report", default="runs/headless_report.json")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args(argv)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as exc:  # already set (e.g. by tests) — spawn kept
        LOGGER.debug("start method already set: %s", exc)

    if args.profile_models:
        from profiling import format_profile_table, profile_all_profiles

        print(format_profile_table(profile_all_profiles()))
        return 0
    if args.validate_demos:
        return cmd_validate_demos(args.validate_demos)
    if args.pretrain:
        return cmd_pretrain_headless(args)
    if args.headless:
        return cmd_headless(args)
    if args.evaluate:
        from evaluation_tool import run_evaluation

        return run_evaluation(args)
    # GUI (default) — Ctrl+C in the console must also shut down cleanly
    from gui import run_gui

    try:
        return run_gui(args)
    except KeyboardInterrupt:
        LOGGER.warning("Ctrl+C received — exiting")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
