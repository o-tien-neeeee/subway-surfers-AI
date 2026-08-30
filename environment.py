"""Game environment: the actor loop plus a synthetic game for headless runs.

Two usage modes share all perception/reward/safety logic:

1. **Live/headless multi-process mode** — ``BotActor.run()`` executes in the
   actor process: it reads the latest frame from the shared ring, preprocesses
   zones, runs the horizon + death detectors, asks the scheduler whether to
   decide, runs local-model inference, presses keys through the safe
   InputController, computes rewards, and ships n-step transitions to the
   learner.  It also owns the death -> respawn state machine and the
   automatic performance downgrader.

2. **Synchronous mode** — ``GameEnvironment`` is a classic reset/step env over
   ``SyntheticGame`` used by headless training, the pipeline smoke test and
   evaluation tooling (no processes, no real input).

``SyntheticGame`` renders a 3-lane endless runner: obstacles spawn at the
horizon and approach the player; lane/blockade obstacles are dodged with
LEFT/RIGHT, low barriers with JUMP, high barriers with SLIDE; a miss triggers
a death (the UI anchor patch changes colour exactly like a game-over overlay)
and only a respawn click revives it.  This is honest scaffolding for CI: it
exercises the full pipeline, and it is NOT the real Poki game.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from action_scheduler import ActionScheduler, PlannedAction
from agent import InferencePolicy, epsilon_for_frame
from config import ACTIONS, BotConfig, NOOP
from death_detector import ColorAnchorDeathDetector, DeathResult, DeathState, RespawnController
from horizon_detector import HorizonDetector, HorizonResult
from input_controller import InputController
from ipc import SharedFrameRing, SharedWeights
from logging_utils import get_logger, put_bounded
from metrics import FpsMeter, LatencyMeter
from models import DuelingDQN
from perception import FrameStack, ZonePreprocessor
from replay_buffer import NStepBuilder
from rewards import SurvivalRewardCalculator

LOGGER = get_logger("actor")


# --------------------------------------------------------------------- #
# Synthetic game (fake capture source)
# --------------------------------------------------------------------- #
class SyntheticGame:
    """Deterministic-ish 3-lane runner rendered with numpy (no real game)."""

    ALIVE_ANCHOR = (206, 66, 66)
    DEAD_ANCHOR = (52, 52, 200)

    def __init__(self, width: int = 480, height: int = 800, fps: int = 30,
                 seed: int = 0, death_frames: int = 12) -> None:
        self.w = width
        self.h = height
        self.fps = fps
        self.rng = np.random.default_rng(seed)
        self.frame_id = 0
        self.player_lane = 1
        self.lanes = 3
        self.obstacles: list[dict[str, float]] = []
        self.spawn_cooldown = 0.6
        self._spawn_timer = 0.0
        self._t = 0.0
        self.dead = False
        self.death_frames = death_frames
        self._dead_for = 0
        self._pending_death = False
        self.total_steps = 0
        self.score = 0
        self._last_jump_t = -1e9
        self._last_slide_t = -1e9
        # "lane" obstacles force a lane change; "low"/"high" barriers sit in
        # the player's lane and are cleared by JUMP/SLIDE respectively.
        self._dodge_window = 0.0
        self._dodge_ok = False

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        self.player_lane = 1
        self.obstacles.clear()
        self.dead = False
        self._dead_for = 0
        self._pending_death = False
        self.score = 0
        self.total_steps = 0
        return self.render()

    # ------------------------------------------------------------------ #
    def step(self, action: int, dt: Optional[float] = None) -> np.ndarray:
        """Advance the game one tick with the player's action."""
        dt = dt if dt is not None else 1.0 / self.fps
        self._t += dt
        self.total_steps += 1
        if not self.dead:
            self.score += dt * 10.0
            if action == 1:      # LEFT
                self.player_lane = max(0, self.player_lane - 1)
            elif action == 2:    # RIGHT
                self.player_lane = min(self.lanes - 1, self.player_lane + 1)
            elif action == 3:    # JUMP (grace window below)
                self._last_jump_t = self._t
            elif action == 4:    # SLIDE
                self._last_slide_t = self._t
            self._spawn_timer -= dt
            if self._spawn_timer <= 0:
                self.spawn_cooldown = max(0.35, self.spawn_cooldown * 0.995)
                self._spawn_timer = self.spawn_cooldown
                kind = str(self.rng.choice(["lane", "lane", "low", "high"]))
                lane = int(self.rng.integers(0, self.lanes))
                self.obstacles.append({"kind": kind, "lane": lane, "prog": 0.0,
                                       "speed": float(self.rng.uniform(0.45, 0.75))})
            survived: list[dict[str, float]] = []
            for ob in self.obstacles:
                ob["prog"] += ob["speed"] * dt
                if ob["prog"] < 0.97:
                    survived.append(ob)
                    continue
                # collision row: in our lane and not dodged -> death
                if ob["lane"] == self.player_lane:
                    dodged = False
                    if ob["kind"] == "low":
                        dodged = (action == 3) or (self._t - self._last_jump_t) < 0.35
                    elif ob["kind"] == "high":
                        dodged = (action == 4) or (self._t - self._last_slide_t) < 0.35
                    if not dodged:
                        self._pending_death = True
            self.obstacles = survived
        else:
            self._dead_for -= 1
            if self._dead_for <= 0:
                # Anchor stays dead until respawn() — mirrors the real
                # game-over screen requiring a click.
                pass
        if self._pending_death and not self.dead:
            self.dead = True
            self._dead_for = 10 ** 9
        return self.render()

    def respawn(self) -> None:
        """Simulates clicking the restart button."""
        self.dead = False
        self._pending_death = False
        self.player_lane = 1
        self.obstacles.clear()
        self._spawn_timer = 0.2

    # ------------------------------------------------------------------ #
    def render(self) -> np.ndarray:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        frame[:, :, :] = (28, 30, 40)                     # asphalt
        lane_w = self.w / self.lanes
        for i in range(1, self.lanes):
            x = int(i * lane_w)
            frame[:, x - 2 : x + 2, :] = 70               # dividers
        # scrolling ground stripes (visual motion for the ground zone)
        off = int((self._t * 240) % 80)
        for y in range(off, self.h, 80):
            frame[y : y + 3, :, :] = (46, 50, 66)
        # obstacles: appear at horizon (small) and grow toward the player
        horizon = int(self.h * 0.25)
        for ob in self.obstacles:
            lane_x = int((ob["lane"] + 0.5) * lane_w)
            size = int(8 + 46 * ob["prog"])
            y = int(horizon + (self.h - horizon - 60) * ob["prog"])
            col = {"lane": (200, 60, 60), "low": (230, 160, 40),
                   "high": (90, 170, 240)}[ob["kind"]]
            frame[y : y + size, lane_x - size // 2 : lane_x + size // 2, :] = col
        # player marker near the bottom
        px = int((self.player_lane + 0.5) * lane_w)
        frame[self.h - 52 : self.h - 14, px - 18 : px + 18, :] = (240, 200, 60)
        # UI anchor patch (top-left) — the death signal
        anchor = self.DEAD_ANCHOR if self.dead else self.ALIVE_ANCHOR
        frame[28:33, 28:33, :] = anchor
        frame[0:24, 0:self.w, :] = (16, 18, 26)           # top bar
        self.frame_id += 1
        return frame


# --------------------------------------------------------------------- #
# Synchronous environment (headless training / tests)
# --------------------------------------------------------------------- #
@dataclass
class StepInfo:
    horizon: HorizonResult
    death: DeathResult
    reward_breakdown: dict[str, float]
    action_step: int


class GameEnvironment:
    """Classic reset/step API over SyntheticGame with full bot perception."""

    def __init__(self, cfg: BotConfig, game: Optional[SyntheticGame] = None) -> None:
        self.cfg = cfg
        self.game = game or SyntheticGame(seed=cfg.seed)
        self.pre = ZonePreprocessor(cfg.perception, cfg.perception.horizon_frac,
                                    anchor_xy=(30, 30))
        self.detector = HorizonDetector(cfg.horizon, cfg.perception.horizon_size)
        death_cfg = self._death_cfg_with_anchor()
        self.death = ColorAnchorDeathDetector(death_cfg)
        self.reward_calc = SurvivalRewardCalculator(cfg.reward, now=lambda: self._t)
        self.stack = FrameStack(cfg.perception.frame_stack, cfg.perception.ground_size)
        self.nstep = NStepBuilder(cfg.rl.n_step, cfg.rl.gamma)
        self._t = 0.0
        self._env_ids: deque[int] = deque(maxlen=4)
        self._last_action = NOOP
        self._prev_action = NOOP
        self.steps = 0
        self.episode = 0

    def _death_cfg_with_anchor(self):
        import copy

        d = copy.copy(self.cfg.death)
        # Synthetic anchor lives at pixel (30,30) and its ALIVE colour is known.
        if not d.anchor_set():
            d.anchor_fx, d.anchor_fy = 30 / self.game.w, 30 / self.game.h
            d.anchor_baseline_rgb = SyntheticGame.ALIVE_ANCHOR
            d.anchor_baseline_std = 2.0
        return d

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        self.game.reset()
        self.detector.reset()
        self.reward_calc.begin_episode(self._t)
        self.nstep.clear()
        self._env_ids.clear()
        self._last_action = NOOP
        self.steps = 0
        self.episode += 1
        return self._ingest(reset=True)

    def _ingest(self, reset: bool = False) -> np.ndarray:
        frame = self.game.render()
        ts = self._t
        z = self.pre.process(frame, self.game.frame_id, ts)
        hz = self.detector.update(z.horizon_gray, z.frame_id, ts)
        dr = self.death.update(z.anchor_patch, z.frame_id, ts)
        fid = self.game.frame_id
        if reset:
            self.stack.reset(z.ground_gray)
            self._env_ids = deque([fid] * 4, maxlen=4)
        else:
            self.stack.push(z.ground_gray)
            self._env_ids.append(fid)
        return self.stack.get()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        self._t += 1.0 / self.cfg.capture.target_fps
        ts = self._t
        # 1. act on the synthetic game
        self.game.step(int(action))
        # 2. ingest the resulting frame (perception + detectors)
        frame = self.game.render()
        z = self.pre.process(frame, self.game.frame_id, ts)
        hz = self.detector.update(z.horizon_gray, z.frame_id, ts)
        dr = self.death.update(z.anchor_patch, z.frame_id, ts)
        died = dr.state is DeathState.DEAD_CONFIRMED
        self.stack.push(z.ground_gray)
        self._env_ids.append(self.game.frame_id)
        obs = self.stack.get()
        # 3. reward for this frame given the action just taken
        breakdown = self.reward_calc.step(
            ts=ts, action=int(action), horizon=hz,
            ground_gray=z.ground_gray, died=died,
        )
        self.steps += 1
        info = StepInfo(horizon=hz, death=dr,
                        reward_breakdown=breakdown.to_dict(), action_step=self.steps)
        return obs, breakdown.total, died, info

    def pop_nstep_transitions(self, obs: np.ndarray, action: int, reward: float,
                              done: bool) -> list:
        env_ids = tuple(self._env_ids)
        return self.nstep.push(obs, env_ids, action, reward, done)

    @property
    def observation(self) -> np.ndarray:
        return self.stack.get()

    def respawn(self) -> None:
        self.game.respawn()
        self.detector.reset()
        # DEEP-FIX: this only reset the game and the death detector.  The
        # reward calculator kept _episode_dead=True, so the -10 death penalty
        # was applied on the FIRST death of the process and never again
        # (verified: death #1 -> -10.0, death #2 -> 0.0).  Every later
        # episode in a headless training/eval run was therefore trained
        # without a death signal.  The n-step window and the frame stack are
        # episode-scoped too and must not leak across the boundary.
        self.episode += 1
        self.steps = 0
        self.reward_calc.begin_episode(self._t)
        self.nstep.clear()
        self._env_ids.clear()
        self._last_action = NOOP


# --------------------------------------------------------------------- #
# Live actor (multi-process mode)
# --------------------------------------------------------------------- #
@dataclass
class ActorEpisodeStats:
    episode_id: int
    survival_s: float
    total_reward: float
    steps: int
    env_frames: int
    fps: float
    action_latency_p95_ms: float
    inference_p95_ms: float
    score: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, float]:
        return {
            "episode_id": self.episode_id, "survival_s": round(self.survival_s, 3),
            "total_reward": round(self.total_reward, 3), "steps": self.steps,
            "env_frames": self.env_frames, "fps": round(self.fps, 2),
            "action_latency_p95_ms": round(self.action_latency_p95_ms, 2),
            "inference_p95_ms": round(self.inference_p95_ms, 2),
            "score": self.score,
        }


class BotActor:
    """The full gameplay loop; runs inside the actor process."""

    def __init__(
        self,
        cfg: BotConfig,
        ring: SharedFrameRing,
        events: dict,
        transition_q: Any,
        metrics_q: Any,
        shared_weights: SharedWeights,
        counters: Any,
        input_backend: str = "auto",
        action_out_q: Any = None,
        seed: int = 0,
        cmd_q: Any = None,
    ) -> None:
        self.cfg = cfg
        self.ring = ring
        self.events = events
        self.transition_q = transition_q
        self.metrics_q = metrics_q
        self.shared_weights = shared_weights
        self.counters = counters
        self.action_out_q = action_out_q  # fake-game action channel (headless)
        # DEEP-FIX: the learner's command queue.  Without it the actor had no
        # way to tell the learner anything (see _downgrade).
        self.cmd_q = cmd_q

        anchor_xy = None
        if cfg.death.anchor_set():
            anchor_xy = (
                int(cfg.death.anchor_fx * cfg.region.width),
                int(cfg.death.anchor_fy * cfg.region.height),
            )
        self.pre = ZonePreprocessor(cfg.perception, cfg.perception.horizon_frac,
                                    anchor_xy,
                                    require_anchor=cfg.death.anchor_set())
        self.horizon = HorizonDetector(cfg.horizon, cfg.perception.horizon_size)
        self.death = ColorAnchorDeathDetector(cfg.death) if cfg.death.anchor_set() else None
        self.reward_calc = SurvivalRewardCalculator(cfg.reward)
        self.stack = FrameStack(cfg.perception.frame_stack, cfg.perception.ground_size)
        self.scheduler = ActionScheduler(cfg.scheduler,
                                          cooldown_ms=cfg.input.cooldown_ms)
        self.scheduler.ttl_ms = float(cfg.input.action_ttl_ms)
        self.input = InputController(cfg.input, backend=input_backend)
        model = DuelingDQN.from_profile(cfg.rl.profile,
                                        cfg.perception.frame_stack,
                                        cfg.perception.ground_size)
        self.policy = InferencePolicy(model, seed=seed)
        self.nstep = NStepBuilder(cfg.rl.n_step, cfg.rl.gamma)

        respawn_abs = None
        if cfg.input.respawn_set():
            respawn_abs = (
                cfg.region.left + int(cfg.input.respawn_fx * cfg.region.width),
                cfg.region.top + int(cfg.input.respawn_fy * cfg.region.height),
            )
        self.respawn_abs = respawn_abs
        self.respawn_ctl = RespawnController(self._respawn_click, cfg.death)

        # metrics
        self.fps_meter = FpsMeter()
        self.infer_ms = LatencyMeter(name="inference")
        self.action_ms = LatencyMeter(name="action")
        self._metrics_last = 0.0
        self._last_report = time.monotonic()
        self._last_processed_id = -1
        self._dropped_total = 0
        self._duplicate_frames = 0
        self._env_ids: deque[int] = deque(maxlen=cfg.perception.frame_stack)
        self._episode_reward = 0.0
        # DEEP-FIX: this was a float sentinel (0.0) meaning "no episode in
        # progress", so a legitimate frame timestamp of exactly 0.0 would
        # make the actor believe no episode had started and _end_episode()
        # would return without publishing anything.  None cannot collide.
        self._episode_start_ts: Optional[float] = None
        self._episode_frames = 0
        self._last_load_check = time.monotonic()
        self._perf_violation_since: Optional[float] = None
        self._profile = cfg.rl.profile
        self._action_step_at_episode_start = self.scheduler.action_step
        self._episode_stats: list[ActorEpisodeStats] = []
        self._shutdown_requested = False

    # ------------------------------------------------------------------ #
    def _put_metrics(self, data: dict[str, Any]) -> None:
        from metrics import metrics_message

        if "type" in data:
            # already an envelope (episode_end / error / watchdog ...)
            put_bounded(self.metrics_q, data)
        else:
            put_bounded(self.metrics_q, metrics_message("actor", data))

    def _put_log(self, level: str, msg: str) -> None:
        put_bounded(self.metrics_q, {"type": "log", "level": level, "src": "actor",
                                      "msg": msg})

    def _respawn_click(self, t: float) -> bool:
        if self.respawn_abs is None:
            self._put_log("warning", "respawn point not calibrated; cannot click")
            return False
        x, y = self.respawn_abs
        ok = self.input.click(x, y, confirm_focus=True)
        if ok and self.action_out_q is not None:
            put_bounded(self.action_out_q, {"type": "respawn"})
        return ok

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        LOGGER.info("actor loop start (profile=%s)", self.cfg.rl.profile)
        self._put_log("info", f"actor started, profile={self.cfg.rl.profile}")
        try:
            while not self._shutdown_requested and not self.events["stop"].is_set():
                if self.events["emergency"].is_set():
                    break
                if self.events["pause"].is_set():
                    self._paused_spin()
                    continue
                frame = self.ring.read_latest()
                if frame is None or frame.frame_id <= self._last_processed_id:
                    time.sleep(0.002)
                    continue
                if self._last_processed_id >= 0 and frame.frame_id > self._last_processed_id + 1:
                    self._dropped_total += frame.frame_id - self._last_processed_id - 1
                self._last_processed_id = frame.frame_id
                self._process_frame(frame)
                self._maybe_report()
            # graceful exit
            self._end_episode(reason="shutdown")
        except Exception as exc:  # full traceback to GUI + log
            from logging_utils import format_exception

            LOGGER.error("actor crashed:\n%s", format_exception(exc))
            self._put_metrics({"type": "error", "src": "actor",
                               "error": f"{type(exc).__name__}: {exc}",
                               "tb": format_exception(exc)})
        finally:
            self.input.release_all("actor exit")
            self.input.dispose()
            self._flush_transitions(final=True)
            LOGGER.info("actor loop end")

    def _paused_spin(self) -> None:
        self.input.release_all("paused")
        self.scheduler.interrupt()
        time.sleep(0.05)
        self._maybe_report()

    # ------------------------------------------------------------------ #
    def _process_frame(self, frame) -> None:
        t_frame = frame.ts
        z = self.pre.process(frame.image, frame.frame_id, t_frame)
        self.counters.env_frame_id.value = frame.frame_id

        if not z.valid or z.ground_gray is None:
            self._put_log("warning", f"invalid frame {frame.frame_id}: {z.reason}")
            time.sleep(0.002)
            return

        hz = self.horizon.update(z.horizon_gray, z.frame_id, t_frame)
        dr = (self.death.update(z.anchor_patch, z.frame_id, t_frame)
              if self.death is not None else None)
        if dr is not None and dr.state is DeathState.DEAD_CONFIRMED:
            self._handle_death(z, hz, t_frame)
            return

        # fresh episode: seed the stack with this frame (no stale pixels)
        if self._episode_start_ts is None:
            self.stack.reset(z.ground_gray)
            self._env_ids = deque([z.frame_id] * self.cfg.perception.frame_stack,
                                  maxlen=self.cfg.perception.frame_stack)
            self._begin_episode(t_frame)
            self.fps_meter.tick(t_frame)
            return

        # live frame: update observation stack
        self.stack.push(z.ground_gray)
        self._env_ids.append(frame.frame_id)
        self.fps_meter.tick(t_frame)
        self._episode_frames += 1

        danger = hz.detected and hz.confidence >= 0.5
        self.counters.danger_flag.value = 1 if danger else 0
        # §9: cadence adapts to CPU load AND horizon confidence together.
        self.scheduler.set_signal(self._cpu_load_estimate(), hz.confidence)
        if self.scheduler.on_frame(danger=danger):
            self._decide(danger)

        planned = self.scheduler.pop_executable()
        if planned is not None:
            self._execute(planned, t_frame)
        # keep the shared action_step counter authoritative for the GUI
        self.counters.action_step.value = self.scheduler.action_step

        reward = self.reward_calc.step(
            ts=t_frame,
            action=self._last_action,
            horizon=hz,
            ground_gray=z.ground_gray,
            prev_ground_gray=self._prev_ground,
            died=False,
        )
        self._prev_ground = z.ground_gray
        self._episode_reward += reward.total
        self._ship_transition(action=self._last_action, reward=reward.total,
                              done=False, ts=t_frame)
        self._maybe_perf_check()

    _prev_ground: Any = None
    _last_action: int = NOOP

    # ------------------------------------------------------------------ #
    def _decide(self, danger: bool) -> None:
        t0 = time.perf_counter()
        self.policy.refresh_weights(self.shared_weights)
        eps = epsilon_for_frame(int(self.counters.env_frame_id.value), self.cfg.rl)
        self.counters.epsilon.value = eps
        action = self.policy.act(self.stack.get(), eps)
        self.infer_ms.observe_ms((time.perf_counter() - t0) * 1000.0)
        self.scheduler.submit(action, danger=danger)

    def _execute(self, planned: PlannedAction, t_frame: float) -> None:
        if not self.input.focus_gate():
            self.scheduler.interrupt()
            return
        # DEEP-FIX: t0 was time.perf_counter() while t_frame is a
        # time.monotonic() timestamp written by the CAPTURE process, and the
        # two were subtracted from each other.  On Linux both clocks share an
        # origin so it happened to work; on Windows (the target platform)
        # perf_counter is QueryPerformanceCounter and monotonic is
        # GetTickCount64, so `t0 - t_frame` is an arbitrary offset -- feeding
        # a p95 that then drives the automatic profile downgrader.  Worse,
        # LatencyMeter.observe_ms() silently discards negative samples, so a
        # negative offset did not just bias the number, it deleted most of
        # the distribution.  One clock, and the queueing term is clamped.
        t0 = time.monotonic()
        ev = self.input.press_action(planned.action, created_ts=planned.created_ts)
        press_ms = (time.monotonic() - t0) * 1000.0
        queue_ms = max(0.0, (t0 - t_frame) * 1000.0)
        self.action_ms.observe_ms(press_ms + queue_ms)
        if ev.pressed or planned.action == NOOP:
            self._last_action = planned.action
            if self.action_out_q is not None:
                put_bounded(self.action_out_q,
                                 {"type": "action", "action": int(planned.action)})

    def _ship_transition(self, action: int, reward: float, done: bool, ts: float) -> None:
        env_ids = tuple(self._env_ids)
        for tr in self.nstep.push(self.stack.get(), env_ids, action, reward, done):
            put_bounded(self.transition_q, tr)

    def _flush_transitions(self, final: bool) -> None:
        """Flush the n-step window on shutdown.

        # DEEP-FIX: two problems here.  (a) ``self.stack.get()`` raises
        RuntimeError when the stack was never seeded (a shutdown before the
        # first valid frame), which turned a clean exit into an actor crash
        # report.  (b) An empty ``_env_ids`` produced a 0-length env-id tuple
        # for a 4-frame stack, which the replay buffer only rejected later,
        # at sample time.
        """
        if not self.nstep.pending:
            return
        if len(self._env_ids) < self.cfg.perception.frame_stack:
            LOGGER.warning(
                "discarding %d pending n-step step(s): the observation stack "
                "was never seeded (shutdown before the first valid frame)",
                self.nstep.pending)
            self.nstep.clear()
            return
        try:
            obs = self.stack.get()
        except RuntimeError as exc:
            LOGGER.warning("cannot flush transitions (%s)", exc)
            self.nstep.clear()
            return
        while self.nstep.pending:
            for tr in self.nstep.push(obs, tuple(self._env_ids), NOOP, 0.0, True):
                put_bounded(self.transition_q, tr)

    # ------------------------------------------------------------------ #
    def _begin_episode(self, t_frame: float) -> None:
        with self.counters.episode_id.get_lock():
            self.counters.episode_id.value += 1
        self._episode_start_ts = t_frame
        self._episode_reward = 0.0
        self._episode_frames = 0
        self.reward_calc.begin_episode(t_frame)
        self.scheduler.reset_episode()
        self.nstep.clear()
        self._prev_ground = None
        self._episode_stats_frames0 = self.counters.env_frame_id.value

    def _end_episode(self, reason: str) -> None:
        if self._episode_start_ts is None:
            return
        # _episode_start_ts is a capture-process time.monotonic() value; the
        # monotonic clock is system-wide on both Linux and Windows so the
        # subtraction is meaningful across processes, but it is clamped so a
        # clock adjustment cannot publish a negative survival time.
        survival = max(0.0, time.monotonic() - self._episode_start_ts)
        stats = ActorEpisodeStats(
            episode_id=int(self.counters.episode_id.value),
            survival_s=survival,
            total_reward=self._episode_reward,
            steps=self.scheduler.action_step - self._action_step_at_episode_start,
            env_frames=self._episode_frames,
            fps=self.fps_meter.fps(),
            action_latency_p95_ms=self.action_ms.snapshot()["p95"],
            inference_p95_ms=self.infer_ms.snapshot()["p95"],
        )
        self._episode_stats.append(stats)
        self._put_metrics({"type": "episode_end", "reason": reason,
                           "data": stats.to_dict()})
        LOGGER.info("episode %d ended (%s): %.1fs reward=%.2f",
                    stats.episode_id, reason, stats.survival_s, stats.total_reward)
        # Publish for the learner's online best-model gate (single writer:
        # actor).  DEEP-FIX: go through the ordered helper -- payload first,
        # id last -- instead of writing the id first and letting the learner
        # pair a new id with the previous episode's numbers.
        if hasattr(self.counters, "publish_episode_result"):
            self.counters.publish_episode_result(
                stats.episode_id, stats.survival_s, stats.total_reward)
        else:  # pragma: no cover - test doubles without the helper
            self.counters.last_episode_survival_s.value = float(stats.survival_s)
            self.counters.last_episode_reward.value = float(stats.total_reward)
            self.counters.last_episode_done_id.value = stats.episode_id
        self._episode_start_ts = None
        self._action_step_at_episode_start = self.scheduler.action_step

    # ------------------------------------------------------------------ #
    # Death / respawn machine (requirement §7)
    # ------------------------------------------------------------------ #
    def _handle_death(self, z, hz, t_frame: float) -> None:
        # 1-4: pause learner, release keys, drop stale commands
        self.events["pause_learning"].set()
        self.events["death"].set()
        self.counters.death_flag.value = 1
        self.input.release_all("death")
        self.scheduler.interrupt()
        self._put_log("warning", f"death confirmed at frame {z.frame_id} "
                                 f"(distance={self.death.last_distance:.1f})")
        # terminal transition with death penalty
        self.stack.push(z.ground_gray)
        self._env_ids.append(z.frame_id)
        reward = self.reward_calc.step(
            ts=t_frame, action=self._last_action, horizon=hz,
            ground_gray=z.ground_gray, died=True,
        )
        self._episode_reward += reward.total
        self._ship_transition(self._last_action, reward.total, True, t_frame)
        self._end_episode(reason="death")

        # 5-8: respawn loop with bounded clicks
        self.respawn_ctl.start()
        last_id = self._last_processed_id
        deadline = time.monotonic() + self.cfg.death.respawn_timeout_s + 1.0
        while not self.events["stop"].is_set() and not self.events["emergency"].is_set():
            if self.events["pause"].is_set():
                self.input.release_all("respawn-paused")
                time.sleep(0.05)
                continue
            frame = self.ring.read_latest()
            if frame is None or frame.frame_id <= last_id:
                time.sleep(0.01)
                if time.monotonic() > deadline:
                    break
                continue
            last_id = frame.frame_id
            zz = self.pre.process(frame.image, frame.frame_id, frame.ts)
            dr = (self.death.update(zz.anchor_patch, zz.frame_id, zz.ts)
                  if self.death is not None else None)
            if dr is None:
                break
            status = self.respawn_ctl.update(dr.state, frame.ts)
            if status.action == "CLICK":
                self._put_log("info", f"respawn click #{self.respawn_ctl.clicks}")
            elif status.action == "RECOVERED":
                self._recover_from_death(zz, frame.ts)
                return
            elif status.action == "FAILED":
                self._respawn_failed(status.detail)
                return
            if time.monotonic() > deadline:
                self._respawn_failed("outer_timeout")
                return
        # stop requested mid-respawn
        self._respawn_failed("stop_requested")

    def _recover_from_death(self, zz, t_frame: float) -> None:
        LOGGER.info("respawn recovered after %d click(s)", self.respawn_ctl.clicks)
        self.events["death"].clear()
        self.events["pause_learning"].clear()
        self.counters.death_flag.value = 0
        self.death_reset_detectors()
        self.stack.reset(zz.ground_gray)
        self._env_ids = deque([zz.frame_id] * self.cfg.perception.frame_stack,
                              maxlen=self.cfg.perception.frame_stack)
        self._begin_episode(t_frame)
        self._put_log("info", "recovered; training resumed")

    def _respawn_failed(self, detail: str) -> None:
        LOGGER.error("respawn failed: %s — pausing (not clicking forever)", detail)
        self.events["pause"].set()
        self.input.release_all("respawn failed")
        put_bounded(self.metrics_q, {
            "type": "error", "src": "respawn",
            "error": f"respawn failed: {detail}",
            "tb": f"RespawnController exhausted: {detail}",
        })

    def death_reset_detectors(self) -> None:
        self.horizon.reset()
        self.reward_calc = SurvivalRewardCalculator(self.cfg.reward)

    # ------------------------------------------------------------------ #
    # Performance watchdog + auto-downgrade (never silently violated)
    # ------------------------------------------------------------------ #
    def _cpu_load_estimate(self) -> float:
        try:
            import psutil

            return min(1.0, psutil.cpu_percent(interval=None) / 100.0)
        except Exception as exc:
            # DEEP-FIX: this swallowed the reason entirely.  Returning a
            # neutral 0.5 is still the right fallback, but the first failure
            # is logged so a missing/broken psutil is visible instead of
            # silently pinning the cadence to the middle of its range.
            if not self._psutil_warned:
                self._psutil_warned = True
                LOGGER.warning("psutil unavailable (%s: %s); cadence will use "
                               "a neutral 0.5 load estimate",
                               type(exc).__name__, exc)
            return 0.5

    _psutil_warned: bool = False
    _ram_violation_since: Optional[float] = None

    def _working_set_gb(self) -> float:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 ** 3)
        except Exception:
            return 0.0

    def _maybe_perf_check(self) -> None:
        now = time.monotonic()
        if now - self._last_load_check < 1.0:
            return
        self._last_load_check = now
        p95 = self.action_ms.snapshot()["p95"]
        fps = self.fps_meter.fps()
        budget = self.cfg.perf.p95_action_latency_ms
        min_fps = self.cfg.perf.min_effective_fps
        violating = (p95 > budget) or (fps > 0 and fps < min_fps)
        if violating:
            if self._perf_violation_since is None:
                self._perf_violation_since = now
            elif now - self._perf_violation_since >= self.cfg.perf.downgrade_window_s:
                self._downgrade(p95, fps)
        else:
            self._perf_violation_since = None

        # DEEP-FIX: cfg.perf.max_working_set_gb was declared and documented
        # ("keep total usage < 4 GB") but read by nothing.  The bot process's
        # own working set is now checked against it, and a sustained overrun
        # is reported once so the operator sees the memory pressure before
        # Windows starts paging and the capture FPS collapses.
        limit_gb = float(self.cfg.perf.max_working_set_gb)
        if limit_gb > 0:
            rss_gb = self._working_set_gb()
            if rss_gb > limit_gb:
                if self._ram_violation_since is None:
                    self._ram_violation_since = now
                elif (now - self._ram_violation_since
                      >= self.cfg.perf.downgrade_window_s
                      and not self._ram_warned):
                    self._ram_warned = True
                    self._put_log(
                        "warning",
                        f"WORKING SET {rss_gb:.2f} GB exceeds "
                        f"perf.max_working_set_gb={limit_gb:.2f} GB — reduce "
                        f"per.capacity or the capture region size")
            else:
                self._ram_violation_since = None
                self._ram_warned = False

    _ram_warned: bool = False

    def _downgrade(self, p95: float, fps: float) -> None:
        lighter = self.cfg.profile_downgrade()
        if lighter is None:
            self._put_log("warning",
                          f"PERF BUDGET EXCEEDED (p95={p95:.0f}ms fps={fps:.1f}) and "
                          f"already on lightest profile — continuing, consider "
                          f"reducing capture size or closing apps")
            self._perf_violation_since = time.monotonic()
            return
        self.cfg.rl.profile = lighter
        self._profile = lighter
        self.counters.set_profile(lighter)
        self._put_log("warning",
                      f"PERF BUDGET EXCEEDED (p95={p95:.0f}ms fps={fps:.1f}): "
                      f"downgrading model profile to '{lighter}'")
        model = DuelingDQN.from_profile(lighter, self.cfg.perception.frame_stack,
                                        self.cfg.perception.ground_size)
        self.policy = InferencePolicy(model, seed=self.cfg.seed)
        # DEEP-FIX: this notice used to go ONLY to transition_q, which the
        # learner never inspects for commands -- Learner.add_transitions()
        # drops anything that is not an NStepTransition, and the whole repo
        # contained exactly one reference to "__cmd__" (this line).  The
        # learner therefore kept training the heavy profile while the actor
        # ran the light one, and because unflatten_into() consumed a blind
        # prefix of the flat vector the actor silently loaded garbage weights
        # (verified: a (4,1,3,3) conv overwritten from a (48,4,8,8) conv).
        # The command now goes to the learner's real command queue, and the
        # actor refuses mismatched weight vectors until the switch lands.
        notice = {"__cmd__": "set_profile", "profile": lighter}
        if self.cmd_q is not None:
            put_bounded(self.cmd_q, {"cmd": "set_profile", "profile": lighter})
        put_bounded(self.transition_q, notice)  # redundant path; idempotent
        self._perf_violation_since = time.monotonic()

    # ------------------------------------------------------------------ #
    def _maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._last_report < self.cfg.perf.report_interval_s:
            return
        self._last_report = now
        self._put_metrics({
            "type": "actor_stats",
            "data": {
                "fps": self.fps_meter.fps(),
                "dropped": self._dropped_total,
                "duplicates": self._duplicate_frames,
                "inference_ms": self.infer_ms.snapshot(),
                "action_ms": self.action_ms.snapshot(),
                "scheduler": self.scheduler.snapshot(),
                "epsilon": float(self.counters.epsilon.value),
                "episode_reward": self._episode_reward,
                "survival_s": (max(0.0, now - self._episode_start_ts)
                               if self._episode_start_ts is not None else 0.0),
                "profile": self._profile,
                "held_keys": self.input.pressed_names(),
                "horizon": {"score": self.horizon._ewma,
                            "no_data": self.horizon.no_data_count},
            },
        })

    # ------------------------------------------------------------------ #
    def request_shutdown(self) -> None:
        self._shutdown_requested = True


def headless_env_from_cfg(cfg: BotConfig, seed: Optional[int] = None) -> GameEnvironment:
    if seed is not None:
        cfg.seed = seed
    return GameEnvironment(cfg)
