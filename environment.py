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
from typing import Any

import numpy as np

from action_scheduler import ActionScheduler, PlannedAction
from agent import InferencePolicy, epsilon_for_frame, epsilon_for_step
from config import NOOP, BotConfig
from death_detector import (
    ColorAnchorDeathDetector,
    DeathResult,
    DeathState,
    RespawnController,
)
from horizon_detector import HorizonDetector, HorizonResult
from input_controller import InputController
from ipc import SharedFrameRing, SharedWeights
from logging_utils import get_logger, put_bounded
from metrics import FpsMeter, LatencyMeter
from models import DuelingDQN
from obstacle_perception import ObstacleTracker
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
    def step(self, action: int, dt: float | None = None) -> np.ndarray:
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
                # Fairer, learnable cadence: obstacles arrive roughly
                # every 0.9-1.6 s at the start, tightening to ~0.7 s — the
                # old 0.35-0.6 s spawn rate made even an oracle die in ~3 s
                # (multiple overlapping obstacles, no reaction window).
                self.spawn_cooldown = max(0.7, self.spawn_cooldown * 0.997)
                self._spawn_timer = self.spawn_cooldown
                kind = str(self.rng.choice(["lane", "lane", "low", "high"]))
                lane = int(self.rng.integers(0, self.lanes))
                # Keep at least one lane open among obstacles that are still
                # far away (never block all three lanes simultaneously) and
                # avoid stacking a fresh obstacle directly on a near one.
                near_lanes = {o["lane"] for o in self.obstacles if o["prog"] < 0.35}
                if len(near_lanes) >= self.lanes - 1:
                    open_lanes = [l for l in range(self.lanes) if l not in near_lanes]
                    if open_lanes:
                        lane = int(self.rng.choice(open_lanes))
                self.obstacles.append({"kind": kind, "lane": lane, "prog": 0.0,
                                       "speed": float(self.rng.uniform(0.45, 0.65))})
            survived: list[dict[str, float]] = []
            for ob in self.obstacles:
                ob["prog"] += ob["speed"] * dt
                if ob["prog"] < 0.97:
                    survived.append(ob)
                    continue
                # collision row: an obstacle in our lane is survivable for
                # low/high barriers via JUMP/SLIDE (with a grace window so a
                # tap a few frames early still counts), and for trains ("lane")
                # via switching lanes before this row.
                if ob["lane"] == self.player_lane:
                    dodged = False
                    if ob["kind"] == "low":
                        dodged = (action == 3) or (self._t - self._last_jump_t) < 0.5
                    elif ob["kind"] == "high":
                        dodged = (action == 4) or (self._t - self._last_slide_t) < 0.5
                    # "lane" (train) is dodged by NOT being in its lane —
                    # being here means the lane change did not happen in time.
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

    def __init__(self, cfg: BotConfig, game: SyntheticGame | None = None) -> None:
        self.cfg = cfg
        self.game = game or SyntheticGame(seed=cfg.seed)
        self.pre = ZonePreprocessor(cfg.perception, cfg.perception.horizon_frac,
                                    anchor_xy=(30, 30))
        self.detector = HorizonDetector(cfg.horizon, cfg.perception.horizon_size)
        death_cfg = self._death_cfg_with_anchor()
        self.death = ColorAnchorDeathDetector(death_cfg)
        self.reward_calc = SurvivalRewardCalculator(cfg.reward, now=lambda: self._t)
        self.stack = FrameStack(cfg.perception.frame_stack, cfg.perception.obs_size)
        self.nstep = NStepBuilder(cfg.rl.n_step, cfg.rl.gamma)
        self.obstacles = ObstacleTracker(player_lane=self.game.player_lane)
        self._t = 0.0
        self._env_ids: deque[int] = deque(maxlen=cfg.perception.frame_stack)
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
        self.obstacles.reset()
        self.episode += 1
        return self._ingest(reset=True)

    def _ingest(self, reset: bool = False):
        frame = self.game.render()
        ts = self._t
        z = self.pre.process(frame, self.game.frame_id, ts)
        # Advance the internal debounce state of detectors even on reset,
        # but the results aren't needed by the caller.
        self.detector.update(z.horizon_gray, z.frame_id, ts)
        self.death.update(z.anchor_patch, z.frame_id, ts)
        fid = self.game.frame_id
        k = self.cfg.perception.frame_stack
        if reset:
            self.stack.reset(z.policy_gray)
            self._env_ids = deque([fid] * k, maxlen=k)
        else:
            self.stack.push(z.policy_gray)
            self._env_ids.append(fid)
        return self.stack.get()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        """Advance ONE AGENT STEP = ``frame_skip`` sub-steps of the game.

        The chosen action is applied at the start and HELD for the skip (a
        fresh tap is not re-sent — Subway Surfers latches lane changes and
        jump/roll animations, matching the live pipeline's tap semantics).
        Rewards are accumulated over the sub-steps; the observation stack
        advances by ONE decision-step frame (Atari frame-skip structure).
        """
        skip = max(1, int(self.cfg.rl.frame_skip))
        dt = 1.0 / self.cfg.capture.target_fps
        total_r = 0.0
        hz = None
        dr = None
        z = None
        died = False
        occ = None
        for _ in range(skip):
            self._t += dt
            ts = self._t
            self.game.step(int(action))          # action held
            frame = self.game.render()
            z = self.pre.process(frame, self.game.frame_id, ts)
            hz = self.detector.update(z.horizon_gray, z.frame_id, ts)
            dr = self.death.update(z.anchor_patch, z.frame_id, ts)
            sub_died = dr.state is DeathState.DEAD_CONFIRMED
            if not sub_died:
                occ = self.obstacles.update(
                    z.policy_gray, player_lane=self.game.player_lane
                )
            breakdown = self.reward_calc.step(
                ts=ts, action=int(action), horizon=hz,
                ground_gray=z.policy_gray, died=sub_died,
                clear=bool(occ.clear) if occ else False,
                danger=bool(occ.danger) if occ else False,
            )
            total_r += breakdown.total
            died = died or sub_died
            if died:
                break
        # stack advances ONE frame per agent step (the frame after the skip)
        self.stack.push(z.policy_gray)
        self._env_ids.append(self.game.frame_id)
        obs = self.stack.get()
        self.steps += 1
        info = StepInfo(horizon=hz, death=dr,
                        reward_breakdown={"total": total_r}, action_step=self.steps)
        return obs, total_r, died, info

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
    ) -> None:
        self.cfg = cfg
        self.ring = ring
        self.events = events
        self.transition_q = transition_q
        self.metrics_q = metrics_q
        self.shared_weights = shared_weights
        self.counters = counters
        self.action_out_q = action_out_q  # fake-game action channel (headless)

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
        # Structured obstacle tracker -> causal clear/danger shaping.
        self.obstacles = ObstacleTracker()
        self.frame_skip = max(1, int(cfg.rl.frame_skip))
        self.stack = FrameStack(cfg.perception.frame_stack, cfg.perception.obs_size)
        self.scheduler = ActionScheduler(cfg.scheduler,
                                          cooldown_ms=cfg.input.cooldown_ms)
        self.scheduler.ttl_ms = float(cfg.input.action_ttl_ms)
        self.input = InputController(cfg.input, backend=input_backend)
        model = DuelingDQN.from_profile(cfg.rl.profile,
                                        cfg.perception.frame_stack,
                                        cfg.perception.obs_size)
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
        self._episode_start_ts = 0.0
        self._episode_frames = 0
        self._last_load_check = time.monotonic()
        self._perf_violation_since: float | None = None
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
        except Exception as exc:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
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

        if not z.valid or z.policy_gray is None:
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
        if self._episode_start_ts == 0.0:
            self.stack.reset(z.policy_gray)
            self._env_ids = deque([z.frame_id] * self.cfg.perception.frame_stack,
                                  maxlen=self.cfg.perception.frame_stack)
            self._begin_episode(t_frame)
            self.fps_meter.tick(t_frame)
            self._step_frames = 0
            self._step_reward = 0.0
            self._step_action = NOOP
            return

        self.fps_meter.tick(t_frame)
        self._episode_frames += 1

        # ---- per-frame perception within the current agent step ---------
        # Structural danger (an obstacle is in our lane right in front of us)
        # and causal clear/danger shaping from the obstacle tracker.
        occ = self.obstacles.update(z.policy_gray)
        clear = bool(occ.clear)
        danger_struct = bool(occ.danger)
        danger = (hz.detected and hz.confidence >= 0.5) or danger_struct
        self.counters.danger_flag.value = 1 if danger else 0
        self.scheduler.set_signal(self._cpu_load_estimate(),
                                  max(hz.confidence, 0.8 if danger_struct else 0.0))

        # ---- decide on the FIRST frame of each agent step ----------------
        # frame-skip structure: one decision every `frame_skip` captured
        # frames (or immediately on danger); the chosen action is held.
        first_frame_of_step = (self._step_frames == 0)
        if first_frame_of_step:
            # Decide from the CURRENT stack (the previous step's final
            # frames); the new frame is pushed below at step completion.
            self._step_action = NOOP
            if self.scheduler.on_frame(danger=danger):
                self._decide(danger)
                planned = self.scheduler.pop_executable()
                if planned is not None:
                    self._execute(planned, t_frame)
                    self._step_action = planned.action
            self.counters.action_step.value = self.scheduler.action_step
            # epsilon is indexed by AGENT DECISION STEPS now (a stable clock
            # under variable cadence), not raw env frames.
            self.counters.epsilon.value = epsilon_for_step(
                self.scheduler.action_step, self.cfg.rl)

        # ---- per-frame reward for the action held this step --------------
        reward = self.reward_calc.step(
            ts=t_frame,
            action=self._step_action,
            horizon=hz,
            ground_gray=z.policy_gray,
            prev_ground_gray=self._prev_ground,
            died=False,
            clear=clear,
            danger=danger_struct,
        )
        self._prev_ground = z.policy_gray
        self._episode_reward += reward.total
        self._step_reward += reward.total
        self._step_frames += 1

        # ---- complete the agent step after `frame_skip` frames -----------
        step_done = self._step_frames >= self.frame_skip
        if step_done:
            # The observation handed to the learner as s_{t+1} is the stack
            # ending on the frame right after the skip (the frame whose
            # processing produced the step reward). Push it once here.
            self.stack.push(z.policy_gray)
            self._env_ids.append(frame.frame_id)
            self._ship_transition(action=self._step_action,
                                  reward=self._step_reward, done=False,
                                  ts=t_frame)
            self._step_frames = 0
            self._step_reward = 0.0
        self._maybe_perf_check()

    _prev_ground: Any = None
    _last_action: int = NOOP
    _step_frames: int = 0
    _step_reward: float = 0.0
    _step_action: int = NOOP

    # ------------------------------------------------------------------ #
    def _decide(self, danger: bool) -> None:
        t0 = time.perf_counter()
        self.policy.refresh_weights(self.shared_weights)
        # epsilon is owned by the frame-skip step clock (set once per agent
        # step in _process_frame); here we just act under the current value.
        eps = float(self.counters.epsilon.value)
        action = self.policy.act(self.stack.get(), eps)
        self.infer_ms.observe_ms((time.perf_counter() - t0) * 1000.0)
        self.scheduler.submit(action, danger=danger)

    def _execute(self, planned: PlannedAction, t_frame: float) -> None:
        if not self.input.focus_gate():
            self.scheduler.interrupt()
            return
        t0 = time.perf_counter()
        ev = self.input.press_action(planned.action, created_ts=planned.created_ts)
        self.action_ms.observe_ms((time.perf_counter() - t0) * 1000.0 + 1000.0 * (t0 - t_frame))
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
        # Emit anything still pending in the n-step window as episode-
        # terminal transitions (zero bootstrap beyond end of episode).
        for tr in self.nstep.push_absorbing():
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
        self._step_frames = 0
        self._step_reward = 0.0
        self._step_action = NOOP
        self.obstacles.reset()
        self._episode_stats_frames0 = self.counters.env_frame_id.value

    def _end_episode(self, reason: str) -> None:
        if self._episode_start_ts == 0.0:
            return
        survival = time.monotonic() - self._episode_start_ts
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
        # publish for the learner's online best-model gate (single writer: actor)
        self.counters.last_episode_done_id.value = stats.episode_id
        self.counters.last_episode_survival_s.value = float(stats.survival_s)
        self.counters.last_episode_reward.value = float(stats.total_reward)
        self._episode_start_ts = 0.0
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
        # terminal transition with death penalty (closing the current
        # in-flight agent step: its summed reward includes the -10).
        self.stack.push(z.policy_gray)
        self._env_ids.append(z.frame_id)
        reward = self.reward_calc.step(
            ts=t_frame, action=self._step_action, horizon=hz,
            ground_gray=z.policy_gray, died=True,
        )
        self._episode_reward += reward.total
        step_r = self._step_reward + reward.total
        self._ship_transition(self._step_action, step_r, True, t_frame)
        self._step_frames = 0
        self._step_reward = 0.0
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
        self.obstacles.reset()
        self._step_frames = 0
        self._step_reward = 0.0
        self._step_action = NOOP

    # ------------------------------------------------------------------ #
    # Performance watchdog + auto-downgrade (never silently violated)
    # ------------------------------------------------------------------ #
    def _cpu_load_estimate(self) -> float:
        try:
            import psutil

            return min(1.0, psutil.cpu_percent(interval=None) / 100.0)
        except Exception:  # noqa: BLE001  (defensive boundary at process/UI edge; error logged, never crashes)
            return 0.5

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
                                        self.cfg.perception.obs_size)
        self.policy = InferencePolicy(model, seed=self.cfg.seed)
        put_bounded(self.transition_q, {"__cmd__": "set_profile", "profile": lighter})
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
                "survival_s": (now - self._episode_start_ts)
                if self._episode_start_ts else 0.0,
                "profile": self._profile,
                "held_keys": self.input.pressed_names(),
                "horizon": {"score": self.horizon._ewma,
                            "no_data": self.horizon.no_data_count},
            },
        })

    # ------------------------------------------------------------------ #
    def request_shutdown(self) -> None:
        self._shutdown_requested = True


def headless_env_from_cfg(cfg: BotConfig, seed: int | None = None) -> GameEnvironment:
    if seed is not None:
        cfg.seed = seed
    return GameEnvironment(cfg)
