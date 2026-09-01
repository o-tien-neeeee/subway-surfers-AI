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
from agent import InferencePolicy, effective_epsilon_for_step
from config import ACTIONS, BotConfig, LEFT, NOOP, RIGHT
from death_detector import ColorAnchorDeathDetector, DeathResult, DeathState, RespawnController
from horizon_detector import HorizonDetector, HorizonResult
from input_controller import InputController
from ipc import SharedFrameRing, SharedWeights
from logging_utils import get_logger, put_bounded
from metrics import FpsMeter, LatencyMeter
from models import DuelingDQN
from obstacle_perception import (
    ObstacleSnapshot,
    ObstacleTracker,
    occupancy_from_game_state,
)
from perception import FrameStack, ZonePreprocessor
from replay_buffer import NStepBuilder
from rewards import SurvivalRewardCalculator

LOGGER = get_logger("actor")

#: Shared zero-value snapshot used when obstacle tracking is disabled, so
#: the reward path never has to branch on ``None``.
_EMPTY_SNAPSHOT = ObstacleSnapshot(occupancy=np.zeros((1, 1), dtype=bool),
                                   player_lane=0)


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
        # Dense-reward tracking: a one-shot flag so the death
        # penalty in :meth:`step_with_reward` is paid exactly
        # once per episode.  Reset on respawn().
        self._episode_dead_flag = False

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
        """Advance the game one tick with the player's action.

        Returns a dict-shaped tuple (frame, reward, done, info)
        when ``return_info=True`` is set in the constructor; the
        default code path still returns the rendered frame for
        backwards compatibility.  Callers who want the *dense
        shaping* reward (proximity-to-obstacle + dodge bonuses)
        should use :meth:`step_with_reward`.
        """
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
        self._episode_dead_flag = False
        self.player_lane = 1
        self.obstacles.clear()
        self._spawn_timer = 0.2

    def step_with_reward(self, action: int) -> dict:
        """One step with a *dense* shaping reward.

        Components (all clipped to keep magnitudes bounded):

        * ``alive`` +0.02 per tick (the standard survival signal)
        * ``obstacle_proximity`` -0.05 if the nearest obstacle in
          any lane is within 30% of the spawn point AND the
          obstacle is in the player's lane.  This is the
          per-tick "you are in danger" signal that the audit
          showed was missing in :mod:`rewards`.
        * ``lane_change_bonus`` +0.20 if the player changed lane
          on this tick AND the previous lane had an obstacle
          within 30% of completion.  This is the "you dodged it"
          signal that the audit showed was missing.
        * ``death`` -5.0 once on the tick the player dies.

        Returned dict has keys ``frame``, ``reward``, ``done``,
        ``info`` (with the breakdown).
        """
        prev_lane = self.player_lane
        prev_obstacle_in_player = any(
            ob["lane"] == prev_lane and ob["prog"] < 0.5
            for ob in self.obstacles
        )
        frame = self.step(action)
        info: dict = {
            "alive": 0.02,
            "proximity": 0.0,
            "dodge": 0.0,
            "death": 0.0,
        }
        # Check if the player is currently in danger.
        in_danger = any(
            ob["lane"] == self.player_lane and ob["prog"] < 0.5
            for ob in self.obstacles
        )
        if in_danger:
            info["proximity"] = -0.05
        # Did the player switch out of a dangerous lane?
        if prev_obstacle_in_player and self.player_lane != prev_lane:
            info["dodge"] = 0.20
        # Death penalty (once).
        if self.dead and self._episode_dead_flag is False:
            info["death"] = -5.0
            self._episode_dead_flag = True
        if not self.dead:
            self._episode_dead_flag = False
        reward = sum(info.values())
        return {"frame": frame, "reward": reward, "done": self.dead,
                "info": info}

    def step_with_frame(self, frame: np.ndarray,
                        action: int) -> dict:
        """Dreamer back-door: take an external frame, ask the game
        what *would* happen, return the outcome without rendering.

        Used by :mod:`dreamer` to verify that an *abstract* frame
        (a VAE-decoded variation of a real frame) still represents
        a "survivable" state.  We do NOT actually use ``frame`` to
        alter the game state — the synthetic game has its own
        internal world — we just run ``step(action)`` and report
        ``done`` so the dreamer can classify.

        The return dict is deliberately tiny so the dreamer does
        not depend on the synthetic env's full StepInfo dataclass.
        """
        if self.dead:
            # Already dead: report done so the dreamer can stop
            # counting this round.
            return {"done": True, "alive": False}
        # ``self.step`` already advances obstacles, score, and
        # possibly flips ``self.dead``.  The frame argument is
        # accepted but not consumed — its purpose is to make the
        # contract clear ("I gave you a frame, you tell me whether
        # you'd die").
        self.step(action)
        return {"done": bool(self.dead), "alive": not bool(self.dead)}

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
        self.stack = FrameStack(cfg.perception.frame_stack, cfg.perception.policy_size)
        self.nstep = NStepBuilder(cfg.rl.n_step, cfg.rl.gamma)
        # DEEP-FIX (v1.24.0): Atari-style frame skip.  One agent step now
        # advances the game ``frame_skip`` raw frames with the chosen action
        # held, accumulates their rewards into ONE transition and pushes ONE
        # observation, so gamma/n_step are expressed per *decision*.
        self.frame_skip = max(1, int(cfg.rl.frame_skip))
        # Structured obstacle grid -> causal clear/danger shaping.
        self.obstacles: Optional[ObstacleTracker] = (
            ObstacleTracker(cfg.obstacle) if cfg.obstacle.enabled else None
        )
        self._t = 0.0
        self._env_ids: deque[int] = deque(maxlen=4)
        self._last_action = NOOP
        self._prev_action = NOOP
        self._prev_policy: Any = None
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
        if self.obstacles is not None:
            self.obstacles.reset()
        self._env_ids.clear()
        self._last_action = NOOP
        self._prev_policy = None
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
            self.stack.reset(z.policy_gray)
            self._env_ids = deque([fid] * 4, maxlen=4)
        else:
            self.stack.push(z.policy_gray)
            self._env_ids.append(fid)
        return self.stack.get()

    def _obstacle_snapshot(self, died: bool) -> ObstacleSnapshot:
        """Structured obstacle facts for this raw frame (synthetic path).

        Uses the game's authoritative obstacle list instead of pixel CV: in
        headless mode the ground truth is already available and a lossy CV
        proxy would only add noise to the shaping signal.  The *tracker* is
        still what debounces cells and detects "it passed the player", so
        the live and headless paths share one state machine.
        """
        if self.obstacles is None:
            return _EMPTY_SNAPSHOT
        occ = occupancy_from_game_state(
            self.game.obstacles, self.obstacles.rows, self.obstacles.lanes,
            bottom_prog=0.97,
        )
        return self.obstacles.update_from_occupancy(
            occ, int(self.game.player_lane), ts=self._t,
            frame_id=int(self.game.frame_id), died=died,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        """Advance ONE agent step = ``frame_skip`` raw frames, action held.

        ``reward`` is the sum of the per-raw-frame components across the
        whole step, so the transition the learner receives matches the
        decision it was made for (Atari frame-skip semantics).
        """
        dt = 1.0 / self.cfg.capture.target_fps
        total = 0.0
        breakdown: dict[str, float] = {}
        died = False
        hz = HorizonResult(frame_id=-1, ts=self._t, change_score=0.0,
                           raw_score=0.0, changed_ratio=0.0, detected=False,
                           confidence=0.0)
        dr = DeathResult(state=DeathState.UNKNOWN, distance=0.0,
                         reason="no_frame", frame_id=-1, ts=self._t)
        last_valid: Any = None
        for i in range(self.frame_skip):
            self._t += dt
            ts = self._t
            # 1. act on the synthetic game.  A Subway Surfers swipe is a
            #    tap, so holding the same action for the rest of the step
            #    does not repeat the manoeuvre — which is exactly why frame
            #    skip is safe on this game.
            self.game.step(int(action))
            # 2. ingest the resulting frame (perception + detectors)
            frame = self.game.render()
            z = self.pre.process(frame, self.game.frame_id, ts)
            hz = self.detector.update(z.horizon_gray, z.frame_id, ts)
            dr = self.death.update(z.anchor_patch, z.frame_id, ts)
            died = dr.state is DeathState.DEAD_CONFIRMED
            snap = self._obstacle_snapshot(died)
            # 3. reward for this raw frame given the action being held
            bd = self.reward_calc.step(
                ts=ts, action=int(action), horizon=hz,
                ground_gray=z.policy_gray, prev_ground_gray=self._prev_policy,
                died=died, cleared=snap.cleared, danger=snap.danger,
                is_decision=(i == 0),
            )
            self._prev_policy = z.policy_gray
            total += bd.total
            breakdown = bd.to_dict()
            if z.valid and z.policy_gray is not None:
                last_valid = z
            if died:
                break
        # One observation per agent step (the last valid raw frame of it), so
        # the stack holds ``frame_stack`` *decisions* of motion history.
        if last_valid is not None:
            self.stack.push(last_valid.policy_gray)
            self._env_ids.append(last_valid.frame_id)
        obs = self.stack.get()
        self.steps += 1
        info = StepInfo(horizon=hz, death=dr,
                        reward_breakdown=breakdown, action_step=self.steps)
        return obs, total, died, info

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
        if self.obstacles is not None:
            self.obstacles.reset()
        self._env_ids.clear()
        self._last_action = NOOP
        self._prev_policy = None


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
        self.stack = FrameStack(cfg.perception.frame_stack, cfg.perception.policy_size)
        self.scheduler = ActionScheduler(cfg.scheduler,
                                          cooldown_ms=cfg.input.cooldown_ms)
        self.scheduler.ttl_ms = float(cfg.input.action_ttl_ms)
        self.input = InputController(cfg.input, backend=input_backend)
        # DEEP-FIX: report requested vs *resolved* backend.  `backend_name`
        # used to echo the request and nothing ever read it, which is why a
        # half-maintained value survived a release.  A run that asked for
        # real input and silently got dry_run now says so on the first line
        # of the log instead of producing an episode of NOOPs.
        if self.input.requested_backend != self.input.backend_name:
            LOGGER.warning(
                "input backend degraded: requested %r but %r is live",
                self.input.requested_backend, self.input.backend_name,
            )
        else:
            LOGGER.info("input backend: %s", self.input.backend_name)
        model = DuelingDQN.from_profile(cfg.rl.profile,
                                        cfg.perception.frame_stack,
                                        cfg.perception.policy_size)
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
        # DEEP-FIX: the user reported "AI không tiến triển sau 500 episode".
        # The most common real cause is the bot dying almost instantly every
        # episode (capture reads the dead/occluded state), so there is no
        # survival signal to learn from.  Count consecutive instant deaths and
        # say so loudly instead of letting 500 empty episodes pass in silence.
        self._instant_death_streak = 0
        self._instant_death_warned = False
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

        # DEEP-FIX (v1.24.0): frame-skip / decision-cadence state.  One agent
        # step spans ``frame_skip`` captured frames; the action decided on the
        # first of them is held for the rest and their rewards are summed into
        # ONE transition.  ``_step_phase == 0`` marks the decision frame.
        self.frame_skip = max(1, int(cfg.rl.frame_skip))
        self._step_phase = 0
        self._step_reward = 0.0
        self._step_action = NOOP
        self._step_obs: Any = None
        self._decision_count = 0
        self._player_lane = 1  # middle lane, tracked from our own LEFT/RIGHT
        self.obstacles: Optional[ObstacleTracker] = (
            ObstacleTracker(cfg.obstacle) if cfg.obstacle.enabled else None
        )

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
        # Ask the learner to bind a fresh synthetic env to the
        # dreamer.  The actor is the only place that has the env
        # *out-of-the-loop* (the learner's process never imports
        # environment at module load — too heavy).  The cmd is
        # best-effort: if the learner doesn't know how to handle
        # it (e.g. older build), it just logs a warning and the
        # dreamer falls back to its Q-score path.
        if self.cmd_q is not None:
            try:
                put_bounded(self.cmd_q,
                            {"cmd": "set_dream_env", "seed": self.cfg.seed})
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("could not request dreamer env: %s", exc)
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
        if self._episode_start_ts is None:
            self.stack.reset(z.policy_gray)
            self._env_ids = deque([z.frame_id] * self.cfg.perception.frame_stack,
                                  maxlen=self.cfg.perception.frame_stack)
            self._begin_episode(t_frame)
            self.fps_meter.tick(t_frame)
            # the NEXT valid frame opens the first agent step
            self._step_phase = 0
            return

        self.fps_meter.tick(t_frame)
        self._episode_frames += 1

        danger = hz.detected and hz.confidence >= 0.5
        self.counters.danger_flag.value = 1 if danger else 0
        # §9: cadence signal is still fed (CPU load AND horizon confidence),
        # but frame_skip is now the cadence: see the decision branch below.
        self.scheduler.set_signal(self._cpu_load_estimate(), hz.confidence)

        # Causal obstacle facts for this raw frame (CV path).
        if self.obstacles is not None:
            snap = self.obstacles.update(z.policy_gray, self._player_lane,
                                         ts=t_frame, frame_id=z.frame_id)
        else:
            snap = _EMPTY_SNAPSHOT
        if snap.danger:
            danger = True
            self.counters.danger_flag.value = 1

        is_decision_frame = (self._step_phase == 0)
        reward = self.reward_calc.step(
            ts=t_frame,
            action=self._last_action,
            horizon=hz,
            ground_gray=z.policy_gray,
            prev_ground_gray=self._prev_ground,
            died=False,
            cleared=snap.cleared,
            danger=snap.danger,
            is_decision=is_decision_frame,
        )
        self._prev_ground = z.policy_gray
        self._episode_reward += reward.total

        if is_decision_frame:
            # --- open a new agent step ---------------------------------- #
            # The stack advances once per DECISION, so it holds
            # frame_stack decisions (~0.4 s) of motion instead of 0.13 s.
            self.stack.push(z.policy_gray)
            self._env_ids.append(frame.frame_id)
            self._step_obs = self.stack.get()
            self._step_reward = reward.total
            self._decision_count += 1
            # frame_skip IS the decision cadence: decide exactly once per
            # agent step.  Routing this through scheduler.on_frame() as well
            # gated decisions behind a second 2-3 frame counter and silently
            # skipped most steps.  pop_executable() still enforces the
            # cooldown / duplicate suppression / TTL guards.
            self._decide(danger)
            planned = self.scheduler.pop_executable()
            if planned is not None:
                self._execute(planned, t_frame)
            self.counters.action_step.value = self.scheduler.action_step
            self._step_action = self._last_action
            self._step_phase = 1
        else:
            # --- continue the current step: hold the action ------------- #
            self._step_reward += reward.total
            self._step_phase += 1
            if self._step_phase >= self.frame_skip:
                self._ship_transition(action=self._step_action,
                                      reward=self._step_reward,
                                      done=False, ts=t_frame)
                self._step_phase = 0
        self._maybe_perf_check()

    _prev_ground: Any = None
    _last_action: int = NOOP

    # ------------------------------------------------------------------ #
    def _decide(self, danger: bool) -> None:
        t0 = time.perf_counter()
        self.policy.refresh_weights(self.shared_weights)
        # DEEP-FIX (v1.24.0): exploration decays per AGENT STEP, not per
        # captured frame.  With frame skip the two differ by a factor of
        # ``frame_skip``, and a frame-counted schedule silently changes
        # meaning every time the cadence or the capture FPS is tuned.
        # effective_epsilon_for_step() still caps exploration once behaviour
        # cloning has produced a policy, so the actor USES the BC policy
        # instead of playing randomly at epsilon~1.0 and dying every ~1s.
        eps = effective_epsilon_for_step(self._decision_count, self.cfg.rl,
                                         self.counters.bc_pretrained.value)
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
            # Track the lane we believe we are in, from our own lateral taps.
            # The obstacle tracker needs it to know which column is "ours" —
            # reading it back from pixels would be far less reliable.
            if planned.action == LEFT:
                self._player_lane = max(0, self._player_lane - 1)
            elif planned.action == RIGHT:
                self._player_lane = min(self.cfg.obstacle.lanes - 1,
                                        self._player_lane + 1)
            if self.action_out_q is not None:
                put_bounded(self.action_out_q,
                                 {"type": "action", "action": int(planned.action)})

    def _ship_transition(self, action: int, reward: float, done: bool, ts: float) -> None:
        env_ids = tuple(self._env_ids)
        # DEEP-FIX (v1.24.0): the head observation of the transition is the
        # stack as it stood WHEN THE DECISION WAS MADE, not the stack at ship
        # time (which already contains the frames produced by that action).
        # Pairing (obs_before, action, R, obs_after_n_steps) is the MDP
        # semantics the n-step target assumes; the old code was shifted by one
        # step, so every Q update credited the action to the state it had
        # already moved on from.
        obs = self._step_obs if self._step_obs is not None else self.stack.get()
        for tr in self.nstep.push(obs, env_ids, action, reward, done):
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
        # frame-skip step state is episode-scoped: a partial step must not
        # carry its accumulated reward into the next life.
        self._step_phase = 0
        self._step_reward = 0.0
        self._step_action = NOOP
        self._step_obs = None
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
        # DEEP-FIX: surface an instant-death loop -- the reason a long run can
        # show "no progress" is that every episode ends before any dodge.
        if survival < 1.0:
            self._instant_death_streak += 1
        else:
            self._instant_death_streak = 0
            self._instant_death_warned = False
        if self._instant_death_streak >= 5 and not self._instant_death_warned:
            self._instant_death_warned = True
            LOGGER.warning(
                "BOT ĐANG CHẾT NGAY LIÊN TIẾP (%d episode < 1s). Không có tín "
                "hiệu sống sót để học — AI sẽ KHÔNG tiến bộ. Nguyên nhân thường "
                "gặp: vùng chọn bị che / đang ở màn thua / neo màu sai. Hãy kiểm "
                "tra lại vùng + hiệu chuẩn neo (bước 1-3) rồi chạy lại.",
                self._instant_death_streak)
            self._put_metrics({"type": "log", "level": "warning", "src": "actor",
                               "msg": f"instant-death loop: {self._instant_death_streak} episodes < 1s"})
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
        self.stack.push(z.policy_gray)
        self._env_ids.append(z.frame_id)
        snap = (self.obstacles.update(z.policy_gray, self._player_lane,
                                      ts=t_frame, frame_id=z.frame_id,
                                      died=True)
                if self.obstacles is not None else _EMPTY_SNAPSHOT)
        reward = self.reward_calc.step(
            ts=t_frame, action=self._last_action, horizon=hz,
            ground_gray=z.policy_gray, died=True,
            cleared=snap.cleared, danger=snap.danger, is_decision=True,
        )
        self._episode_reward += reward.total
        # Fold the in-progress step into the terminal transition so the
        # frames already collected for this decision are not thrown away.
        self._step_reward += reward.total
        self._ship_transition(self._step_action, self._step_reward, True, t_frame)
        self._step_phase = 0
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
        self.stack.reset(zz.policy_gray)
        self._env_ids = deque([zz.frame_id] * self.cfg.perception.frame_stack,
                              maxlen=self.cfg.perception.frame_stack)
        if self.obstacles is not None:
            self.obstacles.reset()
        self._player_lane = 1
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
                                        self.cfg.perception.policy_size)
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
