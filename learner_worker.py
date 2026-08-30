"""Learner process: behaviour cloning + online Double-DQN training.

The learner is the ONLY process allowed to mutate model weights and the
replay buffer (requirement C).  It:

* drains n-step transitions from a bounded queue into PER,
* trains Double-DQN updates at a rate decoupled from action cadence
  (capped by ``max_updates_per_second`` so the CPU keeps headroom for
  Chrome + capture + actor),
* publishes weights into shared memory for the actor's local copy,
* runs behaviour cloning when asked (``pretrain`` command) with an
  episode-based train/val split and class-balanced cross-entropy,
* checkpoints atomically (latest every N updates, best only on metric
  improvement, buffer every M updates and on shutdown),
* reports exceptions with full tracebacks to the GUI instead of dying
  silently, and never lets the GUI crash with it.

Command queue messages: {"cmd": "set_profile"|"pretrain"|"save_buffer"|"shutdown", ...}
"""

from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from agent import DoubleDQNAgent
from checkpoint_manager import CheckpointManager, capture_rng_states, restore_rng_states
# DEEP-FIX: PROFILE_ORDER was imported here and never used.
from config import BotConfig
from ipc import SharedWeights
from logging_utils import drain, format_exception, get_logger, put_bounded, setup_logging
from models import PROFILES, weight_size_for_profile
from replay_buffer import NStepTransition, PrioritizedReplayBuffer

LOGGER = get_logger("learner")


def _torch_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError as exc:
        LOGGER.debug("interop threads already set: %s", exc)


class Learner:
    """Stateful learner core (also unit-testable without processes)."""

    def __init__(self, cfg: BotConfig, shared_weights: SharedWeights, counters: Any,
                 metrics_q: Any, checkpoints_dir: str) -> None:
        self.cfg = cfg
        self.shared_weights = shared_weights
        self.counters = counters
        self.metrics_q = metrics_q
        self.ckpt = CheckpointManager(checkpoints_dir, cfg.rl.profile)
        self.profile = cfg.rl.profile
        self.agent = DoubleDQNAgent(self.profile, cfg.rl, seed=cfg.seed)
        self.buffer = self._load_or_fresh_buffer()
        self.rng = np.random.default_rng(cfg.seed)
        self.update_step = 0
        self.last_metrics: dict[str, float] = {}
        self._last_update_t = 0.0
        self._last_publish = 0.0
        self._publish_every_s = 2.0
        self._publish_every_updates = 20
        self._last_buffer_save_update = 0
        self.bc_history: list[dict[str, Any]] = []
        # -- online best-model tracking (requirement §12) ---------------- #
        # The actor reports each finished episode through SharedCounters;
        # the learner gates best_model.pth on the ROLLING MEAN of the
        # configured metric over the last ``best_metric_window`` episodes.
        self._episode_window: deque[float] = deque(
            maxlen=max(1, cfg.rl.best_metric_window)
        )
        self._last_seen_episode_done_id = 0
        self.best_metric_name = cfg.rl.best_metric
        self.best_rolling: Optional[float] = None
        self._load_checkpoint()

    # ------------------------------------------------------------------ #
    def _load_or_fresh_buffer(self) -> PrioritizedReplayBuffer:
        fresh = lambda: PrioritizedReplayBuffer(  # noqa: E731
            self.cfg.per, frame_size=self.cfg.perception.ground_size,
            gamma=self.cfg.rl.gamma,
        )
        loaded, ok = self.ckpt.buffer_load(
            lambda: PrioritizedReplayBuffer.load(
                str(self.ckpt.buffer_path), self.cfg.per,
                frame_size=self.cfg.perception.ground_size,
                gamma=self.cfg.rl.gamma,
            )
        )
        if ok and loaded is not None:
            LOGGER.info("replay buffer restored: %d transitions", loaded.size)
            return loaded
        return fresh()

    def _load_checkpoint(self) -> None:
        payload = self.ckpt.load_model("latest")
        if payload is None:
            payload = self.ckpt.load_model("best")
        if payload is not None and payload.get("profile") == self.profile:
            try:
                self.agent.load_payload(payload["agent"])
                self.update_step = payload["agent"].get("update_count", 0)
                # restore the persisted best-model gate so a restart cannot
                # overwrite a better historical best_model.pth with a worse run
                extra = payload.get("extra", {}) or {}
                bm = extra.get("best_metric")
                if bm is not None and np.isfinite(float(bm)):
                    self.ckpt.best_metric = float(bm)
                    self.best_rolling = float(bm)
                # DEEP-FIX: actually restore the replay-sampling Generator, so
                # a resumed run continues the same sampling stream instead of
                # restarting it from cfg.seed.
                rng_state = extra.get("rng")
                if isinstance(rng_state, dict) and \
                        restore_rng_states(rng_state, generator=self.rng):
                    LOGGER.info("replay sampling RNG restored from checkpoint")
                LOGGER.info("checkpoint restored (updates=%d, best=%s)",
                            self.update_step, self.ckpt.best_metric)
            except Exception as exc:
                LOGGER.error("checkpoint restore failed, fresh start:\n%s",
                             format_exception(exc))
        else:
            LOGGER.info("no checkpoint for profile %s; fresh networks", self.profile)

    # ------------------------------------------------------------------ #
    def _poll_episode_metric(self) -> Optional[float]:
        """Consume actor-reported episode ends; maybe update best_model.pth.

        Metric semantics (explicit, not "steps"-ambiguous): the metric is the
        per-episode value the ACTOR measured in real time — ``survival_s`` or
        ``total_reward`` (``rl.best_metric``).  The gate is the rolling mean
        over the last ``rl.best_metric_window`` finished episodes.
        """
        # DEEP-FIX: read the episode through the shared seqlock helper.  The
        # actor and the learner used to touch three independent shared values
        # with no ordering between them, so a poll could pair episode N's id
        # with episode N-1's survival time and gate best_model.pth on a number
        # that was never actually measured.
        ep = None
        if hasattr(self.counters, "read_episode_result"):
            ep = self.counters.read_episode_result(self._last_seen_episode_done_id)
        else:  # pragma: no cover - test doubles without the helper
            done_id = int(self.counters.last_episode_done_id.value)
            if done_id > 0 and done_id != self._last_seen_episode_done_id:
                ep = {
                    "episode_id": done_id,
                    "survival_s": float(self.counters.last_episode_survival_s.value),
                    "total_reward": float(self.counters.last_episode_reward.value),
                }
        if ep is None:
            return None
        done_id = int(ep["episode_id"])
        self._last_seen_episode_done_id = done_id
        if self.cfg.rl.best_metric == "total_reward":
            value = float(ep["total_reward"])
        else:
            value = float(ep["survival_s"])
        if not np.isfinite(value) or value <= 0:
            LOGGER.debug("episode %d reported a non-positive metric (%.3f); "
                         "not gating best_model on it", done_id, value)
            return None
        self._episode_window.append(value)
        rolling = float(sum(self._episode_window) / len(self._episode_window))
        # DEEP-FIX: when best_metric was still None the gate opened on the
        # VERY FIRST finished episode, so one lucky run was crowned
        # best_model.pth from a "rolling mean" of a single sample, and every
        # later episode then had to beat that one draw.  The documented rule
        # is a rolling mean over best_metric_window episodes, so the window
        # must actually be full before the first best is written.
        window_full = len(self._episode_window) >= int(self._episode_window.maxlen)
        improved = window_full and (
            self.ckpt.best_metric is None or rolling > self.ckpt.best_metric)
        self.best_rolling = rolling
        if improved:
            extra = {
                "update_step": self.update_step,
                "buffer_size": self.buffer.size,
                "best_metric": rolling,
                "best_metric_name": self.cfg.rl.best_metric,
                "best_metric_window": list(self._episode_window),
                "episode_id": done_id,
            }
            path = self.ckpt.save_model(self.agent.state_payload(), extra,
                                        which="best", metric=rolling,
                                        higher_is_better=True)
            if path is not None:
                LOGGER.info("new best model (%s rolling=%.2f, window=%s)",
                            self.cfg.rl.best_metric, rolling,
                            list(self._episode_window))
                put_bounded(self.metrics_q, {
                    "type": "best_model", "src": "learner",
                    "metric": self.cfg.rl.best_metric,
                    "rolling": rolling, "window": list(self._episode_window),
                })
        return rolling

    # ------------------------------------------------------------------ #
    def set_profile(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile}")
        if profile == self.profile:
            return
        LOGGER.warning("switching profile %s -> %s", self.profile, profile)
        dropped = self.buffer.size
        self.ckpt = CheckpointManager(str(self.ckpt.dir.parent), profile)
        self.profile = profile
        self.agent = DoubleDQNAgent(profile, self.cfg.rl, seed=self.cfg.seed)
        # DEEP-FIX: the replay buffer is profile-independent (frames, actions,
        # rewards, priorities), but it used to be replaced here, silently
        # discarding every transition collected since the last periodic
        # buffer save.  Keep the live buffer and only load from disk when we
        # do not already have one.
        if self.buffer.size == 0:
            self.buffer = self._load_or_fresh_buffer()
        else:
            LOGGER.info("keeping the live replay buffer across the profile "
                        "switch (%d transitions retained)", dropped)
        self._load_checkpoint()
        self.update_step = self.agent.update_count
        self._publish(force=True)

    # ------------------------------------------------------------------ #
    def add_transitions(self, items: list[Any]) -> int:
        n = 0
        for it in items:
            if isinstance(it, NStepTransition):
                self.buffer.add_nstep(it)
                n += 1
        return n

    def can_train(self, now: float) -> bool:
        if self.events_paused:
            return False
        if self.buffer.size < self.cfg.rl.warmup_transitions:
            return False
        min_gap = 1.0 / max(0.1, self.cfg.rl.max_updates_per_second)
        return (now - self._last_update_t) >= min_gap

    events_paused: bool = False

    def train_one(self, now: float) -> Optional[dict[str, float]]:
        beta = self.buffer.beta_for_step(self.update_step)
        t0 = time.perf_counter()
        try:
            batch = self.buffer.sample(self.cfg.rl.batch_size, beta, rng=self.rng)
        except IndexError:
            return None
        sample_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        metrics = self.agent.train_step(batch)
        update_ms = (time.perf_counter() - t1) * 1000.0
        # DEEP-FIX: this used to broadcast the batch MEAN |TD| over every
        # sampled index, which is the one thing prioritised replay must not
        # do — after a single update every sampled slot held the identical
        # priority (measured: all 32 slots == 2.19244182), so PER collapsed
        # towards uniform replay while still paying for the sum tree and the
        # importance-sampling correction of a distribution it no longer had.
        td = metrics.get("td_errors")
        if td is None:  # pragma: no cover - defensive against older agents
            td = metrics["td_error_abs_mean"] * np.ones(len(batch["indices"]))
        self.buffer.update_priorities(batch["indices"], td)
        self.update_step += 1
        self.counters.learner_update_step.value = self.update_step
        self.counters.beta.value = beta
        self.counters.td_loss.value = metrics["loss"]
        self.counters.q_mean.value = metrics["q_mean"]
        self.counters.buffer_size.value = self.buffer.size
        self._last_update_t = now
        metrics["sample_ms"] = sample_ms
        metrics["update_ms"] = update_ms
        metrics["beta"] = beta
        metrics["buffer_size"] = float(self.buffer.size)
        self.last_metrics = metrics
        self._maybe_checkpoint()
        if now - self._last_publish >= self._publish_every_s or \
                self.update_step % self._publish_every_updates == 0:
            self._publish()
        return metrics

    # ------------------------------------------------------------------ #
    def _publish(self, force: bool = False) -> None:
        self.agent.publish(self.shared_weights)
        self._last_publish = time.monotonic()

    def _maybe_checkpoint(self) -> None:
        every = max(50, self.cfg.rl.checkpoint_every_updates)
        if self.update_step % every != 0:
            return
        extra = {
            "update_step": self.update_step,
            "buffer_size": self.buffer.size,
            "epsilon": float(self.counters.epsilon.value),
            "env_frame_id": int(self.counters.env_frame_id.value),
            "perf": self.last_metrics,
            "best_metric": self.ckpt.best_metric,
            "best_metric_name": self.cfg.rl.best_metric,
            # DEEP-FIX: capture the Generator that actually draws the replay
            # batches.  capture_rng_states() used to snapshot a throwaway
            # np.random.default_rng() created inside the function, so the
            # "reproducible resume" the checkpoint advertises was recording
            # the state of a generator nobody uses.
            "rng": capture_rng_states(seed=self.cfg.seed, generator=self.rng),
        }
        self.ckpt.save_model(self.agent.state_payload(), extra, which="latest")
        if self.update_step - self._last_buffer_save_update >= \
                self.cfg.per.save_every_updates:
            if self.ckpt.buffer_save(self.buffer):
                self._last_buffer_save_update = self.update_step

    # ------------------------------------------------------------------ #
    # Behaviour cloning (Phase 1)
    # ------------------------------------------------------------------ #
    def pretrain(self, demos_dir: str, report=print) -> dict[str, Any]:
        from dataset import DemonstrationDataset, validate_directory

        episodes, reports = validate_directory(demos_dir)
        valid = [e for e, r in zip(episodes, reports) if r.ok]
        if len(valid) < self.cfg.bc.min_episodes:
            msg = (f"Only {len(valid)} valid demo episodes in {demos_dir}; "
                   f"need >= {self.cfg.bc.min_episodes}. BC skipped — online "
                   f"learning with warm-up will be used instead.")
            report(msg)
            put_bounded(self.metrics_q, {"type": "log", "level": "warning",
                                         "src": "learner", "msg": msg})
            return {"status": "skipped", "reason": "not_enough_episodes"}
        report(f"BC dataset: {len(valid)}/{len(episodes)} valid episodes")
        ds = DemonstrationDataset(valid, stack=self.cfg.perception.frame_stack)
        train_idx, val_idx = ds.split_by_episode(self.cfg.bc.val_fraction,
                                                 seed=self.cfg.seed)
        if not val_idx:
            val_idx = train_idx  # single-episode fallback (warned by validator)
        w = ds.class_weights(self.cfg.bc.class_balance)
        opt = None
        import torch

        opt = torch.optim.Adam(self.agent.online.parameters(),
                               lr=self.cfg.bc.learning_rate,
                               weight_decay=self.cfg.bc.weight_decay)
        rng = random.Random(self.cfg.seed)
        history: list[dict[str, Any]] = []
        for epoch in range(self.cfg.bc.epochs):
            order = list(train_idx)
            rng.shuffle(order)
            losses, accs = [], []
            for i in range(0, len(order), self.cfg.bc.batch_size):
                chunk = order[i : i + self.cfg.bc.batch_size]
                xs, ys = ds.batch(chunk)
                ws = w[ys]
                m = self.agent.bc_epoch(xs, ys, ws, optimizer=opt)
                losses.append(m["bc_loss"])
                accs.append(m["bc_acc"])
            vx, vy = ds.batch(val_idx[: 4096])
            ev = self.agent.bc_eval(vx, vy)
            row = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "train_acc": float(np.mean(accs)),
                "val_acc": ev["bc_acc"],
                "per_action": ev["per_action"],
            }
            history.append(row)
            report(f"BC epoch {row['epoch']}: loss={row['train_loss']:.4f} "
                   f"train_acc={row['train_acc']:.3f} val_acc={row['val_acc']:.3f}")
            put_bounded(self.metrics_q, {"type": "log", "level": "info",
                                         "src": "learner",
                                         "msg": f"BC epoch {row['epoch']} "
                                                f"val_acc={row['val_acc']:.3f}"})
        self.bc_history = history
        self.agent.sync_target()
        self._publish(force=True)
        extra = {"bc": history[-1], "update_step": self.update_step,
                 "dataset": str(demos_dir)}
        self.ckpt.save_model(self.agent.state_payload(), extra, which="best",
                             metric=history[-1]["val_acc"], higher_is_better=True)
        self.ckpt.save_model(self.agent.state_payload(), extra, which="latest")
        return {"status": "ok", "history": history}

    # ------------------------------------------------------------------ #
    def shutdown_save(self) -> None:
        extra = {"update_step": self.update_step,
                 "buffer_size": self.buffer.size, "shutdown": True,
                 "best_metric": self.ckpt.best_metric,
                 "best_metric_name": self.cfg.rl.best_metric,
                 "rng": capture_rng_states(seed=self.cfg.seed, generator=self.rng)}
        self.ckpt.save_model(self.agent.state_payload(), extra, which="latest")
        self.ckpt.buffer_save(self.buffer)
        LOGGER.info("shutdown save complete (updates=%d, buffer=%d)",
                    self.update_step, self.buffer.size)


def learner_main(
    stop_event: Any,
    pause_event: Any,
    cmd_q: Any,
    transition_q: Any,
    metrics_q: Any,
    shared_weights: SharedWeights,
    counters: Any,
    cfg_dict: dict,
    checkpoints_dir: str,
    log_dir: str = "logs",
) -> None:
    """Entry point of the learner process."""
    setup_logging("learner", log_dir)
    _torch_threads(1)
    cfg = BotConfig.from_dict(cfg_dict)
    try:
        learner = Learner(cfg, shared_weights, counters, metrics_q, checkpoints_dir)
        learner._publish(force=True)
    except Exception as exc:
        put_bounded(metrics_q, {"type": "error", "src": "learner",
                                "error": f"{type(exc).__name__}: {exc}",
                                "tb": format_exception(exc)})
        return
    LOGGER.info("learner start (profile=%s params=%d buffer=%d)",
                learner.profile, learner.agent.count_params(), learner.buffer.size)

    last_report = time.monotonic()
    transitions_since_report = 0
    learner_dropped = [0]
    try:
        while not stop_event.is_set():
            now = time.monotonic()
            did_work = False
            # 1. commands
            for msg in drain(cmd_q, limit=32):
                did_work = True
                cmd = msg.get("cmd") if isinstance(msg, dict) else None
                if cmd == "shutdown":
                    stop_event.set()
                elif cmd == "set_profile":
                    learner.set_profile(msg.get("profile", learner.profile))
                    counters.set_profile(learner.profile)
                elif cmd == "pretrain":
                    res = learner.pretrain(msg.get("demos_dir", "demos"))
                    put_bounded(metrics_q, {"type": "pretrain_done", "result": res})
                elif cmd == "save_buffer":
                    learner.ckpt.buffer_save(learner.buffer)
            # 2. transitions.  DEEP-FIX: the actor's auto-downgrader used to
            # put {"__cmd__": "set_profile", ...} on THIS queue while the
            # learner only ever read commands from cmd_q, and
            # Learner.add_transitions() silently discarded anything that was
            # not an NStepTransition.  The downgrade therefore never reached
            # the learner: the actor swapped to a lighter net and started
            # reading a meaningless PREFIX of the heavier learner's weight
            # vector.  Commands are now accepted on either channel, and
            # add_transitions() reports what it had to drop.
            items = drain(transition_q, limit=128)
            if items:
                did_work = True
                inline = [m for m in items
                          if isinstance(m, dict) and m.get("__cmd__") == "set_profile"]
                for msg in inline:
                    learner.set_profile(msg.get("profile", learner.profile))
                    counters.set_profile(learner.profile)
                    put_bounded(metrics_q, {
                        "type": "log", "level": "warning", "src": "learner",
                        "msg": f"profile switched to {learner.profile} "
                               f"(requested by the actor's auto-downgrader)",
                    })
                kept = learner.add_transitions(items)
                transitions_since_report += kept
                dropped = len(items) - kept - len(inline)
                if dropped > 0:
                    learner_dropped[0] += dropped
            # 3. training (respect pause)
            learner.events_paused = pause_event.is_set()
            if learner.can_train(now):
                learner.train_one(now)
                did_work = True
            # 3.5 online best-model gate (actor-reported episode metrics)
            learner._poll_episode_metric()
            # 4. metrics heartbeat
            if now - last_report >= cfg.perf.report_interval_s:
                last_report = now
                put_bounded(metrics_q, {
                    "type": "metrics", "src": "learner", "t": time.time(),
                    "data": {
                        "learner_updates": learner.update_step,
                        "td_loss": learner.last_metrics.get("loss", 0.0),
                        "q_mean": learner.last_metrics.get("q_mean", 0.0),
                        "grad_norm": learner.last_metrics.get("grad_norm", 0.0),
                        "lr": learner.last_metrics.get("lr", 0.0),
                        "buffer_size": learner.buffer.size,
                        "buffer_mb": learner.buffer.nbytes() / (1024 ** 2),
                        "beta": learner.counters.beta.value,
                        "new_transitions": transitions_since_report,
                        "replay_sample_ms": learner.last_metrics.get("sample_ms", 0.0),
                        "update_ms": learner.last_metrics.get("update_ms", 0.0),
                        "best_metric_name": learner.cfg.rl.best_metric,
                        "best_rolling": (-1.0 if learner.best_rolling is None
                                         else learner.best_rolling),
                        "best_gate": (None if learner.ckpt.best_metric is None
                                      else float(learner.ckpt.best_metric)),
                    },
                    "dropped_transitions": learner_dropped[0],
                })
                transitions_since_report = 0
            # DEEP-FIX: this loop had no idle throttle at all, so a learner
            # that could not train yet (buffer below warmup_transitions, or
            # paused for a death/respawn) spun at 100% of one core doing
            # nothing.  Measured on a 2-core box: 100% CPU, 0 updates, 60%
            # system load -- exactly the starvation the "separate process so
            # Chrome keeps headroom" design was meant to prevent.
            if not did_work:
                time.sleep(0.005)
    except Exception as exc:
        put_bounded(metrics_q, {"type": "error", "src": "learner",
                                "error": f"{type(exc).__name__}: {exc}",
                                "tb": format_exception(exc)})
    finally:
        try:
            learner.shutdown_save()
        except Exception as exc:
            LOGGER.error("shutdown save failed:\n%s", format_exception(exc))
        LOGGER.info("learner stop")


def learner_process_target(*args, **kwargs) -> None:
    learner_main(*args, **kwargs)
