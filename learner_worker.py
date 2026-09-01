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
        # v1.21.0: choose between the standard QR-DQN
        # and the BC-pretrained DQfD agent.  When the
        # user has the ``bc.bc_pretrain`` flag enabled
        # (the new default — see audit_bc_then_rl.py)
        # the learner constructs a :class:`DQfDAgent`
        # so the :meth:`pretrain` flow can hand the
        # expert demos straight to the joint-loss
        # agent.  The DQfD agent is a *drop-in* for
        # the QR-DQN agent: same forward, same TD
        # loss, same target net, same RND hook — the
        # only difference is the BC + margin loss
        # terms that fire whenever a demo buffer has
        # been loaded.
        use_dqfd = bool(getattr(cfg.bc, "bc_pretrain", False))
        if use_dqfd:
            from bc_pretrain import build_dqfd_agent
            from dqfd_agent import DQfDConfig
            dqfd_cfg = DQfDConfig(
                lambda_bc=float(getattr(cfg.bc, "dqfd_lambda_bc", 0.5)),
                lambda_margin=float(getattr(cfg.bc, "dqfd_lambda_margin", 0.1)),
                margin=float(getattr(cfg.bc, "dqfd_margin", 0.8)),
                supervised_decay_episodes=int(getattr(
                    cfg.bc, "dqfd_decay_episodes", 1000)),
                no_exploration=bool(getattr(
                    cfg.rl, "disable_exploration_after_bc", True)),
            )
            self.agent = build_dqfd_agent(
                self.profile, cfg.rl, dqfd_cfg,
                in_frames=cfg.perception.frame_stack,
                size=cfg.perception.policy_size,
                num_quantiles=int(getattr(cfg.rl, "num_quantiles", 51)),
                seed=cfg.seed)
        elif getattr(cfg.rl, "distributional", True):
            from agent_distributional import DistributionalDoubleDQNAgent
            self.agent = DistributionalDoubleDQNAgent(
                self.profile, cfg.rl, num_quantiles=int(
                    getattr(cfg.rl, "num_quantiles", 51)),
                seed=cfg.seed)
        else:
            self.agent = DoubleDQNAgent(self.profile, cfg.rl, seed=cfg.seed)
        # Attach the RND (curiosity) module if the config
        # asked for it.  The RND module is constructed in the
        # same process so the intrinsic-reward tensor never
        # has to cross the IPC boundary.
        self.rnd = None
        if getattr(cfg.rl, "use_rnd", True):
            from rnd import RNDConfig, RNDModule
            r_cfg_raw = getattr(cfg, "rnd", None)
            r_cfg = RNDConfig(
                enabled=getattr(r_cfg_raw, "enabled", True),
                feature_dim=int(getattr(r_cfg_raw, "feature_dim", 128)),
                beta=float(getattr(r_cfg_raw, "beta", 0.5)),
                normalizer_alpha=float(getattr(r_cfg_raw, "normalizer_alpha", 0.99)),
                train_every_n_updates=int(getattr(r_cfg_raw, "train_every_n_updates", 1)),
                target_seed=int(getattr(r_cfg_raw, "target_seed", 12345)),
            ) if r_cfg_raw is not None else RNDConfig()
            self.rnd = RNDModule(r_cfg, in_frames=cfg.perception.frame_stack)
            # Bind the module to the agent (the agent uses it
            # inside train_step to inject the intrinsic bonus
            # into the reward tensor).
            if hasattr(self.agent, "attach_rnd"):
                self.agent.attach_rnd(self.rnd)
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
        # DEEP-FIX: did behaviour cloning ever produce a policy?  Persisted in
        # every checkpoint so a restart keeps exploiting the BC policy instead
        # of silently dropping back to epsilon~1 random play.
        self.bc_done = False
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
        # -- self-imitation (DQfD-style) --------------------------------- #
        # Good episodes are written to <demos_dir>/self/*.npz with the
        # same format as human demos, so the same training pipeline
        # (keypress window + mirror + BC) consumes both pools.  The
        # gate uses the same rolling survival mean the best-model
        # gate already maintains — they cannot disagree on "is this
        # a good run?".
        from self_imitation import SelfImitationConfig, SelfImitationRecorder
        si_cfg = SelfImitationConfig(
            enabled=bool(getattr(cfg.reward, "self_imitation_factor", 1.2) > 0),
            factor=float(getattr(cfg.reward, "self_imitation_factor", 1.2)),
            max_episodes=int(getattr(cfg.reward, "self_imitation_max", 50)),
        )
        # Resolve the demo directory the same way the GUI does — the
        # ``PathsConfig`` keeps the default ``"demos"`` so the
        # recorder and the GUI end up writing to the same place.
        from pathlib import Path as _P
        demos_dir = _P(str(getattr(cfg.paths, "demos_dir", "demos")))
        self.self_imitation = SelfImitationRecorder(si_cfg, demos_dir)
        self._episodes_since_last_self_bc = 0
        # Bounded ring of the most recent transitions; used only when
        # the self-imitation gate opens and we need to dump the
        # current episode to disk.  Capped at 30_000 frames so a long
        # episode does not blow up the learner's working set.  The
        # store is appended in :meth:`add_transitions` and cleared on
        # episode end (whether the episode was saved or not).
        from collections import deque as _dq
        self._recent_transitions: _dq = _dq(maxlen=30_000)
        # ----- Latent-space dreamer (mental rehearsal) -----
        # The third leg of "AI tự học full aggressive": a tiny
        # variational autoencoder that distils the self-imitation
        # pool into a latent space, then *perturbs* a latent to
        # produce an "abstract" frame, then replays the abstract
        # frame through the synthetic env to check whether the
        # policy still survives — that is the "khái quát" the
        # user asked for.  When env-replay is unavailable (no
        # ``_dream_env`` attribute, e.g. in unit tests), the
        # dreamer falls back to the Q-value of the live policy.
        from dreamer import DreamerConfig, DreamerTrainer
        from pathlib import Path as _PathDreamer
        d_cfg_raw = getattr(cfg, "dreamer", None)
        if d_cfg_raw is None:
            d_cfg = DreamerConfig(enabled=False)
        else:
            d_cfg = DreamerConfig(
                enabled=bool(getattr(d_cfg_raw, "enabled", True)),
                latent_dim=int(getattr(d_cfg_raw, "latent_dim", 64)),
                dream_noise_std=float(getattr(d_cfg_raw, "dream_noise_std", 0.30)),
                max_episodes=int(getattr(d_cfg_raw, "max_episodes", 50)),
                beta_kl=float(getattr(d_cfg_raw, "beta_kl", 0.01)),
                frames_per_dream=int(getattr(d_cfg_raw, "frames_per_dream", 32)),
                train_every_n_updates=int(getattr(d_cfg_raw, "train_every_n_updates", 100)),
                dream_every_s=float(getattr(d_cfg_raw, "dream_every_s", 60.0)),
                dreams_per_round=int(getattr(d_cfg_raw, "dreams_per_round", 8)),
                positive_q_threshold=float(getattr(d_cfg_raw, "positive_q_threshold", 0.5)),
                negative_q_threshold=float(getattr(d_cfg_raw, "negative_q_threshold", -0.5)),
            )
        abstract_dir = _PathDreamer(demos_dir) / "abstract"
        self.dreamer = DreamerTrainer(d_cfg, abstract_dir)
        # Wire the self-imitation pool in so the dreamer can pick
        # seed frames.  Done after construction so an empty pool
        # does not block the trainer from being unit-tested.
        self.dreamer.attach_learner(self, self.self_imitation)
        # The env back-door is set by the actor/headless harness at
        # startup; the dreamer checks ``learner._dream_env`` lazily
        # so a missing attribute just means "use the Q fallback".
        self._dream_env: object | None = None
        self._load_checkpoint()

    # ------------------------------------------------------------------ #
    def _load_or_fresh_buffer(self) -> PrioritizedReplayBuffer:
        fresh = lambda: PrioritizedReplayBuffer(  # noqa: E731
            self.cfg.per, frame_size=self.cfg.perception.policy_size,
            gamma=self.cfg.rl.gamma,
        )
        loaded, ok = self.ckpt.buffer_load(
            lambda: PrioritizedReplayBuffer.load(
                str(self.ckpt.buffer_path), self.cfg.per,
                frame_size=self.cfg.perception.policy_size,
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
                # DEEP-FIX: a BC-pretrained checkpoint must keep driving the
                # actor after a restart.  Without this, bc_pretrained reset
                # to 0 on every launch and the actor explored at epsilon~1
                # again, ignoring the BC policy it had just loaded.
                if extra.get("bc") is not None or extra.get("bc_done"):
                    self.bc_done = True
                    self.counters.bc_pretrained.value = 1.0
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
                "bc_done": self.bc_done,
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
        # ---- self-imitation: maybe save the just-finished episode ---- #
        # Run after the best-model gate so the self-imitation decision
        # sees the freshly-updated rolling mean.  Failures here never
        # raise — the actor's next episode is more important than
        # archiving the previous one.
        try:
            self._maybe_self_imitate(done_id, value)
        except Exception as exc:
            LOGGER.warning("self-imitation hook failed: %s", exc)
        return rolling

    # ------------------------------------------------------------------ #
    def _maybe_self_imitate(self, episode_id: int, survival_s: float) -> None:
        """Decide whether to archive the just-finished episode.

        The "decision" is the recorder's job; this function is the
        wiring that drains the recent-transition ring, calls
        :meth:`SelfImitationRecorder.save_episode`, and schedules a
        self-BC if the gate is configured to retrain periodically.
        """
        decision = self.self_imitation.note_episode(episode_id, survival_s)
        if not decision.get("saved"):
            # Always clear the ring on episode end so the next
            # episode does not accumulate against the previous one.
            self._recent_transitions.clear()
            return
        recent = self._consume_recent_episode()
        if recent is None:
            # Learner started after the actor's recent frames had
            # already rotated; without the data the gate can only
            # honestly say "no" — clear the ring and move on.
            self._recent_transitions.clear()
            return
        path = self.self_imitation.save_episode(
            episode_id, survival_s,
            recent["frames"], recent["actions"],
            recent["timestamps"], recent["done"],
            meta={"source": "self_imitation", "config": self.cfg.rl.profile})
        # Drop the recent transitions regardless of save success.
        self._recent_transitions.clear()
        if path is None:
            return
        LOGGER.info("self-imitation: archived episode %d (survival=%.2fs, "
                    "threshold=%.2fs) -> %s",
                    episode_id, survival_s,
                    decision.get("threshold", 0.0), path)
        put_bounded(self.metrics_q, {
            "type": "log", "level": "info", "src": "learner",
            "msg": f"📼 self-imitation: lưu episode {episode_id} "
                   f"(sống {survival_s:.1f}s) — sẽ được BC lại tự động"})
        put_bounded(self.metrics_q, {
            "type": "self_imitation_saved", "src": "learner",
            "path": str(path), "episode_id": episode_id,
            "survival_s": survival_s})
        # Schedule a re-BC run that includes the new episode.
        self._episodes_since_last_self_bc += 1
        if int(self.self_imitation.cfg.bc_every_n_episodes) > 0 and \
                self._episodes_since_last_self_bc >= \
                int(self.self_imitation.cfg.bc_every_n_episodes):
            self._episodes_since_last_self_bc = 0
            self._self_bc_pending = True

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
        # Same distributional-or-scalar choice as the
        # constructor (see __init__).
        if getattr(self.cfg.rl, "distributional", True):
            from agent_distributional import DistributionalDoubleDQNAgent
            self.agent = DistributionalDoubleDQNAgent(
                profile, self.cfg.rl,
                num_quantiles=int(getattr(self.cfg.rl, "num_quantiles", 51)),
                seed=self.cfg.seed)
        else:
            self.agent = DoubleDQNAgent(profile, self.cfg.rl, seed=self.cfg.seed)
        # Re-bind the RND module if we have one (the
        # networks inside RND are not profile-dependent so
        # we can keep the same instance).
        if self.rnd is not None and hasattr(self.agent, "attach_rnd"):
            self.agent.attach_rnd(self.rnd)
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
                # Mirror the latest N transitions into the recent
                # ring so the self-imitation gate can dump the
                # current episode if the just-finished one is good.
                # A full NStepTransition carries the obs stack and
                # the action — that is everything the .npz file needs.
                self._recent_transitions.append(it)
                n += 1
        return n

    def _consume_recent_episode(self) -> Optional[dict[str, Any]]:
        """Drain the recent-transition ring into an episode-shaped dict.

        Returns ``None`` when the ring holds no transitions, which
        happens when the learner starts after the actor already
        published the episode (the latest-frame ring keeps only the
        most recent N slots; the learner started before the actor
        reached a full episode).  In that case the self-imitation
        gate is forced to "no" instead of producing a corrupt file.
        """
        if not self._recent_transitions:
            return None
        frames: list[np.ndarray] = []
        actions: list[int] = []
        timestamps: list[float] = []
        for tr in self._recent_transitions:
            # Use the (current) obs frame, not the next_obs — the
            # .npz format expects "frame -> action -> frame" so the
            # BC network sees (s_t, a_t) pairs.  Frame-stacks are
            # flattened back to single 84x84 frames because the demo
            # format does not store stacks.
            frame = np.asarray(tr.obs[-1], dtype=np.uint8)  # newest slot
            frames.append(frame)
            actions.append(int(tr.action))
            timestamps.append(float(getattr(tr, "env_id", 0)) / 30.0)
        if not frames:
            return None
        n = len(frames)
        done = np.zeros(n, dtype=bool)
        done[-1] = True  # the latest transition was a terminal one
        return {
            "frames": np.stack(frames),
            "actions": np.asarray(actions, dtype=np.int64),
            "timestamps": np.asarray(timestamps, dtype=np.float64),
            "done": done,
        }

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
            "bc_done": self.bc_done,
        }
        self.ckpt.save_model(self.agent.state_payload(), extra, which="latest")
        if self.update_step - self._last_buffer_save_update >= \
                self.cfg.per.save_every_updates:
            if self.ckpt.buffer_save(self.buffer):
                self._last_buffer_save_update = self.update_step

    # ------------------------------------------------------------------ #
    # Behaviour cloning (Phase 1)
    # ------------------------------------------------------------------ #
    def pretrain(self, demos_dir: str, report=print,
                 force: bool = False) -> dict[str, Any]:
        from dataset import DemonstrationDataset, validate_directory

        # DEEP-FIX (BC-first): if behaviour cloning already produced a policy
        # (this session or restored from a checkpoint), do NOT silently re-run
        # it — that would overwrite RL progress with a fresh BC policy.  The
        # GUI's "BC before train" flow calls with force=False; the manual
        # "Tiền-huấn luyện" button passes force=True.
        if self.bc_done and not force:
            msg = ("BC đã chạy trước đó (checkpoint có sẵn) — bỏ qua để không "
                   "ghi đè tiến trình RL. Bấm nút Tiền-huấn luyện để ép chạy lại.")
            report(msg)
            put_bounded(self.metrics_q, {"type": "log", "level": "info",
                                         "src": "learner", "msg": msg})
            return {"status": "already_done"}

        put_bounded(self.metrics_q, {"type": "log", "level": "info",
                                     "src": "learner",
                                     "msg": f"BC: đang kiểm tra demo trong {demos_dir}…"})
        episodes, reports = validate_directory(
            demos_dir,
            expected_size=self.cfg.perception.policy_size)
        valid = [e for e, r in zip(episodes, reports) if r.ok]
        put_bounded(self.metrics_q, {"type": "log", "level": "info",
                                     "src": "learner",
                                     "msg": f"BC: {len(valid)}/{len(episodes)} episode hợp lệ "
                                            f"(cần >= {self.cfg.bc.min_episodes})"})
        if len(valid) < self.cfg.bc.min_episodes:
            msg = (f"Only {len(valid)} valid demo episodes in {demos_dir}; "
                   f"need >= {self.cfg.bc.min_episodes}. BC skipped — online "
                   f"learning with warm-up will be used instead.")
            report(msg)
            put_bounded(self.metrics_q, {"type": "log", "level": "warning",
                                         "src": "learner", "msg": msg})
            return {"status": "skipped", "reason": "not_enough_episodes"}
        report(f"BC dataset: {len(valid)}/{len(episodes)} valid episodes")
        ds = DemonstrationDataset(valid, stack=self.cfg.perception.frame_stack,
                                  dodge_oversample=self.cfg.bc.dodge_oversample)
        # DEEP-FIX ("why did the human press?"): surface how many of each dodge
        # the demos contain, so a dodge the player never demonstrated is obvious
        # (a dodge with 0 presses can never be learned).
        _presses = ds.dodge_press_counts()
        _names = {1: "trái", 2: "phải", 3: "nhảy", 4: "trượt"}
        put_bounded(self.metrics_q, {
            "type": "log", "level": "info", "src": "learner",
            "msg": "🔍 TẠI SAO BẤM — số pha né trong demo: "
                   + ", ".join(f"{_names[a]}={_presses.get(a, 0)}"
                               for a in (1, 2, 3, 4))
                   + f" | oversample khung né x{self.cfg.bc.dodge_oversample}"})
        train_idx, val_idx = ds.split_by_episode(self.cfg.bc.val_fraction,
                                                 seed=self.cfg.seed)
        if not val_idx:
            val_idx = train_idx  # single-episode fallback (warned by validator)
        w = ds.class_weights(self.cfg.bc.class_balance)
        opt = None
        import torch

        # DEEP-FIX (v1.21.0): when the agent is a
        # :class:`DQfDAgent` we use the joint-loss
        # pretrain path (cross-entropy + DQfD arming)
        # INSTEAD of the standard BC path.  The DQfD
        # path also pre-fills the replay buffer with
        # the demos so the BC anchor is sampled
        # throughout online RL.
        if bool(getattr(self.cfg.bc, "bc_pretrain", False)):
            return self._pretrain_dqfd(ds, train_idx, val_idx, w,
                                         report, demos_dir)
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
        put_bounded(self.metrics_q, {"type": "log", "level": "info",
                                     "src": "learner",
                                     "msg": f"BC: hoàn tất {len(history)} epoch, "
                                            f"val_acc={history[-1]['val_acc']:.3f}, đang lưu checkpoint…"})
        # DEEP-FIX ("why did the human press?"): the headline val_acc is usually
        # inflated by NOOP; the per-dodge accuracy is what decides survival.  Log
        # it explicitly so a bot that "knows BC" but cannot dodge is diagnosable.
        _pa = history[-1].get("per_action", {}) if history else {}
        put_bounded(self.metrics_q, {
            "type": "log", "level": "info", "src": "learner",
            "msg": "🎯 accuracy sau BC theo action (né = sống còn): "
                   + ", ".join(f"{_names[a]}={float(_pa.get(a, 0)):.2f}"
                               for a in (1, 2, 3, 4))
                   + f" | NOOP={float(_pa.get(0, 0)):.2f}"})
        self.bc_history = history
        self.agent.sync_target()
        # DEEP-FIX: tell the actor a good policy now exists so it stops playing
        # randomly (epsilon~1) and exploits the BC policy instead.
        self.bc_done = True
        self.counters.bc_pretrained.value = 1.0
        self._publish(force=True)
        extra = {"bc": history[-1], "update_step": self.update_step,
                 "dataset": str(demos_dir), "bc_done": True}
        self.ckpt.save_model(self.agent.state_payload(), extra, which="best",
                             metric=history[-1]["val_acc"], higher_is_better=True)
        self.ckpt.save_model(self.agent.state_payload(), extra, which="latest")
        return {"status": "ok", "history": history}

    # ------------------------------------------------------------------ #
    def _pretrain_dqfd(self, ds, train_idx, val_idx, w,
                        report, demos_dir: str) -> dict[str, Any]:
        """BC pretrain path used when the agent is a
        :class:`DQfDAgent`.  Hands the demo frames +
        actions to :meth:`DQfDAgent.pretrain_demos` and
        pre-fills the replay buffer so the supervised
        + margin terms fire on every subsequent online
        update.

        Why a separate method
        ---------------------
        The standard BC path runs ``self.agent.bc_epoch``
        in a loop.  The DQfD agent does NOT have
        ``bc_epoch`` (it uses ``pretrain_demos`` instead,
        which is the standard Hester 2018 recipe).  The
        demo format is also different: the standard BC
        path uses ``ds.batch(chunk)`` (the dataset
        loader's batching), whereas the DQfD agent
        expects ``[N, F, H, W]`` float32 in [0, 1]
        directly.  Adapting the demo to the right shape
        lives in this method.
        """
        from bc_pretrain import pretrain_and_arm_dqfd
        # Materialise the train+val frames and actions
        # into the ``[N, F, H, W]`` float32 / [N] int64
        # format the DQfD agent expects.
        all_idx = list(train_idx) + list(val_idx)
        if not all_idx:
            return {"status": "skipped", "reason": "no_indices"}
        xs, ys = ds.batch(all_idx)
        # The dataset returns obs in the (B, F*H*W) or
        # (B, F, H, W) layout the dataset uses; we need
        # to figure out which.  The simplest is to ask
        # the dataset directly.
        obs_arr, act_arr = self._materialise_demos(ds, all_idx)
        if len(obs_arr) == 0:
            return {"status": "skipped", "reason": "empty_demos"}
        # Convert to [0, 1] float32 if the dataset returns
        # uint8.
        if obs_arr.dtype == np.uint8:
            obs_arr = obs_arr.astype(np.float32) / 255.0
        report(f"DQfD BC pretrain on {len(obs_arr)} demo frames…")
        put_bounded(self.metrics_q, {
            "type": "log", "level": "info", "src": "learner",
            "msg": f"🧠 DQfD BC pretrain trên {len(obs_arr)} khung hình, "
                   f"agent={type(self.agent).__name__}."})
        result = pretrain_and_arm_dqfd(self.agent, obs_arr, act_arr,
                                          n_epochs=self.cfg.bc.epochs,
                                          batch_size=self.cfg.bc.batch_size,
                                          lr=self.cfg.bc.learning_rate)
        # Pre-fill the replay buffer with the demo
        # transitions so the next sample batch includes
        # them with high priority.  This is the
        # "demo pre-fill" pattern that keeps the BC
        # anchor alive once online RL starts.
        try:
            from bc_pretrain import prefill_replay_with_demos
            # Synthesize 1-step rewards (0 except +1 at
            # the last step of each episode so the n-step
            # builder has a terminal reward).
            rewards = np.zeros(len(act_arr), dtype=np.float32)
            # Find episode boundaries via the done mask.
            done_mask = ds.episode_done_mask(all_idx)
            rewards[done_mask] = 1.0
            n_added = prefill_replay_with_demos(
                self.buffer, obs_arr, act_arr, rewards,
                frame_stack=self.cfg.perception.frame_stack,
                gamma=self.cfg.rl.gamma,
                n_step=self.cfg.rl.n_step,
                priority_boost=100.0)
            report(f"  pre-filled replay buffer with {n_added} "
                   f"n-step demo transitions")
            put_bounded(self.metrics_q, {
                "type": "log", "level": "info", "src": "learner",
                "msg": f"📥 replay buffer pre-filled với {n_added} "
                       f"demo transitions (priority boost 100×)."})
        except Exception as exc:  # noqa: BLE001 - reported
            LOGGER.warning("demo pre-fill failed: %s", exc)
            put_bounded(self.metrics_q, {
                "type": "log", "level": "warning", "src": "learner",
                "msg": f"demo pre-fill lỗi: {exc}"})
        # Build a minimal "history" record so the
        # checkpoint format stays consistent with the
        # standard BC path.
        history = [{
            "epoch": self.cfg.bc.epochs,
            "train_loss": result["bc_loss"],
            "train_acc": 0.0,
            "val_acc": 0.0,
            "per_action": {},
            "n_frames": result["n_frames"],
            "method": "dqfd",
        }]
        self.bc_history = history
        self.agent.sync_target()
        self.bc_done = True
        self.counters.bc_pretrained.value = 1.0
        self._publish(force=True)
        extra = {"bc": history[-1], "update_step": self.update_step,
                 "dataset": str(demos_dir), "bc_done": True,
                 "method": "dqfd"}
        self.ckpt.save_model(self.agent.state_payload(), extra, which="best",
                             metric=result["bc_loss"], higher_is_better=False)
        self.ckpt.save_model(self.agent.state_payload(), extra, which="latest")
        put_bounded(self.metrics_q, {
            "type": "log", "level": "info", "src": "learner",
            "msg": f"✅ DQfD BC xong: loss={result['bc_loss']:.4f} "
                   f"trên {result['n_frames']} frames — actor sẽ "
                   f"exploit (ε=0) cho tới khi policy drift."})
        return {"status": "ok", "history": history, "method": "dqfd",
                "bc_loss": result["bc_loss"]}

    def _materialise_demos(self, ds, idx) -> tuple[np.ndarray, np.ndarray]:
        """Helper: pull (obs, action) arrays for the
        given dataset indices.  We dispatch on the
        dataset's ``stack`` and ``frame_size``
        attributes to build the right output shape.

        The returned obs is ``[N, F, H, W]`` float32
        in [0, 1]; the returned actions is ``[N]``
        int64.
        """
        # The dataset's ``batch`` method returns a
        # concatenation of every (obs, action) for the
        # indices.  We re-batch in chunks to keep peak
        # memory bounded.
        chunk = 256
        obs_chunks, act_chunks = [], []
        for start in range(0, len(idx), chunk):
            sub = idx[start:start + chunk]
            xs, ys = ds.batch(sub)
            obs_chunks.append(np.asarray(xs))
            act_chunks.append(np.asarray(ys))
        if not obs_chunks:
            return np.zeros((0,)), np.zeros((0,))
        obs_arr = np.concatenate(obs_chunks, 0)
        act_arr = np.concatenate(act_chunks, 0)
        # The dataset may return a flattened (N, F*H*W)
        # layout.  Inflate to (N, F, H, W) so the
        # ``pretrain_demos`` reshape path is consistent.
        if obs_arr.ndim == 2:
            F = int(getattr(ds, "stack", 4))
            H = int(getattr(ds, "frame_size", 84))
            total = obs_arr.shape[1]
            assert total == F * H * H, (
                f"demo obs shape {obs_arr.shape} not divisible "
                f"by (F={F}, H={H})")
            obs_arr = obs_arr.reshape(-1, F, H, H)
        return obs_arr, act_arr

    # ------------------------------------------------------------------ #
    def pretrain_with_self_imitation(self, human_dir: str) -> dict[str, Any]:
        """BC on the union of human demos and self-imitation episodes.

        Reads the human demos from ``human_dir`` and the self pool
        from :attr:`self_imitation.out_dir`, validates both, splits by
        EPISODE (so a self copy and its source human demo can never
        land in different folds), trains BC, and saves the new policy
        as ``best_model.pth`` keyed on the val accuracy.  The point
        is the same as :meth:`pretrain` — it bakes the recent good
        runs into the policy without waiting for a manual button
        press.
        """
        from dataset import validate_directory
        from pathlib import Path as _P
        human_eps, human_reps = validate_directory(
            human_dir,
            expected_size=self.cfg.perception.policy_size)
        self_dir = str(self.self_imitation.out_dir)
        self_eps, self_reps = validate_directory(
            self_dir,
            expected_size=self.cfg.perception.policy_size)
        human_valid = [e for e, r in zip(human_eps, human_reps) if r.ok]
        self_valid = [e for e, r in zip(self_eps, self_reps) if r.ok]
        if not human_valid and not self_valid:
            return {"status": "skipped", "reason": "no_valid_episodes"}
        # Reuse the standard BC path by pretending the union came
        # from a single directory.  We could just call :meth:`pretrain`
        # on a synthetic loader but that would re-publish the
        # "BC done" log; calling the BC trainer inline keeps the
        # output quiet and uses the dataset's own validation.
        from dataset import DemonstrationDataset
        from demo_augment import DemoAugmentor
        all_eps = human_valid + self_valid
        ds = DemonstrationDataset(
            all_eps, stack=self.cfg.perception.frame_stack,
            dodge_oversample=self.cfg.bc.dodge_oversample,
            augment=DemoAugmentor(),  # default = keypress window + mirror
        )
        train_idx, val_idx = ds.split_by_episode(self.cfg.bc.val_fraction,
                                                 seed=self.cfg.seed)
        if not val_idx:
            val_idx = train_idx
        import torch
        opt = torch.optim.Adam(self.agent.online.parameters(),
                               lr=self.cfg.bc.learning_rate,
                               weight_decay=self.cfg.bc.weight_decay)
        w = ds.class_weights(self.cfg.bc.class_balance)
        import random
        rng = random.Random(self.cfg.seed)
        history: list[dict[str, Any]] = []
        for epoch in range(self.cfg.bc.epochs):
            order = list(train_idx)
            rng.shuffle(order)
            losses, accs = [], []
            for i in range(0, len(order), self.cfg.bc.batch_size):
                chunk = order[i: i + self.cfg.bc.batch_size]
                xs, ys = ds.batch(chunk)
                ws = w[ys]
                m = self.agent.bc_epoch(xs, ys, ws, optimizer=opt)
                losses.append(m["bc_loss"])
                accs.append(m["bc_acc"])
            vx, vy = ds.batch(val_idx[:4096])
            ev = self.agent.bc_eval(vx, vy)
            history.append({"epoch": epoch + 1,
                            "train_loss": float(np.mean(losses)),
                            "train_acc": float(np.mean(accs)),
                            "val_acc": ev["bc_acc"],
                            "per_action": ev["per_action"]})
        self.bc_history = history
        self.agent.sync_target()
        self.bc_done = True
        self.counters.bc_pretrained.value = 1.0
        self._publish(force=True)
        extra = {"bc": history[-1], "update_step": self.update_step,
                 "dataset": f"human+self({len(self_valid)})",
                 "bc_done": True, "self_imitation": True}
        self.ckpt.save_model(self.agent.state_payload(), extra, which="best",
                             metric=history[-1]["val_acc"],
                             higher_is_better=True)
        self.ckpt.save_model(self.agent.state_payload(), extra,
                             which="latest")
        put_bounded(self.metrics_q, {
            "type": "log", "level": "info", "src": "learner",
            "msg": f"♻ self-BC xong trên {len(human_valid)} human + "
                   f"{len(self_valid)} self, val_acc="
                   f"{history[-1]['val_acc']:.3f}"})
        return {"status": "ok", "history": history,
                "n_human": len(human_valid), "n_self": len(self_valid)}

    # ------------------------------------------------------------------ #
    def shutdown_save(self) -> None:
        extra = {"update_step": self.update_step,
                 "buffer_size": self.buffer.size, "shutdown": True,
                 "bc_done": self.bc_done,
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
                    # DEEP-FIX (BC-first): always emit pretrain_done, even on a
                    # BC error — otherwise the GUI would wait forever and leave
                    # the actor paused (it is held until BC finishes).
                    try:
                        res = learner.pretrain(
                            msg.get("demos_dir", "demos"),
                            force=bool(msg.get("force", False)))
                    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                        res = {"status": "error", "reason": format_exception(exc)}
                        put_bounded(metrics_q, {
                            "type": "log", "level": "error", "src": "learner",
                            "msg": f"BC lỗi: {res['reason']}"})
                    put_bounded(metrics_q, {"type": "pretrain_done", "result": res})
                elif cmd == "pretrain_with_self":
                    # GUI button or auto-scheduler asks to retrain on the
                    # union of human demos and self-imitation episodes.
                    try:
                        res = learner.pretrain_with_self_imitation(
                            msg.get("human_dir",
                                    str(cfg.paths.demos_dir)))
                    except Exception as exc:  # noqa: BLE001
                        res = {"status": "error",
                               "reason": format_exception(exc)}
                        put_bounded(metrics_q, {
                            "type": "log", "level": "error", "src": "learner",
                            "msg": f"self-BC lỗi: {res['reason']}"})
                    put_bounded(metrics_q, {
                        "type": "pretrain_done", "result": res})
                elif cmd == "save_buffer":
                    learner.ckpt.buffer_save(learner.buffer)
                elif cmd == "dream_now":
                    # Manual override from the GUI: trigger a
                    # mental-rehearsal round immediately.  The
                    # round is throttled by the dreamer itself
                    # (``dream_every_s``) so the operator cannot
                    # accidentally pin the CPU by spamming the
                    # button.
                    try:
                        written = learner.dreamer.maybe_dream(time.time())
                        if written:
                            put_bounded(metrics_q, {
                                "type": "log",
                                "level": "info",
                                "src": "learner",
                                "msg": (f"💭 dreamer: đã viết {len(written)} "
                                        f"abstract episode "
                                        f"(positive={learner.dreamer.stats.dreams_positive}, "
                                        f"negative={learner.dreamer.stats.dreams_negative})"),
                            })
                    except Exception as exc:  # noqa: BLE001
                        put_bounded(metrics_q, {
                            "type": "log", "level": "error", "src": "learner",
                            "msg": f"dreamer lỗi: {format_exception(exc)}"})
                elif cmd == "set_dream_env":
                    # The actor process (or the headless harness)
                    # tells the learner to bind a fresh synthetic
                    # env as the dreamer's verification back-door.
                    # We construct the env on the *learner* side
                    # because only the learner has torch (the
                    # dreamer's Q fallback needs the online net)
                    # and creating it here avoids the IPC
                    # serialisation cost of shipping a live env.
                    try:
                        from environment import SyntheticGame
                        seed = int(msg.get("seed", 0))
                        env = SyntheticGame(seed=seed)
                        learner._dream_env = env
                        put_bounded(metrics_q, {
                            "type": "log", "level": "info",
                            "src": "learner",
                            "msg": "dreamer: synthetic env bound "
                                   "(mental-rehearsal will use env-replay)."})
                    except Exception as exc:  # noqa: BLE001
                        put_bounded(metrics_q, {
                            "type": "log", "level": "warning",
                            "src": "learner",
                            "msg": f"dreamer: env bind failed, "
                                   f"fallback to Q-score: {exc}"})
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
            # 3.4 latent-space dreamer: train the tiny VAE on the
            # self-imitation pool.  This is cheap (~10 ms) so it
            # runs in the same heartbeat as the RL update.  The
            # dreamer checks ``train_every_n_updates`` internally
            # so most calls are no-ops.
            try:
                dream_loss = learner.dreamer.maybe_train(
                    learner.update_step)
                if dream_loss is not None:
                    put_bounded(metrics_q, {
                        "type": "dreamer_train",
                        "src": "learner",
                        "loss": dream_loss,
                    })
            except Exception as exc:  # noqa: BLE001 - reported
                put_bounded(metrics_q, {
                    "type": "log", "level": "warning", "src": "learner",
                    "msg": f"dreamer train lỗi: {format_exception(exc)}"})
            # 3.5 online best-model gate (actor-reported episode metrics)
            learner._poll_episode_metric()
            # 3.6 self-BC auto-trigger: if the most recent gate asked
            # for a self-imitation retrain, run it now (the GUI will
            # receive a "pretrain_done" message just like the manual
            # BC button).  The pre-train BC and self-BC are mutually
            # exclusive in spirit (one runs at startup, the other on
            # agent improvement), so the actor is NOT paused here —
            # online RL keeps running while BC trains.
            if getattr(learner, "_self_bc_pending", False):
                learner._self_bc_pending = False
                try:
                    res = learner.pretrain_with_self_imitation(
                        str(cfg.paths.demos_dir))
                    put_bounded(metrics_q, {
                        "type": "pretrain_done", "result": res})
                except Exception as exc:  # noqa: BLE001 - reported
                    put_bounded(metrics_q, {
                        "type": "log", "level": "error", "src": "learner",
                        "msg": f"self-BC lỗi: {format_exception(exc)}"})
            # 4. metrics heartbeat
            if now - last_report >= cfg.perf.report_interval_s:
                last_report = now
                # Mental-rehearsal round (throttled by
                # ``dream_every_s``).  The dreamer uses the live
                # policy and the self-imitation pool; if either is
                # missing, it returns immediately.
                try:
                    learner.dreamer.maybe_dream(now)
                except Exception as exc:  # noqa: BLE001 - reported
                    put_bounded(metrics_q, {
                        "type": "log", "level": "warning", "src": "learner",
                        "msg": f"dreamer round lỗi: {format_exception(exc)}"})
                si_stats = learner.self_imitation.stats()
                d_heart = learner.dreamer.to_heartbeat()
                # RND (curiosity) stats: the predictor loss
                # and the running normaliser.  The GUI shows
                # these so the operator can tell at a glance
                # whether the agent is exploring novel
                # states.
                rnd_heart = (learner.rnd.to_heartbeat()
                              if learner.rnd is not None else {})
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
                        # Self-imitation pool stats; the GUI shows the
                        # on-disk count and "episodes seen" so the
                        # operator can see whether the gate is firing.
                        "self_imitation_seen": si_stats["self_episodes_seen"],
                        "self_imitation_saved": si_stats["self_episodes_saved"],
                        "self_imitation_on_disk": si_stats["self_on_disk"],
                        # Mental-rehearsal pool stats; the GUI shows
                        # how many abstract episodes the dreamer has
                        # written (positive vs negative) and the
                        # current rolling Q mean.
                        "dreams_total": d_heart["dreams_total"],
                        "dreams_positive": d_heart["dreams_positive"],
                        "dreams_negative": d_heart["dreams_negative"],
                        "on_disk_positive": d_heart["on_disk_positive"],
                        "on_disk_negative": d_heart["on_disk_negative"],
                        "dream_q_mean": d_heart["dream_q_mean"],
                        "dreamer_train_loss": d_heart["last_train_loss"],
                        # RND (curiosity) — the operator can
                        # see whether the agent is exploring
                        # novel states.
                        "rnd_norm": rnd_heart.get("rnd_norm", 0.0),
                        "rnd_mean_intrinsic": rnd_heart.get("rnd_mean_intrinsic", 0.0),
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
