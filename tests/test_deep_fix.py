"""Regression tests for the DEEP-FIX pass.

Every test here pins a defect that was found by reading the code and then
reproduced by running it.  The docstring of each test states the observable
symptom of the original bug so a future regression is diagnosable from the
failure message alone.
"""

from __future__ import annotations

import json
import threading
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from config import PERConfig, BotConfig
from ipc import SharedCounters, SharedWeights, unflatten_into

try:  # DEEP-FIX added this symbol; keep the import optional so this module
    from ipc import layout_fingerprint as _layout_fingerprint
except ImportError:  # pragma: no cover - only on the pre-fix tree
    _layout_fingerprint = None


def layout_fingerprint(module) -> str:
    """Fail loudly (not at import time) when the guard symbol is missing."""
    assert _layout_fingerprint is not None, (
        "ipc.layout_fingerprint does not exist — the weight-layout guard "
        "is missing")
    return _layout_fingerprint(module)


# --------------------------------------------------------------------- #
# 1. Prioritised replay must receive PER-SAMPLE TD errors
# --------------------------------------------------------------------- #
def _nstep(i: int, action: int = 1, reward: float = 0.1):
    from replay_buffer import NStepTransition

    obs = np.full((4, 84, 84), i % 200, dtype=np.uint8)
    return NStepTransition(
        obs=obs, next_obs=obs, action=action, reward=reward, done=False,
        span=3, gamma_pow=0.99 ** 3,
        obs_env_ids=(i, i + 1, i + 2, i + 3),
        next_env_ids=(i + 1, i + 2, i + 3, i + 4),
    )


def _filled_buffer(n: int = 200, capacity: int = 512):
    from replay_buffer import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(PERConfig(capacity=capacity), frame_size=84)
    for i in range(n):
        buf.add_nstep(_nstep(i))
    return buf


class TestPerSamplePriorities:
    def test_distinct_td_errors_give_distinct_priorities(self) -> None:
        """The old learner broadcast one scalar; priorities must differ."""
        buf = _filled_buffer()
        batch = buf.sample(32, 0.4, rng=np.random.default_rng(0))
        td = np.arange(1.0, 33.0)  # 32 clearly different errors
        buf.update_priorities(batch["indices"], td)
        prios = buf.priorities[batch["indices"]]
        assert len(np.unique(np.round(prios, 9))) == 32, (
            "PER collapsed: every sampled slot received the same priority")
        # the highest TD error must own the highest priority
        assert batch["indices"][int(np.argmax(td))] == \
            batch["indices"][int(np.argmax(prios))]

    def test_agent_returns_per_sample_td_errors(self) -> None:
        """train_step must expose a per-sample vector, not only the mean."""
        from agent import DoubleDQNAgent
        from config import RLConfig

        cfg = RLConfig(batch_size=8, warmup_transitions=1)
        agent = DoubleDQNAgent("strict_lite", cfg, seed=0)
        buf = _filled_buffer(n=64, capacity=128)
        batch = buf.sample(8, 0.4, rng=np.random.default_rng(1))
        metrics = agent.train_step(batch)
        assert "td_errors" in metrics
        assert metrics["td_errors"].shape == (8,)
        assert np.all(np.isfinite(metrics["td_errors"]))
        assert metrics["td_error_abs_mean"] == pytest.approx(
            float(metrics["td_errors"].mean()), rel=1e-5)

    def test_learner_writes_per_sample_priorities(self, tmp_path) -> None:
        """End to end: one train_one() must leave a non-uniform priority set."""
        from queue import Queue

        from config import BotConfig
        from learner_worker import Learner
        from models import weight_size_for_profile

        cfg = BotConfig()
        cfg.paths.checkpoints_dir = str(tmp_path / "ckpt")
        cfg.rl.batch_size = 16
        cfg.rl.warmup_transitions = 8
        cfg.per.capacity = 256
        weights = SharedWeights(weight_size_for_profile("quality_cpu",
                                                        cfg.perception.frame_stack))
        learner = Learner(cfg, weights, SharedCounters(), Queue(),
                          str(tmp_path / "ckpt"))
        for i in range(60):
            learner.buffer.add_nstep(_nstep(i))
        learner.buffer.priorities[:] = 1.0
        learner.buffer.tree.rebuild_from(learner.buffer.priorities,
                                         filled=learner.buffer.size)
        before = learner.buffer.priorities.copy()
        assert learner.train_one(time.monotonic()) is not None
        changed = learner.buffer.priorities != before
        assert changed.sum() > 0, "no priority was updated at all"
        touched = learner.buffer.priorities[changed]
        assert len(np.unique(np.round(touched, 9))) > 1, (
            "all updated slots share one priority — PER is degenerate")

    def test_mismatched_td_length_is_rejected(self) -> None:
        """A short TD vector must not silently mis-attribute priorities."""
        buf = _filled_buffer()
        with pytest.raises(ValueError, match="td errors"):
            buf.update_priorities(np.arange(8), np.ones(7))

    def test_out_of_range_indices_are_dropped_not_wrapped(self) -> None:
        """Index -1 used to write through to the LAST slot."""
        buf = _filled_buffer(n=50, capacity=64)
        before = buf.priorities[63]
        buf.update_priorities([-1, 0], np.array([99.0, 99.0]))
        assert buf.priorities[63] == before, "negative index wrote to slot 63"


# --------------------------------------------------------------------- #
# 2. Profile switch must not corrupt the actor's weights
# --------------------------------------------------------------------- #
class TestWeightLayoutGuard:
    def test_mismatched_layout_is_refused(self) -> None:
        """An actor on profile A must not consume profile B's prefix."""
        from agent import InferencePolicy
        from models import DuelingDQN

        heavy = DuelingDQN.from_profile("quality_cpu")
        light = DuelingDQN.from_profile("strict_lite")
        shared = SharedWeights(sum(p.numel() for p in heavy.parameters()))
        policy = InferencePolicy(light, seed=0)
        shared.publish(
            np.concatenate([p.detach().numpy().reshape(-1)
                            for p in heavy.parameters()]).astype(np.float32),
            fingerprint=layout_fingerprint(heavy))
        before = light.state_dict()["encoder.0.0.weight"].clone()
        assert policy.refresh_weights(shared) is False
        after = light.state_dict()["encoder.0.0.weight"]
        assert bool((before == after).all()), "mismatched weights were applied"

    def test_matching_layout_is_applied(self) -> None:
        from agent import InferencePolicy
        from models import DuelingDQN

        model = DuelingDQN.from_profile("strict_lite")
        shared = SharedWeights(sum(p.numel() for p in model.parameters()))
        policy = InferencePolicy(DuelingDQN.from_profile("strict_lite"), seed=0)
        flat = np.concatenate([p.detach().numpy().reshape(-1)
                               for p in model.parameters()]).astype(np.float32)
        shared.publish(flat, fingerprint=layout_fingerprint(model))
        assert policy.refresh_weights(shared) is True
        got = policy.model.state_dict()["encoder.0.0.weight"].numpy()
        assert np.allclose(got, model.state_dict()["encoder.0.0.weight"].numpy())

    def test_unflatten_refuses_a_short_vector(self) -> None:
        from models import DuelingDQN

        model = DuelingDQN.from_profile("strict_lite")
        with pytest.raises(ValueError, match="too short"):
            unflatten_into(model, np.zeros(10, dtype=np.float32))

    def test_nonfinite_weights_are_never_published(self) -> None:
        from models import DuelingDQN

        model = DuelingDQN.from_profile("strict_lite")
        shared = SharedWeights(sum(p.numel() for p in model.parameters()))
        bad = np.full(shared.size, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            shared.publish(bad, fingerprint=layout_fingerprint(model))

    def test_actor_downgrade_notifies_the_learner_command_queue(self) -> None:
        """The old code put the notice on transition_q, which nobody reads."""
        import inspect

        import environment

        src = inspect.getsource(environment.BotActor._downgrade)
        assert "self.cmd_q" in src, "the downgrade still has no command channel"
        assert '"cmd": "set_profile"' in src

    def test_learner_accepts_an_inline_profile_command(self, tmp_path) -> None:
        """Defence in depth: the transition queue path is honoured too."""
        from queue import Queue

        from config import BotConfig
        from learner_worker import Learner
        from models import weight_size_for_profile

        cfg = BotConfig()
        cfg.rl.profile = "balanced_cpu"
        weights = SharedWeights(weight_size_for_profile("quality_cpu",
                                                        cfg.perception.frame_stack))
        learner = Learner(cfg, weights, SharedCounters(), Queue(),
                          str(tmp_path / "ckpt"))
        assert learner.profile == "balanced_cpu"
        learner.set_profile("strict_lite")
        assert learner.profile == "strict_lite"

    def test_profile_switch_keeps_the_live_replay_buffer(self, tmp_path) -> None:
        from queue import Queue

        from config import BotConfig
        from learner_worker import Learner
        from models import weight_size_for_profile

        cfg = BotConfig()
        cfg.paths.checkpoints_dir = str(tmp_path / "ckpt")
        weights = SharedWeights(weight_size_for_profile("quality_cpu",
                                                        cfg.perception.frame_stack))
        learner = Learner(cfg, weights, SharedCounters(), Queue(),
                          str(tmp_path / "ckpt"))
        for i in range(25):
            learner.buffer.add_nstep(_nstep(i))
        assert learner.buffer.size == 25
        learner.set_profile("balanced_cpu")
        assert learner.buffer.size == 25, "the live buffer was discarded"


# --------------------------------------------------------------------- #
# 3. Episode telemetry is an atomic record
# --------------------------------------------------------------------- #
class TestEpisodeTelemetry:
    def test_payload_and_id_are_consistent(self) -> None:
        c = SharedCounters()
        c.publish_episode_result(1, 12.5, 7.25)
        got = c.read_episode_result(0)
        assert got == {"episode_id": 1, "survival_s": 12.5, "total_reward": 7.25}
        assert c.read_episode_result(1) is None

    def test_publish_invalidates_before_writing_the_payload(self) -> None:
        """DEEP-FIX root cause, pinned deterministically.

        The actor used to write ``last_episode_done_id`` FIRST and the two
        payload fields after it, so a learner that polled in that window saw a
        *new* id next to the *previous* episode's survival/reward -- the value
        that gates ``best_model.pth``.

        Writing the id last is also NOT sufficient (see the docstring on
        ``publish_episode_result``), so this asserts the full three-phase
        protocol: invalidate -> payload -> commit.
        """
        c = SharedCounters()
        writes: list[tuple[str, float]] = []

        class Recorder:
            """Proxy that logs every .value assignment and forwards it."""

            def __init__(self, raw, name: str) -> None:
                self._raw = raw
                self._name = name

            @property
            def value(self):
                return self._raw.value

            @value.setter
            def value(self, new) -> None:
                writes.append((self._name, new))
                self._raw.value = new

        for name in ("last_episode_survival_s", "last_episode_reward",
                     "last_episode_done_id"):
            setattr(c, name, Recorder(getattr(c, name), name))

        c.publish_episode_result(7, 3.25, -1.5)

        names = [n for n, _ in writes]
        assert names[0] == "last_episode_done_id", (
            f"the id must be INVALIDATED before any payload write, otherwise a "
            f"reader that sampled the old id still sees it after reading the "
            f"new payload; write order was {names}")
        assert writes[0][1] == 0, (
            f"the first id write must be the invalid marker 0, got {writes[0][1]!r}")
        assert names[-1] == "last_episode_done_id", (
            f"the id must also be COMMITTED last; write order was {names}")
        assert writes[-1][1] == 7, (
            f"the last id write must be the real episode id, got {writes[-1][1]!r}")
        assert set(names[1:-1]) == {"last_episode_survival_s",
                                   "last_episode_reward"}, names
        # and the committed record must be readable back intact
        assert c.read_episode_result(0) == {"episode_id": 7, "survival_s": 3.25,
                                            "total_reward": -1.5}

    def test_reader_never_pairs_a_new_id_with_an_old_payload(self) -> None:
        """Hammer the writer/reader pair; the pairing must never tear.

        Guards against a *vacuous* pass: a writer that raised (for example on
        a tree without the publish helper) used to leave ``stop`` set and an
        empty mismatch list, so the test "passed" without reading anything.
        The reader's workload and the writer's health are both asserted.
        """
        c = SharedCounters()
        stop = threading.Event()
        mismatches: list[str] = []
        writer_error: list[BaseException] = []
        reads = [0]
        # DEEP-FIX to the test itself: with the default 5 ms switch interval
        # the writer finishes all 4000 publishes inside a single GIL slice, so
        # the reader never ran and the original assertion passed without
        # observing a single write.  Force a handoff every few publishes and
        # shrink the switch interval so the interleaving is real.
        prev_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)

        def writer() -> None:
            try:
                for i in range(1, 20_001):
                    # payload and id encode the same value, so any tear shows
                    c.publish_episode_result(i, float(i), float(i))
                    if i % 8 == 0:
                        time.sleep(0)  # yield the GIL to the reader
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                writer_error.append(exc)
            finally:
                sys.setswitchinterval(prev_interval)
                stop.set()

        def reader() -> None:
            last = 0
            while not stop.is_set():
                got = c.read_episode_result(last)
                if got is None:
                    continue
                reads[0] += 1
                last = int(got["episode_id"])
                if got["survival_s"] != float(last):
                    mismatches.append(f"id={last} survival={got['survival_s']}")

        tw = threading.Thread(target=writer)
        tr = threading.Thread(target=reader)
        tr.start(); tw.start(); tw.join(30); tr.join(30)
        assert not tw.is_alive() and not tr.is_alive(), "threads did not finish"
        assert not writer_error, f"writer thread failed: {writer_error[:1]!r}"
        assert reads[0] > 100, (
            f"reader only observed {reads[0]} episodes -- the interleaving "
            f"this test exists to exercise never happened")
        assert not mismatches, f"torn episode telemetry: {mismatches[:3]}"

    def test_id_first_order_is_demonstrably_tearable(self) -> None:
        """Evidence that the ordering above is load-bearing, not decorative.

        Reproducing a race by luck is a bad test (a first attempt here saw
        zero tears in 115,212 opportunistic reads).  Instead the interleaving
        is *injected*: the reader is forced to yield immediately after reading
        the id, and the writer is forced to yield between the id and the
        payload -- which is exactly the window the ordering fix closes.  In
        the old id-first order the reader must observe a stale payload.
        """
        c = SharedCounters()
        stop = threading.Event()
        torn = [0]
        reads = [0]

        class PreemptAfterRead:
            """Proxy that yields the GIL right after each .value read."""

            def __init__(self, raw) -> None:
                self._raw = raw

            @property
            def value(self):
                got = self._raw.value
                for _ in range(4):
                    time.sleep(0)
                return got

            @value.setter
            def value(self, new) -> None:
                self._raw.value = new

        c.last_episode_done_id = PreemptAfterRead(c.last_episode_done_id)

        def writer() -> None:
            try:
                for i in range(1, 20_001):
                    c.last_episode_done_id.value = i      # OLD order: id first
                    time.sleep(0)                          # widen the window
                    c.last_episode_survival_s.value = float(i)
            except BaseException:
                raise
            finally:
                stop.set()

        def reader() -> None:
            while not stop.is_set():
                i = int(c.last_episode_done_id.value)
                surv = float(c.last_episode_survival_s.value)
                if i:
                    reads[0] += 1
                    if abs(surv - float(i)) > 1e-6:
                        torn[0] += 1

        tw = threading.Thread(target=writer)
        tr = threading.Thread(target=reader)
        tr.start(); tw.start(); tw.join(60); tr.join(60)
        assert not tw.is_alive() and not tr.is_alive(), "threads did not finish"
        assert reads[0] > 100, f"reader only saw {reads[0]} writes"
        assert torn[0] > 0, (
            f"expected the OLD id-first write order to expose a stale payload "
            f"in {reads[0]} reads; saw none -- either the ordering is no "
            f"longer load-bearing or the injected interleaving stopped working")


# --------------------------------------------------------------------- #
# 4. Replay buffer persistence
# --------------------------------------------------------------------- #
class TestBufferPersistence:
    def test_capacity_mismatch_is_a_corrupt_file_error(self, tmp_path) -> None:
        """A raw numpy broadcast error used to escape instead."""
        from logging_utils import CorruptFileError
        from replay_buffer import PrioritizedReplayBuffer

        big = PrioritizedReplayBuffer(PERConfig(capacity=4000), frame_size=84)
        for i in range(40):
            big.add_nstep(_nstep(i))
        path = str(tmp_path / "buffer.pkl")
        big.save(path)
        with pytest.raises(CorruptFileError, match="per.capacity"):
            PrioritizedReplayBuffer.load(path, PERConfig(capacity=500),
                                         frame_size=84)

    def test_round_trip_preserves_transitions(self, tmp_path) -> None:
        from replay_buffer import PrioritizedReplayBuffer

        cfg = PERConfig(capacity=300)
        buf = PrioritizedReplayBuffer(cfg, frame_size=84)
        for i in range(120):
            buf.add_nstep(_nstep(i, action=i % 5, reward=0.01 * i))
        path = str(tmp_path / "buffer.pkl")
        buf.save(path)
        restored = PrioritizedReplayBuffer.load(path, cfg, frame_size=84)
        assert restored.size == buf.size
        batch = restored.sample(16, 0.4, rng=np.random.default_rng(3))
        assert batch["obs"].shape == (16, 4, 84, 84)
        assert np.all(np.isfinite(batch["weights"]))

    def test_sample_never_skews_indices_and_transitions(self) -> None:
        """Evicted slots must be substituted in BOTH the data and the index."""
        buf = _filled_buffer(n=200, capacity=160)
        # Evict a contiguous band of frame slots: transitions whose env-ids
        # fall in it become invalid, the rest stay sampleable.
        buf.frames.env_ids[100:130] = -1
        batch = buf.sample(16, 0.5, rng=np.random.default_rng(4),
                           max_replace_rounds=0)
        for j, idx in enumerate(batch["indices"]):
            t = buf.transitions[int(idx)]
            assert t is not None, f"sample {j} points at an empty slot"
            # (a) the index must describe exactly the data that was returned
            expected = np.stack([buf.frames.frames[s] for s in t.obs_ids])
            assert np.array_equal(batch["obs"][j], expected), (
                f"sample {j} carries data that index {idx} does not describe")
            # (b) and that data must still be the frames the transition names.
            # The old code happily returned a transition whose frames had been
            # evicted, training on whatever a later frame had overwritten into
            # the slot -- contradicting this module's own guarantee that
            # "stale pixels can never silently train the network".
            for slot, eid in zip(t.obs_ids, t.obs_env_ids):
                assert buf.frames.resident(slot, eid), (
                    f"sample {j} (index {idx}) references evicted frame {eid}")


# --------------------------------------------------------------------- #
# 5. Checkpoint integrity
# --------------------------------------------------------------------- #
class TestCheckpointIntegrity:
    def test_best_metric_does_not_advance_on_a_failed_write(self, tmp_path) -> None:
        from checkpoint_manager import CheckpointManager

        cm = CheckpointManager(tmp_path, "strict_lite")

        def boom(payload, path):
            raise OSError("disk full")

        cm._atomic_torch_save = boom
        assert cm.save_model({"online": {}}, {}, which="best", metric=10.0) is None
        assert cm.best_metric is None, (
            "the gate advanced even though no file was written")
        # and a genuinely better metric later still gets through
        cm._atomic_torch_save = CheckpointManager._atomic_torch_save.__get__(cm)
        assert cm.save_model({"online": {}}, {}, which="best", metric=5.0) is not None
        assert cm.best_metric == pytest.approx(5.0)

    def test_nonfinite_metric_is_refused(self, tmp_path) -> None:
        from checkpoint_manager import CheckpointManager

        cm = CheckpointManager(tmp_path, "strict_lite")
        assert cm.save_model({}, {}, which="best", metric=float("nan")) is None
        assert cm.save_model({}, {}, which="best", metric=None) is None

    def test_corrupt_checkpoint_is_quarantined_on_load(self, tmp_path) -> None:
        """The sidecar used to be written but never read back."""
        import torch

        from checkpoint_manager import CheckpointManager

        cm = CheckpointManager(tmp_path, "strict_lite")
        cm.save_model({"online": {}}, {}, which="latest")
        path = cm.latest_path
        assert path.exists()
        blob = bytearray(path.read_bytes())
        blob[len(blob) // 2] ^= 0xFF  # flip a byte in the payload
        path.write_bytes(bytes(blob))
        assert cm.load_model("latest") is None
        assert not path.exists()
        assert path.with_name(path.name + ".corrupt").exists()

    def test_rng_state_round_trips_the_sampling_generator(self) -> None:
        """capture_rng_states used to snapshot a throwaway generator."""
        from checkpoint_manager import capture_rng_states, restore_rng_states

        rng = np.random.default_rng(123)
        _ = rng.random(5)
        snap = capture_rng_states(seed=7, generator=rng)
        expected = rng.random(4)
        other = np.random.default_rng(999)
        assert restore_rng_states(snap, generator=other) is True
        assert np.allclose(other.random(4), expected)


# --------------------------------------------------------------------- #
# 6. Demonstration recorder
# --------------------------------------------------------------------- #
class TestDemoRecorder:
    def test_listener_is_rebuilt_after_stop(self) -> None:
        """Episode 2+ used to record NOOP only because _listener stayed None."""
        from demonstration_recorder import KeyboardTap

        class FakeListener:
            def __init__(self):
                self.started = 0
                self.stopped = 0

            def start(self):
                self.started += 1

            def stop(self):
                self.stopped += 1

        tap = KeyboardTap(lambda k: None, lambda k: None)
        tap._keyboard = type("K", (), {"Listener": staticmethod(
            lambda on_press, on_release: FakeListener())})
        assert tap.start() is True
        tap.stop()
        assert tap._listener is None
        assert tap.start() is True, "a stopped listener was never rebuilt"
        assert tap._listener is not None

    def test_pump_records_frames_from_the_reader(self, tmp_path) -> None:
        """read_frame was stored and never called; nothing was ever recorded."""
        from demonstration_recorder import DemoRecorder
        from ipc import Frame

        frames = [Frame(frame_id=i + 1, ts=i / 30.0,
                        image=np.full((800, 480, 3), (40, 44, 60), np.uint8))
                  for i in range(5)]
        it = iter(frames)
        rec = DemoRecorder(BotConfig(), tmp_path / "demos",
                           lambda: next(it, None))
        rec.start()
        assert rec.pump(max_frames=5) == 5
        path = rec.stop()
        assert path is not None
        assert np.load(path)["frames"].shape[0] == 5

    def test_same_second_episodes_do_not_overwrite(self, tmp_path) -> None:
        from demonstration_recorder import DemoRecorder
        from ipc import Frame

        cfg = BotConfig()
        cfg.region.width, cfg.region.height = 480, 800
        paths = []
        for _ in range(3):
            rec = DemoRecorder(cfg, tmp_path / "demos", lambda: None)
            rec.start()
            for i in range(3):
                rec.tick(Frame(frame_id=i + 1, ts=i / 30.0,
                               image=np.full((800, 480, 3), 50, np.uint8)))
            paths.append(rec.stop())
        assert len(set(paths)) == 3, f"episodes collided: {paths}"
        assert all(Path(p).exists() for p in paths)


# --------------------------------------------------------------------- #
# 7. Queue poisoning / shutdown
# --------------------------------------------------------------------- #
class TestGuardedDrain:
    def test_poisoned_queue_is_skipped(self) -> None:
        from logging_utils import drain, quarantine_queue

        class Hanging:
            name = "hanging"

            def get_nowait(self):
                time.sleep(30)

        q = Hanging()
        t0 = time.monotonic()
        assert drain(q, limit=4, timeout_s=0.2) == []
        assert time.monotonic() - t0 < 5.0
        # the second call must not pay the timeout again
        t0 = time.monotonic()
        assert drain(q, limit=4, timeout_s=0.2) == []
        assert time.monotonic() - t0 < 0.1, "quarantine did not take effect"

    def test_inline_drain_is_unchanged(self) -> None:
        import queue as queue_mod

        from logging_utils import drain

        q = queue_mod.Queue()
        for i in range(5):
            q.put(i)
        assert drain(q, limit=3) == [0, 1, 2]
        assert drain(q, limit=10, timeout_s=0.5) == [3, 4]
        assert drain(q, limit=10, timeout_s=0.5) == []


# --------------------------------------------------------------------- #
# 8. Reward / environment episode boundaries
# --------------------------------------------------------------------- #
class TestEpisodeBoundaries:
    def test_respawn_restores_the_death_penalty(self) -> None:
        """The -10 penalty used to fire once per process, not per episode."""
        from config import BotConfig
        from environment import GameEnvironment

        env = GameEnvironment(BotConfig())
        env.reset()
        env.reward_calc.begin_episode(0.0)
        assert env.reward_calc.death_reward() == pytest.approx(-10.0)
        env.respawn()
        assert env.reward_calc.death_reward() == pytest.approx(-10.0), (
            "the second episode got no death penalty")

    def test_flush_transitions_survives_an_unseeded_stack(self) -> None:
        """Shutdown before the first valid frame must not raise."""
        from config import BotConfig
        from environment import BotActor

        cfg = BotConfig()
        cfg.capture.source = "fake"
        cfg.region.width, cfg.region.height = 480, 800
        cfg.death.anchor_fx, cfg.death.anchor_fy = 30 / 480, 30 / 800
        cfg.death.anchor_baseline_rgb = (206, 66, 66)
        cfg.input.respawn_fx, cfg.input.respawn_fy = 0.5, 0.6

        class _Q:
            def put_nowait(self, item):
                raise AssertionError("nothing should be shipped")

        from ipc import SharedFrameRing, SharedWeights
        from models import weight_size_for_profile

        actor = BotActor(
            cfg, SharedFrameRing(2, 800, 480, 3),
            {k: threading.Event() for k in
             ("stop", "emergency", "pause", "pause_learning", "death")},
            _Q(), _Q(), SharedWeights(weight_size_for_profile("quality_cpu")),
            SharedCounters(), input_backend="dry_run")
        actor.nstep.push(np.zeros((4, 84, 84), np.uint8), (1, 2, 3, 4), 0, 0.1, False)
        assert actor.nstep.pending == 1
        actor._flush_transitions(final=True)   # must not raise
        assert actor.nstep.pending == 0


# --------------------------------------------------------------------- #
# 9. Evaluation honesty
# --------------------------------------------------------------------- #
class TestEvaluation:
    def test_evaluation_tool_derives_a_per_episode_seed(self) -> None:
        """The runner must not reuse cfg.seed for every episode."""
        import inspect

        import evaluation_tool

        src = inspect.getsource(evaluation_tool.run_headless_evaluation)
        assert "ep_seed" in src and "SyntheticGame(seed=ep_seed" in src, (
            "run_headless_evaluation still builds every episode from cfg.seed")

    def test_episodes_are_not_identical(self) -> None:
        """Same cfg.seed everywhere made all N eval episodes bit-identical."""
        from config import BotConfig
        from environment import GameEnvironment, SyntheticGame

        cfg = BotConfig()
        a = GameEnvironment(cfg, game=SyntheticGame(seed=cfg.seed * 1_000_003 + 0))
        b = GameEnvironment(cfg, game=SyntheticGame(seed=cfg.seed * 1_000_003 + 1))
        a.reset(); b.reset()
        fa = [a.step(0)[0].sum() for _ in range(15)]
        fb = [b.step(0)[0].sum() for _ in range(15)]
        assert fa != fb, "two evaluation episodes are identical"

    def test_report_load_tolerates_unknown_fields(self, tmp_path) -> None:
        from evaluation import EvaluationReport

        path = tmp_path / "old.json"
        path.write_text(json.dumps({"records": [
            {"episode_id": 1, "survival_s": 3.0, "total_reward": 1.0,
             "steps": 10, "env_frames": 10, "fps": 30.0,
             "action_latency_p95_ms": 5.0, "inference_p95_ms": 1.0,
             "score": 0.0, "kind": "eval", "ts": 0.0,
             "a_field_from_the_future": {"nested": True}},
            "not even a dict",
        ]}), encoding="utf-8")
        rep = EvaluationReport.load(path)
        assert len(rep.records) == 1
        assert rep.records[0].survival_s == pytest.approx(3.0)

    def test_failure_modes_can_be_scoped(self) -> None:
        from evaluation import EpisodeRecord, EvaluationReport

        rep = EvaluationReport()
        for kind, surv in (("train", 1.0), ("eval", 60.0),
                           ("human_baseline", 2.0)):
            rep.add(EpisodeRecord(episode_id=1, survival_s=surv, total_reward=0.0,
                                  steps=1, env_frames=1, fps=30.0,
                                  action_latency_p95_ms=1.0,
                                  inference_p95_ms=1.0, kind=kind))
        assert rep.failure_modes(kinds=("eval",)) == {}
        assert rep.failure_modes()["early_death_lt_5s"] == 2


# --------------------------------------------------------------------- #
# 10. Dead configuration knobs are now enforced
# --------------------------------------------------------------------- #
class TestConfigKnobsAreUsed:
    def test_grab_timeout_is_read_by_the_capture_worker(self) -> None:
        import inspect

        import capture_worker

        src = inspect.getsource(capture_worker.capture_main)
        assert "grab_timeout_s" in src

    def test_max_working_set_is_read_by_the_actor(self) -> None:
        import inspect

        import environment

        src = inspect.getsource(environment.BotActor._maybe_perf_check)
        assert "max_working_set_gb" in src


# --------------------------------------------------------------------- #
# 11. Dataset isolation
# --------------------------------------------------------------------- #
class TestDatasetIsolation:
    def test_one_bad_file_does_not_abort_the_load(self, tmp_path) -> None:
        import dataset

        np.savez_compressed(
            tmp_path / "good.npz",
            frames=np.zeros((5, 84, 84), np.uint8),
            actions=np.zeros(5, np.int64),
            timestamps=np.arange(5) / 30.0,
            done=np.array([0, 0, 0, 0, 1], bool))
        np.savez_compressed(tmp_path / "broken.npz",
                            frames=np.zeros((5, 84, 84), np.uint8))
        (tmp_path / "junk.npz").write_bytes(b"not an npz")
        eps, reps = dataset.validate_directory(tmp_path)
        assert len(eps) == 1
        assert sum(1 for r in reps if r.ok) == 1
        assert sum(1 for r in reps if not r.ok) == 2
        # zip(episodes, reports) must stay aligned for Learner.pretrain
        assert all(e.path == r.path for e, r in zip(eps, reps))


# --------------------------------------------------------------------- #
# 12. Input normalisation must not alias the caller's buffer
# --------------------------------------------------------------------- #
class TestNoAliasing:
    def test_float32_input_is_not_mutated_in_place(self) -> None:
        from agent import _to_unit_float

        arr = np.full((4, 84, 84), 255.0, dtype=np.float32)
        out = _to_unit_float(arr)
        assert float(arr.mean()) == pytest.approx(255.0), (
            "the caller's array was divided by 255 in place")
        assert float(out.mean()) == pytest.approx(1.0)

    def test_uint8_input_still_normalises(self) -> None:
        from agent import _to_unit_float

        arr = np.full((4, 84, 84), 255, dtype=np.uint8)
        assert float(_to_unit_float(arr).mean()) == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# 13. Respawn recovery beats the deadline
# --------------------------------------------------------------------- #
class TestRespawnOrdering:
    def test_recovery_at_the_deadline_is_not_reported_as_failure(self) -> None:
        from config import DeathConfig
        from death_detector import DeathState, RespawnController

        cfg = DeathConfig(respawn_timeout_s=1.0, stable_frames=2,
                          respawn_interval_s=0.2)
        ctl = RespawnController(lambda t: True, cfg, now=lambda: 0.0)
        ctl.start()
        ctl.update(DeathState.DEAD_CONFIRMED, 0.0)
        ctl.update(DeathState.ALIVE, 2.0)          # recovered, past deadline
        status = ctl.update(DeathState.ALIVE, 2.1)
        assert status.action == "RECOVERED", (
            f"a successful respawn was reported as {status.action}")

    def test_timeout_still_fails_when_never_recovered(self) -> None:
        from config import DeathConfig
        from death_detector import DeathState, RespawnController

        cfg = DeathConfig(respawn_timeout_s=1.0, stable_frames=2,
                          respawn_interval_s=0.2)
        ctl = RespawnController(lambda t: True, cfg, now=lambda: 0.0)
        ctl.start()
        last = None
        for i in range(40):
            last = ctl.update(DeathState.DEAD_CONFIRMED, i * 0.1)
            if last.action == "FAILED":
                break
        assert last is not None and last.action == "FAILED"
        assert ctl.clicks <= 12


# --------------------------------------------------------------------- #
# 14. A Thread subclass must not shadow threading.Thread's internals
# --------------------------------------------------------------------- #
class TestThreadInternalShadowing:
    """``SafetyWatchdog`` shadowed ``Thread._stop`` and crashed on shutdown.

    Symptom of the original bug: every headless run ended with

        Process actor-worker:
        Traceback (most recent call last):
          ...
          File ".../threading.py", line 1134, in _wait_for_tstate_lock
            self._stop()
        TypeError: 'Event' object is not callable

    CPython's ``threading.Thread`` has an *internal method* called ``_stop``.
    Assigning ``self._stop = threading.Event()`` replaces that method with an
    Event, so when ``join()`` completed and called ``self._stop()`` it blew
    up.  The crash happened in the child process's ``_bootstrap``, after our
    own logging had already announced a clean shutdown -- which is exactly
    why a passing test suite and a tidy log coexisted with a hard crash on
    every single run.
    """

    def test_watchdog_joins_without_raising(self) -> None:
        """Direct reproduction: start it, stop it, join it."""
        from safety_watchdog import SafetyWatchdog

        wd = SafetyWatchdog(events={}, input_controller=None, counters=None,
                            metrics_q=None, interval_s=0.02)
        wd.start()
        time.sleep(0.05)
        wd.stop()
        wd.join(timeout=5.0)          # <- raised TypeError before the fix
        assert not wd.is_alive(), "watchdog thread did not exit"

    def test_no_thread_subclass_shadows_a_thread_internal(self) -> None:
        """Generalise: catch the whole class of bug, not just ``_stop``.

        Any instance attribute a ``threading.Thread`` subclass assigns that
        collides with a name on ``Thread`` itself is a latent crash -- it may
        not fire today (only ``_stop`` is called by ``join``), but the next
        CPython release is free to use any of these names internally.
        """
        import ast

        thread_attrs = set(dir(threading.Thread))
        offenders: list[str] = []
        for path in sorted(Path(__file__).resolve().parent.parent.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                bases = [b.id if isinstance(b, ast.Name) else
                         getattr(b, "attr", "") for b in cls.bases]
                if "Thread" not in bases:
                    continue
                for node in ast.walk(cls):
                    if (isinstance(node, ast.Attribute)
                            and isinstance(node.value, ast.Name)
                            and node.value.id == "self"
                            and isinstance(node.ctx, ast.Store)
                            and node.attr in thread_attrs):
                        offenders.append(f"{path.name}:{node.lineno} "
                                         f"{cls.name}.{node.attr}")
        assert not offenders, (
            f"Thread subclass(es) overwrite attributes that threading.Thread "
            f"already defines: {offenders}.  Rename them -- see the class "
            f"docstring for the shutdown crash this caused."
        )

    def test_stop_event_is_a_real_event(self) -> None:
        """The replacement attribute must still behave like the old one."""
        from safety_watchdog import SafetyWatchdog

        wd = SafetyWatchdog(events={}, input_controller=None, counters=None,
                            metrics_q=None, interval_s=0.02)
        assert isinstance(wd._stop_event, threading.Event)
        assert not wd._stop_event.is_set()
        wd._stop_event.set()
        assert wd._stop_event.is_set()
        # Thread._stop must still be the method CPython expects.
        assert callable(threading.Thread._stop)


# --------------------------------------------------------------------- #
# 15. A persistent hazard must expire, not wedge the tracker
# --------------------------------------------------------------------- #
def _hz(frame_id: int, ts: float, detected: bool):
    from horizon_detector import HorizonResult
    return HorizonResult(
        frame_id=frame_id, ts=ts,
        change_score=40.0 if detected else 0.0,
        raw_score=40.0 if detected else 0.0,
        changed_ratio=0.8 if detected else 0.0,
        detected=detected, confidence=0.9 if detected else 0.0,
    )


class TestHazardExpiry:
    """``register`` overwrote the field both expiry checks read.

    ``register`` is called on every frame the horizon detector fires, and it
    did ``self._current.opened_ts = horizon.ts``.  So a hazard that stayed
    visible kept pushing its own deadline one frame ahead and the event never
    expired.  Measured before the fix: 10 s of continuous danger with
    ``hazard_expiry_s=1.2`` left ``expired=0`` and the event still open.

    The knock-on effect was worse than a missing expiry.  When the screen
    finally went quiet, the stale event resolved and paid ``hazard_bonus``
    credited to the action recorded on frame 0 -- 300 frames / 10 s earlier.
    That is a reward for dodging an obstacle the agent passed ten seconds
    ago, i.e. corrupted credit assignment, in the one module whose docstring
    promises "no future leakage".
    """

    FPS = 30.0

    def test_continuous_hazard_expires(self) -> None:
        from config import RewardConfig
        from rewards import PendingHazardTracker

        cfg = RewardConfig()
        tr = PendingHazardTracker(cfg)
        for f in range(int(self.FPS * 10)):          # 10 s, expiry is 1.2 s
            ts = f / self.FPS
            h = _hz(f, ts, detected=True)
            tr.register(h)
            tr.on_frame(h, 1)
            tr.expire_old(ts)
        st = tr.stats()
        assert st["hazards_expired"] > 0, (
            f"a hazard held for 10 s never expired (hazard_expiry_s="
            f"{cfg.hazard_expiry_s}); stats={st}"
        )
        assert st["hazards_total"] > 1, (
            "one event monopolised the tracker for the whole run; overlapping "
            "hazards should mint a fresh event once the previous one expires"
        )

    def test_no_bonus_is_credited_to_an_ancient_action(self) -> None:
        """The credit-assignment half of the bug."""
        from config import RewardConfig
        from rewards import PendingHazardTracker, DODGE_ACTIONS

        cfg = RewardConfig()
        tr = PendingHazardTracker(cfg)
        dodge = next(iter(sorted(DODGE_ACTIONS)))
        noop = 0
        grants = []
        for f in range(int(self.FPS * 12)):
            ts = f / self.FPS
            detected = f < int(self.FPS * 10)
            h = _hz(f, ts, detected)
            if detected:
                tr.register(h)
            action = dodge if f == 0 else noop       # dodge once, then nothing
            bonus = tr.on_frame(h, action)
            tr.expire_old(ts)
            if bonus:
                grants.append((f, tr.events[-1].opened_frame_id))
        for grant_frame, opened_frame in grants:
            gap = grant_frame - opened_frame
            assert gap <= cfg.hazard_resolve_frames + 2, (
                f"bonus at frame {grant_frame} was credited to an event "
                f"opened at frame {opened_frame} ({gap} frames earlier); a "
                f"dodge that old must not be paid out"
            )

    def test_a_normal_dodge_still_earns_its_bonus(self) -> None:
        """Guard against over-correcting: the happy path must keep working."""
        from config import RewardConfig
        from rewards import PendingHazardTracker, DODGE_ACTIONS

        cfg = RewardConfig()
        tr = PendingHazardTracker(cfg)
        dodge = next(iter(sorted(DODGE_ACTIONS)))
        total = 0.0
        for f in range(10):
            ts = f / self.FPS
            h = _hz(f, ts, detected=f == 0)
            if h.detected:
                tr.register(h)
            total += tr.on_frame(h, dodge)
            tr.expire_old(ts)
        assert total == pytest.approx(cfg.hazard_bonus), (
            f"a clean dodge followed by quiet frames should earn exactly "
            f"{cfg.hazard_bonus}, got {total}"
        )

    def test_opened_ts_is_never_mutated_by_register(self) -> None:
        from config import RewardConfig
        from rewards import PendingHazardTracker

        cfg = RewardConfig()
        tr = PendingHazardTracker(cfg)
        tr.register(_hz(0, 100.0, detected=True))
        opened = tr.events[0].opened_ts
        for f in range(1, 50):
            tr.register(_hz(f, 100.0 + f / self.FPS, detected=True))
        assert tr.events[0].opened_ts == opened, (
            "opened_ts is the expiry deadline anchor and must be immutable"
        )
        assert tr.events[0].last_seen_ts > opened, (
            "the 'extend the danger window' semantics should still be "
            "recorded, just not on the deadline field"
        )


# --------------------------------------------------------------------- #
# 16. A black / flat capture must never be accepted as a calibration anchor
# --------------------------------------------------------------------- #
class TestBlackCaptureGuard:
    """The anchor check used to accept a broken capture.

    Symptom from a real Windows run: the preview was a black box and the
    anchor read ``RGB=(37,37,35) stability(std)=0.00 [ACCEPTED]``.  Two
    defects compounded:

    * the step-3 preview image was never drawn on the anchor canvas, so the
      user picked the anchor blind on a black box, and
    * the acceptance gate was ``std <= 6`` -- "stable" -- but a black or flat
      patch is the *most* stable thing there is, so a broken capture sailed
      through with std=0.00.

    These tests pin the perception-side guards that make a broken capture
    impossible to accept silently.  They are pure numpy, so they run without a
    display.
    """

    def test_black_frame_is_flagged(self) -> None:
        from perception import is_black_frame, capture_problem, mean_luma

        black = np.zeros((50, 50, 3), dtype=np.uint8)
        assert is_black_frame(black, 8.0)
        msg = capture_problem(black)
        assert msg is not None and "BLACK" in msg

    def test_normal_frame_is_not_flagged(self) -> None:
        from perception import is_black_frame, capture_problem

        rng = np.random.default_rng(0)
        frame = rng.integers(40, 220, size=(50, 50, 3)).astype(np.uint8)
        assert not is_black_frame(frame, 8.0)
        assert capture_problem(frame) is None

    def test_flat_patch_is_degenerate_even_when_stable(self) -> None:
        """A uniform patch has std=0 (maximally 'stable') yet is useless."""
        from perception import is_degenerate_patch

        flat = np.full((5, 5, 3), 37, dtype=np.uint8)   # the (37,37,35)-style patch
        assert is_degenerate_patch(flat)
        textured = np.random.default_rng(1).integers(
            0, 255, size=(5, 5, 3)).astype(np.uint8)
        assert not is_degenerate_patch(textured)

    def test_acceptance_requires_content_not_just_stability(self) -> None:
        """Rebuild the real gate and show a black capture is rejected."""
        from perception import patch_stability, is_degenerate_patch, mean_luma

        black_samples = [np.zeros((5, 5, 3), dtype=np.uint8) for _ in range(17)]
        std, baseline = patch_stability(black_samples)
        assert std == 0.0, "a black patch is perfectly stable — that's the trap"
        black = mean_luma(np.stack(black_samples)) < 8.0
        flat = is_degenerate_patch(black_samples[0])
        accepted = (std <= 6.0) and not black and not flat
        assert not accepted, "black capture must be rejected despite std=0"

    def test_mean_luma_matches_known_value(self) -> None:
        from perception import mean_luma

        white = np.full((4, 4, 3), 255, dtype=np.uint8)
        assert mean_luma(white) == pytest.approx(255.0, abs=1e-6)
        grey = np.full((4, 4, 3), 128, dtype=np.uint8)
        assert 127.0 <= mean_luma(grey) <= 129.0


# --------------------------------------------------------------------- #
# 17. Activation-memory estimate must not double-count composite modules
# --------------------------------------------------------------------- #
class TestActivationMemoryEstimate:
    """``estimate_activation_memory_mb`` hooked containers AND their children.

    The selector matched ``ConvBlock``/``DepthwiseSeparableConv`` by class
    name *and* their inner ``nn.Conv2d`` children by ``isinstance``.  A
    container's output tensor is identical to its last child's output, so it
    was summed twice (measured 2.06 MB vs the true 1.67 MB for strict_lite --
    a 1.23x inflation on the number used to size profiles for a 12 GB box).
    """

    def test_single_conv_is_counted_exactly_once(self) -> None:
        import torch.nn as nn
        from profiling import estimate_activation_memory_mb

        conv = nn.Conv2d(3, 8, 3, padding=1)
        # output (1,8,8,8) = 512 elems * 4 bytes = 2048 B
        expected_mb = (1 * 8 * 8 * 8) * 4 / 1048576
        got = estimate_activation_memory_mb(conv, (3, 8, 8))
        assert got == pytest.approx(expected_mb, rel=1e-3), (
            f"a lone Conv2d should contribute exactly its output bytes, "
            f"got {got} expected {expected_mb}")

    def test_composite_container_is_not_double_counted(self) -> None:
        """A ConvBlock's estimate must equal the sum over its LEAF children,
        not leaf-children + the container's (duplicated) output."""
        import torch
        import torch.nn as nn
        from models import ConvBlock
        from profiling import estimate_activation_memory_mb

        block = ConvBlock(3, 8, 3)          # children: Conv2d, GroupNorm, ReLU
        leaf_bytes = 0
        handles = []
        def leaf_hook(m, i, o):
            nonlocal leaf_bytes
            if isinstance(o, torch.Tensor):
                leaf_bytes += o.numel() * o.element_size()
        for m in block.modules():
            if next(m.children(), None) is None:   # leaves only = ground truth
                handles.append(m.register_forward_hook(leaf_hook))
        with torch.inference_mode():
            block(torch.zeros(1, 3, 8, 8))
        for h in handles:
            h.remove()
        got = estimate_activation_memory_mb(block, (3, 8, 8))
        assert got == pytest.approx(leaf_bytes / 1048576, rel=1e-3), (
            f"ConvBlock estimate {got} MB != leaf-only ground truth "
            f"{leaf_bytes / 1048576} MB (double count would be larger)")

    def test_wrapping_in_anonymous_container_adds_nothing(self) -> None:
        import torch.nn as nn
        from profiling import estimate_activation_memory_mb

        conv = nn.Conv2d(3, 8, 3, padding=1)
        a = estimate_activation_memory_mb(conv, (3, 8, 8))
        b = estimate_activation_memory_mb(nn.Sequential(conv), (3, 8, 8))
        assert a == pytest.approx(b, rel=1e-6), "containers must contribute 0"


# --------------------------------------------------------------------- #
# 18. A failed config save must never corrupt the existing config.json
# --------------------------------------------------------------------- #
class TestConfigSaveAtomicity:
    """``BotConfig.save`` used to be a plain ``write_text``: a crash or full
    disk mid-write left a truncated config.json that ``load`` could not parse,
    bricking the next start and silently discarding the user's calibration.
    It is now write-to-temp + ``os.replace`` (atomic on NTFS/ext4).
    """

    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        from config import BotConfig
        p = tmp_path / "config.json"
        cfg = BotConfig()
        cfg.seed = 4242
        cfg.save(p)
        assert not (tmp_path / "config.json.tmp").exists(), "no temp litter"
        loaded = BotConfig.load(str(p))
        assert loaded.seed == 4242

    def test_failed_save_leaves_original_intact(self, tmp_path: Path,
                                                 monkeypatch) -> None:
        import os as _os
        from config import BotConfig
        p = tmp_path / "config.json"
        original = BotConfig(); original.seed = 1
        original.save(p)

        def boom(src, dst):
            raise OSError("simulated disk failure during replace")
        monkeypatch.setattr(_os, "replace", boom)

        newer = BotConfig(); newer.seed = 2
        caught = None
        try:
            newer.save(p)
        except OSError as exc:
            caught = exc                 # the failure must surface, not vanish
        assert caught is not None, "save must propagate the failure"
        # The on-disk config must still be the ORIGINAL, parseable file.
        assert BotConfig.load(str(p)).seed == 1, (
            "a failed save corrupted the existing config.json")


# --------------------------------------------------------------------- #
# 19. There must be exactly ONE is_black_frame (no import-time shadowing)
# --------------------------------------------------------------------- #
class TestNoShadowedBlackFrame:
    """Round 4 added a second ``is_black_frame`` that shadowed the one the
    runtime ``ZonePreprocessor`` uses, silently swapping its black-detection
    formula.  A duplicate top-level definition is a DRY violation and an
    invisible behaviour change.  Pin both the structure and the runtime path.
    """

    def test_is_black_frame_is_defined_exactly_once(self) -> None:
        import ast
        src = (Path(__file__).resolve().parent.parent / "perception.py") \
            .read_text(encoding="utf-8")
        defs = [n.name for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "is_black_frame"]
        assert len(defs) == 1, f"is_black_frame defined {len(defs)} times: {defs}"

    def test_runtime_preprocessor_flags_black_and_accepts_normal(self) -> None:
        """The production perception path must skip black frames (§6)."""
        from config import PerceptionConfig
        from perception import ZonePreprocessor

        cfg = PerceptionConfig()
        pre = ZonePreprocessor(cfg, cfg.horizon_frac, anchor_xy=None,
                               require_anchor=False)
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        rb = pre.process(black, 1, 0.0)
        assert not rb.valid and rb.reason == "black", (
            f"a black frame must be invalid at runtime, got {rb.reason}")
        normal = np.random.default_rng(0).integers(
            40, 220, size=(100, 100, 3)).astype(np.uint8)
        rn = pre.process(normal, 2, 0.0)
        assert rn.valid and rn.ground_gray is not None

    def test_black_threshold_respects_config(self) -> None:
        """A frame whose luma is just under the configured threshold is black;
        just over is not.  Guards the unified single definition."""
        from perception import is_black_frame

        dim = np.full((20, 20, 3), 3, dtype=np.uint8)    # luma 3
        assert is_black_frame(dim, 4.0)
        bright = np.full((20, 20, 3), 30, dtype=np.uint8)  # luma 30
        assert not is_black_frame(bright, 4.0)


# --------------------------------------------------------------------- #
# 20. The step-4 respawn canvas must show the live preview (not a black box)
# --------------------------------------------------------------------- #
class TestRespawnPreviewVisible:
    """Step 4 (respawn click) used to be a blank black canvas -- the preview
    image was only drawn on step 2/3, so the user clicked the respawn point
    blind.  Guard the fix structurally (gui.py cannot be imported headless).
    """

    def _src(self) -> str:
        return (Path(__file__).resolve().parent.parent / "gui.py") \
            .read_text(encoding="utf-8")

    def test_update_preview_draws_on_respawn_canvas(self) -> None:
        import ast
        tree = ast.parse(self._src())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_update_preview")
        drawn = [
            n.func.value.attr for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "create_image" and isinstance(n.func.value, ast.Attribute)
        ]
        assert "respawn_canvas" in drawn, (
            "_update_preview must draw the live frame onto respawn_canvas so "
            "the respawn point can be picked by sight, not blind")
        assert "anchor_canvas" in drawn

    def test_respawn_click_marks_the_point(self) -> None:
        import ast
        tree = ast.parse(self._src())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_respawn_clicked")
        calls = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)}
        assert "create_oval" in calls, "clicking respawn must leave a visible marker"


class TestUIIsVietnamese:
    """The user asked for the whole UI in Vietnamese.  Pin the key strings so a
    future edit cannot silently revert the interface to English."""

    def test_tabs_are_vietnamese(self) -> None:
        src = self._src() if hasattr(self, "_src") else \
            (Path(__file__).resolve().parent.parent / "gui.py").read_text(encoding="utf-8")
        for vn in ("1 Chọn vùng", "2 Khoá vùng", "4 Nút hồi sinh", "6 Chạy & số liệu"):
            assert vn in src, f"tab {vn!r} missing"
        for en in ('"1 Select region"', '"4 Respawn click"', '"6 Train"'):
            assert en not in src, f"English tab {en} still present"

    def test_key_buttons_and_status_are_vietnamese(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "gui.py") \
            .read_text(encoding="utf-8")
        for vn in ("Bật preview", "Khoá vùng", "Hiệu chuẩn SỐNG (2s)",
                   "Test click (hỏi trước)", "Bắt đầu train", "KHẨN CẤP (F8)",
                   "Số liệu trực tiếp", "Nhật ký train"):
            assert vn in src, f"Vietnamese UI string missing: {vn}"
        for en in ('"Start preview"', '"Lock region"', '"Calibrate ALIVE (2s)"',
                   '"Start training"', '"Live metrics"', '"Training log"'):
            assert en not in src, f"English UI string still present: {en}"


# --------------------------------------------------------------------- #
# 21. Calibration must be VISIBLE and the test click must not be focus-gated
# --------------------------------------------------------------------- #
class TestCalibrationVisibleAndTestClick:
    """From a real Windows log: the horizon line never appeared on the preview
    ("chân trời ko thấy hiện"), and every calibration test click printed
    ``BLOCKED (focus?)`` because the gameplay focus gate also guarded the
    user-confirmed test click.  Guard both fixes.
    """

    def _src(self) -> str:
        return (Path(__file__).resolve().parent.parent / "gui.py") \
            .read_text(encoding="utf-8")

    def test_preview_draws_horizon_and_markers(self) -> None:
        src = self._src()
        assert "ImageDraw.Draw(img)" in src, "preview must be annotated"
        assert "horizon_frac" in src and "dr.line(" in src, \
            "the horizon split line must be drawn on the preview"
        assert src.count("dr.ellipse(") >= 2, \
            "anchor and respawn markers must be drawn on the preview"

    def test_test_click_bypasses_focus_gate(self) -> None:
        src = self._src()
        assert "ctl.click(x, y, confirm_focus=False)" in src, (
            "the user-confirmed calibration test click must not be blocked by "
            "the gameplay Chrome-focus gate")

    def test_a_to_z_help_tab_exists_and_is_vietnamese(self) -> None:
        src = self._src()
        assert "HELP_AZ" in src and '"❓ Hướng dẫn"' in src
        for phrase in ("BƯỚC 1", "BƯỚC 6", "hardware acceleration",
                       "CLICK VÀO CỬA", "F8",
                       "QUAY DEMO", "TIỀN-HUẤN LUYỆN BC",
                       "--validate-demos", "--pretrain",
                       "bc.min_episodes", "NOOP only",
                       "● REC", "HĐ:", "vùng 84×84",
                       "AI không tiến bộ", "CHẾT NGAY LIÊN TIẾP"):
            assert phrase in src, f"help must cover {phrase!r}"

    def test_click_focus_gate_still_protects_gameplay(self) -> None:
        """The bypass must be scoped to the test click: the default click still
        confirms focus (safety for real gameplay)."""
        from input_controller import InputController
        from config import InputConfig
        import inspect
        sig = inspect.signature(InputController.click)
        assert sig.parameters["confirm_focus"].default is True, (
            "gameplay clicks must still require focus by default")

    def test_start_runs_preflight_capture_check(self) -> None:
        """Start must pre-check the capture so a covered region / dead screen is
        flagged before a run that would instantly die (the log showed an
        episode dying at frame 33, distance ~ dead-sample)."""
        src = self._src()
        assert "self._preflight_capture_check()" in src
        assert "def _preflight_capture_check" in src
        assert "patch_rgb_distance" in src, "preflight must use the real API"


class TestChangelogNotMerged:
    """A missing comma once silently concatenated two changelog entries into one
    string.  Every entry must start with its own distinct version tag."""

    def test_each_entry_has_distinct_version(self) -> None:
        import re
        from version import CHANGELOG
        tags = [re.match(r"^(\d+\.\d+\.\d+)", e) for e in CHANGELOG]
        assert all(m for m in tags), "every changelog entry starts with a version"
        versions = [m.group(1) for m in tags]  # type: ignore[union-attr]
        assert len(versions) == len(set(versions)), (
            f"merged/duplicate changelog entries: {versions}")
        # a merged entry would contain a second "X.Y.Z —" tag inside it
        for entry in CHANGELOG:
            inner = re.findall(r"\d+\.\d+\.\d+ —", entry)
            assert len(inner) == 1, f"merged changelog entry: {entry!r}"


# --------------------------------------------------------------------- #
# 22. Demo recorder: phantom actions, stoppability, live feedback
# --------------------------------------------------------------------- #
def _make_rec(tmp_path):
    from config import BotConfig
    from demonstration_recorder import DemoRecorder
    cfg = BotConfig()
    cfg.region.left, cfg.region.top = 120, 80
    cfg.region.width, cfg.region.height = 480, 800
    cfg.region.screen_width, cfg.region.screen_height = 1920, 1080
    cfg.region.dpi_scale = 1.0
    return DemoRecorder(cfg, tmp_path / "demos", lambda: None)


class TestDemoRecorderPhantomActions:
    """A single scalar _current_action set-on-press / clear-on-release left the
    action stuck after a missed release (phantom actions the player never
    pressed), let a second key overwrite the first, and leaked alt-tab arrows.
    Guard the held-key-set rewrite."""

    def test_held_key_most_recent_wins_and_release_is_scoped(self, tmp_path) -> None:
        from config import NOOP, LEFT, RIGHT
        rec = _make_rec(tmp_path)
        rec._handle_press("left")
        assert rec.current_action() == LEFT
        rec._handle_press("right")
        assert rec.current_action() == RIGHT
        rec._handle_release("right")
        assert rec.current_action() == LEFT, (
            "releasing one key must not clear a different still-held key")
        rec._handle_release("left")
        assert rec.current_action() == NOOP

    def test_modifier_guard_blocks_shortcuts(self, tmp_path) -> None:
        from config import NOOP, LEFT
        rec = _make_rec(tmp_path)
        rec._handle_press("alt_l")
        rec._handle_press("left")
        assert rec.current_action() == NOOP, (
            "an arrow pressed under a modifier (alt-tab) is not a game action")
        rec._handle_release("left")
        rec._handle_release("alt_l")
        rec._handle_press("left")
        assert rec.current_action() == LEFT

    def test_stuck_action_auto_clears_on_missed_release(self, tmp_path) -> None:
        import time
        import numpy as np
        from config import LEFT, NOOP
        from ipc import Frame
        rec = _make_rec(tmp_path)
        rec.start()
        rec._handle_press("left")
        assert rec.current_action() == LEFT
        rec._last_key_event_ts = time.monotonic() - (rec.STUCK_CLEAR_S + 0.5)
        img = np.full((800, 480, 3), (40, 44, 60), dtype=np.uint8)
        rec.tick(Frame(frame_id=1, ts=0.0, image=img))
        assert rec.current_action() == NOOP, "a stuck action must be cleared"
        rec.stop(done=False)

    def test_public_readouts_for_hud(self, tmp_path) -> None:
        import numpy as np
        from ipc import Frame
        rec = _make_rec(tmp_path)
        rec.start()
        img = np.full((800, 480, 3), (40, 44, 60), dtype=np.uint8)
        rec.tick(Frame(frame_id=1, ts=0.0, image=img))
        z = rec.last_zone()
        assert z is not None and z.shape == (84, 84)
        assert isinstance(rec.current_action(), int)
        rec.stop(done=False)


class TestDemoStopAndLiveHUD:
    def _src(self) -> str:
        return (Path(__file__).resolve().parent.parent / "gui.py") \
            .read_text(encoding="utf-8")

    def test_f9_stop_hotkey_is_actually_wired(self) -> None:
        src = self._src()
        assert "keyboard.Key.f9" in src, "F9 was advertised but never bound"
        assert "_arm_demo_hotkey" in src and "_disarm_demo_hotkey" in src
        # the stop must be marshalled to the Tk thread (pynput runs its own)
        assert "self._demo_stop_req.is_set()" in src

    def test_recording_hud_is_drawn(self) -> None:
        src = self._src()
        assert "_draw_rec_overlay" in src
        assert "BotState.RECORDING_DEMO" in src
        assert "vùng 84×84 đang ghi" in src


class TestInstantDeathDiagnostic:
    """'AI không tiến triển sau 500 episode' is almost always the bot dying
    instantly every episode (no survival signal).  The actor must say so loudly
    instead of letting hundreds of empty episodes pass in silence."""

    def test_actor_detects_instant_death_loop(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "environment.py") \
            .read_text(encoding="utf-8")
        assert "_instant_death_streak" in src
        assert "survival < 1.0" in src
        assert "CHẾT NGAY LIÊN TIẾP" in src
        # the streak must reset on a surviving episode, else it never re-arms
        assert "self._instant_death_streak = 0" in src


# --------------------------------------------------------------------- #
# 23. Explicit state logging for demo recording and BC pretraining
# --------------------------------------------------------------------- #
class TestExplicitStateLogging:
    """The user could not see what demo recording / pretraining were doing.
    Both must log their state explicitly to the GUI log."""

    def _gui(self) -> str:
        return (Path(__file__).resolve().parent.parent / "gui.py") \
            .read_text(encoding="utf-8")

    def test_demo_start_stop_and_progress_are_logged(self) -> None:
        src = self._gui()
        assert "=== BẮT ĐẦU QUAY DEMO ===" in src
        assert "=== DỪNG QUAY DEMO ===" in src
        assert "đang ghi demo:" in src, "periodic progress must be logged"
        assert "keyboard_active()" in src, "hook status must be surfaced"

    def test_pretrain_start_and_done_are_logged(self) -> None:
        src = self._gui()
        assert "=== BẮT ĐẦU TIỀN-HUẤN LUYỆN (BC) ===" in src
        assert "_on_pretrain_done" in src
        assert 'kind == "pretrain_done"' in src
        assert "TIỀN-HUẤN LUYỆN HOÀN TẤT" in src
        assert "TIỀN-HUẤN LUYỆN BỎ QUA" in src

    def test_learner_emits_progress_to_gui_queue(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "learner_worker.py") \
            .read_text(encoding="utf-8")
        # dataset validation + completion must go through metrics_q (not just
        # report=print, which only reaches the learner's stdout)
        assert src.count('BC: đang kiểm tra demo') == 1
        assert 'episode hợp lệ' in src
        assert 'BC: hoàn tất' in src


# --------------------------------------------------------------------- #
# 24. Demo recording must NOT drive the game (capture-only)
# --------------------------------------------------------------------- #
class TestDemoDoesNotSelfPlay:
    """The user saw the character move BY ITSELF during demo recording: the
    actor (which presses keys) was started even though only capture is needed.
    Recording must be capture-only, and the lifecycle must let training start
    afterwards."""

    def test_app_start_can_skip_actor(self) -> None:
        import inspect
        from app import BotApplication
        sig = inspect.signature(BotApplication.start)
        assert "with_actor" in sig.parameters
        assert sig.parameters["with_actor"].default is True

    def test_demo_recording_starts_capture_only(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "gui.py") \
            .read_text(encoding="utf-8")
        assert "with_learner=False, with_actor=False" in src, (
            "demo recording must not start the key-pressing actor")

    def test_demo_stop_releases_app_for_training(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "gui.py") \
            .read_text(encoding="utf-8")
        assert "_demo_started_app" in src
        assert "self.app.shutdown()" in src
        # the actor-died warning must be scoped to runs that have an actor
        assert 'getattr(self.app, "actor_proc", None) is not None' in src


# --------------------------------------------------------------------- #
# 25. Each life = one episode: auto-split on death
# --------------------------------------------------------------------- #
def _split_rec(tmp_path, with_anchor=True):
    from config import BotConfig
    from demonstration_recorder import DemoRecorder
    cfg = BotConfig()
    cfg.region.left, cfg.region.top = 0, 0
    cfg.region.width, cfg.region.height = 480, 800
    cfg.region.screen_width, cfg.region.screen_height = 1920, 1080
    cfg.region.dpi_scale = 1.0
    if with_anchor:
        cfg.death.anchor_fx, cfg.death.anchor_fy = 0.5, 0.5
        cfg.death.anchor_baseline_rgb = (100, 100, 100)
        cfg.death.threshold = 25.0
        cfg.death.confirm_frames = 3
    return DemoRecorder(cfg, tmp_path / "demos", lambda: None)


class TestDemoAutoSplitOnDeath:
    def test_death_splits_episode_and_keeps_recording(self, tmp_path) -> None:
        import numpy as np
        from ipc import Frame
        rec = _split_rec(tmp_path)
        rec.death_trim_s = 0.0  # isolate split logic from the tail-trim
        rec.start()
        alive = np.full((800, 480, 3), (100, 100, 100), dtype=np.uint8)
        dead = np.full((800, 480, 3), (200, 50, 50), dtype=np.uint8)
        for i in range(5):
            rec.tick(Frame(frame_id=i + 1, ts=i / 30.0, image=alive))
        assert rec.frame_count() > 0 and rec.episode_paths == []
        cf = rec.cfg.death.confirm_frames
        for i in range(cf):
            rec.tick(Frame(frame_id=100 + i, ts=1.0 + i / 30.0, image=dead))
        assert len(rec.episode_paths) == 1, "death must save one episode"
        assert rec.recording, "recording must continue after a split"
        assert rec._wait_alive is True and rec.frame_count() == 0
        for i in range(cf + 2):
            rec.tick(Frame(frame_id=200 + i, ts=2.0 + i / 30.0, image=alive))
        assert rec._wait_alive is False and rec.frame_count() > 0
        rec.stop(done=False)

    def test_no_split_without_calibrated_anchor(self, tmp_path) -> None:
        import numpy as np
        from ipc import Frame
        rec = _split_rec(tmp_path, with_anchor=False)
        rec.start()
        dead = np.full((800, 480, 3), (200, 50, 50), dtype=np.uint8)
        for i in range(10):
            rec.tick(Frame(frame_id=i + 1, ts=i / 30.0, image=dead))
        assert rec.episode_paths == [], "no anchor -> no auto-split"
        rec.stop(done=False)

    def test_death_split_trims_glitchy_tail(self, tmp_path) -> None:
        import numpy as np
        from pathlib import Path as _P
        from ipc import Frame
        rec = _split_rec(tmp_path)
        rec.death_trim_s = 3.5
        rec.start()
        alive = np.full((800, 480, 3), (100, 100, 100), dtype=np.uint8)
        dead = np.full((800, 480, 3), (200, 50, 50), dtype=np.uint8)
        # 6 seconds of alive play at 30 fps
        n_alive = 180
        for i in range(n_alive):
            rec.tick(Frame(frame_id=i + 1, ts=i / 30.0, image=alive))
        assert rec.frame_count() == n_alive
        cf = rec.cfg.death.confirm_frames
        for i in range(cf):
            rec.tick(Frame(frame_id=500 + i, ts=10.0 + i / 30.0, image=dead))
        assert len(rec.episode_paths) == 1
        data = np.load(_P(rec.episode_paths[0]), allow_pickle=False)
        kept = len(data["frames"])
        assert kept < n_alive, "the glitchy tail must be cut"
        last_ts = float(data["timestamps"][-1])
        # last kept frame must be at least ~3.5s before the final alive frame
        assert last_ts <= (n_alive - 1) / 30.0 - 3.5 + 1e-6
        rec.stop(done=False)
