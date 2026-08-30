"""Replay buffer tests: PER maths, eviction, persistence, corruption."""

from __future__ import annotations

import numpy as np
import pytest

from config import PERConfig
from logging_utils import CorruptFileError
from replay_buffer import (
    FrameStore,
    NStepBuilder,
    PrioritizedReplayBuffer,
    SumTree,
)


def frame(value: float) -> np.ndarray:
    f = np.zeros((84, 84), dtype=np.uint8)
    f[:, :] = int(value) % 256
    return f


def stack4(base: float) -> np.ndarray:
    return np.stack([frame(base + i) for i in range(4)], axis=0)


def fill_buffer(n: int, buf: PrioritizedReplayBuffer, start_env_id: int = 0) -> int:
    """Add ``n`` synthetic transitions; returns the next unused env id."""
    from replay_buffer import NStepTransition

    env_id = start_env_id
    for i in range(n):
        env_id += 1
        base = float((i * 3) % 200)
        tr = NStepTransition(
            obs=stack4(base), next_obs=stack4(base + 1), action=i % 5,
            reward=0.1, done=(i == n - 1), span=1, gamma_pow=0.99,
            obs_env_ids=(env_id, env_id, env_id, env_id),
            next_env_ids=(env_id + 1,) * 4,
        )
        buf.add_nstep(tr)
    return env_id


class TestSumTree:
    def test_total_and_update(self) -> None:
        t = SumTree(8)
        for i in range(8):
            t.update(i, float(i + 1))
        assert t.total() == pytest.approx(36.0)
        t.update(3, 100.0)  # replaces leaf 3 (value 4)
        assert t.total() == pytest.approx(36.0 - 4.0 + 100.0)

    def test_prefixsum_batch_matches_scalar(self) -> None:
        t = SumTree(16)
        rng = np.random.default_rng(0)
        vals = rng.uniform(0.5, 5.0, 16)
        for i, v in enumerate(vals):
            t.update(i, v)
        qs = np.linspace(0.1, t.total() - 0.1, 50)
        batch = t.find_prefixsum_batch(qs)
        for q, leaf in zip(qs, batch):
            assert leaf == t.find_prefixsum(float(q))

    def test_nan_and_negative_sanitised(self) -> None:
        t = SumTree(4)
        t.update(0, float("nan"))
        t.update(1, float("inf"))
        t.update(2, -5.0)
        t.update(3, 2.0)
        assert np.isfinite(t.total()) and t.total() > 0

    def test_batch_update_consistent(self) -> None:
        t = SumTree(8)
        t.update_batch(np.arange(8), np.ones(8))
        t.update_batch(np.array([2, 5]), np.array([10.0, 20.0]))
        assert t.total() == pytest.approx(6 * 1 + 10 + 20)
        # tree[2] roots leaves 0-3: 1 + 1 + 10 + 1
        assert t.tree[2] == pytest.approx(13.0)


class TestFrameStore:
    def test_dedup_by_env_id(self) -> None:
        fs = FrameStore(64, 84)
        a = fs.add_if_new(frame(1), env_id=100)
        b = fs.add_if_new(frame(1), env_id=100)
        assert a == b, "same env id must reuse the slot (no duplicate storage)"
        c = fs.add_if_new(frame(2), env_id=101)
        assert c != a
        assert fs.written == 2

    def test_eviction_invalidates_old_refs(self) -> None:
        fs = FrameStore(4, 84)
        slot0 = fs.add_if_new(frame(5), 1)
        for i in range(2, 10):
            fs.add_if_new(frame(i), i)
        assert not fs.resident(slot0, 1), "wrapped slot must not validate"

    def test_bounded_memory(self) -> None:
        fs = FrameStore(16, 84)
        for i in range(1000):
            fs.add_if_new(frame(i), i)
        assert fs.nbytes() == 16 * 84 * 84 + 16 * 8


class TestPrioritizedReplayBuffer:
    _eid = 0

    def _fill(self, n: int, buf: PrioritizedReplayBuffer) -> None:
        TestPrioritizedReplayBuffer._eid = fill_buffer(
            n, buf, start_env_id=TestPrioritizedReplayBuffer._eid)

    def test_add_and_sample(self) -> None:
        buf = PrioritizedReplayBuffer(PERConfig(capacity=64), frame_size=84)
        self._eid = 0
        self._fill(32, buf)
        assert buf.size == 32
        rng = np.random.default_rng(0)
        batch = buf.sample(16, beta=0.4, rng=rng)
        assert batch["obs"].shape == (16, 4, 84, 84)
        assert batch["next_obs"].shape == (16, 4, 84, 84)
        assert batch["actions"].shape == (16,)
        assert batch["dones"].sum() <= 1
        assert np.all(batch["weights"] <= 1.0001)
        assert np.all(batch["weights"] > 0)

    def test_priorities_bias_sampling(self) -> None:
        buf = PrioritizedReplayBuffer(PERConfig(capacity=64, alpha=1.0))
        self._eid = 1000
        self._fill(8, buf)
        buf.update_priorities(np.array([0]), np.array([1000.0]))  # huge TD
        rng = np.random.default_rng(1)
        counts = np.zeros(8)
        for _ in range(300):
            b = buf.sample(8, beta=0.4, rng=rng)
            for idx in b["indices"]:
                counts[idx] += 1
        assert counts[0] > counts.mean(), "high priority must be sampled more"

    def test_priority_update_and_is_weights(self) -> None:
        buf = PrioritizedReplayBuffer(PERConfig(capacity=64))
        self._eid = 2000
        self._fill(16, buf)
        rng = np.random.default_rng(2)
        b = buf.sample(8, beta=1.0, rng=rng)
        buf.update_priorities(b["indices"], np.full(8, 50.0))
        b2 = buf.sample(8, beta=1.0, rng=rng)
        # uniform priorities after the update -> weights all equal (1.0)
        assert np.allclose(b2["weights"], 1.0, atol=1e-4)

    def test_nan_priorities_never_corrupt(self) -> None:
        buf = PrioritizedReplayBuffer(PERConfig(capacity=32))
        self._eid = 3000
        self._fill(8, buf)
        buf.update_priorities(np.array([0, 1]), np.array([float("nan"), float("inf")]))
        rng = np.random.default_rng(3)
        b = buf.sample(8, beta=0.4, rng=rng)
        assert np.all(np.isfinite(b["weights"]))

    def test_invalid_transition_rejected(self) -> None:
        from replay_buffer import NStepTransition

        buf = PrioritizedReplayBuffer(PERConfig(capacity=16), frame_size=84)
        with pytest.raises(ValueError):
            buf.add_nstep(NStepTransition(
                obs=stack4(1), next_obs=stack4(2), action=9, reward=0.1,
                done=False, span=1, gamma_pow=0.99,
                obs_env_ids=(1, 1, 1, 1), next_env_ids=(2, 2, 2, 2)))
        with pytest.raises(ValueError):
            buf.add_nstep(NStepTransition(
                obs=stack4(1), next_obs=stack4(2), action=1,
                reward=float("nan"), done=False, span=1, gamma_pow=0.99,
                obs_env_ids=(1, 1, 1, 1), next_env_ids=(2, 2, 2, 2)))

    def test_evicted_transitions_are_not_sampled(self) -> None:
        cfg = PERConfig(capacity=32)
        buf = PrioritizedReplayBuffer(cfg, frame_size=84)
        self._eid = 10_000
        self._fill(32, buf)
        # add more -> frame ring wraps, oldest transitions reference evicted frames
        self._fill(40, buf)
        rng = np.random.default_rng(4)
        b = buf.sample(32, beta=0.4, rng=rng)
        assert len(b["indices"]) == 32
        # every sampled transition must still resolve to resident frames
        for i in b["indices"]:
            t = buf.transitions[int(i)]
            assert t is not None

    def test_memory_is_bounded(self) -> None:
        cfg = PERConfig(capacity=128)
        buf = PrioritizedReplayBuffer(cfg, frame_size=84)
        self._eid = 50_000
        self._fill(128, buf)
        before = buf.nbytes()
        self._fill(128, buf)
        assert buf.nbytes() == before, "memory must not grow past capacity"
        assert buf.size == 128

    def test_action_counts(self) -> None:
        buf = PrioritizedReplayBuffer(PERConfig(capacity=16), frame_size=84)
        self._eid = 60_000
        self._fill(10, buf)
        counts = buf.action_counts()
        assert sum(counts.values()) == 10
        assert all(0 <= a <= 4 for a in counts)


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path) -> None:
        cfg = PERConfig(capacity=64)
        buf = PrioritizedReplayBuffer(cfg, frame_size=84)
        TestPrioritizedReplayBuffer._eid = fill_buffer(32, buf, 70_000)
        buf.update_priorities(np.arange(32), np.linspace(0.1, 5.0, 32))
        path = str(tmp_path / "buffer.pkl")
        buf.save(path)
        loaded = PrioritizedReplayBuffer.load(path, cfg, frame_size=84)
        assert loaded.size == 32
        b1 = buf.sample(8, beta=0.4, rng=np.random.default_rng(0))
        b2 = loaded.sample(8, beta=0.4, rng=np.random.default_rng(0))
        assert np.array_equal(b1["indices"], b2["indices"])
        assert np.allclose(b1["rewards"], b2["rewards"])
        assert np.allclose(b1["obs"], b2["obs"])

    def test_corrupt_buffer_renamed_and_reported(self, tmp_path) -> None:
        cfg = PERConfig(capacity=16)
        buf = PrioritizedReplayBuffer(cfg, frame_size=84)
        TestPrioritizedReplayBuffer._eid = fill_buffer(8, buf, 80_000)
        path = tmp_path / "buffer.pkl"
        buf.save(str(path))
        # corrupt the payload but keep the sidecar hash
        data = bytearray(path.read_bytes())
        data[64] ^= 0xFF
        path.write_bytes(bytes(data))
        with pytest.raises(CorruptFileError):
            PrioritizedReplayBuffer.load(str(path), cfg, frame_size=84)
        assert (tmp_path / "buffer.pkl.corrupt").exists()
        assert not path.exists(), "corrupt file must be renamed, not loaded"

    def test_hash_sidecar_written(self, tmp_path) -> None:
        cfg = PERConfig(capacity=16)
        buf = PrioritizedReplayBuffer(cfg, frame_size=84)
        TestPrioritizedReplayBuffer._eid = fill_buffer(4, buf, 90_000)
        path = tmp_path / "buffer.pkl"
        buf.save(str(path))
        assert (tmp_path / "buffer.pkl.sha256").exists()


class TestNStepBuilder:
    def test_n1_passthrough(self) -> None:
        # n=1: G_t = r_t + γ Q(s_{t+1}, ·).  Builder emits step t as soon as
        # the next observation exists; span=1 (one reward: r_t itself),
        # γ^1 applied to the bootstrap target.
        b = NStepBuilder(1, 0.9)
        out = b.push(stack4(0), (1, 1, 1, 1), 2, 0.5, False)
        assert out == []
        out = b.push(stack4(1), (2, 2, 2, 2), 0, 0.25, False)
        assert len(out) == 1
        assert out[0].reward == pytest.approx(0.5)
        assert out[0].span == 1
        assert out[0].done is False
        assert out[0].gamma_pow == pytest.approx(0.9)
        assert np.allclose(out[0].obs, stack4(0))
        assert np.allclose(out[0].next_obs, stack4(1))

    def test_n3_discount(self) -> None:
        b = NStepBuilder(3, 0.5)
        b.push(stack4(0), (1,) * 4, 0, 1.0, False)
        b.push(stack4(1), (2,) * 4, 0, 2.0, False)
        b.push(stack4(2), (3,) * 4, 0, 4.0, False)
        out = b.push(stack4(3), (4,) * 4, 0, 8.0, False)  # window complete
        assert len(out) == 1
        # n=3: R = r_t + γ r_{t+1} + γ^2 r_{t+2} = 1 + .5*2 + .25*4 = 3.0
        # Bootstrap from s_{t+3}; γ^3 = 0.125.
        assert out[0].reward == pytest.approx(3.0)
        assert out[0].span == 3
        assert out[0].gamma_pow == pytest.approx(0.125)
        assert np.allclose(out[0].next_obs, stack4(3))

    def test_terminal_flush_marks_drained_as_done(self) -> None:
        """On episode end every pending transition inside the n-step window
        has a terminal bootstrap (done=True, gamma_pow=0) — otherwise Q
        would leak across episode boundaries.  Rewards are accumulated
        forward from each head to the death step."""
        b = NStepBuilder(3, 0.9)
        b.push(stack4(0), (1,) * 4, 0, 1.0, False)
        b.push(stack4(1), (2,) * 4, 0, 1.0, False)
        out = b.push(stack4(2), (3,) * 4, 0, -10.0, True)
        # All three pending transitions terminate at death:
        #   step0: r = 1 + 0.9*1 + 0.9^2*(-10) = -6.2  done=True
        #   step1: r = 1 + 0.9*(-10)                = -8.0  done=True
        #   step2: r = -10                           = -10.0 done=True
        assert len(out) == 3
        assert all(t.done for t in out)
        assert all(t.gamma_pow == 0.0 for t in out)
        assert out[0].reward == pytest.approx(1 + 0.9 * 1 + 0.81 * (-10.0))
        assert out[1].reward == pytest.approx(1 + 0.9 * (-10.0))
        assert out[2].reward == pytest.approx(-10.0)

    def test_nothing_crosses_episode_boundary(self) -> None:
        b = NStepBuilder(3, 0.9)
        b.push(stack4(0), (1,) * 4, 0, 1.0, True)
        b.clear()
        out = b.push(stack4(9), (99,) * 4, 0, 5.0, True)
        assert len(out) == 1
        assert out[0].reward == pytest.approx(5.0)
        assert out[0].done is True
        assert out[0].gamma_pow == 0.0
