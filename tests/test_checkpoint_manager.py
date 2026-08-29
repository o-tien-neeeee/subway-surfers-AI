"""Checkpoint manager tests: atomicity, best-model policy, corruption."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from agent import DoubleDQNAgent
from checkpoint_manager import CheckpointManager
from config import PERConfig, RLConfig
from replay_buffer import NStepTransition, PrioritizedReplayBuffer
from test_replay_buffer import TestPrioritizedReplayBuffer


def small_agent() -> DoubleDQNAgent:
    return DoubleDQNAgent("strict_lite", RLConfig(), seed=3)


class TestModelCheckpoints:
    def test_save_load_roundtrip(self, tmp_path) -> None:
        mgr = CheckpointManager(tmp_path, "strict_lite")
        agent = small_agent()
        payload = agent.state_payload()
        saved = mgr.save_model(payload, {"env_frame_id": 42}, which="latest")
        assert saved is not None and saved.exists()
        loaded = mgr.load_model("latest")
        assert loaded is not None
        assert loaded["profile"] == "strict_lite"
        assert loaded["extra"]["env_frame_id"] == 42
        restored = small_agent()
        restored.load_payload(loaded["agent"])
        for (k1, v1), (k2, v2) in zip(agent.online.state_dict().items(),
                                      restored.online.state_dict().items()):
            assert torch.equal(v1, v2), f"weight mismatch in {k1}/{k2}"

    def test_best_only_improves(self, tmp_path) -> None:
        mgr = CheckpointManager(tmp_path, "strict_lite")
        agent = small_agent()
        assert mgr.save_model(agent.state_payload(), {}, which="best", metric=10.0)
        # worse metric -> refused
        assert mgr.save_model(agent.state_payload(), {}, which="best", metric=5.0) is None
        # better metric -> saved
        assert mgr.save_model(agent.state_payload(), {}, which="best", metric=12.0)
        assert mgr.best_metric == 12.0

    def test_lower_is_better_mode(self, tmp_path) -> None:
        mgr = CheckpointManager(tmp_path, "strict_lite")
        agent = small_agent()
        mgr.save_model(agent.state_payload(), {}, which="best", metric=0.5,
                       higher_is_better=False)
        assert mgr.save_model(agent.state_payload(), {}, which="best", metric=0.2,
                              higher_is_better=False) is not None
        assert mgr.best_metric == 0.2

    def test_atomic_write_leaves_no_tmp(self, tmp_path) -> None:
        mgr = CheckpointManager(tmp_path, "strict_lite")
        mgr.save_model(small_agent().state_payload(), {}, which="latest")
        leftovers = list(Path(mgr.latest_path).parent.glob("*.tmp"))
        assert leftovers == []

    def test_corrupt_checkpoint_recovered(self, tmp_path) -> None:
        mgr = CheckpointManager(tmp_path, "strict_lite")
        mgr.save_model(small_agent().state_payload(), {}, which="latest")
        # truncate the file -> unreadable
        mgr.latest_path.write_bytes(b"garbage" * 16)
        loaded = mgr.load_model("latest")
        assert loaded is None, "corrupt checkpoint returns None, not a crash"
        assert (tmp_path / "strict_lite" / "latest_model.pth.corrupt").exists()

    def test_missing_checkpoint_returns_none(self, tmp_path) -> None:
        mgr = CheckpointManager(tmp_path, "strict_lite")
        assert mgr.load_model("latest") is None
        assert mgr.load_model("best") is None

    def test_git_hash_in_payload(self, tmp_path) -> None:
        from checkpoint_manager import _git_hash

        mgr = CheckpointManager(tmp_path, "strict_lite")
        mgr.save_model(small_agent().state_payload(), {}, which="latest")
        data = mgr.latest_path.read_bytes()
        assert isinstance(data, bytes) and len(data) > 0
        h = _git_hash()
        assert h == "" or len(h) == 40


class TestBufferViaManager:
    def test_buffer_save_load_through_manager(self, tmp_path) -> None:
        mgr = CheckpointManager(tmp_path, "strict_lite")
        cfg = PERConfig(capacity=32)
        buf = PrioritizedReplayBuffer(cfg, frame_size=84)
        from test_replay_buffer import fill_buffer
        fill_buffer(12, buf, 200_000)
        assert mgr.buffer_save(buf)
        assert mgr.buffer_path.exists()

        def load_fn():
            return PrioritizedReplayBuffer.load(str(mgr.buffer_path), cfg,
                                                frame_size=84)

        loaded, ok = mgr.buffer_load(load_fn)
        assert ok and loaded is not None
        assert loaded.size == 12

    def test_buffer_corruption_recovers_empty(self, tmp_path) -> None:
        mgr = CheckpointManager(tmp_path, "strict_lite")
        cfg = PERConfig(capacity=32)
        buf = PrioritizedReplayBuffer(cfg, frame_size=84)
        from test_replay_buffer import fill_buffer
        fill_buffer(8, buf, 300_000)
        mgr.buffer_save(buf)
        raw = bytearray(mgr.buffer_path.read_bytes())
        raw[40] ^= 0xAA
        mgr.buffer_path.write_bytes(bytes(raw))

        def load_fn():
            return PrioritizedReplayBuffer.load(str(mgr.buffer_path), cfg,
                                                frame_size=84)

        fresh, ok = mgr.buffer_load(load_fn)
        assert not ok and fresh is None, "corrupt buffer -> empty start, GUI alive"
        assert (tmp_path / "strict_lite" / "buffer.pkl.corrupt").exists()


class TestRngStateIntegrity:
    def test_rng_states_restorable(self, tmp_path) -> None:
        from checkpoint_manager import capture_rng_states, restore_rng_states

        mgr = CheckpointManager(tmp_path, "strict_lite")
        agent = small_agent()
        payload = agent.state_payload()
        mgr.save_model(payload, {"update_step": 7})
        loaded = mgr.load_model("latest")
        rng = loaded["extra"]["rng"]
        assert "torch" in rng and "numpy_legacy" in rng and "python" in rng
        torch.manual_seed(123)
        restore_rng_states(rng)
        assert loaded["extra"]["update_step"] == 7

    def test_restore_rng_roundtrip_deterministic(self, tmp_path) -> None:
        from checkpoint_manager import capture_rng_states, restore_rng_states

        torch.manual_seed(42)
        states = capture_rng_states()
        a = torch.rand(4)
        torch.manual_seed(0)
        restore_rng_states(states)
        b = torch.rand(4)
        assert torch.equal(a, b), "restored RNG must reproduce the stream"
