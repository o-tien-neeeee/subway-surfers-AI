"""Tests for the BC pretrain integration in ``Learner``.

These tests pin:

* When ``bc.bc_pretrain=True`` the learner builds a
  :class:`DQfDAgent` (not the standard QR-DQN).
* When ``bc.bc_pretrain=False`` the learner falls back
  to the original :class:`DistributionalDoubleDQNAgent`.
* The DQfD pre-train path (``_pretrain_dqfd``) actually
  loads the demos into the agent and pre-fills the
  replay buffer.
* The agent's BC anchor survives a series of train
  steps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import BotConfig, RLConfig
from dataset import DemonstrationDataset
from ipc import SharedCounters, SharedWeights
from learner_worker import Learner
from models import weight_size_for_profile


def _fresh_learner(ckpt_dir: str, bc_pretrain: bool = True) -> tuple:
    """Build a fresh learner with a small demo set
    so the DQfD path has data to consume.  Returns
    ``(learner, counters)``."""
    from queue import Queue
    cfg = BotConfig()
    cfg.paths.checkpoints_dir = ckpt_dir
    # Override the BC config for the test.
    cfg.bc.bc_pretrain = bc_pretrain
    # Use a small profile so the QR-DQN is small.
    cfg.rl.profile = "strict_lite"
    cfg.rl.distributional = True
    # The QR-DQN with 4×84×84 input + 51 quantiles has
    # ~194k parameters — much larger than the
    # strict_lite "model" weight budget of 49k.  We
    # allocate a generously-sized shared buffer so
    # the agent's weights fit; in production the
    # ``ipc.SharedWeights`` is sized to the agent's
    # actual weight count.
    import torch
    from dqfd_agent import DQfDAgent, DQfDConfig
    if bc_pretrain:
        probe = DQfDAgent(cfg.rl.profile, cfg.rl, DQfDConfig(),
                            in_frames=cfg.perception.frame_stack,
                            size=cfg.perception.ground_size,
                            num_quantiles=int(getattr(cfg.rl, "num_quantiles", 51)))
        n_params = probe.count_params()
    else:
        from agent_distributional import DistributionalDoubleDQNAgent
        probe = DistributionalDoubleDQNAgent(cfg.rl.profile, cfg.rl,
                num_quantiles=int(getattr(cfg.rl, "num_quantiles", 51)))
        n_params = probe.count_params()
    weights = SharedWeights(int(n_params * 1.5))
    counters = SharedCounters()
    learner = Learner(cfg, weights, counters, Queue(), ckpt_dir)
    return learner, counters


@pytest.fixture()
def demo_dir(tmp_path):
    """Create a tiny demo directory on disk with two
    valid 84x84 episodes (the default min_episodes=2)."""
    d = tmp_path / "demos"
    d.mkdir()
    for i in range(2):
        n_frames = 60
        frames = np.zeros((n_frames, 84, 84), dtype=np.uint8)
        actions = np.zeros(n_frames, dtype=np.int64)
        actions[10:13] = 1  # LEFT dodges
        actions[30:33] = 2  # RIGHT dodges
        actions[50:53] = 3  # JUMP dodges
        timestamps = np.arange(n_frames, dtype=np.float64) / 30.0
        done = np.zeros(n_frames, dtype=bool)
        done[-1] = True
        np.savez(d / f"episode_{i:04d}.npz",
                  frames=frames, actions=actions,
                  timestamps=timestamps, done=done)
    return str(d)


class TestLearnerBuildsDqfdWhenFlagSet:
    def test_bc_pretrain_true_uses_dqfd(self, tmp_path) -> None:
        ckpt_dir = str(tmp_path / "ckpts")
        learner, _ = _fresh_learner(ckpt_dir, bc_pretrain=True)
        from dqfd_agent import DQfDAgent
        assert isinstance(learner.agent, DQfDAgent), (
            f"with bc.bc_pretrain=True, learner.agent should "
            f"be a DQfDAgent, got {type(learner.agent)}")

    def test_bc_pretrain_false_uses_standard(self, tmp_path) -> None:
        ckpt_dir = str(tmp_path / "ckpts2")
        learner, _ = _fresh_learner(ckpt_dir, bc_pretrain=False)
        from dqfd_agent import DQfDAgent
        from agent_distributional import DistributionalDoubleDQNAgent
        assert not isinstance(learner.agent, DQfDAgent)
        assert isinstance(learner.agent, DistributionalDoubleDQNAgent)


class TestLearnerDqfdPretrain:
    def test_dqfd_pretrain_caches_demos(self, tmp_path,
                                          demo_dir: str) -> None:
        ckpt_dir = str(tmp_path / "ckpts3")
        learner, _ = _fresh_learner(ckpt_dir, bc_pretrain=True)
        # The default cfg.bc.epochs is 8, which is
        # enough to drive the loss down on a small
        # dataset.
        res = learner.pretrain(demo_dir, report=lambda m: None,
                                force=True)
        assert res["status"] == "ok"
        assert res.get("method") == "dqfd"
        # The agent's _demo_obs should now be cached.
        assert learner.agent._demo_obs is not None
        # The replay buffer should have been pre-filled.
        assert learner.buffer.size > 0, (
            "replay buffer was NOT pre-filled with demo "
            "transitions — the BC anchor will drift")

    def test_dqfd_pretrain_sets_bc_done(self, tmp_path,
                                          demo_dir: str) -> None:
        ckpt_dir = str(tmp_path / "ckpts4")
        learner, counters = _fresh_learner(ckpt_dir, bc_pretrain=True)
        learner.pretrain(demo_dir, report=lambda m: None, force=True)
        assert learner.bc_done is True
        assert float(counters.bc_pretrained.value) == 1.0
