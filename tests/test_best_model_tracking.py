"""Online best-model tracking (requirement §12).

``best_model.pth`` must be updated during ONLINE training exactly when the
configured episode metric (rolling mean over a window of finished episodes)
improves — and a restart must not overwrite a historical best with a worse
run.  The actor reports episodes through SharedCounters; the learner polls
them; both sides are exercised here in-process.
"""

from __future__ import annotations

from queue import Queue

import pytest

from config import BotConfig
from ipc import SharedCounters, SharedWeights
from learner_worker import Learner
from models import weight_size_for_profile


def make_learner(tmp_path, best_metric: str = "survival_s",
                 window: int = 3) -> tuple[Learner, SharedCounters]:
    cfg = BotConfig()
    cfg.paths.checkpoints_dir = str(tmp_path / "checkpoints")
    cfg.rl.best_metric = best_metric
    cfg.rl.best_metric_window = window
    weights = SharedWeights(weight_size_for_profile("quality_cpu",
                                                    cfg.perception.frame_stack))
    counters = SharedCounters()
    learner = Learner(cfg, weights, counters, Queue(), str(tmp_path / "checkpoints"))
    return learner, counters


def report_episode(counters: SharedCounters, episode_id: int,
                   survival_s: float, reward: float) -> None:
    counters.last_episode_done_id.value = episode_id
    counters.last_episode_survival_s.value = survival_s
    counters.last_episode_reward.value = reward


class TestBestModelTracking:
    def test_first_episode_saves_best(self, tmp_path) -> None:
        learner, counters = make_learner(tmp_path)
        assert learner.ckpt.best_metric is None
        report_episode(counters, 1, survival_s=5.0, reward=3.0)
        rolling = learner._poll_episode_metric()
        assert rolling == pytest.approx(5.0)
        assert learner.ckpt.best_metric == pytest.approx(5.0)
        assert (tmp_path / "checkpoints" / "strict_lite" / "best_model.pth").exists()

    def test_rolling_mean_gates_saving(self, tmp_path) -> None:
        learner, counters = make_learner(tmp_path, window=3)
        report_episode(counters, 1, 10.0, 6.0)
        assert learner._poll_episode_metric() == pytest.approx(10.0)
        report_episode(counters, 2, 12.0, 7.0)
        assert learner._poll_episode_metric() == pytest.approx(11.0)
        best_after_two = learner.ckpt.best_metric
        # episode 3: rolling mean (10+12+4)/3 = 8.67 < 11 -> best NOT updated
        report_episode(counters, 3, 4.0, 2.0)
        learner._poll_episode_metric()
        assert learner.ckpt.best_metric == best_after_two
        # episode 4: window now [12, 4, 14] -> mean 10.0 < 11 -> still refused
        report_episode(counters, 4, 14.0, 9.0)
        learner._poll_episode_metric()
        assert learner.ckpt.best_metric == best_after_two
        # episode 5: window [4, 14, 20] -> mean 12.67 > 11 -> updated
        report_episode(counters, 5, 20.0, 12.0)
        learner._poll_episode_metric()
        assert learner.ckpt.best_metric == pytest.approx(38.0 / 3.0)

    def test_same_episode_id_polled_once(self, tmp_path) -> None:
        learner, counters = make_learner(tmp_path)
        report_episode(counters, 7, 9.0, 5.0)
        assert learner._poll_episode_metric() is not None
        assert learner._poll_episode_metric() is None  # no new episode

    def test_metric_selectable_total_reward(self, tmp_path) -> None:
        learner, counters = make_learner(tmp_path, best_metric="total_reward")
        report_episode(counters, 1, survival_s=1.0, reward=8.0)
        rolling = learner._poll_episode_metric()
        assert rolling == pytest.approx(8.0)
        assert learner.cfg.rl.best_metric == "total_reward"

    def test_nonfinite_and_nonpositive_ignored(self, tmp_path) -> None:
        learner, counters = make_learner(tmp_path)
        report_episode(counters, 1, 0.0, 0.0)
        assert learner._poll_episode_metric() is None
        assert learner.ckpt.best_metric is None

    def test_best_survives_restart_without_regression(self, tmp_path) -> None:
        learner1, counters1 = make_learner(tmp_path)
        report_episode(counters1, 1, 25.0, 15.0)
        learner1._poll_episode_metric()
        best_v1 = learner1.ckpt.best_metric
        # simulate a restart: a fresh Learner over the same checkpoint dir
        learner2, counters2 = make_learner(tmp_path)
        assert learner2.ckpt.best_metric == pytest.approx(best_v1)
        # a WORSE episode after the restart must not overwrite the best
        report_episode(counters2, 2, 3.0, 1.0)
        learner2._poll_episode_metric()
        assert learner2.ckpt.best_metric == pytest.approx(best_v1)
        report_episode(counters2, 3, 30.0, 18.0)
        learner2._poll_episode_metric()
        assert learner2.ckpt.best_metric == pytest.approx(best_v1)  # 16.5 < 25
        # window [3, 30, 50] -> rolling 27.67 > 25 -> the best finally moves
        report_episode(counters2, 4, 50.0, 25.0)
        learner2._poll_episode_metric()
        assert learner2.ckpt.best_metric > best_v1

    def test_window_size_from_config(self, tmp_path) -> None:
        learner, _ = make_learner(tmp_path, window=5)
        assert learner._episode_window.maxlen == 5

    def test_config_validation_rejects_unknown_metric(self) -> None:
        cfg = BotConfig()
        cfg.rl.best_metric = "coins"
        with pytest.raises(Exception, match="best_metric"):
            cfg.validate()
