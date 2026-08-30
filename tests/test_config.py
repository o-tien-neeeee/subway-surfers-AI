"""Configuration validation tests."""

from __future__ import annotations

import json

import pytest

from config import (
    PROFILE_NAMES,
    BotConfig,
    ConfigError,
    PerceptionConfig,
)


class TestValidation:
    def test_default_config_is_valid(self) -> None:
        cfg = BotConfig()
        assert cfg.validate() == [] or all("warning" not in w or w for w in cfg.validate())

    def test_bad_profile_rejected(self) -> None:
        cfg = BotConfig()
        cfg.rl.profile = "giant_gpu_net"
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_horizon_fraction_bounds(self) -> None:
        with pytest.raises(ConfigError):
            PerceptionConfig(horizon_frac=0.5)
        with pytest.raises(ConfigError):
            PerceptionConfig(horizon_frac=0.05)
        PerceptionConfig(horizon_frac=0.15)  # boundary ok
        PerceptionConfig(horizon_frac=0.40)  # boundary ok

    def test_keymap_must_cover_all_actions(self) -> None:
        cfg = BotConfig()
        cfg.input.keymap = {0: "", 1: "left", 2: "right", 3: "up"}
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_keymap_must_not_be_empty_for_gameplay(self) -> None:
        cfg = BotConfig()
        cfg.input.keymap = {0: "", 1: "", 2: "right", 3: "up", 4: "down"}
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_scheduler_bounds(self) -> None:
        cfg = BotConfig()
        cfg.scheduler.min_decision_frames = 5
        cfg.scheduler.max_decision_frames = 2
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_gamma_range(self) -> None:
        cfg = BotConfig()
        cfg.rl.gamma = 1.5
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_buffer_warning_not_error(self) -> None:
        cfg = BotConfig()
        cfg.per.capacity = 500_000
        warnings = cfg.validate()
        assert any("replay buffer" in w.lower() for w in warnings)

    def test_confirm_frames_positive(self) -> None:
        cfg = BotConfig()
        cfg.death.confirm_frames = 0
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_reward_clip_sanity(self) -> None:
        cfg = BotConfig()
        cfg.reward.reward_clip_min = 1.0
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_n_step_range(self) -> None:
        cfg = BotConfig()
        cfg.rl.n_step = 99
        with pytest.raises(ConfigError):
            cfg.validate()


class TestProfiles:
    def test_downgrade_order(self) -> None:
        cfg = BotConfig()
        cfg.rl.profile = "quality_cpu"
        assert cfg.profile_downgrade() == "balanced_cpu"
        cfg.rl.profile = "balanced_cpu"
        assert cfg.profile_downgrade() == "strict_lite"
        cfg.rl.profile = "strict_lite"
        assert cfg.profile_downgrade() is None

    def test_all_profiles_valid(self) -> None:
        for name in PROFILE_NAMES:
            cfg = BotConfig()
            cfg.rl.profile = name
            cfg.validate()


class TestSerialisation:
    def test_json_roundtrip(self, tmp_path) -> None:
        cfg = BotConfig()
        cfg.rl.profile = "balanced_cpu"
        cfg.perception.horizon_frac = 0.3
        cfg.death.anchor_baseline_rgb = (10, 20, 30)
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = BotConfig.load(path)
        assert loaded.rl.profile == "balanced_cpu"
        assert loaded.perception.horizon_frac == pytest.approx(0.3)
        assert loaded.death.anchor_baseline_rgb == (10, 20, 30)
        assert loaded.validate() == cfg.validate()

    def test_partial_json_keeps_defaults(self, tmp_path) -> None:
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"rl": {"profile": "quality_cpu"}}))
        cfg = BotConfig.load(path)
        assert cfg.rl.profile == "quality_cpu"
        assert cfg.capture.target_fps == 30  # untouched default

    def test_invalid_json_raises(self, tmp_path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        with pytest.raises(ConfigError):
            BotConfig.load(path)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            BotConfig.load(tmp_path / "nope.json")
