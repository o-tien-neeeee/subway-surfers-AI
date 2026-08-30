"""Demonstration-recorder metadata + episode file integrity (requirement §10).

The recorder must store the preprocessed uint8 frames, the action, the
timestamp, the episode's done flag, optional score/death/confidence, and the
browser geometry + configuration metadata — without ever duplicating full
resolution frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import BotConfig
from demonstration_recorder import DemoRecorder
from ipc import Frame


def make_recorder(tmp_path: Path, cfg: BotConfig | None = None) -> DemoRecorder:
    cfg = cfg or BotConfig()
    cfg.region.left, cfg.region.top = 120, 80
    cfg.region.width, cfg.region.height = 480, 800
    cfg.region.screen_width, cfg.region.screen_height = 1920, 1080
    cfg.region.dpi_scale = 1.25
    frames = iter([])

    def read_frame():
        try:
            return next(frames)
        except StopIteration:
            return None

    return DemoRecorder(cfg, tmp_path / "demos", read_frame)


def feed(rec: DemoRecorder, tmp_path: Path, n: int = 6, action: int = 3) -> None:
    rec.start()
    for i in range(n):
        image = np.full((800, 480, 3), (40, 44, 60), dtype=np.uint8)
        rec.tick(Frame(frame_id=i + 1, ts=float(i) / 30.0, image=image),
                 death_state="ALIVE", confidence=0.1, score=float(i))
    rec._current_action = action


class TestRecorderMetadata:
    def test_episode_file_shape_and_meta(self, tmp_path) -> None:
        rec = make_recorder(tmp_path)
        feed(rec, tmp_path)
        path = rec.stop(done=True)
        assert path is not None and Path(path).exists()
        data = np.load(path, allow_pickle=False)
        assert data["frames"].dtype == np.uint8
        assert data["frames"].shape[1:] == (84, 84)  # preprocessed, not full-res
        assert data["actions"].shape == data["frames"].shape[:1]
        assert data["timestamps"].shape == data["actions"].shape
        assert data["done"].dtype == np.bool_
        assert bool(data["done"][-1]) and not bool(data["done"][:-1].any())
        meta = json.loads(str(data["meta"]))
        assert meta["fps_target"] == 30
        assert meta["region"]["dpi_scale"] == 1.25
        assert meta["region"]["screen_width"] == 1920
        assert meta["region"]["left"] == 120
        assert meta["profile"]
        assert "4" in meta["keymap"]

    def test_optional_columns_present(self, tmp_path) -> None:
        rec = make_recorder(tmp_path)
        feed(rec, tmp_path)
        path = rec.stop()
        data = np.load(path, allow_pickle=False)
        assert data["score"].shape == data["actions"].shape
        assert data["confidence"].shape == data["actions"].shape
        assert set(np.unique(data["death_state"].astype(str))) == {"ALIVE"}

    def test_zero_frames_saves_nothing(self, tmp_path) -> None:
        rec = make_recorder(tmp_path)
        rec.start()
        assert rec.stop() is None

    def test_actions_recorded_from_key_simulation(self, tmp_path) -> None:
        rec = make_recorder(tmp_path)
        rec.start()
        rec._handle_press("left")
        for i in range(4):
            image = np.full((800, 480, 3), (40, 44, 60), dtype=np.uint8)
            rec.tick(Frame(frame_id=i + 1, ts=float(i) / 30.0, image=image))
        rec._handle_release("left")
        path = rec.stop()
        actions = np.load(path)["actions"]
        assert (actions == 1).all()  # LEFT mapped from the "left" key press

    def test_atomic_write_leaves_no_tmp(self, tmp_path) -> None:
        rec = make_recorder(tmp_path)
        feed(rec, tmp_path)
        path = rec.stop()
        assert path is not None
        assert not list((tmp_path / "demos").glob("*.tmp"))
