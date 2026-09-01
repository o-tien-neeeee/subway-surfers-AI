"""Tests for the recorder-side keypress-window filter.

The recorder and the dataset are two layers of the same rule ("only
keep frames around a human key press"); they MUST agree so a recording
made with the recorder's filter ON is still usable by the dataset
when its filter is also ON (and vice-versa).  These tests pin the
recorder-side contract.
"""

from __future__ import annotations

import json
import numpy as np

from config import BotConfig, NOOP, LEFT, RIGHT, JUMP
from demonstration_recorder import DemoRecorder
from ipc import Frame


def _make_recorder(tmp_path, cfg: BotConfig = None) -> DemoRecorder:
    cfg = cfg or BotConfig()
    cfg.region.left, cfg.region.top = 120, 80
    cfg.region.width, cfg.region.height = 480, 800
    cfg.region.screen_width, cfg.region.screen_height = 1920, 1080
    cfg.region.dpi_scale = 1.25
    return DemoRecorder(cfg, tmp_path / "demos", lambda: None)


def _feed(rec: DemoRecorder, n: int, action_at: dict[int, int] | None = None,
          hold_for: int = 0, start_id: int = 1) -> None:
    """Push n frames into a started recorder; optionally press keys.

    The recorder is started by the test; the press simulation has to
    go through ``_handle_press`` (NOT ``_current_action = X``) so the
    press-index bookkeeping actually fires.

    ``hold_for`` controls how many frames the key is held after the
    press (0 = a single-frame tap that gets cleared on the next tick;
    the test only really needs the single-tap semantics to verify the
    window rule).  Subway Surfers' dodges are usually 1-frame taps so
    the default is the realistic one.
    """
    action_at = action_at or {}
    key_to_action = {LEFT: "left", RIGHT: "right", JUMP: "up"}
    for i in range(n):
        image = np.full((800, 480, 3), (40, 44, 60), dtype=np.uint8)
        if i in action_at:
            key = key_to_action[action_at[i]]
            rec._handle_press(key)
        # Release any held key the configured number of frames after
        # its press; default 0 (single-frame tap) keeps the action
        # column honest for the window test.
        for press_idx, act in action_at.items():
            if i == press_idx + hold_for:
                key = key_to_action[act]
                rec._handle_release(key)
        rec.tick(Frame(frame_id=start_id + i, ts=float(i) / 30.0, image=image))


class TestKeypressWindowFilter:
    def test_keypress_window_off_keeps_everything(self, tmp_path) -> None:
        rec = _make_recorder(tmp_path)
        # The default cfg has keypress_window on; explicitly disable.
        rec.cfg.demo_augment.keypress_window = False
        rec.start()
        _feed(rec, 10)  # all NOOP
        path = rec.stop(done=True)
        data = np.load(path, allow_pickle=False)
        assert data["frames"].shape[0] == 10

    def test_keypress_window_on_with_no_press_saves_all(self, tmp_path) -> None:
        # An episode with zero recorded presses still has to land on
        # disk so the user can see "I forgot to play" — but the
        # downstream dataset can refuse it.
        rec = _make_recorder(tmp_path)
        rec.cfg.demo_augment.keypress_window = True
        rec.start()
        _feed(rec, 10)  # all NOOP, no presses
        path = rec.stop(done=True)
        assert path is not None
        data = np.load(path, allow_pickle=False)
        assert data["frames"].shape[0] == 10

    def test_keypress_window_drops_noop_stretches(self, tmp_path) -> None:
        rec = _make_recorder(tmp_path)
        rec.cfg.demo_augment.keypress_window = True
        rec.cfg.demo_augment.keypress_pre = 2
        rec.cfg.demo_augment.keypress_post = 2
        rec.start()
        # 20 frames, single LEFT at index 5 (held for 1 frame so the
        # action column reads exactly one LEFT row in the window).
        _feed(rec, 20, action_at={5: LEFT}, hold_for=1)
        path = rec.stop(done=True)
        data = np.load(path, allow_pickle=False)
        # pre=2, post=2 around idx 5 -> 5 frames (3, 4, 5, 6, 7).
        assert data["frames"].shape[0] == 5
        meta = json.loads(str(data["meta"]))
        assert meta["press_count"] == 1
        assert meta["keypress_window"] is True
        # The action column records the CURRENTLY-HELD key.  A
        # single-frame tap (press index 5, release at 6) yields
        # action=LEFT only on row 5; rows 3, 4, 6, 7 are NOOP.  This
        # is the *honest* mapping the BC loss needs — "I saw the
        # obstacle, I pressed LEFT" — and matches the dataset's
        # keypress_windows() rule.
        kept_actions = data["actions"].tolist()
        assert sum(1 for a in kept_actions if a == LEFT) == 1
        assert sum(1 for a in kept_actions if a == NOOP) == 4

    def test_two_presses_merge_windows(self, tmp_path) -> None:
        rec = _make_recorder(tmp_path)
        rec.cfg.demo_augment.keypress_window = True
        rec.cfg.demo_augment.keypress_pre = 1
        rec.cfg.demo_augment.keypress_post = 1
        rec.start()
        # Presses at indices 3 and 5 with pre=1, post=1 -> windows
        # 2..4 and 4..6 -> merged to 2..6 = 5 frames.  Each press is
        # released on the next frame so the action column has two
        # non-zero entries, not one continuous LEFT.
        _feed(rec, 10, action_at={3: LEFT, 5: RIGHT}, hold_for=1)
        path = rec.stop(done=True)
        data = np.load(path, allow_pickle=False)
        assert data["frames"].shape[0] == 5
        meta = json.loads(str(data["meta"]))
        assert meta["press_count"] == 2

    def test_press_at_episode_boundary_clamps_window(self, tmp_path) -> None:
        rec = _make_recorder(tmp_path)
        rec.cfg.demo_augment.keypress_window = True
        rec.cfg.demo_augment.keypress_pre = 5
        rec.cfg.demo_augment.keypress_post = 5
        rec.start()
        # Press at the very first frame: pre=5 clamps to 0, post=5
        # gives 0..5 (6 frames), then only 5 frames exist.  Hold
        # for one frame so the JUMP action is recorded.
        _feed(rec, 5, action_at={0: JUMP}, hold_for=1)
        path = rec.stop(done=True)
        data = np.load(path, allow_pickle=False)
        assert data["frames"].shape[0] == 5

    def test_press_count_visible_to_gui(self, tmp_path) -> None:
        rec = _make_recorder(tmp_path)
        rec.cfg.demo_augment.keypress_window = False  # don't filter, just count
        rec.start()
        assert rec.press_count == 0
        rec._handle_press("left")
        rec._handle_press("up")
        assert rec.press_count == 2
        rec._handle_press("right")
        rec._handle_press("down")
        assert rec.press_count == 4
        rec.stop(done=True)

    def test_meta_records_keypress_window_flag(self, tmp_path) -> None:
        rec = _make_recorder(tmp_path)
        rec.cfg.demo_augment.keypress_window = True
        rec.cfg.demo_augment.keypress_pre = 7
        rec.cfg.demo_augment.keypress_post = 3
        rec.start()
        _feed(rec, 10, action_at={4: LEFT}, hold_for=1)
        path = rec.stop(done=True)
        meta = json.loads(str(np.load(path, allow_pickle=False)["meta"]))
        assert meta["keypress_window"] is True
        assert meta["keypress_pre"] == 7
        assert meta["keypress_post"] == 3
        assert meta["press_count"] == 1
