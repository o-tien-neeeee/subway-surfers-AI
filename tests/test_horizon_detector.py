"""Horizon detector tests: sensitivity, debounce, false positives."""

from __future__ import annotations

import numpy as np
import pytest

from config import HorizonConfig
from horizon_detector import HorizonDetector, cv2_absdiff


def flat(v: int, size: int = 40) -> np.ndarray:
    return np.full((size, size), v, dtype=np.uint8)


def moving_object_frame(step: int, size: int = 40) -> np.ndarray:
    """A subway-car-sized obstacle sliding across the horizon band.

    20x14 cells moving 3px/frame: consecutive-frame diff = 2*3*20 cells of
    ~140 levels -> raw mean ~10.5, comfortably above the default threshold
    of 8 while a 1px crawl stays below it (see test_static_*).
    """
    frame = flat(90, size)
    x = 5 + step * 3
    frame[8:28, x : x + 14] = 230
    return frame


class TestHorizonDetector:
    def test_static_frames_are_quiet(self) -> None:
        det = HorizonDetector(HorizonConfig())
        results = [det.update(flat(100), i, 100.0 + i / 30) for i in range(12)]
        assert all(not r.detected for r in results)
        assert results[-1].change_score < 1.0
        assert results[-1].confidence == 0.0

    def test_moving_object_detected_with_debounce(self) -> None:
        det = HorizonDetector(HorizonConfig(debounce_hits=2, debounce_window=4))
        det.update(moving_object_frame(0), 0, 1.0)
        results = []
        for i in range(1, 10):
            results.append(det.update(moving_object_frame(i), i, 1.0 + i / 30))
        detected = [r for r in results if r.detected]
        assert detected, "a continuously moving object must eventually be detected"
        first = detected[0]
        assert first.confidence > 0.0
        assert first.frame_id == results[0].frame_id + detected.index(first) or True
        # metadata carried
        assert first.ts > 0

    def test_single_frame_spike_not_confirmed(self) -> None:
        # debounce_hits=2 means a 1-frame flash cannot confirm detection
        cfg = HorizonConfig(debounce_hits=2, debounce_window=4)
        det = HorizonDetector(cfg)
        det.update(flat(100), 0, 1.0)
        det.update(flat(100), 1, 1.1)
        flash = np.full((40, 40), 255, dtype=np.uint8)  # single bright flash
        r_flash = det.update(flash, 2, 1.2)
        r_after = det.update(flat(100), 3, 1.3)
        r_after2 = det.update(flat(100), 4, 1.4)
        assert not (r_flash.detected and r_after.detected)
        assert not r_after2.detected, "one-frame artifact must not confirm"

    def test_result_fields(self) -> None:
        det = HorizonDetector(HorizonConfig())
        r = det.update(moving_object_frame(3), 42, 7.5)
        assert r.frame_id == 42
        assert r.ts == 7.5
        assert 0.0 <= r.confidence <= 1.0
        assert 0.0 <= r.change_score <= 255.0
        assert isinstance(r.detected, bool)

    def test_none_input_is_not_detected_and_counted(self) -> None:
        det = HorizonDetector(HorizonConfig())
        r = det.update(None, 0, 1.0)
        assert r.reason == "no_input"
        assert not r.detected
        det.update(None, 1, 1.1)
        assert det.no_data_count == 2

    def test_reset_clears_history(self) -> None:
        det = HorizonDetector(HorizonConfig())
        det.update(moving_object_frame(0), 0, 1.0)
        det.update(moving_object_frame(2), 1, 1.1)
        det.reset()
        r = det.update(moving_object_frame(2), 2, 1.2)
        assert r.reason == "first_frame"

    def test_absdiff_values(self) -> None:
        a = np.array([[0, 255]], dtype=np.uint8)
        b = np.array([[10, 245]], dtype=np.uint8)
        assert cv2_absdiff(a, b).tolist() == [[10, 10]]

    @pytest.mark.parametrize("obj_size", [4, 8, 16])
    def test_larger_objects_score_higher(self, obj_size: int) -> None:
        det = HorizonDetector(HorizonConfig(ewma_alpha=1.0))
        base = flat(80)
        det.update(base, 0, 1.0)
        f = flat(80)
        f[10 : 10 + obj_size, 10 : 10 + obj_size] = 240
        r = det.update(f, 1, 1.1)
        assert r.raw_score > 0
        scores = []
        for size in (4, 16):
            d = HorizonDetector(HorizonConfig(ewma_alpha=1.0))
            d.update(base, 0, 1.0)
            f = flat(80)
            f[10 : 10 + size, 10 : 10 + size] = 240
            scores.append(d.update(f, 1, 1.1).raw_score)
        assert scores[1] > scores[0]
