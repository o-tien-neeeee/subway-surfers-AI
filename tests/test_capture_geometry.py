"""Screen-geometry drift detection in the capture worker (requirement §6).

A moved/resized window, a resolution change or a DPI change silently shifts
everything the calibrated coordinates point at.  The capture worker therefore
re-checks the virtual screen against the calibrated reference every few
seconds and warns exactly once per change.  The pure decision function is
tested here without a display; the polling loop reuses it.
"""

from __future__ import annotations

from dataclasses import dataclass

from capture_worker import geometry_matches


@dataclass
class FakeRegion:
    screen_width: int = 0
    screen_height: int = 0


class TestGeometryMatches:
    def test_matching_geometry_true(self) -> None:
        region = FakeRegion(screen_width=1920, screen_height=1080)
        assert geometry_matches(region, 1920, 1080) is True

    def test_resolution_change_detected(self) -> None:
        region = FakeRegion(screen_width=1920, screen_height=1080)
        assert geometry_matches(region, 2560, 1440) is False

    def test_dpi_change_detected(self) -> None:
        # a DPI change typically shows up as a smaller effective desktop
        region = FakeRegion(screen_width=1920, screen_height=1080)
        assert geometry_matches(region, 1536, 864) is False

    def test_no_reference_is_unknown(self) -> None:
        assert geometry_matches(FakeRegion(0, 0), 1920, 1080) is None

    def test_no_display_is_unknown(self) -> None:
        region = FakeRegion(screen_width=1920, screen_height=1080)
        assert geometry_matches(region, None, None) is None

    def test_partial_dimension_change_detected(self) -> None:
        region = FakeRegion(screen_width=1920, screen_height=1080)
        assert geometry_matches(region, 1920, 900) is False

    def test_helper_never_raises_on_odd_types(self) -> None:
        region = FakeRegion(screen_width=1920, screen_height=1080)
        assert geometry_matches(region, 0, 0) is False
