"""Tests for browser helper functions."""

from __future__ import annotations

from tankpit_bot.browser import (
    CDPClock,
    get_current_time_ms,
)


def test_get_current_time_ms_returns_int() -> None:
    """Test get_current_time_ms returns an integer."""
    result = get_current_time_ms()
    assert type(result) is int
    assert result > 0


def test_cdp_clock_anchors_to_unix_time() -> None:
    """A fresh clock anchors its first reading to the wall clock."""
    current_unix_ms = get_current_time_ms()
    cdp_seconds = 12345.678
    result = CDPClock().to_unix_ms(cdp_seconds)
    # Result should be approximately current Unix time
    # (within 100ms to account for test execution time)
    expected_offset = current_unix_ms - int(cdp_seconds * 1000)
    expected = int(cdp_seconds * 1000) + expected_offset
    assert abs(result - expected) < 100


def test_cdp_clock_offset_persists_across_readings() -> None:
    """The anchor is set once, so later readings keep their spacing."""
    clock = CDPClock()
    # First call establishes the offset
    result1 = clock.to_unix_ms(100.0)
    # Second call uses same offset, so difference should be exactly 1000ms
    result2 = clock.to_unix_ms(101.0)
    assert result2 - result1 == 1000


def test_each_cdp_clock_anchors_independently() -> None:
    """Two clocks anchor separately -- the offset is per session.

    A second browser session's CDP origin differs from the first's, so
    sharing one anchor would misdate every frame it reads
    ([[session-state-deglobalisation]] step 4).
    """
    first = CDPClock()
    first.to_unix_ms(100.0)

    second = CDPClock()

    # The second clock anchors on ITS first reading, so a far-future CDP
    # timestamp still maps to roughly now rather than to now + 900 s.
    assert abs(second.to_unix_ms(1000.0) - get_current_time_ms()) < 100
