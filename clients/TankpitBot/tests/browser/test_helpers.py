"""Tests for browser helper functions."""

from __future__ import annotations

from tankpit_bot.browser import (
    cdp_timestamp_to_ms,
    get_current_time_ms,
    reset_cdp_time_offset,
)


def test_get_current_time_ms_returns_int() -> None:
    """Test get_current_time_ms returns an integer."""
    result = get_current_time_ms()
    assert type(result) is int
    assert result > 0


def test_cdp_timestamp_to_ms() -> None:
    """Test cdp_timestamp_to_ms converts CDP time to Unix time."""
    reset_cdp_time_offset()
    current_unix_ms = get_current_time_ms()
    cdp_seconds = 12345.678
    result = cdp_timestamp_to_ms(cdp_seconds)
    # Result should be approximately current Unix time
    # (within 100ms to account for test execution time)
    expected_offset = current_unix_ms - int(cdp_seconds * 1000)
    expected = int(cdp_seconds * 1000) + expected_offset
    assert abs(result - expected) < 100


def test_cdp_timestamp_offset_persists() -> None:
    """Test CDP time offset is calculated once and reused."""
    reset_cdp_time_offset()
    # First call establishes the offset
    result1 = cdp_timestamp_to_ms(100.0)
    # Second call uses same offset, so difference should be exactly 1000ms
    result2 = cdp_timestamp_to_ms(101.0)
    assert result2 - result1 == 1000
