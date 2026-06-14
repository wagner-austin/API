"""Coverage tests for diagnostics/account_stats.py: stats_marker_present."""

from __future__ import annotations

from tankpit_bot.diagnostics.account_stats import stats_marker_present


def test_stats_marker_present_with_marker() -> None:
    """stats_marker_present returns True when Statistics: header is present."""
    page_text = " Rank: private (160)\n Statistics:\n Play time: 41:25:23\n"
    assert stats_marker_present(page_text) is True


def test_stats_marker_present_without_marker() -> None:
    """stats_marker_present returns False when header is absent."""
    page_text = "LOCATION: 131,126\nName: Artax\n"
    assert stats_marker_present(page_text) is False


def test_stats_marker_present_empty_text() -> None:
    """stats_marker_present returns False for empty string."""
    assert stats_marker_present("") is False
