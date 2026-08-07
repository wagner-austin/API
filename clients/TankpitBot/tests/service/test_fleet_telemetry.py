"""Fleet telemetry: the cached stats/activity summaries behind the page."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service.fleet import FleetError, FleetManager
from tankpit_bot.service.fleet_telemetry import (
    TELEMETRY_CACHE_TTL_MS,
    FleetTelemetry,
)

_EVENT_LINES = "\n".join(
    [
        '{"timestamp":"2026-08-06T20:00:00","level":"INFO","logger":"l",'
        '"mode":"bot","channel":"STATE","message":"INITIALIZING"}',
        '{"timestamp":"2026-08-06T20:00:10","level":"INFO","logger":"l",'
        '"mode":"bot","channel":"WORLD","message":"Fuel: 1100 -> 1055 (-45)",'
        '"tick_n":133,"bot_state":"TELEPORT/CLOSE"}',
        '{"timestamp":"2026-08-06T20:00:12","level":"INFO","logger":"l",'
        '"mode":"bot","channel":"COMBAT","message":"kill registered",'
        '"tick_n":134,"bot_state":"HUNT/ENGAGE"}',
        '{"timestamp":"2026-08-06T20:00:12","level":"INFO","logger":"l",'
        '"mode":"bot","channel":"AI","message":"shoot orange-5 at (231,29)",'
        '"tick_n":134,"bot_state":"HUNT/ENGAGE"}',
    ]
)


class _CountingReader:
    """read_text fake that counts invocations."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def __call__(self, path: Path) -> str:
        _ = path
        self.calls += 1
        return self.text


class _FixedClock:
    """get_current_time_ms fake with a settable now."""

    def __init__(self) -> None:
        self.now_ms = 1_000_000

    def __call__(self) -> int:
        return self.now_ms


def test_stats_reduces_the_digest_with_timeline_and_inventory() -> None:
    """Stats carry the digest numbers plus timeline and inventory."""
    reader = _CountingReader(_EVENT_LINES)
    original_read = top_hooks.read_text
    top_hooks.read_text = reader
    try:
        summary = FleetTelemetry().stats("alpha")
    finally:
        top_hooks.read_text = original_read

    assert summary["available"] is True
    assert summary["kills"] == 1
    assert summary["deaths"] == 0
    assert summary["duration_s"] == 12
    assert summary["timeline_kills"] == [1]
    assert summary["inventory_last"] == []


def test_activity_reads_the_live_tail() -> None:
    """Activity carries state, tick, fuel, and the channel feed in order."""
    reader = _CountingReader(_EVENT_LINES)
    original_read = top_hooks.read_text
    top_hooks.read_text = reader
    try:
        tail = FleetTelemetry().activity("alpha")
    finally:
        top_hooks.read_text = original_read

    assert tail["available"] is True
    assert tail["state"] == "HUNT/ENGAGE"
    assert tail["tick"] == 134
    assert tail["fuel"] == 1055
    feed = tail["feed"]
    assert isinstance(feed, list) and len(feed) == 4
    first = feed[0]
    last = feed[-1]
    assert isinstance(first, dict) and first["channel"] == "STATE"
    assert isinstance(last, dict) and last["message"] == "shoot orange-5 at (231,29)"
    assert last["time"] == "20:00:12"


def test_summaries_are_cached_for_the_ttl_window() -> None:
    """Fast polling costs one events parse per cache window, not per hit."""
    reader = _CountingReader(_EVENT_LINES)
    clock = _FixedClock()
    original_read = top_hooks.read_text
    original_time = top_hooks.get_current_time_ms
    top_hooks.read_text = reader
    top_hooks.get_current_time_ms = clock
    try:
        telemetry = FleetTelemetry()
        telemetry.stats("alpha")
        telemetry.stats("alpha")
        within_window = reader.calls
        clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
        telemetry.stats("alpha")
        after_expiry = reader.calls
    finally:
        top_hooks.read_text = original_read
        top_hooks.get_current_time_ms = original_time

    assert within_window == 1
    assert after_expiry == 2


def test_missing_or_empty_events_are_unavailable() -> None:
    """No file and no lines both answer available=False, cached too."""

    def raise_missing(path: Path) -> str:
        raise OSError(f"no events at {path}")

    original_read = top_hooks.read_text
    top_hooks.read_text = raise_missing
    try:
        telemetry = FleetTelemetry()
        stats = telemetry.stats("alpha")
        activity = telemetry.activity("alpha")
    finally:
        top_hooks.read_text = original_read
    assert stats == {"available": False}
    assert activity == {"available": False}

    empty_reader = _CountingReader("")
    top_hooks.read_text = empty_reader
    try:
        empty_activity = FleetTelemetry().activity("alpha")
    finally:
        top_hooks.read_text = original_read
    assert empty_activity == {"available": False}


def test_manager_activity_requires_a_registered_instance() -> None:
    """The manager gate: ghosts 404 before any file is touched."""
    manager = FleetManager()
    with pytest.raises(FleetError, match="unknown instance"):
        manager.activity("ghost")
