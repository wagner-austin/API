"""Fleet telemetry: the cached stats/activity summaries behind the page."""

from __future__ import annotations

import pytest

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_telemetry import (
    TELEMETRY_CACHE_TTL_MS,
    FleetTelemetry,
)
from tests.service._artifact_fixtures import FakeArtifact

_EVENT_LINES = [
    '{"timestamp":"2026-08-06T20:00:00","level":"INFO","logger":"l",'
    '"mode":"bot","channel":"STATE","message":"INITIALIZING"}',
    '{"timestamp":"2026-08-06T20:00:01","level":"INFO","logger":"l",'
    '"mode":"bot","channel":"DIAGNOSTIC","message":"diagnostic_kind=tank_identity",'
    '"diagnostic_kind":"tank_identity","tank_id":601}',
    '{"timestamp":"2026-08-06T20:00:10","level":"INFO","logger":"l",'
    '"mode":"bot","channel":"WORLD","message":"Fuel: 1100 -> 1055 (-45)",'
    '"tick_n":133,"bot_state":"TELEPORT/CLOSE"}',
    '{"timestamp":"2026-08-06T20:00:11","level":"INFO","logger":"l",'
    '"mode":"bot","channel":"COMBAT","message":"engaging orange-5",'
    '"tick_n":134,"bot_state":"HUNT/ENGAGE"}',
    '{"timestamp":"2026-08-06T20:00:12","level":"INFO","logger":"l",'
    '"mode":"bot","channel":"DIAGNOSTIC","message":"diagnostic_kind=tank_deactivated",'
    '"diagnostic_kind":"tank_deactivated","victim_id":529,"killer_id":601,'
    '"tick_n":134,"bot_state":"HUNT/ENGAGE"}',
    '{"timestamp":"2026-08-06T20:00:12","level":"INFO","logger":"l",'
    '"mode":"bot","channel":"AI","message":"shoot orange-5 at (231,29)",'
    '"tick_n":134,"bot_state":"HUNT/ENGAGE"}',
]


def _ai_line(second: int) -> str:
    """Build one AI-channel event line.

    Args:
        second: Seconds field of the timestamp, also the decision id.

    Returns:
        One JSONL event line.
    """
    return (
        f'{{"timestamp":"2026-08-06T20:00:{second:02d}","level":"INFO","logger":"l",'
        f'"mode":"bot","channel":"AI","message":"decision {second}"}}'
    )


class _FixedClock:
    """get_current_time_ms fake with a settable now."""

    def __init__(self) -> None:
        self.now_ms = 1_000_000

    def __call__(self) -> int:
        return self.now_ms


@pytest.fixture()
def clock() -> _FixedClock:
    """Freeze the telemetry clock so the TTL is a test-controlled value.

    Returns:
        The settable clock, already installed. The autouse hook-restore
        fixture puts ``get_current_time_ms`` back.
    """
    fixed = _FixedClock()
    top_hooks.get_current_time_ms = fixed
    return fixed


def test_stats_reduces_the_digest_with_timeline_and_inventory(
    artifact: FakeArtifact,
) -> None:
    """Stats carry the digest numbers plus timeline and inventory."""
    artifact.start_run(_EVENT_LINES)

    summary = FleetTelemetry().stats("alpha")

    assert summary["available"] is True
    assert summary["kills"] == 1
    assert summary["deaths"] == 0
    assert summary["duration_s"] == 12
    assert summary["timeline_kills"] == [1]
    assert summary["inventory_last"] == []


def test_activity_reads_the_live_tail(artifact: FakeArtifact) -> None:
    """Activity carries state, tick, fuel, and the channel feed in order."""
    artifact.start_run(_EVENT_LINES)

    tail = FleetTelemetry().activity("alpha")

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


def test_summaries_are_cached_for_the_ttl_window(
    artifact: FakeArtifact,
    clock: _FixedClock,
) -> None:
    """Fast polling costs one artifact read per cache window, not per hit."""
    artifact.start_run(_EVENT_LINES)
    telemetry = FleetTelemetry()

    telemetry.stats("alpha")
    telemetry.stats("alpha")
    within_window = len(artifact.read_offsets)
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    telemetry.stats("alpha")
    after_expiry = len(artifact.read_offsets)

    assert within_window == 1
    assert after_expiry == 2


def test_a_growing_run_is_read_forward_not_re_read(
    artifact: FakeArtifact,
    clock: _FixedClock,
) -> None:
    """The second poll reads only the bytes the bot appended since the first.

    This is the whole point of the 2026-09-01 change: a live bot's
    artifact reaches megabytes, and the page polls it every second. A
    reader that starts at zero every time is what made reconnecting to
    a running fleet take forever.
    """
    artifact.start_run(_EVENT_LINES)
    telemetry = FleetTelemetry()

    first = telemetry.stats("alpha")
    consumed_by_first = artifact.bytes_served
    artifact.append([_ai_line(20)])
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    second = telemetry.stats("alpha")

    appended = artifact.bytes_served - consumed_by_first
    assert artifact.read_offsets == [0, consumed_by_first]
    assert appended == len(_ai_line(20)) + 1
    # The fold continued rather than restarting: the new line extends
    # the run's span instead of being counted as a fresh one.
    assert first["duration_s"] == 12
    assert second["duration_s"] == 20
    assert second["kills"] == 1


def test_both_summaries_share_one_cursor(
    artifact: FakeArtifact,
    clock: _FixedClock,
) -> None:
    """Stats then activity fold the same records exactly once."""
    artifact.start_run(_EVENT_LINES)
    telemetry = FleetTelemetry()

    stats = telemetry.stats("alpha")
    activity = telemetry.activity("alpha")

    # Two reads (one per summary) but only the first carries bytes:
    # the second finds the cursor already at the end.
    assert artifact.read_offsets[0] == 0
    assert artifact.read_offsets[1] == artifact.bytes_served
    assert stats["kills"] == 1
    assert activity["tick"] == 134


def test_a_new_run_under_the_same_path_restarts_the_fold(
    artifact: FakeArtifact,
    clock: _FixedClock,
) -> None:
    """A replaced artifact resets the digest instead of continuing it."""
    artifact.start_run(_EVENT_LINES)
    telemetry = FleetTelemetry()
    finished = telemetry.stats("alpha")

    artifact.start_run(
        [
            '{"timestamp":"2026-08-07T09:00:00","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"STATE","message":"INITIALIZING"}'
        ]
    )
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    restarted = telemetry.stats("alpha")

    assert finished["kills"] == 1
    assert restarted["kills"] == 0
    assert restarted["started_at"] == "2026-08-07T09:00:00"
    assert artifact.read_offsets[-1] == 0


def test_a_line_still_being_written_is_withheld_until_complete(
    artifact: FakeArtifact,
    clock: _FixedClock,
) -> None:
    """A poll landing mid-append decodes nothing from the partial line."""
    artifact.start_run(_EVENT_LINES)
    telemetry = FleetTelemetry()
    telemetry.stats("alpha")

    half = _ai_line(20)
    artifact.append_partial(half[: len(half) // 2])
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    mid_write = telemetry.stats("alpha")

    artifact.append_partial(half[len(half) // 2 :] + "\n")
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    completed = telemetry.stats("alpha")

    assert mid_write["duration_s"] == 12
    assert completed["duration_s"] == 20


def test_missing_or_empty_events_are_unavailable(artifact: FakeArtifact) -> None:
    """No file and no lines both answer available=False, cached too."""
    telemetry = FleetTelemetry()

    assert telemetry.stats("alpha") == {"available": False}
    assert telemetry.activity("alpha") == {"available": False}

    artifact.start_run([])
    assert FleetTelemetry().activity("alpha") == {"available": False}


def test_a_malformed_record_keeps_failing_instead_of_folding_a_hole(
    artifact: FakeArtifact,
    clock: _FixedClock,
) -> None:
    """A record the fold rejects spoils the run rather than being skipped.

    The cursor has already passed the bad line by the time the fold
    fails, so a stream that carried on would serve a digest missing
    those records and say nothing about it.
    """
    artifact.start_run(_EVENT_LINES)
    telemetry = FleetTelemetry()
    assert telemetry.stats("alpha")["available"] is True

    artifact.append(
        [
            '{"timestamp":"2026-08-06T20:00:20","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"DIAGNOSTIC","message":"bad sample",'
            '"diagnostic_kind":"inventory_sample","armor":"many","dual":0,'
            '"missile":0,"homing":0,"radar":0}'
        ]
    )
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    spoiled = telemetry.stats("alpha")

    artifact.append([_ai_line(30)])
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    still_spoiled = telemetry.stats("alpha")

    assert spoiled == {"available": False}
    assert still_spoiled == {"available": False}


def test_forget_drops_the_cursor_and_the_cached_summaries(
    artifact: FakeArtifact,
    clock: _FixedClock,
) -> None:
    """A removed instance stops costing the manager a cursor."""
    artifact.start_run(_EVENT_LINES)
    telemetry = FleetTelemetry()
    telemetry.stats("alpha")
    telemetry.activity("alpha")

    telemetry.forget("alpha")
    reads_before = len(artifact.read_offsets)
    again = telemetry.stats("alpha")

    # A fresh cursor: the re-read starts at zero, and it is a read
    # rather than a cache hit even inside the TTL window.
    assert len(artifact.read_offsets) == reads_before + 1
    assert artifact.read_offsets[-1] == 0
    assert again["kills"] == 1


def test_activity_caps_the_feed_and_skips_other_channels(
    artifact: FakeArtifact,
) -> None:
    """Activity skips non-feed channels and keeps the newest six lines."""
    artifact.start_run(
        [_ai_line(second) for second in range(1, 9)]
        # Newest line is a non-feed channel: the feed must skip it
        # instead of surfacing raw protocol noise.
        + [
            '{"timestamp":"2026-08-06T20:00:09","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"PROTO","message":"raw frame"}'
        ]
    )

    tail = FleetTelemetry().activity("alpha")

    feed = tail["feed"]
    assert isinstance(feed, list) and len(feed) == 6
    newest = feed[-1]
    assert isinstance(newest, dict) and newest["message"] == "decision 8"
    oldest = feed[0]
    assert isinstance(oldest, dict) and oldest["message"] == "decision 3"


def test_activity_is_cached_within_the_window(
    artifact: FakeArtifact,
    clock: _FixedClock,
) -> None:
    """A second activity poll inside the window is a cache hit."""
    artifact.start_run([_ai_line(1)])
    telemetry = FleetTelemetry()

    first = telemetry.activity("alpha")
    reads_after_first = len(artifact.read_offsets)
    second = telemetry.activity("alpha")

    assert second == first
    assert len(artifact.read_offsets) == reads_after_first


def test_activity_without_fuel_state_or_tick_reads_sentinels(
    artifact: FakeArtifact,
) -> None:
    """Streams that never state fuel/state/tick answer the sentinels."""
    artifact.start_run(
        [
            '{"timestamp":"2026-08-06T20:00:00","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"AI","message":"no telemetry fields"}'
        ]
    )

    tail = FleetTelemetry().activity("alpha")

    assert tail["available"] is True
    assert tail["fuel"] == -1
    assert tail["state"] == ""
    assert tail["tick"] == -1


def test_a_fuel_line_without_a_number_reads_the_sentinel(
    artifact: FakeArtifact,
) -> None:
    """A Fuel: line whose total is not a plain number answers -1."""
    artifact.start_run(
        [
            '{"timestamp":"2026-08-06T20:00:00","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"WORLD","message":"Fuel: 1100 -> unknown"}'
        ]
    )

    assert FleetTelemetry().activity("alpha")["fuel"] == -1


def test_manager_activity_requires_a_registered_instance() -> None:
    """The manager gate: ghosts 404 before any file is touched."""
    manager = FleetManager()
    with pytest.raises(FleetError, match="unknown instance"):
        manager.activity("ghost")


def test_spawn_derives_the_instance_from_the_account() -> None:
    """No name given: the account (or the default) names the instance."""
    from tankpit_bot.service import _test_hooks as service_hooks
    from tankpit_bot.service.fleet_config import derive_instance
    from tests.service._fleet_fixtures import (
        _FakeSpawner,
        _restore_account_hooks,
        _with_configured_accounts,
    )

    fake = _FakeSpawner()
    original_spawn = service_hooks.spawn_bot_process
    service_hooks.spawn_bot_process = fake
    originals = _with_configured_accounts()
    try:
        manager = FleetManager()
        named = manager.spawn(instance="", account="second", kills=0, seconds=0)
        default = manager.spawn(instance="", account="", kills=0, seconds=0)
        assert derive_instance("We!rd Name") == "we-rd-name"
    finally:
        _restore_account_hooks(originals)
        service_hooks.spawn_bot_process = original_spawn

    assert named["instance"] == "second"
    assert default["instance"] == "artax"
