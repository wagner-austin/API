"""Tests for the smoke script's happy path and setup.

``test_smoke_script.py`` was 776 lines; the failure-outcome tests are
now a sibling over the shared record builders.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
)

from scripts import (
    smoke,
)
from tests._smoke_records import (
    _login_records,
    _map_data_processed_record,
    _smoke_record,
)


class TestDecodeSmokeRecord:
    """Tests for ``_decode_smoke_record`` (record decoder boundary)."""

    def test_returns_record_with_fields_minus_reserved_keys(self) -> None:
        """Reserved record-level keys are stripped from the fields view."""
        rec = _smoke_record(
            1,
            "AI",
            "HUNT score=0.5",
            combat_target_x=131,
            combat_target_y=124,
        )
        assert rec.fields == {"combat_target_x": 131, "combat_target_y": 124}
        assert "channel" not in rec.fields
        assert "message" not in rec.fields
        assert rec.channel == "AI"
        assert rec.line_no == 1

    def test_raises_jsontypeerror_on_missing_channel(self) -> None:
        """JSONTypeError propagates unchanged when ``channel`` is absent."""
        parsed: JSONObject = {
            "timestamp": "2026-06-20T15:00:00",
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "message": "noop",
        }
        with pytest.raises(JSONTypeError):
            smoke._decode_smoke_record(line_no=1, raw="{}", parsed=parsed)

    def test_raises_jsontypeerror_on_missing_message(self) -> None:
        """JSONTypeError propagates when ``message`` is absent."""
        parsed: JSONObject = {
            "timestamp": "2026-06-20T15:00:00",
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "channel": "STATE",
        }
        with pytest.raises(JSONTypeError):
            smoke._decode_smoke_record(line_no=2, raw="{}", parsed=parsed)

    def test_raises_jsontypeerror_on_missing_timestamp(self) -> None:
        """JSONTypeError propagates when ``timestamp`` is absent."""
        parsed: JSONObject = {
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "channel": "STATE",
            "message": "noop",
        }
        with pytest.raises(JSONTypeError):
            smoke._decode_smoke_record(line_no=3, raw="{}", parsed=parsed)


class TestParseIsoTimestampSeconds:
    """Tests for the ISO-ish timestamp parser."""

    def test_returns_seconds_for_full_timestamp(self) -> None:
        """ISO timestamps parse to total seconds-of-day."""
        assert smoke.parse_iso_timestamp_seconds("2026-06-20T01:02:03") == pytest.approx(3723.0)

    def test_accepts_fractional_seconds(self) -> None:
        """Fractional seconds carry through to the float result."""
        assert smoke.parse_iso_timestamp_seconds("2026-06-20T00:00:01.5") == pytest.approx(1.5)

    def test_raises_without_t_separator(self) -> None:
        """Timestamps without 'T' are rejected at parse time."""
        with pytest.raises(ValueError, match="T"):
            smoke.parse_iso_timestamp_seconds("2026-06-20 01:02:03")

    def test_raises_without_seconds_component(self) -> None:
        """Timestamps with only HH:MM are rejected (need seconds)."""
        with pytest.raises(ValueError, match="seconds"):
            smoke.parse_iso_timestamp_seconds("2026-06-20T01:02")


class TestAssertLoginCompleted:
    """Tests for assertion 1 (login ladder)."""

    def test_passes_on_full_ladder(self) -> None:
        """The full ladder produces ``None``."""
        assert smoke.assert_login_completed(_login_records()) is None

    def test_passes_when_extra_state_events_appear_before(self) -> None:
        """Extra STATE events before the ladder do not break the check."""
        records = [
            _smoke_record(0, "STATE", "BOOTING"),
            *_login_records(),
        ]
        assert smoke.assert_login_completed(records) is None

    def test_fails_when_first_transition_missing(self) -> None:
        """Missing the first transition fails with a clear message."""
        records = [
            _smoke_record(1, "STATE", "INITIALIZING"),
            _smoke_record(2, "STATE", "WAITING_FOR_POSITION -> IDLE"),
        ]
        failure = smoke.assert_login_completed(records)
        if failure is None:
            raise AssertionError("missing first transition must fail the gate")
        assert "login ladder" in failure["message"]
        assert failure["pivot"] == 1

    def test_fails_when_second_transition_missing(self) -> None:
        """Missing the second transition fails with a clear message."""
        records = [_smoke_record(1, "STATE", "INITIALIZING -> WAITING_FOR_POSITION")]
        failure = smoke.assert_login_completed(records)
        if failure is None:
            raise AssertionError("missing second transition must fail the gate")
        assert "login ladder" in failure["message"]
        assert failure["pivot"] == 0

    def test_fails_when_no_state_events(self) -> None:
        """No STATE events at all fails the check with pivot=0."""
        records = [_smoke_record(1, "AI", "HUNT score=0", combat_target_x=0, combat_target_y=0)]
        failure = smoke.assert_login_completed(records)
        if failure is None:
            raise AssertionError("no STATE events must fail the login ladder gate")
        assert failure["pivot"] == 0


class TestAssertMapOpenClearedViaMapData:
    """Tests for assertion 2 (map_open cleared via map_data_processed)."""

    def test_passes_when_one_map_open_clears_via_map_data(self) -> None:
        """One matching WIRE_COMPLETE event satisfies the gate."""
        records = [_map_data_processed_record(1, "2026-06-20T15:00:05")]
        assert smoke.assert_map_open_cleared_via_map_data(records) is None

    def test_fails_when_no_map_open_events_at_all(self) -> None:
        """No map_open WIRE_COMPLETE events fails the gate."""
        failure = smoke.assert_map_open_cleared_via_map_data([])
        if failure is None:
            raise AssertionError("empty records must fail the map_open gate")
        assert "map_open" in failure["message"]

    def test_fails_when_map_open_clears_via_stall_timeout(self) -> None:
        """A map_open that cleared via stall_timeout fails the gate.

        Regression: this is exactly the failure mode the 2026-06-20 fix
        cured -- the dispatcher wasn't marking map_data_processed, so
        every map_open cleared via the 10s stall_timeout instead.
        """
        records = [
            _smoke_record(
                1,
                "WIRE_COMPLETE",
                "map_open completed in 10000ms via stall_timeout",
                action_kind="map_open",
                duration_ms=10000,
                signal="stall_timeout",
            )
        ]
        failure = smoke.assert_map_open_cleared_via_map_data(records)
        if failure is None:
            raise AssertionError("stall-cleared map_open must fail the gate")
        assert "map_data_processed" in failure["message"]

    def test_ignores_wire_complete_for_other_action_kinds(self) -> None:
        """Non-map_open WIRE_COMPLETE events do not satisfy the gate."""
        records = [
            _smoke_record(
                1,
                "WIRE_COMPLETE",
                "teleport completed in 250ms via teleport_landed",
                action_kind="teleport",
                duration_ms=250,
                signal="teleport_landed",
            )
        ]
        failure = smoke.assert_map_open_cleared_via_map_data(records)
        if failure is None:
            raise AssertionError("teleport-only completion must fail the gate")


class TestAssertHuntScoredTarget:
    """Tests for assertion 3 (HUNT scored a non-zero target)."""

    def test_passes_when_combat_target_x_is_non_zero(self) -> None:
        """Any HUNT score event with target_x != 0 satisfies the gate."""
        records = [
            _smoke_record(
                1,
                "AI",
                "HUNT score=0.8 target=(131,124)",
                combat_target_x=131,
                combat_target_y=124,
            )
        ]
        assert smoke.assert_hunt_scored_target(records) is None

    def test_passes_when_only_combat_target_y_is_non_zero(self) -> None:
        """target_y alone is enough -- guards against axis-bias misses."""
        records = [
            _smoke_record(
                1,
                "AI",
                "HUNT score=0.4",
                combat_target_x=0,
                combat_target_y=124,
            )
        ]
        assert smoke.assert_hunt_scored_target(records) is None

    def test_fails_when_no_hunt_events(self) -> None:
        """No HUNT events at all fails the gate (pivot=0)."""
        records = [_smoke_record(1, "AI", "decided IDLE")]
        failure = smoke.assert_hunt_scored_target(records)
        if failure is None:
            raise AssertionError("no HUNT events must fail the gate")
        assert failure["pivot"] == 0

    def test_fails_when_every_hunt_score_has_zero_target(self) -> None:
        """Every HUNT event with both coords 0 fails the gate."""
        records = [
            _smoke_record(
                1,
                "AI",
                "HUNT score=0",
                combat_target_x=0,
                combat_target_y=0,
            ),
            _smoke_record(
                2,
                "AI",
                "HUNT score=0",
                combat_target_x=0,
                combat_target_y=0,
            ),
        ]
        failure = smoke.assert_hunt_scored_target(records)
        if failure is None:
            raise AssertionError("all-zero HUNT events must fail the gate")
        assert failure["pivot"] == 1
