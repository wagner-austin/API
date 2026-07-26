"""Tests for match state event schemas.

The codec is the boundary between Kafka and the domain, so what it accepts
is what the rest of the pipeline is forced to handle. These tests pin both
sides: a well-formed snapshot round-trips exactly, and anything short of
that is rejected here rather than deeper in feature extraction.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str

from covenant_radar_api.domains.esports.schemas import (
    MatchEventV1,
    decode_match_event,
    encode_match_event,
    make_match_event,
)
from tests.domains.esports._test_esports_fixtures import make_snapshot

_VALID_FIELDS: MatchEventV1 = {
    "type": "esports.match_state.v1",
    "event_id": "evt-200",
    "match_id": "match-7",
    "game_number": 2,
    "game_time_seconds": 1500,
    "blue_kills": 11,
    "red_kills": 4,
    "blue_gold": 45000,
    "red_gold": 38000,
    "blue_towers": 5,
    "red_towers": 1,
    "blue_dragons": 3,
    "red_dragons": 1,
    "blue_barons": 1,
    "red_barons": 0,
    "timestamp": "2026-07-25T18:25:00Z",
}


class TestMakeMatchEvent:
    """Tests for the make_match_event factory."""

    def test_sets_the_discriminator(self) -> None:
        """The factory stamps the type, so no caller can get it wrong."""
        event = make_snapshot()

        assert event["type"] == "esports.match_state.v1"

    def test_carries_every_field_given(self) -> None:
        """Each argument reaches the field of the same name.

        Fifteen fields assigned positionally in one dict literal is exactly
        where a transposition hides, so every one is checked.
        """
        event = make_match_event(
            event_id="evt-001",
            match_id="match-7",
            game_number=2,
            game_time_seconds=1500,
            blue_kills=11,
            red_kills=4,
            blue_gold=45000,
            red_gold=38000,
            blue_towers=5,
            red_towers=1,
            blue_dragons=3,
            red_dragons=1,
            blue_barons=1,
            red_barons=0,
            timestamp="2026-07-25T18:25:00Z",
        )

        assert event["event_id"] == "evt-001"
        assert event["match_id"] == "match-7"
        assert event["game_number"] == 2
        assert event["game_time_seconds"] == 1500
        assert event["blue_kills"] == 11
        assert event["red_kills"] == 4
        assert event["blue_gold"] == 45000
        assert event["red_gold"] == 38000
        assert event["blue_towers"] == 5
        assert event["red_towers"] == 1
        assert event["blue_dragons"] == 3
        assert event["red_dragons"] == 1
        assert event["blue_barons"] == 1
        assert event["red_barons"] == 0
        assert event["timestamp"] == "2026-07-25T18:25:00Z"

    def test_opening_snapshot_is_all_zeros(self) -> None:
        """The state at the first tick of a game is representable."""
        event = make_snapshot(game_time_seconds=0)

        assert event["game_time_seconds"] == 0
        assert event["blue_kills"] == 0
        assert event["red_gold"] == 0


class TestEncodeMatchEvent:
    """Tests for encode_match_event."""

    def test_round_trip_preserves_every_field(self) -> None:
        """Encode then decode returns the same values.

        Gold is carried as whole units precisely so this holds exactly; a
        float field would make the round trip approximate.
        """
        original = make_snapshot(
            event_id="evt-100",
            match_id="match-3",
            game_time_seconds=1234,
            blue_kills=9,
            red_kills=9,
            blue_gold=40001,
            red_gold=39999,
            blue_towers=2,
            red_towers=3,
            blue_dragons=1,
            red_dragons=2,
            blue_barons=1,
            red_barons=1,
        )

        decoded = decode_match_event(encode_match_event(original))

        assert decoded == original

    def test_produces_json_naming_the_type_and_match(self) -> None:
        """The encoded payload identifies what it is and what it describes."""
        result = encode_match_event(make_snapshot(event_id="evt-101", match_id="match-9"))

        assert "esports.match_state.v1" in result
        assert "evt-101" in result
        assert "match-9" in result


class TestDecodeMatchEvent:
    """Tests for decode_match_event."""

    def test_decodes_a_valid_payload(self) -> None:
        """A well-formed payload decodes to the values it carried."""
        event = decode_match_event(dump_json_str(_VALID_FIELDS))

        assert event == _VALID_FIELDS

    def test_wrong_type_is_rejected(self) -> None:
        """A payload from another domain fails at the discriminator.

        Weather observations and match snapshots share no field names, so
        without this check the failure would surface as a missing-key error
        naming an unrelated field.
        """
        payload = dump_json_str({**_VALID_FIELDS, "type": "weather.observation.v1"})

        with pytest.raises(JSONTypeError, match=r"Expected 'esports\.match_state\.v1'"):
            decode_match_event(payload)

    @pytest.mark.parametrize(
        "missing",
        [
            "event_id",
            "match_id",
            "game_number",
            "game_time_seconds",
            "blue_kills",
            "red_kills",
            "blue_gold",
            "red_gold",
            "blue_towers",
            "red_towers",
            "blue_dragons",
            "red_dragons",
            "blue_barons",
            "red_barons",
            "timestamp",
        ],
    )
    def test_missing_field_is_rejected(self, missing: str) -> None:
        """Every field is required; none may be absent.

        A missing count would otherwise have to be defaulted somewhere, and
        a defaulted zero is indistinguishable from a real zero scoreline.

        Args:
            missing: Name of the field omitted from the payload.
        """
        fields = {key: value for key, value in _VALID_FIELDS.items() if key != missing}

        with pytest.raises(JSONTypeError, match=missing):
            decode_match_event(dump_json_str(fields))

    def test_gold_as_a_string_is_rejected(self) -> None:
        """A numeric field carrying text fails rather than being coerced.

        Coercion here would put a string into arithmetic downstream, where
        the error would name a numpy operation instead of the payload.
        """
        payload = dump_json_str({**_VALID_FIELDS, "blue_gold": "45000"})

        with pytest.raises(JSONTypeError, match="blue_gold"):
            decode_match_event(payload)

    def test_match_id_as_a_number_is_rejected(self) -> None:
        """The partition key must be a string, since Kafka keys on it."""
        payload = dump_json_str({**_VALID_FIELDS, "match_id": 7})

        with pytest.raises(JSONTypeError, match="match_id"):
            decode_match_event(payload)

    def test_invalid_json_is_rejected(self) -> None:
        """A payload that is not JSON fails as invalid JSON."""
        with pytest.raises(InvalidJsonError):
            decode_match_event("not valid json")

    def test_non_object_json_is_rejected(self) -> None:
        """Valid JSON that is not an object cannot carry an event."""
        with pytest.raises(JSONTypeError):
            decode_match_event('"just a string"')
