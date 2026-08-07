"""Tests for the tank-observation encode/decode pair."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
)

from tankpit_bot.state.types import (
    decode_tank_observation,
    encode_tank_observation,
    make_tank_observation,
)


class TestTankObservationCodec:
    """``encode_tank_observation``/``decode_tank_observation`` round-trip."""

    def test_round_trip_full(self) -> None:
        """Every field round-trips through encode + decode."""
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(40, 50),
            team=2,
            rank=3,
            damage_state=1,
            direction=8,
            name="player",
            is_bot=True,
        )
        encoded = encode_tank_observation(obs)
        decoded = decode_tank_observation(encoded)
        assert decoded == obs

    def test_round_trip_all_nullable_none(self) -> None:
        """``None`` aspects survive the round-trip as ``None``."""
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
        )
        encoded = encode_tank_observation(obs)
        decoded = decode_tank_observation(encoded)
        assert decoded == obs

    def test_decode_rejects_missing_required_field(self) -> None:
        """Missing required field raises ``JSONTypeError``."""
        data: JSONObject = {"tank_id": 42}
        with pytest.raises(JSONTypeError):
            decode_tank_observation(data)

    def test_decode_rejects_malformed_position_wrong_length(self) -> None:
        """Position list with wrong length raises ``JSONTypeError``."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": [1, 2, 3],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="2-element list"):
            decode_tank_observation(data)

    def test_decode_rejects_malformed_position_non_int(self) -> None:
        """Position with non-int component raises ``JSONTypeError``."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": ["x", 2],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match=r"position\[0\] must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_malformed_position_y_non_int(self) -> None:
        """Position with non-int y raises ``JSONTypeError``."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": [1, "y"],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match=r"position\[1\] must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_malformed_position_not_list(self) -> None:
        """Position that is not a list raises ``JSONTypeError``."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": "10,20",
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="2-element list"):
            decode_tank_observation(data)

    def test_decode_rejects_optional_int_with_wrong_type(self) -> None:
        """Optional int fields reject non-int values."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": None,
            "team": "two",
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="team must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_optional_str_with_wrong_type(self) -> None:
        """Optional string fields reject non-string values."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": None,
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": 123,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="name must be str"):
            decode_tank_observation(data)

    def test_decode_rejects_optional_bool_with_wrong_type(self) -> None:
        """Optional bool fields reject non-bool values."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": None,
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": "yes",
        }
        with pytest.raises(JSONTypeError, match="is_bot must be bool"):
            decode_tank_observation(data)

    def test_decode_rejects_unknown_storage_source(self) -> None:
        """Unknown ``storage_source`` raises via shared validator."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "storage_source": "moon",
            "position": None,
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="storage_source must be"):
            decode_tank_observation(data)

    def test_decode_rejects_bool_in_optional_int_field(self) -> None:
        """Booleans must not be accepted by optional int validators.

        Python's ``bool`` is a subclass of ``int``, so the optional-int
        validator explicitly rejects ``bool`` to avoid silent
        misinterpretation.
        """
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": None,
            "team": True,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="team must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_bool_in_position_x_int_field(self) -> None:
        """Booleans must not pass the position-component int check.

        ``isinstance(True, int)`` is True; the validator must guard
        against that or callers could pass ``[True, False]`` as a
        position.
        """
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": [True, 5],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match=r"position\[0\] must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_bool_in_position_y_int_field(self) -> None:
        """Position y must reject booleans for the same reason as x."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": [5, True],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match=r"position\[1\] must be int"):
            decode_tank_observation(data)
