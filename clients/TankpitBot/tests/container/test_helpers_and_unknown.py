"""Tests for miscellaneous container decoder functionality.

Tests for DecodeLevel, MESSAGE_TYPE_LEVELS, ContainerDecodeError, and unknown container.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    MESSAGE_TYPE_LEVELS,
    ContainerDecodeError,
    ContainerMessageType,
    DecodeLevel,
    UnknownContainerDict,
    decode_unknown_container,
    get_decode_level,
)
from tests.container.test_data import (
    UNKNOWN_8_BYTES,
)


class TestContainerDecodeError:
    """Tests for ContainerDecodeError exception."""

    def test_error_can_be_raised_and_caught(self) -> None:
        """ContainerDecodeError can be raised and caught as Exception."""
        with pytest.raises(Exception) as exc:
            raise ContainerDecodeError("test message")
        assert str(exc.value) == "test message"

    def test_error_message_stored(self) -> None:
        """Error message is stored in message attribute."""
        error = ContainerDecodeError("test message")
        assert error.message == "test message"

    def test_error_str_representation(self) -> None:
        """Error has proper string representation."""
        error = ContainerDecodeError("test message")
        assert str(error) == "test message"


class TestDecodeUnknownContainer:
    """Tests for unknown container decoding."""

    def test_preserves_data(self) -> None:
        """Preserves data for unknown structures."""
        result = decode_unknown_container(UNKNOWN_8_BYTES)
        assert result["msg_type"] == "unknown_container"
        assert result["subtype"] == 0x7E
        assert result["length"] == 8
        assert result["data"] == UNKNOWN_8_BYTES

    def test_raises_on_empty_data(self) -> None:
        """Raises on empty data."""
        with pytest.raises(ContainerDecodeError):
            decode_unknown_container(b"")

    def test_unknown_container_dict_keys(self) -> None:
        """UnknownContainerDict has expected keys."""
        result: UnknownContainerDict = decode_unknown_container(UNKNOWN_8_BYTES)
        assert result["msg_type"] == "unknown_container"
        assert result["subtype"] == 0x7E
        assert result["length"] == 8
        assert result["data"] == UNKNOWN_8_BYTES


class TestDecodeLevel:
    """Tests for DecodeLevel enum."""

    def test_level_values_are_weights(self) -> None:
        """DecodeLevel values are integer weights for stats calculation."""
        assert DecodeLevel.UNKNOWN.value == 0
        assert DecodeLevel.IDENTIFIED.value == 25
        assert DecodeLevel.PARTIAL.value == 50
        assert DecodeLevel.FULL.value == 100

    def test_level_ordering(self) -> None:
        """DecodeLevel values are ordered from least to most understanding."""
        assert DecodeLevel.UNKNOWN < DecodeLevel.IDENTIFIED
        assert DecodeLevel.IDENTIFIED < DecodeLevel.PARTIAL
        assert DecodeLevel.PARTIAL < DecodeLevel.FULL


class TestGetDecodeLevel:
    """Tests for get_decode_level function."""

    def test_full_level_for_container_pickup(self) -> None:
        """Container pickup has FULL decode level."""
        level = get_decode_level(ContainerMessageType.CONTAINER_PICKUP)
        assert level == DecodeLevel.FULL

    # RADAR_RESPONSE enum entry deleted 2026-06-19 with the rest of the
    # container radar chain.

    # TIP_NOTIFICATION / CHUNK_DATA / WORLD_STATE enum entries deleted
    # 2026-06-19 with the rest of the dead container blob chain.

    def test_unknown_for_unknown_type(self) -> None:
        """UNKNOWN type has UNKNOWN decode level."""
        level = get_decode_level(ContainerMessageType.UNKNOWN)
        assert level == DecodeLevel.UNKNOWN


class TestMessageTypeLevels:
    """Tests for MESSAGE_TYPE_LEVELS registry."""

    def test_all_message_types_have_level(self) -> None:
        """All ContainerMessageType values are in MESSAGE_TYPE_LEVELS."""
        for msg_type in ContainerMessageType:
            assert msg_type in MESSAGE_TYPE_LEVELS, f"{msg_type} missing from registry"

    def test_registry_values_are_decode_levels(self) -> None:
        """All values in MESSAGE_TYPE_LEVELS are DecodeLevel enum values."""
        for msg_type, level in MESSAGE_TYPE_LEVELS.items():
            assert level in DecodeLevel, f"{msg_type} has invalid level {level}"
