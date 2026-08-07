"""Tests for :mod:`tankpit_bot.service.types` and codecs.

Covers every factory, translator, and codec branch for the bot
service wire surface. No mocks — the tests validate the real code
paths end-to-end.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.bus.session_status import (
    WIRE_MODES,
    idle_session_status,
    make_live_stats,
    make_session_status,
    manual_to_wire_mode,
    wire_mode_to_manual,
    zero_live_stats,
)
from tankpit_bot.service.types import make_mode_command
from tankpit_bot.service.types_codecs import (
    decode_live_stats,
    decode_mode_command,
    decode_session_status,
    encode_live_stats,
    encode_mode_command,
    encode_session_status,
)

# =============================================================================
# WIRE_MODES constant
# =============================================================================


class TestWireModes:
    """Tests for :data:`WIRE_MODES`."""

    def test_all_four_modes_present(self) -> None:
        """All four wire modes are exposed in order."""
        assert WIRE_MODES == ("UNSET", "HUNT", "COLLECT", "AUTO")


# =============================================================================
# ModeCommandDict
# =============================================================================


class TestModeCommand:
    """Factory + encode/decode for :class:`ModeCommandDict`."""

    def test_make_mode_command(self) -> None:
        """Factory populates the manual_mode field."""
        cmd = make_mode_command("HUNT")
        assert cmd["manual_mode"] == "HUNT"

    def test_encode_decode_roundtrip_each_mode(self) -> None:
        """Every wire mode round-trips cleanly through the codec."""
        for mode in WIRE_MODES:
            original = make_mode_command(mode)
            encoded = encode_mode_command(original)
            decoded = decode_mode_command(encoded)
            assert decoded == original

    def test_decode_invalid_mode_raises(self) -> None:
        """Decode rejects a value outside :data:`WIRE_MODES`."""
        data: JSONObject = {"manual_mode": "INVALID"}
        with pytest.raises(ValueError, match="must be one of"):
            decode_mode_command(data)

    def test_decode_missing_field_raises(self) -> None:
        """Decode rejects a missing ``manual_mode`` field."""
        data: JSONObject = {}
        with pytest.raises(JSONTypeError):
            decode_mode_command(data)

    def test_decode_wrong_type_raises(self) -> None:
        """Decode rejects a non-string ``manual_mode`` value."""
        data: JSONObject = {"manual_mode": 7}
        with pytest.raises(JSONTypeError):
            decode_mode_command(data)


# =============================================================================
# LiveStatsDict
# =============================================================================


class TestLiveStats:
    """Factory + encode/decode for :class:`LiveStatsDict`."""

    def test_make_live_stats(self) -> None:
        """Factory populates every counter."""
        stats = make_live_stats(kills=3, hits=47, misses=12, radars_used=8, teleports=15)
        assert stats["kills"] == 3
        assert stats["hits"] == 47
        assert stats["misses"] == 12
        assert stats["radars_used"] == 8
        assert stats["teleports"] == 15

    def test_zero_live_stats(self) -> None:
        """The zero factory populates every counter with 0."""
        stats = zero_live_stats()
        assert stats["kills"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["radars_used"] == 0
        assert stats["teleports"] == 0

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode reproduces the original stats dict."""
        original = make_live_stats(kills=2, hits=9, misses=4, radars_used=6, teleports=11)
        encoded = encode_live_stats(original)
        decoded = decode_live_stats(encoded)
        assert decoded == original

    def test_decode_missing_kills_raises(self) -> None:
        """Missing ``kills`` field surfaces as JSONTypeError."""
        data: JSONObject = {"hits": 0, "misses": 0, "radars_used": 0, "teleports": 0}
        with pytest.raises(JSONTypeError):
            decode_live_stats(data)

    def test_decode_missing_hits_raises(self) -> None:
        """Missing ``hits`` field surfaces as JSONTypeError."""
        data: JSONObject = {"kills": 0, "misses": 0, "radars_used": 0, "teleports": 0}
        with pytest.raises(JSONTypeError):
            decode_live_stats(data)

    def test_decode_missing_misses_raises(self) -> None:
        """Missing ``misses`` field surfaces as JSONTypeError."""
        data: JSONObject = {"kills": 0, "hits": 0, "radars_used": 0, "teleports": 0}
        with pytest.raises(JSONTypeError):
            decode_live_stats(data)

    def test_decode_missing_radars_used_raises(self) -> None:
        """Missing ``radars_used`` field surfaces as JSONTypeError."""
        data: JSONObject = {"kills": 0, "hits": 0, "misses": 0, "teleports": 0}
        with pytest.raises(JSONTypeError):
            decode_live_stats(data)

    def test_decode_missing_teleports_raises(self) -> None:
        """Missing ``teleports`` field surfaces as JSONTypeError."""
        data: JSONObject = {"kills": 0, "hits": 0, "misses": 0, "radars_used": 0}
        with pytest.raises(JSONTypeError):
            decode_live_stats(data)

    def test_decode_wrong_type_raises(self) -> None:
        """Non-int counter surfaces as JSONTypeError."""
        data: JSONObject = {
            "kills": "one",
            "hits": 0,
            "misses": 0,
            "radars_used": 0,
            "teleports": 0,
        }
        with pytest.raises(JSONTypeError):
            decode_live_stats(data)


# =============================================================================
# SessionStatusDict
# =============================================================================


class TestSessionStatus:
    """Factory + encode/decode for :class:`SessionStatusDict`."""

    def test_make_session_status(self) -> None:
        """Factory wires every field through."""
        stats = make_live_stats(kills=1, hits=2, misses=3, radars_used=4, teleports=5)
        status = make_session_status(
            running=True,
            manual_mode="HUNT",
            active_mode="HUNT",
            active_mode_state="ACQUIRE",
            session_started_ms=1000,
            tick_timestamp_ms=1200,
            stats=stats,
        )
        assert status["running"] is True
        assert status["manual_mode"] == "HUNT"
        assert status["active_mode"] == "HUNT"
        assert status["active_mode_state"] == "ACQUIRE"
        assert status["session_started_ms"] == 1000
        assert status["tick_timestamp_ms"] == 1200
        assert status["stats"] == stats

    def test_idle_session_status(self) -> None:
        """Idle status is not running and every counter is zero."""
        status = idle_session_status(tick_timestamp_ms=42)
        assert status["running"] is False
        assert status["manual_mode"] == "AUTO"
        assert status["active_mode"] == "UNSET"
        assert status["active_mode_state"] == ""
        assert status["session_started_ms"] == 0
        assert status["tick_timestamp_ms"] == 42
        assert status["stats"] == zero_live_stats()

    def test_encode_decode_roundtrip_running(self) -> None:
        """Encode/decode reproduces a running-session snapshot."""
        stats = make_live_stats(kills=6, hits=18, misses=9, radars_used=3, teleports=7)
        original = make_session_status(
            running=True,
            manual_mode="COLLECT",
            active_mode="COLLECT",
            active_mode_state="APPROACH",
            session_started_ms=555,
            tick_timestamp_ms=999,
            stats=stats,
        )
        encoded = encode_session_status(original)
        decoded = decode_session_status(encoded)
        assert decoded == original

    def test_encode_decode_roundtrip_idle(self) -> None:
        """Encode/decode reproduces an idle status."""
        original = idle_session_status(tick_timestamp_ms=17)
        encoded = encode_session_status(original)
        decoded = decode_session_status(encoded)
        assert decoded == original

    def test_decode_missing_stats_raises(self) -> None:
        """A missing ``stats`` field surfaces as ValueError."""
        original = idle_session_status(tick_timestamp_ms=1)
        encoded = encode_session_status(original)
        del encoded["stats"]
        with pytest.raises(ValueError, match="stats must be an object"):
            decode_session_status(encoded)

    def test_decode_stats_wrong_type_raises(self) -> None:
        """A non-object ``stats`` field surfaces as ValueError."""
        original = idle_session_status(tick_timestamp_ms=1)
        encoded = encode_session_status(original)
        encoded["stats"] = [0, 0, 0, 0, 0]
        with pytest.raises(ValueError, match="stats must be an object"):
            decode_session_status(encoded)

    def test_decode_invalid_mode_state_pair_raises(self) -> None:
        """An invalid (active_mode, active_mode_state) pair raises."""
        original = idle_session_status(tick_timestamp_ms=1)
        encoded = encode_session_status(original)
        encoded["active_mode"] = "HUNT"
        encoded["active_mode_state"] = "SENSE"  # SENSE is a COLLECT substate
        with pytest.raises(ValueError, match="is invalid for active_mode"):
            decode_session_status(encoded)

    def test_decode_invalid_manual_mode_raises(self) -> None:
        """An invalid ``manual_mode`` surfaces as ValueError."""
        original = idle_session_status(tick_timestamp_ms=1)
        encoded = encode_session_status(original)
        encoded["manual_mode"] = "INVALID"
        with pytest.raises(ValueError, match="must be one of"):
            decode_session_status(encoded)

    def test_decode_missing_running_raises(self) -> None:
        """Missing ``running`` surfaces as JSONTypeError."""
        original = idle_session_status(tick_timestamp_ms=1)
        encoded = encode_session_status(original)
        del encoded["running"]
        with pytest.raises(JSONTypeError):
            decode_session_status(encoded)


# =============================================================================
# wire_mode_to_manual + manual_to_wire_mode
# =============================================================================


class TestWireModeTranslators:
    """Tests for wire ↔ manual translation."""

    def test_wire_auto_becomes_none(self) -> None:
        """AUTO on the wire maps to auto-arbitration (None)."""
        assert wire_mode_to_manual("AUTO") is None

    def test_wire_unset_passes_through(self) -> None:
        """UNSET on the wire maps to UNSET."""
        assert wire_mode_to_manual("UNSET") == "UNSET"

    def test_wire_hunt_passes_through(self) -> None:
        """HUNT on the wire maps to HUNT."""
        assert wire_mode_to_manual("HUNT") == "HUNT"

    def test_wire_collect_passes_through(self) -> None:
        """COLLECT on the wire maps to COLLECT."""
        assert wire_mode_to_manual("COLLECT") == "COLLECT"

    def test_manual_none_becomes_auto(self) -> None:
        """None maps back to AUTO on the wire."""
        assert manual_to_wire_mode(None) == "AUTO"

    def test_manual_unset_passes_through(self) -> None:
        """UNSET maps back to UNSET on the wire."""
        assert manual_to_wire_mode("UNSET") == "UNSET"

    def test_manual_hunt_passes_through(self) -> None:
        """HUNT maps back to HUNT on the wire."""
        assert manual_to_wire_mode("HUNT") == "HUNT"

    def test_manual_collect_passes_through(self) -> None:
        """COLLECT maps back to COLLECT on the wire."""
        assert manual_to_wire_mode("COLLECT") == "COLLECT"

    def test_wire_manual_roundtrip(self) -> None:
        """Every WireMode round-trips through manual → wire."""
        for wire in WIRE_MODES:
            manual = wire_mode_to_manual(wire)
            assert manual_to_wire_mode(manual) == wire
