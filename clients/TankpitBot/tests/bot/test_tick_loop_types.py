"""Tests for TickDecisionDict encode/decode round-trip."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.bot.ai.types import make_behavior_score, make_initial_ai_state
from tankpit_bot.bot.tick_loop_types import (
    decode_tick_decision,
    encode_tick_decision,
    make_tick_decision,
)
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_move_command,
    make_pickup_move_command,
    make_radar_command,
    make_teleport_command,
)


class TestMakeTickDecision:
    """Tests for make_tick_decision factory."""

    def test_make_with_move_command(self) -> None:
        """Factory creates TickDecisionDict with move command."""
        cmd = make_move_command(100, 200)
        behavior = make_behavior_score("HUNT", 50, 100, 200, "patrol_waypoint")
        ai_state = make_initial_ai_state()
        decision = make_tick_decision(cmd, behavior, ai_state, [2, 5])
        assert decision["command"]["cmd_type"] == "move"
        assert decision["command"]["target_x"] == 100
        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["desired_equipment"] == [2, 5]

    def test_make_with_radar_command(self) -> None:
        """Factory creates TickDecisionDict with radar command."""
        cmd = make_radar_command()
        behavior = make_behavior_score("HUNT", 0, 0, 0, "proactive_radar")
        ai_state = make_initial_ai_state()
        decision = make_tick_decision(cmd, behavior, ai_state, [5])
        assert decision["command"]["cmd_type"] == "radar"
        assert decision["desired_equipment"] == [5]

    def test_desired_equipment_sorted(self) -> None:
        """Factory sorts desired_equipment list."""
        cmd = make_radar_command()
        behavior = make_behavior_score("HUNT", 0, 0, 0, "test")
        ai_state = make_initial_ai_state()
        decision = make_tick_decision(cmd, behavior, ai_state, [5, 2])
        assert decision["desired_equipment"] == [2, 5]


class TestEncodeDecodeRoundTrip:
    """Tests for encode/decode round-trip on all command types."""

    def test_roundtrip_move(self) -> None:
        """Encode then decode produces identical TickDecisionDict with move."""
        cmd = make_move_command(50, 75)
        behavior = make_behavior_score("HUNT", 800, 50, 75, "enemy nearby")
        ai_state = make_initial_ai_state()
        original = make_tick_decision(cmd, behavior, ai_state, [2, 4, 5])
        encoded = encode_tick_decision(original)
        decoded = decode_tick_decision(encoded)
        assert decoded == original

    def test_roundtrip_shoot(self) -> None:
        """Encode then decode produces identical TickDecisionDict with shoot."""
        from tankpit_bot.bot.types import make_shoot_command

        cmd = make_shoot_command(128, 64)
        behavior = make_behavior_score("HUNT", 900, 128, 64, "target acquired")
        ai_state = make_initial_ai_state()
        original = make_tick_decision(cmd, behavior, ai_state, [2, 5])
        encoded = encode_tick_decision(original)
        decoded = decode_tick_decision(encoded)
        assert decoded == original

    def test_roundtrip_radar(self) -> None:
        """Encode then decode produces identical TickDecisionDict with radar."""
        cmd = make_radar_command()
        behavior = make_behavior_score("HUNT", 0, 0, 0, "proactive_radar")
        ai_state = make_initial_ai_state()
        original = make_tick_decision(cmd, behavior, ai_state, [5])
        encoded = encode_tick_decision(original)
        decoded = decode_tick_decision(encoded)
        assert decoded == original

    def test_roundtrip_map_open(self) -> None:
        """Encode then decode produces identical TickDecisionDict with map_open."""
        cmd = make_map_open_command()
        behavior = make_behavior_score("HUNT", 0, 0, 0, "map_open_enemies")
        ai_state = make_initial_ai_state()
        original = make_tick_decision(cmd, behavior, ai_state, [5])
        encoded = encode_tick_decision(original)
        decoded = decode_tick_decision(encoded)
        assert decoded == original

    def test_roundtrip_pickup_move(self) -> None:
        """Encode then decode produces identical TickDecisionDict with pickup_move."""
        cmd = make_pickup_move_command(80, 90)
        behavior = make_behavior_score("COLLECT_FUEL", 600, 80, 90, "low fuel")
        ai_state = make_initial_ai_state()
        original = make_tick_decision(cmd, behavior, ai_state, [1, 5])
        encoded = encode_tick_decision(original)
        decoded = decode_tick_decision(encoded)
        assert decoded == original

    def test_roundtrip_teleport(self) -> None:
        """Encode then decode produces identical TickDecisionDict with teleport."""
        cmd = make_teleport_command(200, 200)
        behavior = make_behavior_score("HUNT", 50, 200, 200, "teleport_search")
        ai_state = make_initial_ai_state()
        original = make_tick_decision(cmd, behavior, ai_state, [1, 5])
        encoded = encode_tick_decision(original)
        decoded = decode_tick_decision(encoded)
        assert decoded == original


class TestDecodeValidation:
    """Tests for decode_tick_decision validation errors."""

    def test_decode_invalid_command_raises(self) -> None:
        """Decode rejects non-dict command."""
        data: JSONObject = {
            "command": "not_a_dict",
            "behavior": {},
            "updated_ai_state": {},
            "desired_equipment": [],
        }
        with pytest.raises(ValueError, match="command must be an object"):
            decode_tick_decision(data)

    def test_decode_invalid_behavior_raises(self) -> None:
        """Decode rejects non-dict behavior."""
        data: JSONObject = {
            "command": {"cmd_type": "radar"},
            "behavior": "not_a_dict",
            "updated_ai_state": {},
            "desired_equipment": [],
        }
        with pytest.raises(ValueError, match="behavior must be an object"):
            decode_tick_decision(data)

    def test_decode_invalid_ai_state_raises(self) -> None:
        """Decode rejects non-dict updated_ai_state."""
        cmd = make_radar_command()
        behavior = make_behavior_score("HUNT", 0, 0, 0, "test")
        data: JSONObject = {
            "command": {"cmd_type": "radar"},
            "behavior": encode_tick_decision(
                make_tick_decision(cmd, behavior, make_initial_ai_state(), [5])
            )["behavior"],
            "updated_ai_state": "not_a_dict",
            "desired_equipment": [5],
        }
        with pytest.raises(ValueError, match="updated_ai_state must be an object"):
            decode_tick_decision(data)

    def test_decode_invalid_equipment_not_list(self) -> None:
        """Decode rejects non-list desired_equipment."""
        cmd = make_radar_command()
        behavior = make_behavior_score("HUNT", 0, 0, 0, "test")
        ai_state = make_initial_ai_state()
        full = encode_tick_decision(make_tick_decision(cmd, behavior, ai_state, [5]))
        full["desired_equipment"] = "not_a_list"
        with pytest.raises(ValueError, match="must be a list"):
            decode_tick_decision(full)

    def test_decode_invalid_equipment_non_int(self) -> None:
        """Decode rejects desired_equipment with non-int items."""
        cmd = make_radar_command()
        behavior = make_behavior_score("HUNT", 0, 0, 0, "test")
        ai_state = make_initial_ai_state()
        full = encode_tick_decision(make_tick_decision(cmd, behavior, ai_state, [5]))
        full["desired_equipment"] = ["not_int"]
        with pytest.raises(ValueError, match="must be int"):
            decode_tick_decision(full)

    def test_decode_unknown_cmd_type_raises(self) -> None:
        """Decode rejects unknown cmd_type."""
        cmd = make_radar_command()
        behavior = make_behavior_score("HUNT", 0, 0, 0, "test")
        ai_state = make_initial_ai_state()
        full = encode_tick_decision(make_tick_decision(cmd, behavior, ai_state, [5]))
        # Replace command with a dict that has unknown cmd_type + required fields
        full["command"] = {"cmd_type": "UNKNOWN", "target_x": 0, "target_y": 0}
        with pytest.raises(ValueError, match="Unknown cmd_type"):
            decode_tick_decision(full)
