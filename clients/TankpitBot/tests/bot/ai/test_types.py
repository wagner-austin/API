"""Tests for the AI vocabulary types.

Behaviour scores, enemy threats, path steps, and config.
``test_types.py`` was 649 lines; the AI-state suite is now a sibling.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.bot.ai.scoring_types import (
    BEHAVIOR_MODES,
    make_behavior_score,
)
from tankpit_bot.bot.ai.types import (
    make_default_ai_config,
    make_initial_ai_state,
    make_respawn_ai_state,
)
from tankpit_bot.bot.ai.types_codecs import (
    decode_ai_config,
    decode_ai_state,
    decode_behavior_score,
    decode_enemy_threat,
    decode_path_step,
    encode_ai_config,
    encode_ai_state,
    encode_behavior_score,
    encode_enemy_threat,
    encode_path_step,
)
from tankpit_bot.bot.ai.world_types import (
    make_enemy_threat,
    make_path_step,
)


class TestAIState:
    """Tests for AIStateDict factory and encode/decode."""

    def test_make_respawn_ai_state_carries_session_scope_only(self) -> None:
        """Death resets life-scoped state; the seven session-scoped
        fields survive — run bot-20260803-180918's summary printed 23
        shots against 223 actual because the old inline carry-list
        dropped the hit/miss/reject counters."""
        previous = make_initial_ai_state()
        previous["session_kill_count"] = 14
        previous["session_hit_count"] = 200
        previous["session_miss_count"] = 6
        previous["session_reject_count"] = 17
        previous["wind_down"] = True
        previous["greeted_tank_ids"] = {"2678": 1}
        previous["visited_tank_ids"] = {"984": 1}
        previous["mode"] = "HUNT"

        fresh = make_respawn_ai_state(previous)

        assert fresh["session_kill_count"] == 14
        assert fresh["session_hit_count"] == 200
        assert fresh["session_miss_count"] == 6
        assert fresh["session_reject_count"] == 17
        assert fresh["wind_down"] is True
        assert fresh["greeted_tank_ids"] == {"2678": 1}
        assert fresh["visited_tank_ids"] == {"984": 1}
        assert fresh["mode"] == make_initial_ai_state()["mode"]
        assert fresh["mode"] != "HUNT"

    def test_make_initial_ai_state_defaults(self) -> None:
        """Initial state uses default config and unset durable mode."""
        state = make_initial_ai_state()
        assert state["mode"] == "UNSET"
        assert state["mode_state"] == ""
        assert state["mode_started_ms"] == 0
        assert state["last_scan_ms"] == 1
        assert state["last_shoot_ms"] == 0
        assert state["combat_target_id"] == -1
        assert state["config"]["fuel_low_threshold"] == 200
        assert state["manual_mode"] is None
        assert state["live_radars_used"] == 0
        assert state["live_teleports"] == 0

    def test_make_initial_ai_state_custom_config(self) -> None:
        """Initial state accepts custom config."""
        from tankpit_bot.bot.ai.types import AIConfigDict

        config = make_default_ai_config()
        custom = AIConfigDict(
            fuel_low_threshold=400,
            hunt_min_fuel=config["hunt_min_fuel"],
            combat_range=config["combat_range"],
            scan_cooldown_ms=config["scan_cooldown_ms"],
            shot_feedback_timeout_ms=config["shot_feedback_timeout_ms"],
            action_stall_timeout_ms=config["action_stall_timeout_ms"],
            kill_cooldown_ms=config["kill_cooldown_ms"],
            map_open_cooldown_ms=config["map_open_cooldown_ms"],
            dual_break_threshold=config["dual_break_threshold"],
            radar_break_threshold=config["radar_break_threshold"],
            engagement_fuel_budget=config["engagement_fuel_budget"],
            patrol_waypoints=config["patrol_waypoints"],
            priority_target_name=config["priority_target_name"],
            human_target_min_rank=config["human_target_min_rank"],
            human_target_max_rank=config["human_target_max_rank"],
            role=config["role"],
        )
        state = make_initial_ai_state(custom)
        assert state["config"]["fuel_low_threshold"] == 400

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces identical AIStateDict."""
        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        decoded = decode_ai_state(encoded)
        assert decoded == original

    def test_decode_invalid_config_raises(self) -> None:
        """Decode rejects non-dict config."""
        data: JSONObject = {
            "config": "not_a_dict",
            "mode": "UNSET",
            "mode_state": "",
            "mode_started_ms": 0,
            "last_scan_ms": 0,
            "last_shoot_ms": 0,
            "combat_target_id": -1,
            "wind_down": False,
            "combat_target_x": 0,
            "combat_target_y": 0,
        }
        with pytest.raises(ValueError, match="config must be an object"):
            decode_ai_state(data)

    def test_decode_invalid_mode_raises(self) -> None:
        """Decode rejects invalid durable mode."""
        config = encode_ai_config(make_default_ai_config())
        data: JSONObject = {
            "config": config,
            "mode": "INVALID",
            "mode_state": "",
            "mode_started_ms": 0,
            "last_scan_ms": 0,
            "last_shoot_ms": 0,
            "combat_target_id": -1,
            "wind_down": False,
            "combat_target_x": 0,
            "combat_target_y": 0,
        }
        with pytest.raises(ValueError, match="must be one of"):
            decode_ai_state(data)

    def test_decode_invalid_mode_state_pair_raises(self) -> None:
        """Decode rejects durable mode/substate pairs that do not match."""
        config = encode_ai_config(make_default_ai_config())
        data: JSONObject = {
            "config": config,
            "mode": "HUNT",
            "mode_state": "SEARCH",
            "mode_started_ms": 0,
            "last_scan_ms": 0,
            "last_shoot_ms": 0,
            "last_map_open_ms": 0,
            "combat_target_id": -1,
            "wind_down": False,
            "combat_target_x": 0,
            "combat_target_y": 0,
            "killed_tank_ids": {},
            "blocked_combat_targets": {},
            "last_shot_target_id": -1,
            "last_shot_target_name": "",
            "resource_target_kind": "",
            "resource_target_x": 0,
            "resource_target_y": 0,
        }
        with pytest.raises(ValueError, match="invalid for mode"):
            decode_ai_state(data)

    def test_decode_killed_tank_ids_not_dict_raises(self) -> None:
        """Decode rejects non-dict killed_tank_ids."""
        config = encode_ai_config(make_default_ai_config())
        data: JSONObject = {
            "config": config,
            "mode": "UNSET",
            "mode_state": "",
            "mode_started_ms": 0,
            "last_scan_ms": 0,
            "last_shoot_ms": 0,
            "last_map_open_ms": 0,
            "combat_target_id": -1,
            "wind_down": False,
            "combat_target_x": 0,
            "combat_target_y": 0,
            "killed_tank_ids": "not_a_dict",
            "break_escape_until_fuel": 0,
            "blocked_combat_targets": {},
            "last_shot_target_id": -1,
            "last_shot_target_name": "",
        }
        with pytest.raises(ValueError, match="killed_tank_ids must be an object"):
            decode_ai_state(data)


class TestBehaviorModes:
    """Tests for BEHAVIOR_MODES constant."""

    def test_all_modes_present(self) -> None:
        """Behavior modes are defined."""
        assert len(BEHAVIOR_MODES) == 2
        assert "HUNT" in BEHAVIOR_MODES
        assert "COLLECT" in BEHAVIOR_MODES


class TestBehaviorScore:
    """Tests for BehaviorScoreDict factory and encode/decode."""

    def test_make_behavior_score(self) -> None:
        """Factory creates correct BehaviorScoreDict."""
        score = make_behavior_score("HUNT", 800, 100, 150, "find_target")
        assert score["mode"] == "HUNT"
        assert score["score"] == 800
        assert score["target_x"] == 100
        assert score["target_y"] == 150
        assert score["reason_kind"] == "find_target"
        assert score["reason_context"] == {}

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces identical BehaviorScoreDict."""
        original = make_behavior_score("COLLECT", 600, 50, 75, "fuel_collect")
        encoded = encode_behavior_score(original)
        decoded = decode_behavior_score(encoded)
        assert decoded == original

    def test_roundtrip_with_reason_context(self) -> None:
        """A reason context map survives encode/decode."""
        original = make_behavior_score(
            "HUNT", 800, 100, 150, "shoot_target", reason_context={"target_name": "orange-3"}
        )
        decoded = decode_behavior_score(encode_behavior_score(original))
        assert decoded == original

    def test_decode_invalid_reason_kind_raises(self) -> None:
        """Decode rejects an unknown reason kind."""
        data: JSONObject = {
            "mode": "HUNT",
            "score": 100,
            "target_x": 0,
            "target_y": 0,
            "target_id": 0,
            "reason_kind": "vibes",
            "reason_context": {},
        }
        with pytest.raises(JSONTypeError, match="reason_kind must be one of"):
            decode_behavior_score(data)

    def test_decode_invalid_reason_context_value_raises(self) -> None:
        """Decode rejects non-scalar reason context values."""
        data: JSONObject = {
            "mode": "HUNT",
            "score": 100,
            "target_x": 0,
            "target_y": 0,
            "target_id": 0,
            "reason_kind": "manual_hold",
            "reason_context": {"flag": True},
        }
        with pytest.raises(JSONTypeError, match="must be str or int"):
            decode_behavior_score(data)

    def test_decode_invalid_mode_raises(self) -> None:
        """Decode rejects invalid BehaviorMode."""
        data: JSONObject = {
            "mode": "INVALID",
            "score": 100,
            "target_x": 0,
            "target_y": 0,
            "reason": "test",
        }
        with pytest.raises(ValueError, match="must be one of"):
            decode_behavior_score(data)

    def test_decode_missing_field_raises(self) -> None:
        """Decode rejects missing required fields."""
        data: JSONObject = {"mode": "HUNT", "score": 100}
        with pytest.raises(JSONTypeError):
            decode_behavior_score(data)


class TestEnemyThreat:
    """Tests for EnemyThreatDict factory and encode/decode."""

    def test_make_enemy_threat(self) -> None:
        """Factory creates correct EnemyThreatDict."""
        threat = make_enemy_threat(
            tank_id=536,
            x=100,
            y=120,
            distance=30,
            damage_state=2,
            rank=3,
            team=0,
            name="red-1",
            is_bot=True,
            timestamp_ms=8000,
            last_wire_seen_ms=4200,
            last_position_update_ms=4200,
        )
        assert threat["tank_id"] == 536
        assert threat["x"] == 100
        assert threat["distance"] == 30
        assert threat["damage_state"] == 2
        assert threat["rank"] == 3
        assert threat["name"] == "red-1"
        assert threat["is_bot"] is True
        assert threat["timestamp_ms"] == 8000
        assert threat["last_wire_seen_ms"] == 4200

    def test_make_enemy_threat_defaults_wire_stamp_zero(self) -> None:
        """last_wire_seen_ms defaults to zero (never wire-confirmed)."""
        threat = make_enemy_threat(
            tank_id=1,
            x=0,
            y=0,
            distance=0,
            damage_state=0,
            rank=0,
            team=1,
            name="n",
            is_bot=True,
        )
        assert threat["last_wire_seen_ms"] == 0

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces identical EnemyThreatDict."""
        original = make_enemy_threat(
            tank_id=100,
            x=50,
            y=60,
            distance=15,
            damage_state=0,
            rank=5,
            team=1,
            name="test",
            is_bot=False,
            timestamp_ms=7000,
            last_wire_seen_ms=6500,
            last_position_update_ms=6500,
        )
        encoded = encode_enemy_threat(original)
        assert encoded["last_wire_seen_ms"] == 6500
        decoded = decode_enemy_threat(encoded)
        assert decoded == original

    def test_decode_missing_field_raises(self) -> None:
        """Decode rejects missing required fields."""
        data: JSONObject = {"tank_id": 1, "x": 0}
        with pytest.raises(JSONTypeError):
            decode_enemy_threat(data)


class TestPathStep:
    """Tests for PathStepDict factory and encode/decode."""

    def test_make_path_step(self) -> None:
        """Factory creates correct PathStepDict."""
        step = make_path_step(10, 20)
        assert step["x"] == 10
        assert step["y"] == 20

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces identical PathStepDict."""
        original = make_path_step(128, 64)
        encoded = encode_path_step(original)
        decoded = decode_path_step(encoded)
        assert decoded == original

    def test_decode_missing_field_raises(self) -> None:
        """Decode rejects missing required fields."""
        data: JSONObject = {"x": 10}
        with pytest.raises(JSONTypeError):
            decode_path_step(data)


class TestAIConfig:
    """Tests for AIConfigDict factory and encode/decode."""

    def test_make_default_ai_config(self) -> None:
        """Default config has sensible values."""
        config = make_default_ai_config()
        assert config["fuel_low_threshold"] == 200
        assert config["hunt_min_fuel"] == 100
        assert config["combat_range"] == 20
        assert config["scan_cooldown_ms"] == 5000
        assert config["shot_feedback_timeout_ms"] == 4000
        assert config["action_stall_timeout_ms"] == 10000
        assert "teleport_fuel_cost" not in config
        assert len(config["patrol_waypoints"]) == 4
        assert config["dual_break_threshold"] == 4
        assert config["engagement_fuel_budget"] == 450

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces identical AIConfigDict."""
        original = make_default_ai_config()
        encoded = encode_ai_config(original)
        decoded = decode_ai_config(encoded)
        assert decoded == original

    def test_decode_invalid_waypoints_not_list_raises(self) -> None:
        """Decode rejects non-list patrol_waypoints."""
        data: JSONObject = {
            "fuel_low_threshold": 200,
            "hunt_min_fuel": 400,
            "combat_range": 20,
            "scan_cooldown_ms": 5000,
            "shot_feedback_timeout_ms": 4000,
            "action_stall_timeout_ms": 10000,
            "kill_cooldown_ms": 20000,
            "map_open_cooldown_ms": 5000,
            "dual_break_threshold": 4,
            "patrol_waypoints": "not_a_list",
        }
        with pytest.raises(ValueError, match="must be a list"):
            decode_ai_config(data)

    def test_decode_invalid_waypoint_format_raises(self) -> None:
        """Decode rejects waypoints that are not [x, y] pairs."""
        data: JSONObject = {
            "fuel_low_threshold": 200,
            "hunt_min_fuel": 400,
            "combat_range": 20,
            "scan_cooldown_ms": 5000,
            "shot_feedback_timeout_ms": 4000,
            "action_stall_timeout_ms": 10000,
            "kill_cooldown_ms": 20000,
            "map_open_cooldown_ms": 5000,
            "dual_break_threshold": 4,
            "patrol_waypoints": [[1, 2, 3]],
        }
        with pytest.raises(ValueError, match="must be"):
            decode_ai_config(data)

    def test_decode_invalid_waypoint_type_raises(self) -> None:
        """Decode rejects waypoints with non-int coordinates."""
        data: JSONObject = {
            "fuel_low_threshold": 200,
            "hunt_min_fuel": 400,
            "combat_range": 20,
            "scan_cooldown_ms": 5000,
            "shot_feedback_timeout_ms": 4000,
            "action_stall_timeout_ms": 10000,
            "kill_cooldown_ms": 20000,
            "map_open_cooldown_ms": 5000,
            "dual_break_threshold": 4,
            "patrol_waypoints": [["a", "b"]],
        }
        with pytest.raises(ValueError, match="must be int"):
            decode_ai_config(data)

    def test_decode_missing_field_raises(self) -> None:
        """Decode rejects missing required fields."""
        data: JSONObject = {"fuel_low_threshold": 200}
        with pytest.raises(JSONTypeError):
            decode_ai_config(data)
