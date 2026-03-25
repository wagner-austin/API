"""Tests for AI system core types."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.bot.ai.types import (
    BEHAVIOR_MODES,
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
    make_behavior_score,
    make_default_ai_config,
    make_enemy_threat,
    make_initial_ai_state,
    make_path_step,
)

# =============================================================================
# BehaviorMode
# =============================================================================


class TestBehaviorModes:
    """Tests for BEHAVIOR_MODES constant."""

    def test_all_modes_present(self) -> None:
        """All six behavior modes are defined."""
        assert len(BEHAVIOR_MODES) == 6
        assert "HUNT" in BEHAVIOR_MODES
        assert "COLLECT_FUEL" in BEHAVIOR_MODES
        assert "COLLECT_EQUIPMENT" in BEHAVIOR_MODES
        assert "DEPOSIT_FUEL" in BEHAVIOR_MODES
        assert "PATROL" in BEHAVIOR_MODES
        assert "DEFEND" in BEHAVIOR_MODES


# =============================================================================
# BehaviorScoreDict
# =============================================================================


class TestBehaviorScore:
    """Tests for BehaviorScoreDict factory and encode/decode."""

    def test_make_behavior_score(self) -> None:
        """Factory creates correct BehaviorScoreDict."""
        score = make_behavior_score("HUNT", 800, 100, 150, "enemy nearby")
        assert score["mode"] == "HUNT"
        assert score["score"] == 800
        assert score["target_x"] == 100
        assert score["target_y"] == 150
        assert score["reason"] == "enemy nearby"

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces identical BehaviorScoreDict."""
        original = make_behavior_score("COLLECT_FUEL", 600, 50, 75, "low fuel")
        encoded = encode_behavior_score(original)
        decoded = decode_behavior_score(encoded)
        assert decoded == original

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


# =============================================================================
# EnemyThreatDict
# =============================================================================


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
        )
        assert threat["tank_id"] == 536
        assert threat["x"] == 100
        assert threat["distance"] == 30
        assert threat["damage_state"] == 2
        assert threat["rank"] == 3
        assert threat["name"] == "red-1"
        assert threat["is_bot"] is True

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
        )
        encoded = encode_enemy_threat(original)
        decoded = decode_enemy_threat(encoded)
        assert decoded == original

    def test_decode_missing_field_raises(self) -> None:
        """Decode rejects missing required fields."""
        data: JSONObject = {"tank_id": 1, "x": 0}
        with pytest.raises(JSONTypeError):
            decode_enemy_threat(data)


# =============================================================================
# PathStepDict
# =============================================================================


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


# =============================================================================
# AIConfigDict
# =============================================================================


class TestAIConfig:
    """Tests for AIConfigDict factory and encode/decode."""

    def test_make_default_ai_config(self) -> None:
        """Default config has sensible values."""
        config = make_default_ai_config()
        assert config["fuel_critical_threshold"] == 200
        assert config["fuel_low_threshold"] == 500
        assert config["fuel_full_threshold"] == 1200
        assert config["hunt_min_fuel"] == 400
        assert config["combat_range"] == 20
        assert config["scan_cooldown_ms"] == 5000
        assert config["shoot_cooldown_ms"] == 2000
        assert len(config["patrol_waypoints"]) == 4

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces identical AIConfigDict."""
        original = make_default_ai_config()
        encoded = encode_ai_config(original)
        decoded = decode_ai_config(encoded)
        assert decoded == original

    def test_decode_invalid_waypoints_not_list_raises(self) -> None:
        """Decode rejects non-list patrol_waypoints."""
        data: JSONObject = {
            "fuel_critical_threshold": 200,
            "fuel_low_threshold": 500,
            "fuel_full_threshold": 1200,
            "hunt_min_fuel": 400,
            "combat_range": 20,
            "scan_cooldown_ms": 5000,
            "shoot_cooldown_ms": 2000,
            "patrol_waypoints": "not_a_list",
        }
        with pytest.raises(ValueError, match="must be a list"):
            decode_ai_config(data)

    def test_decode_invalid_waypoint_format_raises(self) -> None:
        """Decode rejects waypoints that are not [x, y] pairs."""
        data: JSONObject = {
            "fuel_critical_threshold": 200,
            "fuel_low_threshold": 500,
            "fuel_full_threshold": 1200,
            "hunt_min_fuel": 400,
            "combat_range": 20,
            "scan_cooldown_ms": 5000,
            "shoot_cooldown_ms": 2000,
            "patrol_waypoints": [[1, 2, 3]],
        }
        with pytest.raises(ValueError, match="must be"):
            decode_ai_config(data)

    def test_decode_invalid_waypoint_type_raises(self) -> None:
        """Decode rejects waypoints with non-int coordinates."""
        data: JSONObject = {
            "fuel_critical_threshold": 200,
            "fuel_low_threshold": 500,
            "fuel_full_threshold": 1200,
            "hunt_min_fuel": 400,
            "combat_range": 20,
            "scan_cooldown_ms": 5000,
            "shoot_cooldown_ms": 2000,
            "patrol_waypoints": [["a", "b"]],
        }
        with pytest.raises(ValueError, match="must be int"):
            decode_ai_config(data)

    def test_decode_missing_field_raises(self) -> None:
        """Decode rejects missing required fields."""
        data: JSONObject = {"fuel_low_threshold": 200}
        with pytest.raises(JSONTypeError):
            decode_ai_config(data)


# =============================================================================
# AIStateDict
# =============================================================================


class TestAIState:
    """Tests for AIStateDict factory and encode/decode."""

    def test_make_initial_ai_state_defaults(self) -> None:
        """Initial state uses default config and PATROL mode."""
        state = make_initial_ai_state()
        assert state["active_mode"] == "PATROL"
        assert state["patrol_waypoint_index"] == 0
        assert state["last_scan_ms"] == 0
        assert state["last_shoot_ms"] == 0
        assert state["combat_target_id"] == -1
        assert state["ticks_in_mode"] == 0
        assert state["config"]["fuel_critical_threshold"] == 200
        assert state["config"]["fuel_low_threshold"] == 500

    def test_make_initial_ai_state_custom_config(self) -> None:
        """Initial state accepts custom config."""
        from tankpit_bot.bot.ai.types import AIConfigDict

        config = make_default_ai_config()
        custom = AIConfigDict(
            fuel_critical_threshold=150,
            fuel_low_threshold=400,
            fuel_full_threshold=config["fuel_full_threshold"],
            hunt_min_fuel=config["hunt_min_fuel"],
            combat_range=config["combat_range"],
            scan_cooldown_ms=config["scan_cooldown_ms"],
            shoot_cooldown_ms=config["shoot_cooldown_ms"],
            patrol_waypoints=config["patrol_waypoints"],
        )
        state = make_initial_ai_state(custom)
        assert state["config"]["fuel_critical_threshold"] == 150
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
            "active_mode": "PATROL",
            "patrol_waypoint_index": 0,
            "last_scan_ms": 0,
            "last_shoot_ms": 0,
            "combat_target_id": -1,
            "combat_target_x": 0,
            "combat_target_y": 0,
            "ticks_in_mode": 0,
        }
        with pytest.raises(ValueError, match="config must be an object"):
            decode_ai_state(data)

    def test_decode_invalid_mode_raises(self) -> None:
        """Decode rejects invalid active_mode."""
        config = encode_ai_config(make_default_ai_config())
        data: JSONObject = {
            "config": config,
            "active_mode": "INVALID",
            "patrol_waypoint_index": 0,
            "last_scan_ms": 0,
            "last_shoot_ms": 0,
            "combat_target_id": -1,
            "combat_target_x": 0,
            "combat_target_y": 0,
            "ticks_in_mode": 0,
        }
        with pytest.raises(ValueError, match="must be one of"):
            decode_ai_state(data)
