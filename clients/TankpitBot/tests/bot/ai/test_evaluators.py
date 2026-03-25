"""Tests for AI behavior evaluators."""

from __future__ import annotations

from tankpit_bot.bot.ai.evaluators import (
    score_collect_equipment,
    score_collect_fuel,
    score_defend,
    score_deposit_fuel,
    score_hunt,
    score_patrol,
    select_best_behavior,
)
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
from tankpit_bot.state.types import (
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
    make_tank_state,
)


def _empty_world() -> WorldStateDict:
    """Create an empty world state."""
    return WorldStateDict(
        self_state=None,
        tanks={},
        containers={},
        mines={},
        terrain={},
        viewport=ViewportStateDict(left=0, top=0, width=18, height=18),
        timestamp_ms=0,
    )


def _self(fuel: int = 800, x: int = 100, y: int = 100) -> SelfStateDict:
    """Create a self state with given fuel."""
    return make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=0,
        rank=4,
        fuel=fuel,
        leaderboard_position=1,
    )


def _ai_state() -> AIStateDict:
    """Create default AI state."""
    return make_initial_ai_state()


# =============================================================================
# score_hunt
# =============================================================================


class TestScoreHunt:
    """Tests for HUNT evaluator."""

    def test_no_enemies(self) -> None:
        """Score is 0 when no enemies exist."""
        world = _empty_world()
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        threats = analyze_threats(world, self_state)
        result = score_hunt(world, self_state, ai_state, threats)
        assert result["mode"] == "HUNT"
        assert result["score"] == 0

    def test_fuel_too_low(self) -> None:
        """Score is 0 when fuel below hunt_min_fuel."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=105,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="enemy",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=100)  # Below hunt_min_fuel=400
        ai_state = _ai_state()
        threats = analyze_threats(world, self_state)
        result = score_hunt(world, self_state, ai_state, threats)
        assert result["score"] == 0
        assert "fuel too low" in result["reason"]

    def test_enemy_out_of_range(self) -> None:
        """Score is 0 when enemies are beyond combat_range."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=200,
            y=200,
            team=1,
            rank=0,
            damage_state=0,
            name="far-enemy",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        threats = analyze_threats(world, self_state)
        result = score_hunt(world, self_state, ai_state, threats)
        assert result["score"] == 0

    def test_enemy_in_range(self) -> None:
        """Score is high when enemy is within combat_range."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=110,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="close-enemy",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        threats = analyze_threats(world, self_state)
        result = score_hunt(world, self_state, ai_state, threats)
        assert result["score"] >= 700
        assert result["target_x"] == 110
        assert result["target_y"] == 100

    def test_damaged_enemy_scores_higher(self) -> None:
        """Damaged enemies increase hunt score."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=110,
            y=100,
            team=1,
            rank=0,
            damage_state=3,
            name="crit-enemy",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        threats = analyze_threats(world, self_state)
        result = score_hunt(world, self_state, ai_state, threats)
        assert result["score"] >= 800  # Higher due to damage bonus


# =============================================================================
# score_collect_fuel
# =============================================================================


class TestScoreCollectFuel:
    """Tests for COLLECT_FUEL evaluator."""

    def test_no_fuel_visible(self) -> None:
        """Score is 0 when no fuel containers exist."""
        world = _empty_world()
        self_state = _self(fuel=100)
        ai_state = _ai_state()
        result = score_collect_fuel(world, self_state, ai_state)
        assert result["score"] == 0

    def test_fuel_full(self) -> None:
        """Score is 0 when fuel is at full threshold."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        self_state = _self(fuel=1200)  # At fuel_full_threshold
        ai_state = _ai_state()
        result = score_collect_fuel(world, self_state, ai_state)
        assert result["score"] == 0

    def test_critical_fuel(self) -> None:
        """Score is emergency when fuel is below critical threshold."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        self_state = _self(fuel=50)  # Below fuel_critical_threshold=200
        ai_state = _ai_state()
        result = score_collect_fuel(world, self_state, ai_state)
        assert result["score"] == 950
        assert result["target_x"] == 110

    def test_critical_fuel_prefers_high_volume(self) -> None:
        """When critical, prefers high-volume container over closer one."""
        world = _empty_world()
        world["containers"]["105,100"] = make_container_state(
            x=105, y=100, is_fuel=True, volume=100
        )
        world["containers"]["150,100"] = make_container_state(
            x=150, y=100, is_fuel=True, volume=1000
        )
        self_state = _self(fuel=50)
        ai_state = _ai_state()
        result = score_collect_fuel(world, self_state, ai_state)
        assert result["score"] == 950
        # Should target the high-volume container at 150
        assert result["target_x"] == 150

    def test_low_fuel_not_critical(self) -> None:
        """Score is high (700-900) when fuel is below low threshold but not critical."""
        world = _empty_world()
        world["containers"]["120,100"] = make_container_state(
            x=120, y=100, is_fuel=True, volume=300
        )
        self_state = _self(fuel=300)  # Between critical=200 and low=500
        ai_state = _ai_state()
        result = score_collect_fuel(world, self_state, ai_state)
        assert 700 <= result["score"] <= 900
        assert result["target_x"] == 120
        assert "fuel low" in result["reason"]

    def test_low_fuel_no_fuel_visible(self) -> None:
        """Score is 0 when fuel is low but no containers visible."""
        world = _empty_world()
        self_state = _self(fuel=300)
        ai_state = _ai_state()
        result = score_collect_fuel(world, self_state, ai_state)
        assert result["score"] == 0

    def test_moderate_fuel(self) -> None:
        """Score scales linearly between low and full thresholds."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        self_state = _self(fuel=700)  # Mid-range
        ai_state = _ai_state()
        result = score_collect_fuel(world, self_state, ai_state)
        assert 200 < result["score"] < 700


# =============================================================================
# score_collect_equipment
# =============================================================================


class TestScoreCollectEquipment:
    """Tests for COLLECT_EQUIPMENT evaluator."""

    def test_no_equipment_visible(self) -> None:
        """Score is 0 when no equipment containers exist."""
        world = _empty_world()
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        result = score_collect_equipment(world, self_state, ai_state)
        assert result["score"] == 0

    def test_fuel_too_low(self) -> None:
        """Score is 0 when fuel is below critical threshold."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=False,
            volume=0,
        )
        self_state = _self(fuel=50)  # Below fuel_critical_threshold=200
        ai_state = _ai_state()
        result = score_collect_equipment(world, self_state, ai_state)
        assert result["score"] == 0

    def test_equipment_nearby(self) -> None:
        """Score is moderate when equipment is nearby and fuel adequate."""
        world = _empty_world()
        world["containers"]["105,100"] = make_container_state(
            x=105,
            y=100,
            is_fuel=False,
            volume=0,
        )
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        result = score_collect_equipment(world, self_state, ai_state)
        assert 400 <= result["score"] <= 600
        assert result["target_x"] == 105


# =============================================================================
# score_deposit_fuel
# =============================================================================


class TestScoreDepositFuel:
    """Tests for DEPOSIT_FUEL evaluator."""

    def test_fuel_not_full(self) -> None:
        """Score is 0 when fuel below full threshold."""
        world = _empty_world()
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        result = score_deposit_fuel(world, self_state, ai_state)
        assert result["score"] == 0

    def test_no_deposit_target(self) -> None:
        """Score is 0 when no fuel container available for deposit."""
        world = _empty_world()
        self_state = _self(fuel=1500)  # Above full threshold
        ai_state = _ai_state()
        result = score_deposit_fuel(world, self_state, ai_state)
        assert result["score"] == 0

    def test_fuel_full_with_target(self) -> None:
        """Score is high when fuel exceeds threshold and deposit available."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        self_state = _self(fuel=1500)
        ai_state = _ai_state()
        result = score_deposit_fuel(world, self_state, ai_state)
        assert result["score"] >= 600
        assert result["target_x"] == 110

    def test_higher_surplus_higher_score(self) -> None:
        """More fuel surplus increases deposit score."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        ai_state = _ai_state()

        result_low = score_deposit_fuel(world, _self(fuel=1250), ai_state)
        result_high = score_deposit_fuel(world, _self(fuel=1800), ai_state)
        assert result_high["score"] > result_low["score"]


# =============================================================================
# score_patrol
# =============================================================================


class TestScorePatrol:
    """Tests for PATROL evaluator."""

    def test_default_score(self) -> None:
        """Patrol always returns a low constant score."""
        ai_state = _ai_state()
        result = score_patrol(ai_state)
        assert result["mode"] == "PATROL"
        assert result["score"] == 100

    def test_targets_current_waypoint(self) -> None:
        """Patrol targets the current waypoint from config."""
        ai_state = _ai_state()
        config = ai_state["config"]
        wx, wy = config["patrol_waypoints"][0]
        result = score_patrol(ai_state)
        assert result["target_x"] == wx
        assert result["target_y"] == wy

    def test_wraps_waypoint_index(self) -> None:
        """Waypoint index wraps around circuit length."""
        ai_state = _ai_state()
        ai_state["patrol_waypoint_index"] = 100  # Way beyond 4 waypoints
        config = ai_state["config"]
        expected_idx = 100 % len(config["patrol_waypoints"])
        wx, wy = config["patrol_waypoints"][expected_idx]
        result = score_patrol(ai_state)
        assert result["target_x"] == wx
        assert result["target_y"] == wy


# =============================================================================
# score_defend
# =============================================================================


class TestScoreDefend:
    """Tests for DEFEND evaluator."""

    def test_no_close_threats(self) -> None:
        """Score is 0 when no enemies within half combat range."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=200,
            y=200,
            team=1,
            rank=0,
            damage_state=0,
            name="far",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=100)
        ai_state = _ai_state()
        threats = analyze_threats(world, self_state)
        result = score_defend(self_state, ai_state, threats)
        assert result["score"] == 0

    def test_close_threat_low_fuel(self) -> None:
        """High score when enemy is very close and fuel is low."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=105,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="close",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=100)  # Below hunt_min_fuel
        ai_state = _ai_state()
        threats = analyze_threats(world, self_state)
        result = score_defend(self_state, ai_state, threats)
        assert result["score"] == 850

    def test_close_threat_adequate_fuel(self) -> None:
        """Moderate score when enemy is close but fuel is adequate."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=105,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="close",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        threats = analyze_threats(world, self_state)
        result = score_defend(self_state, ai_state, threats)
        assert result["score"] == 500


# =============================================================================
# select_best_behavior
# =============================================================================


class TestSelectBestBehavior:
    """Tests for select_best_behavior."""

    def test_patrol_default(self) -> None:
        """Patrol wins when nothing else is active."""
        world = _empty_world()
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        result = select_best_behavior(world, self_state, ai_state)
        assert result["mode"] == "PATROL"

    def test_critical_fuel_overrides_all(self) -> None:
        """COLLECT_FUEL wins when fuel is critical and fuel is visible."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        # Also add an enemy in range
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=105,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="enemy",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=50)  # Critical
        ai_state = _ai_state()
        result = select_best_behavior(world, self_state, ai_state)
        assert result["mode"] == "COLLECT_FUEL"

    def test_hunt_when_enemy_in_range(self) -> None:
        """HUNT wins when enemy is in range and fuel is adequate."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=110,
            y=100,
            team=1,
            rank=0,
            damage_state=2,
            name="damaged",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=800)
        ai_state = _ai_state()
        result = select_best_behavior(world, self_state, ai_state)
        assert result["mode"] == "HUNT"

    def test_deposit_when_fuel_full(self) -> None:
        """DEPOSIT_FUEL wins when fuel is above threshold with deposit target."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        self_state = _self(fuel=1500)
        ai_state = _ai_state()
        result = select_best_behavior(world, self_state, ai_state)
        assert result["mode"] == "DEPOSIT_FUEL"

    def test_defend_low_fuel_close_enemy(self) -> None:
        """DEFEND wins when fuel is low and enemy is very close."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=105,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="close",
            is_bot=True,
            is_self=False,
        )
        self_state = _self(fuel=100)  # Below hunt_min_fuel
        ai_state = _ai_state()
        result = select_best_behavior(world, self_state, ai_state)
        assert result["mode"] == "DEFEND"
