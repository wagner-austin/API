"""Tests for AI tactical decision functions."""

from __future__ import annotations

from tankpit_bot.bot.ai.tactics import (
    compute_desired_equipment,
    find_teleport_target,
    should_proactive_radar,
    should_teleport_search,
)
from tankpit_bot.bot.ai.types import make_behavior_score, make_default_ai_config
from tankpit_bot.state.types import (
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
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


def _self(x: int = 100, y: int = 100) -> SelfStateDict:
    """Create a self state at given position."""
    return make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=0,
        rank=4,
        fuel=800,
        leaderboard_position=1,
    )


# =============================================================================
# should_proactive_radar
# =============================================================================


class TestShouldProactiveRadar:
    """Tests for proactive radar triggering."""

    def test_fuel_above_threshold_returns_false(self) -> None:
        """High fuel never triggers proactive radar."""
        config = make_default_ai_config()
        world = _empty_world()
        # fuel_low_threshold=500, buffer=200 → cutoff at 700
        result = should_proactive_radar(800, world, 0, 10000, config)
        assert result is False

    def test_fuel_near_low_no_fuel_visible_triggers(self) -> None:
        """Fuel near low threshold with no visible fuel triggers radar."""
        config = make_default_ai_config()
        world = _empty_world()
        # fuel=600 < 500+200=700, no containers, cooldown elapsed
        result = should_proactive_radar(600, world, 0, 10000, config)
        assert result is True

    def test_fuel_visible_blocks_radar(self) -> None:
        """Visible fuel containers prevent proactive radar."""
        config = make_default_ai_config()
        world = _empty_world()
        world["containers"]["50,50"] = make_container_state(50, 50, is_fuel=True, volume=500)
        result = should_proactive_radar(600, world, 0, 10000, config)
        assert result is False

    def test_scan_cooldown_blocks_radar(self) -> None:
        """Recent scan prevents proactive radar."""
        config = make_default_ai_config()
        world = _empty_world()
        # Scanned at 8000, now=10000 → age=2000 < cooldown=5000
        result = should_proactive_radar(600, world, 8000, 10000, config)
        assert result is False

    def test_equipment_container_does_not_count_as_fuel(self) -> None:
        """Equipment containers do not satisfy the fuel check."""
        config = make_default_ai_config()
        world = _empty_world()
        world["containers"]["50,50"] = make_container_state(50, 50, is_fuel=False, volume=0)
        result = should_proactive_radar(600, world, 0, 10000, config)
        assert result is True


# =============================================================================
# should_teleport_search
# =============================================================================


class TestShouldTeleportSearch:
    """Tests for teleport search triggering."""

    def test_fuel_above_threshold_returns_false(self) -> None:
        """High fuel never triggers teleport search."""
        config = make_default_ai_config()
        world = _empty_world()
        behavior = make_behavior_score("PATROL", 50, 64, 64, "patrol")
        result = should_teleport_search(behavior, 800, world, 0, 10000, config)
        assert result is False

    def test_containers_present_returns_false(self) -> None:
        """Containers in area prevent teleport search."""
        config = make_default_ai_config()
        world = _empty_world()
        world["containers"]["50,50"] = make_container_state(50, 50, is_fuel=True, volume=500)
        behavior = make_behavior_score("PATROL", 50, 64, 64, "patrol")
        result = should_teleport_search(behavior, 300, world, 9000, 10000, config)
        assert result is False

    def test_scan_too_old_returns_false(self) -> None:
        """Stale scan prevents teleport (area not confirmed empty)."""
        config = make_default_ai_config()
        world = _empty_world()
        behavior = make_behavior_score("PATROL", 50, 64, 64, "patrol")
        # Scanned at 0, now=10000 → age=10000 >= cooldown=5000
        result = should_teleport_search(behavior, 300, world, 0, 10000, config)
        assert result is False

    def test_high_priority_behavior_returns_false(self) -> None:
        """High-score behavior prevents teleport override."""
        config = make_default_ai_config()
        world = _empty_world()
        behavior = make_behavior_score("HUNT", 500, 110, 100, "enemy nearby")
        # Recent scan, low fuel, no containers — but behavior score too high
        result = should_teleport_search(behavior, 300, world, 9000, 10000, config)
        assert result is False

    def test_all_conditions_met_triggers(self) -> None:
        """All conditions met triggers teleport search."""
        config = make_default_ai_config()
        world = _empty_world()
        behavior = make_behavior_score("PATROL", 50, 64, 64, "patrol")
        # Low fuel, empty area, recent scan, low-priority behavior
        result = should_teleport_search(behavior, 300, world, 9000, 10000, config)
        assert result is True


# =============================================================================
# find_teleport_target
# =============================================================================


class TestFindTeleportTarget:
    """Tests for teleport target selection."""

    def test_picks_farthest_waypoint(self) -> None:
        """Selects the farthest waypoint from current position."""
        config = make_default_ai_config()
        # Default waypoints: (64,64), (192,64), (192,192), (64,192)
        self_state = _self(x=60, y=60)
        tx, ty = find_teleport_target(config, self_state)
        # Farthest from (60,60) is (192,192) — distance 264
        assert (tx, ty) == (192, 192)

    def test_picks_farthest_from_opposite_corner(self) -> None:
        """Picks farthest from opposite starting corner."""
        config = make_default_ai_config()
        self_state = _self(x=190, y=190)
        tx, ty = find_teleport_target(config, self_state)
        # Farthest from (190,190) is (64,64) — distance 252
        assert (tx, ty) == (64, 64)

    def test_single_waypoint_returns_it(self) -> None:
        """Single waypoint is always returned."""
        from tankpit_bot.bot.ai.types import AIConfigDict

        config = AIConfigDict(
            fuel_critical_threshold=200,
            fuel_low_threshold=500,
            fuel_full_threshold=1200,
            hunt_min_fuel=400,
            combat_range=20,
            scan_cooldown_ms=5000,
            shoot_cooldown_ms=2000,
            patrol_waypoints=[(128, 128)],
        )
        self_state = _self(x=10, y=10)
        tx, ty = find_teleport_target(config, self_state)
        assert (tx, ty) == (128, 128)


# =============================================================================
# compute_desired_equipment
# =============================================================================


class TestComputeDesiredEquipment:
    """Tests for equipment desired-set computation."""

    def test_patrol_only_radar(self) -> None:
        """PATROL mode only needs radar."""
        result = compute_desired_equipment("PATROL", 800, 0, 200)
        assert result == {5}

    def test_hunt_healthy_dual_and_radar(self) -> None:
        """HUNT with healthy enemy needs radar and dual."""
        result = compute_desired_equipment("HUNT", 800, 0, 200)
        assert result == {2, 5}

    def test_hunt_critical_adds_homing(self) -> None:
        """HUNT with critically damaged enemy adds homing."""
        result = compute_desired_equipment("HUNT", 800, 3, 200)
        assert result == {2, 4, 5}

    def test_defend_enables_shields(self) -> None:
        """DEFEND mode enables shields and radar."""
        result = compute_desired_equipment("DEFEND", 800, 0, 200)
        assert result == {1, 5}

    def test_collect_fuel_critical_enables_shields(self) -> None:
        """COLLECT_FUEL with critical fuel enables shields."""
        result = compute_desired_equipment("COLLECT_FUEL", 100, 0, 200)
        assert result == {1, 5}

    def test_collect_fuel_normal_no_shields(self) -> None:
        """COLLECT_FUEL with adequate fuel has no shields."""
        result = compute_desired_equipment("COLLECT_FUEL", 400, 0, 200)
        assert result == {5}

    def test_teleport_always_enables_shields(self) -> None:
        """Teleport flag enables shields regardless of mode."""
        result = compute_desired_equipment("PATROL", 800, 0, 200, is_teleport=True)
        assert result == {1, 5}

    def test_hunt_damage_below_critical_no_homing(self) -> None:
        """HUNT with damage_state=2 does not enable homing."""
        result = compute_desired_equipment("HUNT", 800, 2, 200)
        assert result == {2, 5}
