"""Tests for AI tactical decision functions."""

from __future__ import annotations

from tankpit_bot.bot.ai.tactics import (
    compute_desired_equipment,
    should_map_open_for_enemies,
    should_proactive_radar,
)
from tankpit_bot.bot.ai.types import make_default_ai_config
from tankpit_bot.state.types import (
    SelfStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
    make_tank_state,
    make_viewport_state,
)


def _empty_world() -> WorldStateDict:
    """Create an empty world state."""
    return WorldStateDict(
        self_state=None,
        tanks={},
        containers={},
        mines={},
        terrain={},
        viewport=make_viewport_state(left=0, top=0, width=18, height=18),
        scanned_tiles={},
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
    """Tests for proactive radar triggering (fuel discovery only)."""

    def test_fuel_above_threshold_returns_false(self) -> None:
        """High fuel never triggers proactive radar."""
        config = make_default_ai_config()
        world = _empty_world()
        result = should_proactive_radar(500, world, 0, 10000, config)
        assert result is False

    def test_fuel_at_threshold_triggers(self) -> None:
        """Fuel at low threshold with no containers triggers radar."""
        config = make_default_ai_config()
        world = _empty_world()
        result = should_proactive_radar(150, world, 0, 10000, config)
        assert result is True

    def test_fuel_below_threshold_triggers(self) -> None:
        """Fuel below low threshold with no containers triggers radar."""
        config = make_default_ai_config()
        world = _empty_world()
        result = should_proactive_radar(150, world, 0, 10000, config)
        assert result is True

    def test_containers_visible_blocks_radar(self) -> None:
        """Any visible container blocks radar (collect first)."""
        config = make_default_ai_config()
        world = _empty_world()
        world["containers"]["5,5"] = make_container_state(5, 5, is_fuel=True, volume=500)
        result = should_proactive_radar(150, world, 0, 10000, config)
        assert result is False

    def test_equipment_container_does_not_block_radar(self) -> None:
        """Equipment containers do NOT block radar — only fuel containers do."""
        config = make_default_ai_config()
        world = _empty_world()
        world["containers"]["50,50"] = make_container_state(50, 50, is_fuel=False, volume=0)
        result = should_proactive_radar(150, world, 0, 10000, config)
        assert result is True

    def test_off_viewport_fuel_container_does_not_block_radar(self) -> None:
        """Remembered off-viewport fuel does not count as visible fuel."""
        config = make_default_ai_config()
        world = _empty_world()
        world["containers"]["50,50"] = make_container_state(50, 50, is_fuel=True, volume=500)
        world["viewport"] = make_viewport_state(left=100, top=100, width=18, height=18)

        result = should_proactive_radar(150, world, 0, 10000, config)

        assert result is True

    def test_scan_cooldown_blocks_radar(self) -> None:
        """Recent scan prevents proactive radar."""
        config = make_default_ai_config()
        world = _empty_world()
        result = should_proactive_radar(150, world, 8000, 10000, config)
        assert result is False

    def test_skips_self_and_invalidated_enemies(self) -> None:
        """Self, same-team, and origin-position tanks are skipped."""
        config = make_default_ai_config()
        world = _empty_world()
        world["self_state"] = _self()
        # Self tank — skipped by is_self check
        world["tanks"]["1"] = make_tank_state(
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=4,
            name="Me",
            is_self=True,
            is_bot=False,
            damage_state=0,
            timestamp_ms=0,
        )
        # Same-team tank — skipped by team check
        world["tanks"]["2"] = make_tank_state(
            tank_id=2,
            x=120,
            y=120,
            team=0,
            rank=2,
            name="Ally",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=0,
        )
        # Enemy at (0,0) — skipped as invalidated position
        world["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=0,
            y=0,
            team=2,
            rank=1,
            name="Dead",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=0,
        )
        # No live visible enemies → radar triggers
        result = should_proactive_radar(150, world, 0, 10000, config)
        assert result is True


# =============================================================================
# should_map_open_for_enemies
# =============================================================================


class TestShouldMapOpenForEnemies:
    """Tests for map open enemy discovery triggering."""

    def test_no_enemies_visible_triggers(self) -> None:
        """Triggers when no live enemies are visible and cooldown elapsed."""
        config = make_default_ai_config()
        world = _empty_world()
        self_state = _self()
        # No tanks, cooldown elapsed (last_map_open=0, now=10000)
        result = should_map_open_for_enemies(world, self_state, 0, 10000, config)
        assert result is True

    def test_cooldown_blocks(self) -> None:
        """Recent map open prevents triggering."""
        config = make_default_ai_config()
        world = _empty_world()
        self_state = _self()
        # last_map_open=8000, now=10000 → age=2000 < cooldown=5000
        result = should_map_open_for_enemies(world, self_state, 8000, 10000, config)
        assert result is False

    def test_live_enemy_visible_blocks(self) -> None:
        """Live enemy tank visible prevents triggering."""
        config = make_default_ai_config()
        world = _empty_world()
        world["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=5,
            y=5,
            team=2,
            rank=1,
            name="Enemy",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=0,
        )
        self_state = _self()
        result = should_map_open_for_enemies(world, self_state, 0, 10000, config)
        assert result is False

    def test_off_viewport_enemy_does_not_block(self) -> None:
        """Remembered off-viewport enemies do not count as visible enemies."""
        config = make_default_ai_config()
        world = _empty_world()
        world["viewport"] = make_viewport_state(left=100, top=100, width=18, height=18)
        world["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=20,
            y=20,
            team=2,
            rank=1,
            name="Enemy",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=0,
        )
        self_state = _self()

        result = should_map_open_for_enemies(world, self_state, 0, 10000, config)

        assert result is True

    def test_dead_enemy_at_origin_ignored(self) -> None:
        """Dead enemy at (0,0) does not count as visible."""
        config = make_default_ai_config()
        world = _empty_world()
        world["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=0,
            y=0,
            team=2,
            rank=1,
            name="DeadEnemy",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=0,
        )
        self_state = _self()
        result = should_map_open_for_enemies(world, self_state, 0, 10000, config)
        assert result is True

    def test_teammate_ignored(self) -> None:
        """Teammates do not count as visible enemies."""
        config = make_default_ai_config()
        world = _empty_world()
        # Same team (team=0) as _self()
        world["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=105,
            y=105,
            team=0,
            rank=1,
            name="Ally",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=0,
        )
        self_state = _self()
        result = should_map_open_for_enemies(world, self_state, 0, 10000, config)
        assert result is True

    def test_self_tank_ignored(self) -> None:
        """Self tank does not count as visible enemy."""
        config = make_default_ai_config()
        world = _empty_world()
        world["tanks"]["1"] = make_tank_state(
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=4,
            name="Self",
            is_self=True,
            is_bot=False,
            damage_state=0,
            timestamp_ms=0,
        )
        self_state = _self()
        result = should_map_open_for_enemies(world, self_state, 0, 10000, config)
        assert result is True


# =============================================================================
# compute_desired_equipment
# =============================================================================


class TestComputeDesiredEquipment:
    """Tests for equipment desired-set computation."""

    def test_patrol_dual_and_radar(self) -> None:
        """PATROL mode has dual, homing, and radar always on."""
        result = compute_desired_equipment("PATROL", 800)
        assert result == {2, 4, 5}

    def test_hunt_dual_and_radar(self) -> None:
        """HUNT mode has dual, homing, and radar."""
        result = compute_desired_equipment("HUNT", 800)
        assert result == {2, 4, 5}

    def test_defend_dual_and_radar(self) -> None:
        """DEFEND mode has dual, homing, and radar (no shields)."""
        result = compute_desired_equipment("DEFEND", 800)
        assert result == {2, 4, 5}

    def test_collect_fuel_dual_and_radar(self) -> None:
        """COLLECT has dual, homing, and radar regardless of fuel level."""
        result = compute_desired_equipment("COLLECT", 100)
        assert result == {2, 4, 5}

    def test_dual_shots_depleted_drops_dual(self) -> None:
        """When dual shots count=0, slot 2 is not included."""
        result = compute_desired_equipment("HUNT", 800, dual_shots_count=0)
        assert result == {4, 5}

    def test_homing_shots_depleted_drops_homing(self) -> None:
        """When homing shots count=0, slot 4 is not included."""
        result = compute_desired_equipment("HUNT", 800, homing_shots_count=0)
        assert result == {2, 5}

    def test_no_shields_ever(self) -> None:
        """Shields are never included in desired equipment."""
        result = compute_desired_equipment("HUNT", 800)
        assert 1 not in result
