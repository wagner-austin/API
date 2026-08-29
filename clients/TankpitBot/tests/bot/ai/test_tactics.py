"""Tests for AI tactical decision functions."""

from __future__ import annotations

from tankpit_bot.bot.ai.tactics import (
    combat_radar_min,
    compute_desired_equipment,
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


class TestRadarHoardRule:
    """The extra-radar slot follows the hoard rule (2026-08-28)."""

    def test_below_the_bar_outside_hunt_disables_radar(self) -> None:
        """Restock-phase presses must serve the free built-in 5x5.

        The 2026-08-28 validation run burned all 16 gained radars
        mid-restock and never reached the hunt bar (income-burn
        deadlock); below the bar the slot stays off outside HUNT.
        """
        result = compute_desired_equipment("COLLECT", 800, extra_radars_count=19, rank=1)
        assert 5 not in result

    def test_at_the_bar_outside_hunt_enables_radar(self) -> None:
        """Reaching the bar re-arms paid scanning."""
        result = compute_desired_equipment("COLLECT", 800, extra_radars_count=20, rank=1)
        assert 5 in result

    def test_hunt_keeps_radar_enabled_below_the_bar(self) -> None:
        """Combat scanning is what the bar was saved FOR."""
        result = compute_desired_equipment("HUNT", 800, extra_radars_count=3, rank=1)
        assert 5 in result

    def test_the_bar_is_rank_derived(self) -> None:
        """combat_radar_min follows the inventory-capacity ladder."""
        assert combat_radar_min(0) == 15
        assert combat_radar_min(1) == 20
        assert combat_radar_min(8) == 55

    def test_zero_extras_enables_the_free_builtin_scan(self) -> None:
        """At zero stock the slot is on: the free 5x5 only serves enabled.

        The 2026-08-28 trial proved a press with the slot disabled is
        a total no-op (no extras, no fuel, no scan) -- 369 dead
        presses; the free grid-walk half of the doctrine lives at
        exactly zero stock.
        """
        result = compute_desired_equipment("COLLECT", 800, extra_radars_count=0, rank=1)
        assert 5 in result
