"""Tests for AI tick orchestrator."""

from __future__ import annotations

from tankpit_bot.bot.ai.loop import ai_tick
from tankpit_bot.bot.ai.types import make_initial_ai_state
from tankpit_bot.state.types import (
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
    make_tank_state,
)


def _empty_world() -> WorldStateDict:
    """Create empty world state."""
    return WorldStateDict(
        self_state=None,
        tanks={},
        containers={},
        mines={},
        terrain={},
        viewport=ViewportStateDict(left=0, top=0, width=18, height=18),
        timestamp_ms=0,
    )


class TestAITick:
    """Tests for ai_tick integration."""

    def test_returns_all_three_values(self) -> None:
        """ai_tick returns (state, command, behavior) tuple."""
        world = _empty_world()
        self_state = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=4,
            fuel=800,
            leaderboard_position=1,
        )
        ai_state = make_initial_ai_state()

        new_state, _command, behavior = ai_tick(world, self_state, ai_state, 1000)

        assert new_state["active_mode"] == behavior["mode"]
        assert behavior["score"] >= 0

    def test_hunt_fallback_when_nothing_to_do(self) -> None:
        """Fallback is HUNT with score=0 when nothing is happening."""
        world = _empty_world()
        self_state = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=4,
            fuel=800,
            leaderboard_position=1,
        )
        ai_state = make_initial_ai_state()

        new_state, _command, behavior = ai_tick(world, self_state, ai_state, 1000)

        assert behavior["mode"] == "HUNT"
        assert behavior["score"] == 0
        assert new_state["active_mode"] == "HUNT"

    def test_collect_fuel_when_critical(self) -> None:
        """COLLECT_FUEL activates when fuel is critical and container visible."""
        world = _empty_world()
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=50,
        )
        self_state = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=4,
            fuel=50,
            leaderboard_position=1,
        )
        ai_state = make_initial_ai_state()

        _, command, behavior = ai_tick(world, self_state, ai_state, 1000)

        assert behavior["mode"] == "COLLECT_FUEL"
        assert command["cmd_type"] == "pickup_move"
        assert command["target_x"] == 110

    def test_hunt_when_enemy_nearby(self) -> None:
        """HUNT activates when enemy is in range with adequate fuel."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=110,
            y=100,
            team=1,
            rank=0,
            damage_state=2,
            name="enemy",
            is_bot=True,
            is_self=False,
        )
        self_state = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=4,
            fuel=800,
            leaderboard_position=1,
        )
        ai_state = make_initial_ai_state()

        new_state, _, behavior = ai_tick(world, self_state, ai_state, 6000)

        assert behavior["mode"] == "HUNT"
        assert new_state["active_mode"] == "HUNT"

    def test_fuel_priority_over_hunt(self) -> None:
        """COLLECT_FUEL beats HUNT when fuel is low and containers visible."""
        world = _empty_world()
        world["tanks"]["10"] = make_tank_state(
            tank_id=10,
            x=105,
            y=100,
            team=1,
            rank=0,
            damage_state=0,
            name="close-enemy",
            is_bot=True,
            is_self=False,
        )
        world["containers"]["110,100"] = make_container_state(
            x=110,
            y=100,
            is_fuel=True,
            volume=500,
        )
        self_state = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=4,
            fuel=300,
            leaderboard_position=1,
        )
        ai_state = make_initial_ai_state()

        _, _, behavior = ai_tick(world, self_state, ai_state, 5000)

        # Fuel collection wins over hunt when fuel is low
        assert behavior["mode"] == "COLLECT_FUEL"

    def test_sequential_ticks_accumulate(self) -> None:
        """Multiple ticks accumulate ticks_in_mode."""
        world = _empty_world()
        self_state = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=4,
            fuel=800,
            leaderboard_position=1,
        )
        ai_state = make_initial_ai_state()

        state1, _, _ = ai_tick(world, self_state, ai_state, 1000)
        state2, _, _ = ai_tick(world, self_state, state1, 2000)
        state3, _, _ = ai_tick(world, self_state, state2, 3000)

        assert state3["ticks_in_mode"] == 3  # tick1=0→1, tick2=1→2, tick3=2→3
