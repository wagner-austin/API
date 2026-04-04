"""Tests for ai.combat_strategy — combat decision coverage gaps."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import try_combat
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import (
    make_initial_ai_state,
)
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import SelfStateDict, TankStateDict, WorldStateDict


def _make_inventory(dual_count: int = 30, default_count: int = 30) -> InventoryState:
    """Build an inventory with configurable dual count."""
    item = InventoryItem(count=default_count, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=InventoryItem(count=dual_count, enabled=True),
        missile_shots=item,
        homing_shots=item,
        extra_radars=item,
    )


def _make_world(
    fuel: int = 800,
    tanks: dict[str, TankStateDict] | None = None,
) -> tuple[WorldStateDict, SelfStateDict]:
    """Build a minimal world state."""
    self_state = SelfStateDict(
        tank_id=1,
        x=100,
        y=100,
        team=1,
        rank=0,
        fuel=fuel,
        leaderboard_position=0,
    )
    return (
        WorldStateDict(
            self_state=self_state,
            tanks=tanks or {},
            containers={},
            mines={},
            terrain={},
            viewport={"left": 92, "top": 92, "width": 16, "height": 16},
            scanned_viewports={"92,92": 100000},
            timestamp_ms=100000,
        ),
        self_state,
    )


class TestTryCombatLowFuel:
    """Tests for try_combat returning None on low fuel without active combat."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_low_fuel_no_combat_returns_none(self) -> None:
        """Low fuel with no active combat returns None."""
        world, self_state = _make_world(fuel=300)
        ai_state = make_initial_ai_state()
        inventory = _make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = try_combat(ctx)

        assert result is None


class TestTryCombatEquipmentReserve:
    """Tests for try_combat refusing new fights when equipment reserve low."""

    def setup_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state."""
        reset_world_state()

    def test_new_fight_blocked_by_low_equipment_reserve(self) -> None:
        """New fights blocked when equipment reserve is below resume threshold."""
        tanks: dict[str, TankStateDict] = {
            "50": TankStateDict(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = _make_world(fuel=800, tanks=tanks)
        ai_state = make_initial_ai_state()
        inventory = _make_inventory(dual_count=10)

        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

        result = try_combat(ctx)

        assert result is None
