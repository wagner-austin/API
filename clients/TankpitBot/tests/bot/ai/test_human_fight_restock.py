"""Partial mid-fight restock bar for held human fights (2026-07-31 ruling)."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.mode_gates import (
    held_human_combat_lock,
    human_fight_resume_fuel_floor,
    human_fight_resume_permitted,
    should_enter_hunt,
    should_exit_collect,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.state.types import TankStateDict
from tankpit_bot.types.modes import AIMode
from tests.bot.ai._support import make_enemy_tank, make_scanned_ai_state, make_world


def _inventory(*, duals: int, homings: int, radars: int) -> InventoryState:
    return InventoryState(
        armor_shields=InventoryItem(count=0, enabled=False),
        dual_shots=InventoryItem(count=duals, enabled=True),
        missile_shots=InventoryItem(count=0, enabled=False),
        homing_shots=InventoryItem(count=homings, enabled=True),
        extra_radars=InventoryItem(count=radars, enabled=True),
    )


def _locked_ctx(
    *,
    fuel: int,
    inventory: InventoryState,
    target_name: str = "Yuppler",
    target_present: bool = True,
    wind_down: bool = False,
    mode: AIMode = "COLLECT",
) -> DecideCtx:
    tanks: dict[str, TankStateDict] = {}
    if target_present:
        tanks["50"] = make_enemy_tank(tank_id=50, x=150, y=150, name=target_name)
    world, self_state = make_world(fuel=fuel, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": mode,
            "mode_state": "",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": 150,
            "combat_target_y": 150,
            "wind_down": wind_down,
        }
    )
    return DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")


# The fixture self tank is rank 2: capacity 1200, inventory cap 30.
# Resume bar: fuel >= 750, duals/homings >= 15 (half cap), radars >= 10
# (min(combat_radar_min 25, 2 x radar_break 5)).


def test_resume_fuel_floor_is_750_at_rank_two_defaults() -> None:
    ctx = _locked_ctx(fuel=800, inventory=_inventory(duals=15, homings=15, radars=10))
    assert human_fight_resume_fuel_floor(ctx) == 750


def test_partial_bar_exits_collect_during_a_held_human_fight() -> None:
    """Fuel 750 + half-cap weapons + 10 radars resumes -- far below full."""
    ctx = _locked_ctx(fuel=750, inventory=_inventory(duals=15, homings=15, radars=10))

    assert human_fight_resume_permitted(ctx) is True
    assert should_exit_collect(ctx) is True


def test_partial_bar_holds_below_the_weapon_floor() -> None:
    ctx = _locked_ctx(fuel=750, inventory=_inventory(duals=14, homings=15, radars=10))

    assert human_fight_resume_permitted(ctx) is False
    assert should_exit_collect(ctx) is False


def test_partial_bar_holds_below_the_homing_floor() -> None:
    ctx = _locked_ctx(fuel=750, inventory=_inventory(duals=15, homings=14, radars=10))

    assert human_fight_resume_permitted(ctx) is False


def test_partial_bar_holds_below_the_radar_floor() -> None:
    """Radars resume at 10 -- twice the break of 5, never a zero-width band."""
    ctx = _locked_ctx(fuel=750, inventory=_inventory(duals=15, homings=15, radars=9))

    assert human_fight_resume_permitted(ctx) is False


def test_partial_bar_holds_below_the_resume_fuel_floor() -> None:
    ctx = _locked_ctx(fuel=749, inventory=_inventory(duals=15, homings=15, radars=10))

    assert human_fight_resume_permitted(ctx) is False


def test_bot_fight_keeps_the_full_restock_bar() -> None:
    """A practice-bot lock at partial stock stays in COLLECT (full bar)."""
    ctx = _locked_ctx(
        fuel=750,
        inventory=_inventory(duals=15, homings=15, radars=10),
        target_name="red-7",
    )

    assert held_human_combat_lock(ctx) is False
    assert should_exit_collect(ctx) is False


def test_vanished_target_keeps_the_full_restock_bar() -> None:
    """A lock whose target left the game reads as no human lock."""
    ctx = _locked_ctx(
        fuel=750,
        inventory=_inventory(duals=15, homings=15, radars=10),
        target_present=False,
    )

    assert held_human_combat_lock(ctx) is False
    assert should_exit_collect(ctx) is False


def test_wind_down_keeps_the_full_bar_even_mid_human_fight() -> None:
    """session_complete must leave a fully stocked tank."""
    ctx = _locked_ctx(
        fuel=750,
        inventory=_inventory(duals=15, homings=15, radars=10),
        wind_down=True,
    )

    assert should_exit_collect(ctx) is False


def test_hunt_reenters_at_the_partial_bar_with_a_held_human_lock() -> None:
    """Without this override the partial COLLECT exit deadlocks arbitration."""
    ctx = _locked_ctx(fuel=750, inventory=_inventory(duals=15, homings=15, radars=10))

    assert should_enter_hunt(ctx) is True


def test_hunt_entry_stays_full_bar_without_a_held_lock() -> None:
    world, self_state = make_world(fuel=750)
    ai_state = make_scanned_ai_state()
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        _inventory(duals=15, homings=15, radars=10),
        100000,
        None,
        "",
    )

    assert should_enter_hunt(ctx) is False


def test_weapon_emergency_still_vetoes_the_partial_reentry() -> None:
    """Duals below the break threshold keep COLLECT ownership."""
    ctx = _locked_ctx(fuel=750, inventory=_inventory(duals=3, homings=15, radars=10))

    assert should_enter_hunt(ctx) is False


def test_owner_selection_hands_the_tick_back_to_hunt_at_the_partial_bar() -> None:
    """End-to-end: a COLLECT owner mid-human-fight flips to HUNT once resumable."""
    from tankpit_bot.bot.ai_strategy import _select_owner_mode

    ctx = _locked_ctx(
        fuel=750,
        inventory=_inventory(duals=15, homings=15, radars=10),
        mode="COLLECT",
    )

    assert _select_owner_mode(ctx) == "HUNT"
