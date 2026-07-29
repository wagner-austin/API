"""Tests for human classification and target priority."""

from __future__ import annotations

from tankpit_bot.bot.ai.humans import (
    PRIORITY_BOT,
    PRIORITY_HUMAN,
    PRIORITY_NAMED,
    is_human_name,
    is_practice_bot_name,
    threat_priority_tier,
)
from tankpit_bot.bot.ai.threats import find_acquisition_target
from tankpit_bot.state.types import (
    SelfStateDict,
    TankStateDict,
    make_tank_state,
)
from tests.bot.ai._support import make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_practice_bot_names_match_the_color_number_pattern() -> None:
    """Every corpus bot name shape classifies as bot; humans never do."""
    assert is_practice_bot_name("orange-1") is True
    assert is_practice_bot_name("red-6") is True
    assert is_practice_bot_name("purple-9") is True
    assert is_practice_bot_name("blue-12") is True
    assert is_practice_bot_name("guest") is False
    assert is_practice_bot_name("Yuppler") is False
    assert is_practice_bot_name("orange") is False
    assert is_practice_bot_name("orange-") is False
    assert is_practice_bot_name("green-1") is False


def test_human_name_requires_a_non_empty_non_bot_name() -> None:
    """Unsynced empty names are unknown, never human (no phantom chases)."""
    assert is_human_name("guest") is True
    assert is_human_name("Yuppler") is True
    assert is_human_name("green-1") is True
    assert is_human_name("orange-1") is False
    assert is_human_name("") is False


def test_priority_tiers_order_named_over_human_over_bot() -> None:
    """The named account outranks other humans; humans outrank bots."""
    assert threat_priority_tier("Yuppler", "yuppler") == PRIORITY_NAMED
    assert threat_priority_tier("guest", "yuppler") == PRIORITY_HUMAN
    assert threat_priority_tier("orange-1", "yuppler") == PRIORITY_BOT
    assert threat_priority_tier("", "yuppler") == PRIORITY_BOT
    assert threat_priority_tier("Yuppler", "") == PRIORITY_HUMAN


def _map_fresh_enemy(
    tank_id: int,
    x: int,
    y: int,
    name: str,
) -> TankStateDict:
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=3,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
    )


def _acquire(
    tanks: dict[str, TankStateDict],
    *,
    priority_target_name: str = "",
) -> tuple[SelfStateDict, str]:
    world, self_state = make_world(fuel=1100, tanks=tanks)
    winner = find_acquisition_target(
        world,
        self_state,
        {},
        {},
        InMemoryTerrainMap(),
        100000,
        5000,
        engagement_reserve_fuel=650,
        priority_target_name=priority_target_name,
    )
    if winner is None:
        raise AssertionError("expected an acquisition winner")
    return self_state, winner["name"]


def test_acquisition_prefers_a_distant_human_over_a_near_bot() -> None:
    """User doctrine 2026-07-28: any human who logs in outranks bots."""
    tanks = {
        "50": _map_fresh_enemy(50, 105, 100, "orange-1"),
        "2627": _map_fresh_enemy(2627, 160, 100, "guest"),
    }
    _, picked = _acquire(tanks)
    assert picked == "guest"


def test_acquisition_prefers_the_named_account_over_another_human() -> None:
    """The configured priority target outranks even other humans."""
    tanks = {
        "2627": _map_fresh_enemy(2627, 110, 100, "guest"),
        "2700": _map_fresh_enemy(2700, 170, 100, "Yuppler"),
    }
    _, picked = _acquire(tanks, priority_target_name="yuppler")
    assert picked == "Yuppler"


def test_acquisition_stays_nearest_first_among_bots() -> None:
    """With no humans present, farming order is unchanged."""
    tanks = {
        "50": _map_fresh_enemy(50, 105, 100, "orange-1"),
        "51": _map_fresh_enemy(51, 140, 100, "red-6"),
    }
    _, picked = _acquire(tanks)
    assert picked == "orange-1"
