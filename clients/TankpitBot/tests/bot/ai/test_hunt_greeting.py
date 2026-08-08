"""The greeting approach of the human-consent contract."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.hunt_mode import decide_hunt_mode
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict
from tests.bot.ai._support import (
    consent_human,
    make_inventory,
    make_map_known_enemy,
    make_scanned_ai_state,
    make_world,
)


def test_greeting_approach_lands_a_few_tiles_off_an_unconsented_human() -> None:
    """The greet visit teleports to the stand-off band, no lock taken.

    User ruling 2026-07-30: "make sure we teleport to them. and that
    we've said hello first ... we want to see them. and not an
    adjacent teleport. a few tiles off." The approach fires for a
    map-fresh unconsented human before any bot farming; the HELLO
    itself attaches on arrival via the viewport-encounter greeting.
    """
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_map_known_enemy(tank_id=90, x=140, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "greet_approach"
    ring = abs(decision["command"]["target_x"] - 140) + abs(decision["command"]["target_y"] - 100)
    assert 5 <= ring <= 7
    assert decision["updated_ai_state"]["combat_target_id"] == -1


def test_visited_unconsented_human_is_left_alone() -> None:
    """After the stand-off visit the latch stops re-approaching.

    The visit map is ``visited_tank_ids`` — deliberately NOT the
    HELLO latch (user ruling 2026-07-31: the hello can fire from
    anywhere; only the visit is one-per-human proximity work), so an
    early long-range HELLO can never cancel the visit.
    """
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_map_known_enemy(tank_id=90, x=140, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
            "visited_tank_ids": {"90": 90000},
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["behavior"]["reason_kind"] != "greet_approach"
    assert decision["behavior"]["reason_context"].get("target_name") != "Yuppler"


def test_greeting_approach_declines_when_teleport_unaffordable() -> None:
    """A greet visit the tank cannot pay for is skipped, not forced."""
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=101, y=100, name="red-5"),
        "90": make_map_known_enemy(tank_id=90, x=240, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=60, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    # At fuel 60 nothing is affordable: the greet visit is skipped
    # (never forced) and the acquire path exits no_viable_targets.
    with pytest.raises(SessionExitError) as exc_info:
        decide_hunt_mode(ctx)

    assert exc_info.value.reason == "no_viable_targets"


def test_greeting_scan_ignores_map_stale_humans() -> None:
    """A stale human gets no blind visit; the nearest fresh one wins.

    The second fresh human sits farther than the first so the scan's
    keep-the-nearest branch is exercised both ways.
    """
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    ws = WorldService()
    tanks: dict[str, TankStateDict] = {
        "60": make_map_known_enemy(tank_id=60, x=115, y=100, name="red-5"),
        "90": make_map_known_enemy(tank_id=90, x=140, y=100, name="Yuppler", timestamp_ms=10),
        "91": make_map_known_enemy(tank_id=91, x=130, y=100, name="guest"),
        "92": make_map_known_enemy(tank_id=92, x=200, y=100, name="Visitor"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["behavior"]["reason_kind"] == "greet_approach"
    assert decision["behavior"]["reason_context"]["target_name"] == "guest"


def test_consented_human_map_winner_is_teleport_acquired() -> None:
    """A consented human as the map winner takes the normal acquire path."""
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    ws = WorldService()
    consent_human(ws, 90)
    tanks: dict[str, TankStateDict] = {
        "90": make_map_known_enemy(tank_id=90, x=140, y=100, name="Yuppler"),
    }
    world, self_state = make_world(fuel=1100, tanks=tanks)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ACQUIRE",
            "mode_started_ms": 90000,
            "last_map_open_ms": 99500,
        }
    )
    ctx = DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    decision = decide_hunt_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["updated_ai_state"]["combat_target_id"] == 90
