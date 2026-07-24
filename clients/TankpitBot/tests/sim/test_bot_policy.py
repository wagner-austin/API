"""The archive-mined practice-bot policy ([[enemy-bot-behavior]])."""

from __future__ import annotations

from tankpit_bot.sim.bot_policy import (
    BOT_TELEPORT_OFF_HITS,
    decide_practice_bot,
    make_practice_bot_state,
    note_hit_on_bot,
    teleport_off_threshold,
)
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

BOT_ID = 4


def _arena(rank: int = 0) -> SimWorldDict:
    """One practice bot at (100, 100)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][BOT_ID] = make_sim_tank(BOT_ID, 1, rank, 100, 100, 800)
    return world


def test_calm_bot_holds_still() -> None:
    """The mined default: no hit, no pending return -> no command."""
    world = _arena()
    state = make_practice_bot_state()
    assert decide_practice_bot(state, world, InMemoryTerrainMap(), BOT_ID) is None


def test_dead_bot_never_acts() -> None:
    world = _arena()
    world["tanks"][BOT_ID]["alive"] = False
    state = make_practice_bot_state()
    note_hit_on_bot(state, 101, 100)
    assert decide_practice_bot(state, world, InMemoryTerrainMap(), BOT_ID) is None


def test_hit_queues_one_return_single_at_the_attacker_tile() -> None:
    """The mined law: one next-tick single at the attacker's tile."""
    world = _arena()
    state = make_practice_bot_state()
    note_hit_on_bot(state, 101, 100)
    command = decide_practice_bot(state, world, InMemoryTerrainMap(), BOT_ID)
    if command is None:
        raise AssertionError("a hit bot must return fire")
    assert (command["kind"], command["x"], command["y"]) == ("shoot", 101, 100)
    assert decide_practice_bot(state, world, InMemoryTerrainMap(), BOT_ID) is None


def test_threshold_reached_teleports_off_and_resets() -> None:
    """At the rank threshold the bot escapes beyond the viewport."""
    world = _arena(rank=0)
    state = make_practice_bot_state()
    for _ in range(BOT_TELEPORT_OFF_HITS[0]):
        note_hit_on_bot(state, 101, 100)
    command = decide_practice_bot(state, world, InMemoryTerrainMap(), BOT_ID)
    if command is None:
        raise AssertionError("a bot over threshold must teleport off")
    assert command["kind"] == "teleport"
    distance = max(abs(command["x"] - 100), abs(command["y"] - 100))
    assert distance >= 12
    assert (state["hits_taken"], state["has_pending_return"]) == (0, False)


def test_sealed_band_falls_back_to_the_pending_return() -> None:
    """No open escape tile -> the queued single still fires."""
    world = _arena(rank=0)
    state = make_practice_bot_state()
    for _ in range(BOT_TELEPORT_OFF_HITS[0]):
        note_hit_on_bot(state, 99, 100)
    sealed = InMemoryTerrainMap(default="#")
    command = decide_practice_bot(state, world, sealed, BOT_ID)
    if command is None:
        raise AssertionError("the pending return must fire when escape is sealed")
    assert (command["kind"], command["x"], command["y"]) == ("shoot", 99, 100)
    assert state["hits_taken"] == BOT_TELEPORT_OFF_HITS[0]


def test_thresholds_follow_the_mined_table() -> None:
    """7 recruit / 8 private (archive modes); 9 corporal (guide);
    unknown ranks fall back to the highest known row."""
    assert teleport_off_threshold(0) == 7
    assert teleport_off_threshold(1) == 8
    assert teleport_off_threshold(2) == 9
    assert teleport_off_threshold(8) == 9
