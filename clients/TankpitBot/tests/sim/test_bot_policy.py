"""The archive-mined practice-bot policy ([[enemy-bot-behavior]])."""

from __future__ import annotations

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.commands import CMD_SHOOT
from tankpit_bot.sim.bot_policy import (
    BOT_TELEPORT_OFF_HITS,
    MIN_RESPAWN_DISPLACEMENT,
    decide_practice_bot,
    make_practice_bot_state,
    note_hit_for_team_aggro,
    note_hit_on_bot,
    queue_return,
    reactivate_practice_bot,
    teleport_off_threshold,
)
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.server import CORPSE_WINDOW_TICKS, SimServer
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
    assert distance >= 16
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


def test_reactivation_respawns_far_at_full_fuel() -> None:
    """The measured law: same id, full fuel, >= 24 tiles from corpse."""
    world = _arena(rank=1)
    world["tick"] = 5
    tank = world["tanks"][BOT_ID]
    tank["alive"] = False
    tank["fuel"] = 0
    reactivate_practice_bot(world, InMemoryTerrainMap(), BOT_ID)
    assert tank["alive"] is True
    assert tank["fuel"] == fuel_capacity(1)
    displacement = max(abs(tank["x"] - 100), abs(tank["y"] - 100))
    assert displacement >= MIN_RESPAWN_DISPLACEMENT


def test_reactivation_rescatters_when_the_draw_lands_near_the_corpse() -> None:
    """A scatter draw inside the measured 24-tile floor re-scatters to
    the opposite half of the map (corpse (0,100), tick 20 draws (8,120))."""
    world = _arena(rank=0)
    world["tick"] = 20
    tank = world["tanks"][BOT_ID]
    tank["x"] = 0
    tank["y"] = 100
    tank["alive"] = False
    tank["fuel"] = 0
    reactivate_practice_bot(world, InMemoryTerrainMap(), BOT_ID)
    assert tank["alive"] is True
    displacement = max(abs(tank["x"] - 0), abs(tank["y"] - 100))
    assert displacement >= MIN_RESPAWN_DISPLACEMENT


def test_reactivation_on_sealed_terrain_stays_in_place() -> None:
    """No open tile anywhere -> alive at full fuel on the corpse tile."""
    world = _arena(rank=0)
    tank = world["tanks"][BOT_ID]
    tank["alive"] = False
    tank["fuel"] = 0
    reactivate_practice_bot(world, InMemoryTerrainMap(default="#"), BOT_ID)
    assert tank["alive"] is True
    assert tank["fuel"] == fuel_capacity(0)
    assert (tank["x"], tank["y"]) == (100, 100)


def test_roster_bot_reactivates_when_its_corpse_clears() -> None:
    """Server integration: kill -> 0x41 -> corpse window -> 0x58 +
    same-id reactivation far away; a non-roster corpse stays dead."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 100, 100, 1000)
    world["tanks"][BOT_ID] = make_sim_tank(BOT_ID, 1, 0, 101, 100, 45)
    world["tanks"][BOT_ID]["counts"] = [0, 0, 0, 0, 0]
    server = SimServer(world, InMemoryTerrainMap(), client_id=9, roster_ids=frozenset({BOT_ID}))
    server.queue_command(
        9,
        ClientCommandDict(kind="shoot", command=CMD_SHOOT, x=101, y=100, target_id=BOT_ID, slot=0),
    )
    first = server.advance_tick()
    assert [m["msg_type"] for m in first if m["msg_type"] == 0x41] == [0x41]
    assert world["tanks"][BOT_ID]["alive"] is False
    for _ in range(CORPSE_WINDOW_TICKS - 1):
        batch = server.advance_tick()
        assert [m for m in batch if m["msg_type"] == 0x58] == []
    boundary = server.advance_tick()
    removes = [m for m in boundary if m["msg_type"] == 0x58]
    assert [m["tank_id"] for m in removes] == [BOT_ID]
    bot = world["tanks"][BOT_ID]
    assert bot["alive"] is True
    assert bot["fuel"] == fuel_capacity(0)
    assert max(abs(bot["x"] - 101), abs(bot["y"] - 100)) >= MIN_RESPAWN_DISPLACEMENT
    synced = [m for m in boundary if m["msg_type"] == 0x2E and m["tank_id"] == BOT_ID]
    assert len(synced) == 1


def _aggro_arena() -> SimWorldDict:
    """Player (team 2) at (100,99); purple victim + teammates; blue ally."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 100, 99, 900)  # the player
    world["tanks"][BOT_ID] = make_sim_tank(BOT_ID, 1, 0, 100, 100, 800)  # victim
    world["tanks"][5] = make_sim_tank(5, 1, 0, 106, 100, 800)  # victim teammate, in sight
    world["tanks"][6] = make_sim_tank(6, 1, 0, 120, 100, 800)  # victim teammate, out of sight
    world["tanks"][7] = make_sim_tank(7, 2, 0, 100, 105, 800)  # attacker-team bot, in sight
    return world


def test_team_aggro_queues_gang_up_and_assist_within_sight() -> None:
    """The victim's sighted teammate targets the attacker; the
    attacker's sighted bot teammate targets the victim; the
    out-of-sight teammate never joins (129/129 archive shots <= 8)."""
    world = _aggro_arena()
    states = {
        5: make_practice_bot_state(),
        6: make_practice_bot_state(),
        7: make_practice_bot_state(),
    }
    responders = note_hit_for_team_aggro(world, states, BOT_ID, 9)
    assert responders == [5, 7]
    assert states[5]["has_pending_return"] is True
    assert (states[5]["pending_return_x"], states[5]["pending_return_y"]) == (100, 99)
    assert states[7]["has_pending_return"] is True
    assert (states[7]["pending_return_x"], states[7]["pending_return_y"]) == (100, 100)
    assert states[6]["has_pending_return"] is False
    assert states[5]["hits_taken"] == 0


def test_team_aggro_ignores_same_team_hits_and_dead_or_missing_bots() -> None:
    """Friendly fire never ignites aggro; dead and unknown responders
    are skipped; the victim and attacker never respond to themselves."""
    world = _aggro_arena()
    states = {5: make_practice_bot_state(), 8: make_practice_bot_state()}
    assert note_hit_for_team_aggro(world, states, BOT_ID, 5) == []
    world["tanks"][5]["alive"] = False
    assert note_hit_for_team_aggro(world, states, BOT_ID, 9) == []
    assert note_hit_for_team_aggro(world, states, 99, 9) == []
    assert states[8]["has_pending_return"] is False


def test_queue_return_refreshes_rather_than_stacks() -> None:
    """Two hits before the tick leave ONE pending shot at the newest
    tile — the shot-for-shot ratio ceiling."""
    state = make_practice_bot_state()
    queue_return(state, 10, 10)
    queue_return(state, 12, 10)
    assert state["has_pending_return"] is True
    assert (state["pending_return_x"], state["pending_return_y"]) == (12, 10)


def test_team_aggro_skips_principals_third_teams_and_far_allies() -> None:
    """The victim and attacker never respond to their own hit, a
    third-team bystander stays neutral, and an attacker-team bot out
    of sight of the victim never assists."""
    world = _aggro_arena()
    world["tanks"][11] = make_sim_tank(11, 3, 0, 101, 100, 800)  # third team, adjacent
    world["tanks"][12] = make_sim_tank(12, 2, 0, 120, 120, 800)  # ally, out of sight
    states = {
        9: make_practice_bot_state(),
        BOT_ID: make_practice_bot_state(),
        11: make_practice_bot_state(),
        12: make_practice_bot_state(),
    }
    assert note_hit_for_team_aggro(world, states, BOT_ID, 9) == []
    assert all(not s["has_pending_return"] for s in states.values())


def test_practice_room_ignores_shots_from_unknown_shooters() -> None:
    """A 0x53 whose shooter left the world never ignites anything."""
    from tankpit_bot.protocol.types import ShootEventDict
    from tankpit_bot.sim.practice_room import PracticeRoomDriver

    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 100, 100, 900)
    driver = PracticeRoomDriver(world, InMemoryTerrainMap(), 9, ((510, 1, 0, 108, 100),))
    ghost_shot = ShootEventDict(
        msg_type=0x53,
        team=0,
        shooter_id=777,
        source_x=90,
        source_y=90,
        target_x=100,
        target_y=100,
        aim_x=100,
        aim_y=100,
        weapon=0,
    )
    driver.note_batch(world, [ghost_shot])
    assert all(not state["has_pending_return"] for state in driver.states.values())


def test_round_resolution_orders_by_ascending_tank_id() -> None:
    """The measured within-round law: lower ids resolve first even
    when their commands were queued last (1,820/1,825 archive bursts;
    the sim's old queue-order emission was the only violator)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 100, 100, 900)
    world["tanks"][510] = make_sim_tank(510, 1, 0, 101, 100, 800)
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    server.queue_command(
        9,
        ClientCommandDict(kind="shoot", command=CMD_SHOOT, x=101, y=100, target_id=510, slot=0),
    )
    server.queue_command(
        510,
        ClientCommandDict(kind="shoot", command=CMD_SHOOT, x=100, y=100, target_id=9, slot=0),
    )
    batch = server.advance_tick()
    shooters = [m["shooter_id"] for m in batch if m["msg_type"] == 0x53]
    assert shooters == [9, 510] or shooters == sorted(shooters)
    assert shooters == sorted(shooters)
