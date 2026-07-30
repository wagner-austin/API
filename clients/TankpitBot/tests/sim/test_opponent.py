"""The scripted opponent and the per-recipient supervisor discipline."""

from __future__ import annotations

from tankpit_bot.sim.opponent import decide_opponent, maybe_revive_opponent
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _arena() -> SimWorldDict:
    """Client 9 at (10, 10), armed enemy 11 at (15, 10)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][11] = make_sim_tank(11, 1, 8, 15, 10, 2000)
    world["tanks"][11]["counts"] = [5, 25, 0, 25, 5]
    return world


def test_opponent_beats_are_deterministic() -> None:
    """tick%4: dodge, shoot, hold, shoot — pure function of the tick."""
    world = _arena()
    world["tick"] = 0
    dodge = decide_opponent(world, 11, 9)
    if dodge is None:
        raise AssertionError("dodge beat must produce a move")
    assert (dodge["kind"], dodge["x"], dodge["y"]) == ("move", 16, 10)
    world["tick"] = 1
    shot = decide_opponent(world, 11, 9)
    if shot is None:
        raise AssertionError("shoot beat must produce a shot")
    assert (shot["kind"], shot["x"], shot["y"]) == ("shoot", 10, 10)
    world["tick"] = 2
    assert decide_opponent(world, 11, 9) is None
    world["tick"] = 3
    third = decide_opponent(world, 11, 9)
    if third is None:
        raise AssertionError("second shoot beat must produce a shot")
    assert third["kind"] == "shoot"
    world["tick"] = 4
    west = decide_opponent(world, 11, 9)
    if west is None:
        raise AssertionError("alternating dodge must produce a move")
    assert (west["kind"], west["x"]) == ("move", 14)


def test_opponent_holds_when_blind_or_dead() -> None:
    """No sight, no fight: out-of-viewport clients and corpses hold."""
    world = _arena()
    world["tick"] = 1
    world["tanks"][9]["x"] = 40
    assert decide_opponent(world, 11, 9) is None
    world["tanks"][9]["x"] = 10
    world["tanks"][11]["alive"] = False
    assert decide_opponent(world, 11, 9) is None
    world["tanks"][11]["alive"] = True
    world["tanks"][9]["alive"] = False
    assert decide_opponent(world, 11, 9) is None


def test_enemy_rejections_never_leak_into_the_client_stream() -> None:
    """0x52 is per-connection: another tank's reject stays silent.

    The enemy walks into a wall (cant_go) and hops without fuel
    (insufficient) — the client's batch must carry NO supervisor
    message for either.
    """
    world = _arena()
    walls = {(16, 10): "#"}
    server = SimServer(world, InMemoryTerrainMap(terrain_data=walls), client_id=9)
    from tankpit_bot.sim.commands import ClientCommandDict

    server.queue_command(
        11,
        ClientCommandDict(kind="move", command=112, x=16, y=10, target_id=0, slot=0, message_id=0),
    )
    blocked = server.advance_tick()
    assert [m for m in blocked if m["msg_type"] == 0x52] == []
    world["tanks"][11]["fuel"] = 2
    server.queue_command(
        11,
        ClientCommandDict(
            kind="teleport", command=116, x=40, y=40, target_id=0, slot=0, message_id=0
        ),
    )
    poor = server.advance_tick()
    assert [m for m in poor if m["msg_type"] == 0x52] == []
    server.queue_command(
        11,
        ClientCommandDict(kind="move", command=112, x=20, y=14, target_id=0, slot=0, message_id=0),
    )
    broke = server.advance_tick()
    assert [m for m in broke if m["msg_type"] == 0x52] == []
    server.queue_command(
        11,
        ClientCommandDict(
            kind="pickup_fuel", command=100, x=20, y=10, target_id=0, slot=0, message_id=0
        ),
    )
    ghost = server.advance_tick()
    assert [m for m in ghost if m["msg_type"] == 0x52] == []


def test_revival_activates_a_new_tank_near_the_client() -> None:
    """A dead opponent returns as a NEW id, announced by 0x21.

    Real respawns join with a fresh wire tank id (that is what
    persistent_tank_id bridges), so the killed id stays a corpse and
    the replacement activates within the reachable ring band.
    """
    world = _arena()
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    world["tanks"][11]["alive"] = False
    world["tick"] = 2
    new_id = maybe_revive_opponent(server, 11, 9)
    assert new_id == 12
    fresh = world["tanks"][12]
    assert fresh["alive"] is True
    assert fresh["team"] == 1
    assert world["tanks"][11]["alive"] is False
    reach = max(abs(fresh["x"] - 10), abs(fresh["y"] - 10))
    assert 6 <= reach <= 24
    batch = server.advance_tick()
    announcement = batch[0]
    assert announcement["msg_type"] == 0x21
    assert announcement["tank_id"] == 12


def test_revival_holds_while_alive_off_beat_or_sealed() -> None:
    """No revival for the living, off the beat, or on a closed map."""
    world = _arena()
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    world["tick"] = 2
    assert maybe_revive_opponent(server, 11, 9) == 11
    world["tanks"][11]["alive"] = False
    world["tick"] = 3
    assert maybe_revive_opponent(server, 11, 9) == 11
    sealed = InMemoryTerrainMap(terrain_data={(x, y): "#" for x in range(256) for y in range(256)})
    walled = SimServer(world, sealed, client_id=9)
    world["tick"] = 4
    assert maybe_revive_opponent(walled, 11, 9) == 11
    assert 12 not in world["tanks"]


def test_enemy_equipment_grant_resolves_silently() -> None:
    """Another tank's 0x67 must not leak — production reads it as SELF."""
    from tankpit_bot.sim.commands import ClientCommandDict
    from tankpit_bot.sim.world import SimEquipmentDict

    world = _arena()
    world["tanks"][11]["counts"] = [0, 0, 0, 0, 0]
    world["equipment"].append(SimEquipmentDict(x=16, y=10))
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    server.queue_command(
        11,
        ClientCommandDict(kind="move", command=112, x=16, y=10, target_id=0, slot=0, message_id=0),
    )
    messages = server.advance_tick()
    assert [m for m in messages if m["msg_type"] in (0x67, 0x49)] == []
    assert any(count > 0 for count in world["tanks"][11]["counts"])
    assert world["equipment"] == []
