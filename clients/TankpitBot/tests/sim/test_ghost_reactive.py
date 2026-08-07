"""Reactive ghosts — the certified roster policy under the timeline.

A bot-named ghost carries ``sim/bot_policy`` reactions (2026-08-03):
the live client's shots draw the mined shot-for-shot return fire even
where the recording holds no answer, while recorded events keep
per-tick authority over the policy. Split from ``test_ghost.py``
(already at the 600-line cap).
"""

from __future__ import annotations

from tankpit_bot.protocol.types import BinaryMessage, ShootEventDict
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.ghost import GhostEventDict, GhostSpecDict
from tankpit_bot.sim.practice_room import PracticeRoomDriver
from tankpit_bot.sim.run_boot import _queue_round_opponents
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

_CLIENT = 9
_GHOST = 500


def _shoot(x: int, y: int) -> ClientCommandDict:
    """A shoot command at one tile."""
    return ClientCommandDict(
        kind="shoot", command=115, x=x, y=y, target_id=0, slot=0, message_id=0, direction=0
    )


def _arena() -> tuple[SimWorldDict, SimServer]:
    """Client at (100,100), bot-named ghost at (103,100)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][_CLIENT] = make_sim_tank(_CLIENT, 2, 1, 100, 100, 1100)
    world["tanks"][_GHOST] = make_sim_tank(_GHOST, 1, 1, 103, 100, 1100, name="orange-2")
    server = SimServer(
        world, InMemoryTerrainMap(), client_id=_CLIENT, roster_ids=frozenset({_GHOST})
    )
    return world, server


def _spec(events: list[GhostEventDict]) -> GhostSpecDict:
    """A minimal compiled spec carrying only a timeline."""
    return GhostSpecDict(
        client_team=2,
        client_rank=1,
        client_x=100,
        client_y=100,
        client_fuel=1100,
        client_counts=[25] * 5,
        ghosts=[],
        events=events,
        recorded_path={},
        containers=[],
        equipment=[],
        dot_atlas=[],
        ticks=10,
        unplaced_tanks=0,
    )


def _ghost_shots(batch: list[BinaryMessage]) -> list[ShootEventDict]:
    """The tick's 0x53 events fired by the ghost."""
    shots: list[ShootEventDict] = []
    for message in batch:
        if message["msg_type"] == 0x53 and message["shooter_id"] == _GHOST:
            shots.append(message)
    return shots


def test_hit_ghost_returns_fire_by_the_certified_policy() -> None:
    """A client hit on a bot-named ghost draws the mined single next
    tick — where the recording has no answer at all."""
    world, server = _arena()
    driver = PracticeRoomDriver(frozenset({_GHOST}))

    server.queue_command(_CLIENT, _shoot(103, 100))
    batch = server.advance_tick()
    driver.note_batch(world, batch)
    assert _ghost_shots(batch) == []

    _queue_round_opponents(server, driver, False, _spec([]), 11, 0)
    batch = server.advance_tick()
    returns = _ghost_shots(batch)
    assert len(returns) == 1
    assert (returns[0]["target_x"], returns[0]["target_y"]) == (100, 100)


def test_recorded_event_takes_the_tick_over_the_policy() -> None:
    """A ghost with a recorded action this tick yields its policy
    command — recorded authority wins per tick."""
    world, server = _arena()
    driver = PracticeRoomDriver(frozenset({_GHOST}))

    server.queue_command(_CLIENT, _shoot(103, 100))
    driver.note_batch(world, server.advance_tick())

    recorded = GhostEventDict(tick=0, tank_id=_GHOST, kind="shoot", x=90, y=90, message_id=0)
    _queue_round_opponents(server, driver, False, _spec([recorded]), 11, 0)
    batch = server.advance_tick()
    shots = _ghost_shots(batch)
    assert len(shots) == 1
    assert (shots[0]["target_x"], shots[0]["target_y"]) == (90, 90)

    # The queued return was withheld, not lost: the next quiet tick
    # still answers it (the policy state kept the pending return).
    _queue_round_opponents(server, driver, False, _spec([]), 11, 1)
    batch = server.advance_tick()
    late = _ghost_shots(batch)
    assert len(late) == 1
    assert (late[0]["target_x"], late[0]["target_y"]) == (100, 100)
