"""Shared server builder and emission filters for the sim-server tests."""

from __future__ import annotations

from tankpit_bot.protocol.types import (
    BinaryMessage,
    InventoryDict,
    ShootEventDict,
    SupervisorDict,
    TankStatusSyncDict,
)
from tankpit_bot.sim.commands import (
    ClientCommandDict,
)
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import (
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _move(x: int, y: int) -> ClientCommandDict:
    """A decoded move command to (x, y)."""
    return ClientCommandDict(
        kind="move", command=112, x=x, y=y, target_id=0, slot=0, message_id=0, direction=0
    )


def _shoot(x: int, y: int) -> ClientCommandDict:
    """A decoded shoot command at (x, y)."""
    return ClientCommandDict(
        kind="shoot", command=115, x=x, y=y, target_id=0, slot=0, message_id=0, direction=0
    )


def _server() -> SimServer:
    """Client tank 9 at (10, 10) and enemy 11 at (15, 10)."""
    world: SimWorldDict = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 15, 10, 500)
    return SimServer(world, InMemoryTerrainMap(), client_id=9)


def _kinds(messages: list[BinaryMessage]) -> list[int | str]:
    """The msg_type of every message, in emission order."""
    return [message["msg_type"] for message in messages]


def _shots(messages: list[BinaryMessage]) -> list[ShootEventDict]:
    """All 0x53 echoes in the batch."""
    shots: list[ShootEventDict] = []
    for message in messages:
        if message["msg_type"] == 0x53:
            shots.append(message)
    return shots


def _syncs(messages: list[BinaryMessage]) -> list[TankStatusSyncDict]:
    """All 0x2E fuel syncs in the batch."""
    syncs: list[TankStatusSyncDict] = []
    for message in messages:
        if message["msg_type"] == 0x2E:
            syncs.append(message)
    return syncs


def _supervisors(messages: list[BinaryMessage]) -> list[SupervisorDict]:
    """All 0x52 command-result messages in the batch."""
    results: list[SupervisorDict] = []
    for message in messages:
        if message["msg_type"] == 0x52:
            results.append(message)
    return results


def _snapshots(messages: list[BinaryMessage]) -> list[InventoryDict]:
    """All 0x49 inventory snapshots in the batch."""
    snapshots: list[InventoryDict] = []
    for message in messages:
        if message["msg_type"] == 0x49:
            snapshots.append(message)
    return snapshots


def _command(kind_command: tuple[str, int], x: int = 0, y: int = 0) -> ClientCommandDict:
    """A decoded client command of the given (kind, byte) pair."""
    kind, command = kind_command
    move_kind: ClientCommandDict = ClientCommandDict(
        kind="move", command=command, x=x, y=y, target_id=0, slot=0, message_id=0, direction=0
    )
    if kind == "teleport":
        return ClientCommandDict(
            kind="teleport",
            command=command,
            x=x,
            y=y,
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
        )
    if kind == "radar":
        return ClientCommandDict(
            kind="radar", command=command, x=x, y=y, target_id=0, slot=0, message_id=0, direction=0
        )
    if kind == "mine":
        return ClientCommandDict(
            kind="mine", command=command, x=x, y=y, target_id=0, slot=0, message_id=0, direction=0
        )
    if kind == "map_open":
        return ClientCommandDict(
            kind="map_open",
            command=command,
            x=x,
            y=y,
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
        )
    if kind == "pickup_fuel":
        return ClientCommandDict(
            kind="pickup_fuel",
            command=command,
            x=x,
            y=y,
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
        )
    return move_kind


def _statistics_key() -> ClientCommandDict:
    """The decoded ``CMD_STATISTICS`` key press."""
    return ClientCommandDict(
        kind="statistics", command=118, x=0, y=0, target_id=0, slot=0, message_id=0, direction=0
    )
