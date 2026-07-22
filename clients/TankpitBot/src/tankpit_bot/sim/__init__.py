"""Wiki-derived game simulator (Phase 4, [[physics-module-roadmap]]).

A fake TankPit server whose every rule cites a measured wiki law.
Build step (b): world state, the deterministic quadrant-keyed router,
instant movement, queue-model shot resolution, and the 2-second tick
processor. The wire transport (step c) and laws 4-8 (step d) follow.
"""

from __future__ import annotations

from tankpit_bot.sim.combat import ShotOutcomeDict, process_shot
from tankpit_bot.sim.commands import (
    ClientCommandDict,
    ClientCommandKind,
    SimError,
    decode_client_command,
)
from tankpit_bot.sim.movement import MoveOutcomeDict, PickupRecordDict, process_move
from tankpit_bot.sim.pathfind import PassableFn, route
from tankpit_bot.sim.server import TICK_MS, SimServer
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimMineDict,
    SimTankDict,
    SimWorldDict,
    decode_sim_tank,
    decode_sim_world,
    encode_sim_tank,
    encode_sim_world,
    make_sim_tank,
    make_sim_world,
)

__all__ = [
    "TICK_MS",
    "ClientCommandDict",
    "ClientCommandKind",
    "MoveOutcomeDict",
    "PassableFn",
    "PickupRecordDict",
    "ShotOutcomeDict",
    "SimContainerDict",
    "SimError",
    "SimMineDict",
    "SimServer",
    "SimTankDict",
    "SimWorldDict",
    "decode_client_command",
    "decode_sim_tank",
    "decode_sim_world",
    "encode_sim_tank",
    "encode_sim_world",
    "make_sim_tank",
    "make_sim_world",
    "process_move",
    "process_shot",
    "route",
]
