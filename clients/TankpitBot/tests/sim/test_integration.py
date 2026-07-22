"""End-to-end wiring: sim wire bytes through the PRODUCTION pipeline.

These tests are the step-(c) proof that the simulator is not a
parallel universe: every byte the sim's transport emits is consumed
by the real ingestion path (``sniffer.decoders.process_received_message``
— frame split, XOR, decoders, world-state dispatch), the bot's world
beliefs are asserted against the sim's ground truth, real
production-encoded command bytes drive the sim, and the real planner
decides on sim-fed state.
"""

from __future__ import annotations

import base64

from tests.in_memory_terrain_map import InMemoryTerrainMap

from tankpit_bot.bot.ai.types import make_initial_ai_state
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.protocol.commands import build_move_command
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.transport import decode_client_payload, encode_tick_payload
from tankpit_bot.sim.world import SimContainerDict, make_sim_tank, make_sim_world
from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.xor import (
    build_global_xor_table,
    get_global_xor_table,
    reset_xor_state,
)
from tankpit_bot.state.types import SelfStateDict

_MAGIC = "simmagic"
_CLIENT = 9
_ENEMY = 11


def _established_self() -> SelfStateDict:
    """Return the bot's self-state, failing the test when absent.

    Returns:
        The established self-state.

    Raises:
        AssertionError: If the wire never established self-state.
    """
    self_state = get_world_service().world_state["self_state"]
    if self_state is None:
        raise AssertionError("the sim wire never established self_state")
    return self_state


def _boot() -> tuple[SimServer, bytes]:
    """Reset globals, build the XOR table, and join a fresh sim world.

    Returns:
        The sim server and the session XOR table, with the join
        handshake already ingested by the production pipeline.

    Raises:
        RuntimeError: If the repo's XOR static key is unavailable.
    """
    reset_world_state()
    reset_xor_state()
    build_global_xor_table(_MAGIC)
    table = get_global_xor_table()
    if table is None:
        raise RuntimeError("XOR static key unavailable — cannot run wire integration")
    world = make_sim_world("field01_r.gif")
    world["tanks"][_CLIENT] = make_sim_tank(_CLIENT, 2, 1, 100, 100, 800)
    world["tanks"][_CLIENT]["counts"] = [25, 25, 25, 25, 25]
    world["tanks"][_ENEMY] = make_sim_tank(_ENEMY, 1, 1, 110, 100, 500)
    world["containers"].append(SimContainerDict(x=103, y=100, volume=300))
    server = SimServer(world, InMemoryTerrainMap(), client_id=_CLIENT)
    _deliver(server.handshake(), table)
    return server, table


def _deliver(messages: list[BinaryMessage], table: bytes) -> None:
    """Push one message batch through the production ingestion path.

    Args:
        messages: The sim's decoded batch.
        table: Session XOR table.
    """
    process_received_message(encode_tick_payload(messages, table))


def test_handshake_reaches_production_world_state() -> None:
    """The join burst establishes self-state and the enemy registry."""
    server, _table = _boot()
    world = get_world_service().world_state
    self_state = _established_self()
    assert (self_state["x"], self_state["y"]) == (100, 100)
    assert self_state["fuel"] == 800
    enemy = world["tanks"][str(_ENEMY)]
    assert (enemy["x"], enemy["y"]) == (110, 100)
    assert server.world["tick"] == 0


def test_production_command_bytes_drive_the_sim_and_beliefs_track_truth() -> None:
    """A real client frame moves the sim; the wire keeps beliefs true.

    The command bytes come from the PRODUCTION ``build_move_command``
    (XOR'd the way the command sender transmits them), the sim decodes
    them through its transport, and after the tick's wire batch the
    bot's believed position and fuel equal the sim's ground truth —
    including the container pickup at the destination.
    """
    server, table = _boot()
    framed = build_move_command(103, 100)
    wire = framed[:3] + bytes(
        byte ^ (table[i] if i < len(table) else 0) for i, byte in enumerate(framed[3:])
    )
    commands = decode_client_payload(base64.b64encode(wire).decode("ascii"), table)
    assert [command["kind"] for command in commands] == ["move"]
    server.queue_command(_CLIENT, commands[0])
    _deliver(server.advance_tick(), table)

    truth = server.world["tanks"][_CLIENT]
    assert (truth["x"], truth["y"]) == (103, 100)
    assert truth["fuel"] == 800 - 3 + 300
    self_state = _established_self()
    assert (self_state["x"], self_state["y"]) == (truth["x"], truth["y"])
    assert self_state["fuel"] == truth["fuel"]


def test_victim_fuel_sync_does_not_leak_into_self_belief() -> None:
    """An enemy hit's short-form sync must not overwrite own fuel.

    The client has duals loaded, so the sim's weapon selection fires a
    dual (90 damage) — and the victim's 0x2E sync must arrive
    short-form, leaving the bot's own fuel belief untouched.
    """
    server, table = _boot()
    server.queue_command(
        _CLIENT, ClientCommandDict(kind="shoot", command=115, x=110, y=100, target_id=0)
    )
    _deliver(server.advance_tick(), table)
    assert server.world["tanks"][_ENEMY]["fuel"] == 500 - 90
    world = get_world_service().world_state
    assert _established_self()["fuel"] == 800
    assert world["tanks"][str(_ENEMY)]["damage_state"] == 3


def test_real_planner_decides_on_sim_fed_state() -> None:
    """The production planner produces a decision from sim-fed beliefs."""
    server, _table = _boot()
    del server
    service = get_world_service()
    self_state = _established_self()
    decision = decide(
        world=service.world_state,
        self_state=self_state,
        ai_state=make_initial_ai_state(),
        inventory=service.inventory_state,
        timestamp_ms=get_current_time_ms(),
        terrain=None,
        combat_feedback="",
    )
    assert decision["behavior"]["mode"] in ("HUNT", "COLLECT")
