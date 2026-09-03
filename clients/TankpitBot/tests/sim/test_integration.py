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

from tankpit_bot.bot.ai.types import make_initial_ai_state
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.physics.capacity import damage_tier
from tankpit_bot.protocol.command_builders import build_move_command
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.transport import decode_client_payload, encode_tick_payload
from tankpit_bot.sim.world import SimContainerDict, make_sim_tank, make_sim_world
from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import SelfStateDict
from tankpit_bot.state.types.tank import has_known_position
from tests.in_memory_terrain_map import InMemoryTerrainMap

_MAGIC = "simmagic"
_CLIENT = 9
_ENEMY = 11


def _established_self(ws: WorldService) -> SelfStateDict:
    """Return the bot's self-state, failing the test when absent.

    Args:
        ws: The world service the sim wire fed.

    Returns:
        The established self-state.

    Raises:
        AssertionError: If the wire never established self-state.
    """
    self_state = ws.world_state["self_state"]
    if self_state is None:
        raise AssertionError("the sim wire never established self_state")
    return self_state


def _boot(ws: WorldService) -> tuple[SimServer, bytes]:
    """Build the XOR table and join a fresh sim world into ``ws``.

    Args:
        ws: The world service the handshake is ingested into.

    Returns:
        The sim server and the session XOR table, with the join
        handshake already ingested by the production pipeline.

    Raises:
        XorStaticKeyUnavailableError: If the repo's XOR static key is
            unavailable.
    """
    table = build_session_xor_table(_MAGIC)
    world = make_sim_world("field01_r.gif")
    world["tanks"][_CLIENT] = make_sim_tank(_CLIENT, 2, 1, 100, 100, 800)
    world["tanks"][_CLIENT]["counts"] = [25, 25, 25, 25, 25]
    world["tanks"][_ENEMY] = make_sim_tank(_ENEMY, 1, 1, 107, 100, 500)
    world["containers"].append(SimContainerDict(x=103, y=100, volume=300, dotted=True))
    server = SimServer(world, InMemoryTerrainMap(), client_id=_CLIENT)
    _deliver(ws, server.handshake(), table)
    return server, table


def _deliver(ws: WorldService, messages: list[BinaryMessage], table: bytes) -> None:
    """Push one message batch through the production ingestion path.

    Args:
        ws: The world service the batch is ingested into.
        messages: The sim's decoded batch.
        table: Session XOR table.
    """
    process_received_message(ws, encode_tick_payload(messages, table), table)


def test_handshake_reaches_production_world_state() -> None:
    """The join burst establishes self-state and a position-less roster.

    Self-state is complete from the burst — including fuel, which
    arrives on the burst's own 0x2E status sync, not a 0x44
    ([[recipient-policy]]).

    The enemy is REGISTERED by its 0x21 identity and carries NO
    position. That is the login-roster law, measured twice by
    independent methods: the 2026-08-04 first-sight probe (3 captures,
    113 tanks, every one 0x21-first, 9-46 s to its first position) and
    the 2026-09-01 burst sweep (identity run pure 0x21 in 340 of 340
    sessions). The ``(0, 0)`` phantom is the normal opening state of
    every tank, which is what ``has_known_position`` exists to answer
    ([[tank-freshness-model]]).

    Before the burst was corrected the sim shipped positions the real
    server never sends, so this path — the one production actually
    walks on every join — was never exercised in sim.
    """
    ws = WorldService()
    server, _table = _boot(ws)
    world = ws.world_state
    self_state = _established_self(ws)
    assert (self_state["x"], self_state["y"]) == (100, 100)
    assert self_state["fuel"] == 800
    enemy = world["tanks"][str(_ENEMY)]
    assert enemy["team"] == 1
    assert not has_known_position(enemy)
    assert (enemy["x"], enemy["y"]) == (0, 0)
    assert server.world["tanks"][_ENEMY]["x"] == 107
    assert server.world["tick"] == 0


def test_production_command_bytes_drive_the_sim_and_beliefs_track_truth() -> None:
    """A real client frame moves the sim; the wire keeps beliefs true.

    The command bytes come from the PRODUCTION ``build_move_command``
    (XOR'd the way the command sender transmits them), the sim decodes
    them through its transport, and after the tick's wire batch the
    bot's believed position and fuel equal the sim's ground truth —
    including the container pickup at the destination.
    """
    ws = WorldService()
    server, table = _boot(ws)
    framed = build_move_command(103, 100)
    wire = framed[:3] + bytes(
        byte ^ (table[i] if i < len(table) else 0) for i, byte in enumerate(framed[3:])
    )
    commands = decode_client_payload(base64.b64encode(wire).decode("ascii"), table)
    assert [command["kind"] for command in commands] == ["move"]
    server.queue_command(_CLIENT, commands[0])
    _deliver(ws, server.advance_tick(), table)

    truth = server.world["tanks"][_CLIENT]
    assert (truth["x"], truth["y"]) == (103, 100)
    assert truth["fuel"] == 800 - 3 + 300
    self_state = _established_self(ws)
    assert (self_state["x"], self_state["y"]) == (truth["x"], truth["y"])
    assert self_state["fuel"] == truth["fuel"]


def test_victim_fuel_sync_does_not_leak_into_self_belief() -> None:
    """An enemy hit's short-form sync must not overwrite own fuel.

    The client has duals loaded, so the sim's weapon selection fires a
    dual (90 damage) — and the victim's 0x2E sync must arrive
    short-form, leaving the bot's own fuel belief untouched.
    """
    ws = WorldService()
    server, table = _boot(ws)
    server.queue_command(
        _CLIENT,
        ClientCommandDict(
            kind="shoot", command=115, x=107, y=100, target_id=0, slot=0, message_id=0, direction=0
        ),
    )
    _deliver(ws, server.advance_tick(), table)
    assert server.world["tanks"][_ENEMY]["fuel"] == 500 - 90
    world = ws.world_state
    assert _established_self(ws)["fuel"] == 800
    assert world["tanks"][str(_ENEMY)]["damage_state"] == damage_tier(500 - 90, 1)


def test_real_planner_decides_on_sim_fed_state() -> None:
    """The production planner produces a decision from sim-fed beliefs."""
    ws = WorldService()
    server, _table = _boot(ws)
    del server
    service = ws
    self_state = _established_self(ws)
    decision = decide(
        world=service.world_state,
        self_state=self_state,
        ai_state=make_initial_ai_state(),
        inventory=service.inventory_state,
        timestamp_ms=get_current_time_ms(),
        terrain=None,
        combat_feedback="",
        ws=ws,
    )
    assert decision["behavior"]["mode"] in ("HUNT", "COLLECT")
