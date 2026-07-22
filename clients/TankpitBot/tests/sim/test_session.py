"""The live seam: the PRODUCTION tick loop plays against the sim.

The smoke test here is the step-(c) acceptance: a real ``Bot`` (no
browser), the real ``_tick_once``, the real command service and
encoders — only ``bot._cdp`` is the sim link. Every command the
planner dispatches travels as genuine wire bytes into the sim; every
sim response travels as genuine wire bytes back through the
production ingestion path.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, require_dict

from tankpit_bot.action_lab.page_client_snapshot import decode_page_client_snapshot
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.tick_loop import _tick_once
from tankpit_bot.protocol.commands import CMD_MAP_OPEN, build_query_command
from tankpit_bot.protocol.helpers import EncodeError
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.session import SimCDPSession, deliver_batch
from tankpit_bot.sim.world import SimContainerDict, make_sim_tank, make_sim_world
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.xor import (
    build_global_xor_table,
    get_global_xor_table,
    reset_xor_state,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_MAGIC = "simmagic"
_CLIENT = 9
_ENEMY = 11


def _boot() -> tuple[Bot, SimServer, SimCDPSession, bytes]:
    """Build a real Bot wired to a fresh sim world over the seam.

    Returns:
        The bot, the sim server, the sim CDP link, and the XOR table,
        with the join handshake already delivered to the bot's buffer.

    Raises:
        RuntimeError: If the repo's XOR static key is unavailable.
    """
    reset_world_state()
    reset_xor_state()
    build_global_xor_table(_MAGIC)
    table = get_global_xor_table()
    if table is None:
        raise RuntimeError("XOR static key unavailable — cannot run the seam smoke")
    world = make_sim_world("field01_r.gif")
    world["tanks"][_CLIENT] = make_sim_tank(_CLIENT, 2, 1, 100, 100, 800)
    world["tanks"][_CLIENT]["counts"] = [25, 25, 25, 25, 25]
    world["tanks"][_ENEMY] = make_sim_tank(_ENEMY, 1, 8, 110, 100, 1800)
    world["containers"].append(SimContainerDict(x=103, y=100, volume=300))
    world["containers"].append(SimContainerDict(x=97, y=104, volume=400))
    world["containers"].append(SimContainerDict(x=106, y=95, volume=400))
    server = SimServer(world, InMemoryTerrainMap(), client_id=_CLIENT)
    bot = Bot("https://sim.tankpit.local/", headless=True)
    bot._magic = _MAGIC
    bot._on_magic_captured(_MAGIC)
    link = SimCDPSession(server, table)
    bot._cdp = link
    deliver_batch(bot._cdp_message_buffer, server.handshake(), table)
    return bot, server, link, table


def test_snapshot_answers_from_sim_truth() -> None:
    """The snapshot query decodes and reports a live, present client."""
    _bot, _server, link, _table = _boot()
    result = link.send(
        "Runtime.evaluate",
        {"expression": "window.__tankpitActiveGame && (...)", "returnByValue": True},
    )
    value = require_dict(result, "result")["value"]
    snapshot = decode_page_client_snapshot(require_dict({"snapshot": value}, "snapshot"))
    assert snapshot["client_present"] is True
    assert snapshot["ws_ready_state"] == 1
    assert snapshot["map_visible"] is False
    assert snapshot["self_fields"] == {}


def test_link_rejects_unmodeled_traffic_loudly() -> None:
    """Unknown evaluate expressions and missing params raise, never guess."""
    _bot, _server, link, _table = _boot()
    assert link.send("Network.enable", {}) == {"result": {"value": None}}
    with pytest.raises(EncodeError):
        link.send("Runtime.evaluate", None)
    with pytest.raises(EncodeError):
        link.send("Runtime.evaluate", {"expression": "totally.unknown()"})
    meta = link.send("Runtime.evaluate", {"expression": "window.__sentFrameMetaQueue.shift()"})
    assert meta == {"result": {"value": None}}
    raw = link.send("Runtime.evaluate", {"expression": "window.__rawMsgs.slice(-500)"})
    assert raw == {"result": {"value": []}}
    body = link.send("Runtime.evaluate", {"expression": "document.body.innerText"})
    assert body == {"result": {"value": ""}}

    def handler(params: JSONObject) -> None:
        """Unused event handler (the sim delivers via the buffer)."""
        del params

    link.on("Network.webSocketFrameReceived", handler)
    link.detach()
    assert link._detached is True


def test_production_command_service_reaches_the_sim_queue() -> None:
    """The bot's real command service (XOR and all) drives the sim."""
    bot, server, link, _table = _boot()
    del server
    assert bot._send_bytes(build_query_command(CMD_MAP_OPEN), "map_open") is True
    assert link.sent_commands == ["map_open"]
    assert link.map_visible is True


def test_real_tick_loop_plays_a_session_against_the_sim() -> None:
    """The production ``_tick_once`` runs whole rounds against the sim.

    Each round is one bot tick followed by one sim tick with its wire
    batch delivered through the production ingestion path. After the
    rounds, the bot must have dispatched real commands over the seam
    and its believed position and fuel must equal the sim's ground
    truth exactly.
    """
    bot, server, link, table = _boot()
    for _ in range(12):
        _tick_once(bot)
        deliver_batch(bot._cdp_message_buffer, server.advance_tick(), table)
    _tick_once(bot)
    assert link.sent_commands != []
    truth = server.world["tanks"][_CLIENT]
    world = get_world_service().world_state
    self_state = world["self_state"]
    if self_state is None:
        raise AssertionError("the seam never established self_state")
    assert (self_state["x"], self_state["y"]) == (truth["x"], truth["y"])
    assert self_state["fuel"] == truth["fuel"]
