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
from tankpit_bot.bot.tick_loop import _tick_once
from tankpit_bot.protocol.commands import CMD_MAP_OPEN, build_query_command
from tankpit_bot.protocol.helpers import EncodeError
from tankpit_bot.sim.session import deliver_batch
from tankpit_bot.sniffer.world_state import get_world_service
from tests.sim.seam import SEAM_CLIENT_ID, boot_seam

_CLIENT = SEAM_CLIENT_ID


def test_snapshot_answers_from_sim_truth() -> None:
    """The snapshot query decodes and reports a live, present client."""
    _bot, _server, link, _table = boot_seam()
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
    _bot, _server, link, _table = boot_seam()
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


def test_empty_batch_delivers_and_records_nothing() -> None:
    """An empty sim batch appends no frame and records no traffic.

    Live ticks always carry the per-tank sync cadence, so the empty
    case only arises when a caller has nothing to say — it must not
    fabricate an empty wire frame or a phantom capture record.
    """
    bot, _server, link, _table = boot_seam()
    buffered = len(bot._cdp_message_buffer)
    logged = len(link.wire_log)
    deliver_batch(bot._cdp_message_buffer, [], link)
    assert len(bot._cdp_message_buffer) == buffered
    assert len(link.wire_log) == logged


def test_production_command_service_reaches_the_sim_queue() -> None:
    """The bot's real command service (XOR and all) drives the sim."""
    bot, server, link, _table = boot_seam()
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
    bot, server, link, table = boot_seam()
    del table
    for _ in range(12):
        _tick_once(bot)
        deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
    _tick_once(bot)
    assert link.sent_commands != []
    truth = server.world["tanks"][_CLIENT]
    world = get_world_service().world_state
    self_state = world["self_state"]
    if self_state is None:
        raise AssertionError("the seam never established self_state")
    assert (self_state["x"], self_state["y"]) == (truth["x"], truth["y"])
    assert self_state["fuel"] == truth["fuel"]
