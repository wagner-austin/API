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

from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.browser.page_client_snapshot import decode_page_client_snapshot
from tankpit_bot.capture.xor import xor_decode_body
from tankpit_bot.protocol.command_builders import build_query_command
from tankpit_bot.protocol.commands import (
    CMD_ENTER_GAME,
    CMD_INVENTORY,
    CMD_KEEPALIVE,
    CMD_MAP_OPEN,
    CMD_UNMODELLED_COMBAT,
    COMMAND_PREFIX,
    TYPE_QUERY,
)
from tankpit_bot.protocol.types import DeactivationDict
from tankpit_bot.sim.commands import SimError
from tankpit_bot.sim.session import deliver_batch
from tankpit_bot.wire.helpers import EncodeError
from tests.sim.seam import SEAM_CLIENT_ID, SEAM_ENEMY_ID, boot_seam

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


def test_own_deactivation_ends_the_session_through_the_production_exit() -> None:
    """An own 0x41 raises the ``deactivated`` session exit next tick.

    The wire has carried own-kill 0x41s since 2026-07-19; nothing
    consumed them for self-death until the sim CLI showed a killed
    bot ticking forever. The dispatch records the fact and the tick
    loop converts it — a corpse has no decisions left.
    """
    bot, _server, link, _table = boot_seam()
    ws = bot.world
    for _ in range(2):
        _tick_once(bot)
        deliver_batch(bot._cdp_message_buffer, _server.advance_tick(), link)
    death = DeactivationDict(
        msg_type=0x41,
        status=1,
        victim_id=SEAM_CLIENT_ID,
        promo_eligible=False,
        killer_id=SEAM_ENEMY_ID,
        is_mine_kill=False,
    )
    deliver_batch(bot._cdp_message_buffer, [death], link)
    # User contract 2026-07-30: a death is a respawn wait, not an
    # exit — the corpse's beliefs reset, the dead self record drops,
    # and the loop idles until fresh sync (or the 60 s deadline).
    _tick_once(bot)
    assert ws.self_deactivated is False
    assert ws.world_state["self_state"] is None
    assert bot._ai_state["combat_target_id"] == -1
    assert bot._respawn_deadline_ms > 0
    # Waiting ticks idle quietly while the deadline is in the future.
    _tick_once(bot)
    assert bot._respawn_deadline_ms > 0
    # A fresh self sync IS the respawn: the wait clears and the next
    # arbitration runs with a factory-reset tactical state.
    from tankpit_bot.state import make_self_state

    ws.world_state["self_state"] = make_self_state(
        tank_id=SEAM_CLIENT_ID,
        x=10,
        y=10,
        team=2,
        rank=0,
        fuel=1000,
        leaderboard_position=1,
    )
    _tick_once(bot)
    assert bot._respawn_deadline_ms == 0
    # With no respawn law in the sim, the deadline is the exit path.
    ws.world_state["self_state"] = None
    bot._respawn_deadline_ms = 1
    with pytest.raises(SessionExitError) as exit_info:
        _tick_once(bot)
    assert exit_info.value.reason == "deactivated"


def test_real_tick_loop_plays_a_session_against_the_sim() -> None:
    """The production ``_tick_once`` runs whole rounds against the sim.

    Each round is one bot tick followed by one sim tick with its wire
    batch delivered through the production ingestion path. After the
    rounds, the bot must have dispatched real commands over the seam
    and its believed position and fuel must equal the sim's ground
    truth exactly.
    """
    bot, server, link, table = boot_seam()
    ws = bot.world
    del table
    for _ in range(12):
        _tick_once(bot)
        deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
    _tick_once(bot)
    assert link.sent_commands != []
    truth = server.world["tanks"][_CLIENT]
    world = ws.world_state
    self_state = world["self_state"]
    if self_state is None:
        raise AssertionError("the seam never established self_state")
    assert (self_state["x"], self_state["y"]) == (truth["x"], truth["y"])
    assert self_state["fuel"] == truth["fuel"]


def _query_frame(table: bytes, command: int) -> bytes:
    """One query-family frame as a real page client puts it on the wire.

    ``[!]`` then the XOR-ciphered ``[TYPE_QUERY][command]``. The frame
    must be ENCIPHERED, because the seam deciphers everything it
    routes — ``build_query_command`` returns plaintext with a length
    header and is the bot's send path, not the page's. The cipher is
    symmetric, so the decode helper enciphers, exactly as
    ``SimCDPSession`` builds its own page frames.

    Args:
        table: The session's XOR table.
        command: The query command byte.

    Returns:
        The frame body, lead byte included.
    """
    return bytes([COMMAND_PREFIX]) + xor_decode_body(bytes([TYPE_QUERY, command]), table)


def test_a_page_client_heartbeat_does_not_kill_the_session() -> None:
    """THE CRASH, AT THE SEAM A REAL BROWSER ACTUALLY ENTERS THROUGH.

    Every client frame lands in ``route_client_payload``, which decodes
    it and hands it straight to ``queue_command`` — and that raised
    ``SimError`` for any kind outside ``_SUPPORTED_KINDS``. The
    keep-alive decoded to ``other``, so a real client killed the
    server with its first one, seconds after connecting.

    [[client-commands]] has carried this command (JS class ``dc``) all
    along. Our bot has never SENT one, and that is the whole reason a
    sim soaked for months never met it: the corpus we validated
    against was written by the only client that does not send it.
    """
    _bot, server, link, table = boot_seam()

    link.send_page_frame(_query_frame(table, CMD_KEEPALIVE))

    assert link.sent_commands[-1] == "keepalive"
    assert server.advance_tick() == server.advance_tick()


def test_a_heartbeat_leaves_the_next_real_command_untouched() -> None:
    """A client beats constantly; the beat must cost the next command nothing.

    Asserted against a control session that sent the same real command
    with no heartbeat beside it, so the comparison is the map answer
    itself rather than a shape written down by hand.
    """
    _control_bot, control, control_link, control_table = boot_seam()
    control_link.send_page_frame(_query_frame(control_table, CMD_MAP_OPEN))
    expected = [message["msg_type"] for message in control.advance_tick()]

    _bot, server, link, table = boot_seam()
    link.send_page_frame(_query_frame(table, CMD_KEEPALIVE))
    link.send_page_frame(_query_frame(table, CMD_MAP_OPEN))
    link.send_page_frame(_query_frame(table, CMD_KEEPALIVE))

    assert [message["msg_type"] for message in server.advance_tick()] == expected


def test_enter_game_is_answered_with_the_join_burst() -> None:
    """THE JOIN BURST IS AN ANSWER, NOT A PUSH.

    343 archived sends, every one answered, self-caused tokens
    ``49 49 5A 3Dself`` per send — the tail of the burst
    ``handshake`` builds. The sim pushed that burst unprompted at
    connect because OUR bot never sends this command: ``enter_game()``
    sat in two production classes with zero callers while the bot
    joined through the lobby instead ([[client-commands]]).

    Like the keep-alive, this decoded to ``other`` and so killed the
    server on arrival — the second of three such commands found in the
    same sweep.
    """
    _bot, server, link, table = boot_seam()

    link.send_page_frame(_query_frame(table, CMD_ENTER_GAME))
    burst = server.advance_tick()

    assert link.sent_commands[-1] == "enter_game"
    kinds = [message["msg_type"] for message in burst]
    assert kinds[:2] == [0x21, 0x3E]
    assert kinds.count(0x49) >= 2
    assert 0x5A in kinds


def test_an_inventory_request_is_answered_with_a_snapshot() -> None:
    """The 'i' key draws a 0x49.

    Four archived sends, every one answered with an inventory —
    thin, but the command's own name and its answer agree.
    """
    _bot, server, link, table = boot_seam()

    link.send_page_frame(_query_frame(table, CMD_INVENTORY))
    answered = server.advance_tick()

    assert link.sent_commands[-1] == "inventory"
    snapshots = [m for m in answered if m["msg_type"] == 0x49]
    assert len(snapshots) == 1
    assert snapshots[0]["show"] is True


def test_a_command_with_no_measured_law_is_refused_by_name() -> None:
    """0x44 is real, observed seven times, and NOT modelled.

    Its payloads vary every send and the sim has no law for it, so it
    refuses rather than inventing a response — and the refusal names
    the command and its byte instead of the build phase the old
    message described.
    """
    _bot, _server, link, table = boot_seam()

    with pytest.raises(SimError, match="no modelled law"):
        link.send_page_frame(_query_frame(table, CMD_UNMODELLED_COMBAT))
