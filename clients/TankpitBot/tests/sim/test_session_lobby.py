"""The seam's lobby half: routing, the page-client frames, and refusals."""

from __future__ import annotations

import pytest

from tankpit_bot._test_hooks import ensure_autoscroll_off
from tankpit_bot.browser.room_join import join_room
from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.sim.lobby import SIM_ACCOUNT, SimLobby
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.session import SimCDPSession, deliver_batch
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.wire.helpers import EncodeError
from tests.in_memory_terrain_map import InMemoryTerrainMap

_MAGIC = "lobbymagic5uk3et4epiexu"


def _world() -> SimWorldDict:
    """One client tank at (10, 10)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 2, 1, 10, 10, 1100)
    return world


def _link(*, with_lobby: bool = True) -> SimCDPSession:
    """A seam link, with or without its lobby half."""
    server = SimServer(_world(), InMemoryTerrainMap(), client_id=9)
    lobby = SimLobby(SIM_ACCOUNT) if with_lobby else None
    return SimCDPSession(server, _MAGIC, lobby)


def _received_bodies(link: SimCDPSession) -> list[bytes]:
    """Every received frame body the link recorded, in order."""
    bodies: list[bytes] = []
    for captured in link.wire_log:
        if captured["direction"] == "received":
            bodies.extend(split_payload_frames(captured["payload"]))
    return bodies


def _sent_bodies(link: SimCDPSession) -> list[bytes]:
    """Every sent frame body the link recorded, in order."""
    bodies: list[bytes] = []
    for captured in link.wire_log:
        if captured["direction"] == "sent":
            bodies.extend(split_payload_frames(captured["payload"]))
    return bodies


def test_the_table_is_derived_from_the_magic() -> None:
    """A link holds one cipher fact, not two that can disagree."""
    from tankpit_bot.capture.xor import build_session_xor_table

    link = _link()
    assert link.magic == _MAGIC
    assert link.table == build_session_xor_table(_MAGIC)


def test_opening_the_lobby_sends_auth_and_takes_the_room_list() -> None:
    """The link stands in for the page client, so it sends the AUTH.

    The room list arrives as the server's ANSWER rather than as
    pre-seeded state ([[session-state-deglobalisation]]).
    """
    link = _link()
    link.open_lobby()

    assert [body[:5] for body in _sent_bodies(link)] == [b"%AUTH"]
    assert [body[:2] for body in _received_bodies(link)] == [b"+1", b"+5"]


def test_each_lobby_reply_is_its_own_payload() -> None:
    """One frame per payload, because the page hook pushes per message.

    ``decode_captured_body`` treats a second frame in one payload as
    corruption, so a batched room list would reach the join flow as an
    error rather than as two rooms.
    """
    link = _link()
    link.open_lobby()

    received = [c for c in link.wire_log if c["direction"] == "received"]
    assert [len(split_payload_frames(c["payload"])) for c in received] == [1, 1]
    assert link.raw_messages == [c["payload"] for c in received]


def test_the_production_join_flow_reaches_a_room_over_the_seam() -> None:
    """The REAL ``join_room`` drives the sim lobby end to end."""
    link = _link()
    link.open_lobby()

    assert join_room(link, link) is True

    lobby = link.lobby
    if lobby is None:
        raise AssertionError("the link was built with a lobby")
    assert lobby.entered_room_id == "1"
    assert get_world_service().selected_room == "1"


def test_the_join_flow_registers_the_rooms_field_image() -> None:
    """The selected room's terrain comes from the room list, not a hack.

    ``_boot`` used to hand-register a room named ``sim`` because the
    lobby was skipped; without a selected room the bot's decision
    terrain stays ``None`` for the whole session.
    """
    link = _link()
    link.open_lobby()
    join_room(link, link)

    assert get_world_service().room_images["1"] == "field01.gif"


def test_command_frames_still_route_to_the_server() -> None:
    """One socket, two protocols: '!' frames are commands as before."""
    from tankpit_bot.capture.xor import xor_decode_body
    from tankpit_bot.protocol.commands import CMD_MOVE, COMMAND_PREFIX, TYPE_MOVEMENT

    link = _link()
    link.open_lobby()
    plaintext = bytes([TYPE_MOVEMENT, CMD_MOVE, 12, 10])
    body = bytes([COMMAND_PREFIX]) + xor_decode_body(plaintext, link.table)

    link.send("Runtime.evaluate", {"expression": f"atob('{_b64(_frame(body))}')"})

    assert link.sent_commands == ["move"]
    assert link.server.world["tanks"][9]["x"] != 12  # queued, not yet ticked


def test_a_plaintext_frame_without_a_lobby_is_refused() -> None:
    """An in-room link has no lobby, and silence would eat the join."""
    link = _link(with_lobby=False)

    with pytest.raises(EncodeError, match="plaintext frame 0x2A"):
        link.send("Runtime.evaluate", {"expression": f"atob('{_b64(_frame(b'*1'))}')"})


def test_opening_a_lobbyless_link_is_refused() -> None:
    """The page-client's opening ritual needs a lobby to talk to."""
    with pytest.raises(EncodeError, match="open_lobby"):
        _link(with_lobby=False).open_lobby()


def test_the_magic_and_tpclient_queries_answer_from_sim_truth() -> None:
    """The room-entry step asks for the magic, the script, and its key."""
    from tankpit_bot.browser.cdp_helpers import (
        get_magic_key,
        get_tpclient_url,
        load_tpclient_static_key,
    )

    link = _link()

    assert get_magic_key(link) == _MAGIC
    url = get_tpclient_url(link)
    assert url.endswith("tpclient.js")
    assert len(load_tpclient_static_key(link, url)) == 1000


def test_pressing_a_sends_the_flipped_state_and_takes_the_echo() -> None:
    """The toggle is a SET, not a bare toggle: the archive's 75 sends
    are all ``A1`` or ``A0`` and every one comes back identical."""
    link = _link()
    link.open_lobby()

    link.keyboard.press("a")
    assert link.autoscroll_enabled is True
    link.keyboard.press("a")
    assert link.autoscroll_enabled is False

    toggles = [body for body in _sent_bodies(link) if body[:1] == b"A"]
    echoes = [body for body in _received_bodies(link) if body[:1] == b"A"]
    assert toggles == [b"A1", b"A0"]
    assert echoes == [b"A1", b"A0"]


def test_pressing_any_other_key_is_refused() -> None:
    """A silently-ignored press would let a probe believe it acted."""
    with pytest.raises(EncodeError, match="unmodeled key press"):
        _link().keyboard.press("b")


def test_the_production_enforcer_verifies_autoscroll_off_over_the_seam() -> None:
    """The real press-and-verify dance runs against the sim's channel."""
    link = _link()
    link.open_lobby()
    deliver_batch([], link.server.handshake(), link)
    _drain(link)

    ensure_autoscroll_off(link, link.wire_log)

    assert link.autoscroll_enabled is False


def test_the_production_graceful_quit_reaches_the_lobby() -> None:
    """A session used to just stop; now the bot's own quit lands.

    ``build_quit_command`` is the bot's, not a sim invention — the
    plain un-XOR'd ``-`` it sends at teardown so the server records a
    deliberate lobby exit instead of a socket drop.
    """
    from tankpit_bot.protocol.commands import build_quit_command

    link = _link()
    link.open_lobby()

    link.route_client_payload(_b64(build_quit_command()))

    assert _sent_bodies(link)[-1] == b"-"
    lobby = link.lobby
    if lobby is None:
        raise AssertionError("the link was built with a lobby")
    assert lobby.quit is True


def test_the_statistics_key_press_puts_the_command_on_the_wire() -> None:
    """Pressing ``c`` is what the PAGE turns into CMD_STATISTICS.

    The account-stats capture dispatches the key through CDP and reads
    the panel; in a browser the page's script sends the command frame,
    which is what the server's 0x56 answers. Only the down edge sends —
    the capture dispatches keyDown then keyUp
    ([[session-state-deglobalisation]]).
    """
    link = _link()
    link.open_lobby()

    for event_type in ("keyDown", "keyUp"):
        link.send("Input.dispatchKeyEvent", {"type": event_type, "key": "c"})

    assert link.sent_commands == ["statistics"]
    assert [m["msg_type"] for m in link.server.advance_tick()].count(0x56) == 1


def test_an_unmodeled_dispatched_key_is_refused() -> None:
    """A swallowed press would let a probe believe it acted."""
    link = _link()
    with pytest.raises(EncodeError, match="unmodeled dispatched key"):
        link.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "z"})


def test_a_key_event_without_params_is_refused() -> None:
    """A dispatch with nothing to dispatch is a harness bug."""
    with pytest.raises(EncodeError, match="without params"):
        _link().send("Input.dispatchKeyEvent", None)


def test_waiting_never_actually_sleeps() -> None:
    """The sim answers synchronously; a real sleep is dead soak time."""
    link = _link()
    link.wait_for_timeout(10_000.0)
    assert link.url.startswith("https://")


def _frame(body: bytes) -> bytes:
    """Length-prefix one frame body."""
    from tankpit_bot.wire.helpers import pack16

    return pack16(len(body)) + body


def _b64(data: bytes) -> str:
    """Base64 for the injected-send expression."""
    import base64

    return base64.b64encode(data).decode("ascii")


def _drain(link: SimCDPSession) -> None:
    """Feed the handshake into the world service, as a tick would.

    ``ensure_autoscroll_off`` waits for ``self_state``, which only the
    processed handshake provides.
    """
    from tankpit_bot.sniffer.decoders import process_received_message

    for captured in link.wire_log:
        if captured["direction"] == "received":
            process_received_message(get_world_service(), captured["payload"], link.table)
