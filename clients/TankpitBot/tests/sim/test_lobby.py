"""Law 0 — the sim lobby: room list, select, enter, toggles, quit."""

from __future__ import annotations

import pytest

from tankpit_bot.parser import RoomInfo
from tankpit_bot.parser_messages import parse_room_info
from tankpit_bot.protocol.decoders import decode_join_confirm, try_decode_plaintext_ack
from tankpit_bot.sim.lobby import (
    QUIT_BODY,
    SIM_ACCOUNT,
    SIM_ROOMS,
    SimAccountDict,
    SimLobby,
    build_auth_frame,
)

_MAGIC = "simmagic5uk3et4epiexu"
_AUTH = build_auth_frame("62913", "0" * 32, "0", _MAGIC)


def _lobby() -> SimLobby:
    """A lobby over the standard sim rooms and account."""
    return SimLobby(SIM_ACCOUNT)


def test_auth_is_answered_with_one_frame_per_advertised_room() -> None:
    """The archive's post-auth burst is the room list, one row each.

    Every one of the 285 archived sessions opens ``%AUTH`` then two
    ROOM_LIST rows ([[session-state-deglobalisation]]).
    """
    rows = _lobby().handle_frame(_AUTH)

    assert [row.decode("utf-8") for row in rows] == [
        "+1|Practice|1|0,0,0,0,0,0,0|2|p|field01.gif|2026",
        "+5|World (Desert)|5|1,1,1,0,1,0,0|2|n|field05.gif|2026",
    ]


def test_room_rows_parse_with_the_production_room_parser() -> None:
    """The rows the sim emits are rows the bot can actually read."""
    rows = _lobby().handle_frame(_AUTH)

    parsed = [parse_room_info(row.decode("utf-8")[1:]) for row in rows]
    assert [entry["name"] for entry in parsed] == ["Practice", "World (Desert)"]
    assert [entry["image"] for entry in parsed] == ["field01.gif", "field05.gif"]
    assert parsed[0]["default_troop"] == 2


def test_select_answers_with_a_join_confirm_the_decoder_accepts() -> None:
    """``*1`` draws ``=1|...``, and the production decoder reads it."""
    frames = _lobby().handle_frame(b"*1")

    assert len(frames) == 1
    confirm = decode_join_confirm(frames[0])
    assert confirm["name"] == SIM_ACCOUNT["name"]
    assert confirm["join_date"] == SIM_ACCOUNT["join_date"]


def test_selecting_a_room_that_is_not_advertised_answers_nothing() -> None:
    """An unlisted room id draws no confirm — the client times out."""
    assert _lobby().handle_frame(b"*99") == []


def test_entering_a_room_answers_the_response_and_records_the_room() -> None:
    """``+1|troop|x|y|metadata`` draws ``$1|0``, the only observed code.

    All 286 archived enter responses are ``$1|0``.
    """
    lobby = _lobby()
    assert lobby.entered_room_id is None

    frames = lobby.handle_frame(b"+1|2|128|128|metadata-tail")

    assert [frame.decode("utf-8") for frame in frames] == ["$1|0"]
    assert lobby.entered_room_id == "1"


def test_entering_an_unadvertised_room_answers_nothing_and_enters_nothing() -> None:
    """A room the lobby never listed cannot be entered."""
    lobby = _lobby()

    assert lobby.handle_frame(b"+99|2|128|128|metadata") == []
    assert lobby.entered_room_id is None


def test_the_autoscroll_toggle_is_echoed_back_verbatim() -> None:
    """The server echoes ``A1``/``A0``; it does not re-derive them.

    The archive's 75 toggle sends are all one or the other and every
    one comes back identical — the setting is SET by the client, not
    toggled by the server.
    """
    lobby = _lobby()

    assert lobby.handle_frame(b"A1") == [b"A1"]
    assert lobby.handle_frame(b"A0") == [b"A0"]


def test_the_toggle_echo_is_a_decodable_ack() -> None:
    """What comes back is what the production ack predicate reads."""
    echo = _lobby().handle_frame(b"A1")[0]

    ack = try_decode_plaintext_ack(echo)
    if ack is None:
        raise AssertionError("the toggle echo must decode as a plaintext ack")
    assert ack == {"msg_type": "autoscroll_ack", "enabled": True}


def test_quit_is_recorded_and_answered_with_silence() -> None:
    """The client's ``-`` ends the session; the server says nothing."""
    lobby = _lobby()
    assert lobby.quit is False

    assert lobby.handle_frame(QUIT_BODY) == []
    assert lobby.quit is True


def test_an_unknown_frame_draws_no_reply() -> None:
    """A frame the lobby has no rule for is not answered.

    Silence rather than a raise: the lobby shares a socket with the
    command channel, and the caller has already decided this frame is
    not a command.
    """
    assert _lobby().handle_frame(b"\x99unknown") == []


def test_a_custom_room_set_replaces_the_default_advertisement() -> None:
    """The advertised rooms are a parameter, not a module constant."""
    only = RoomInfo(
        room_id="7",
        name="Sandbox",
        field_id=7,
        game_modes="0,0,0,0,0,0,0",
        default_troop=1,
        mode_code="p",
        image="field07.gif",
        year="2026",
    )
    lobby = SimLobby(SIM_ACCOUNT, (only,))

    assert [row.decode("utf-8") for row in lobby.handle_frame(_AUTH)] == [
        "+7|Sandbox|7|0,0,0,0,0,0,0|1|p|field07.gif|2026"
    ]
    assert lobby.handle_frame(b"*1") == []


def test_the_auth_frame_carries_the_session_magic_last() -> None:
    """The production extractor lifts the magic straight back out.

    The AUTH frame is the page client's, and the bot learns the
    session cipher from exactly this tail — so a sim that writes one
    has to write one the real reader can read.
    """
    from tankpit_bot.protocol.codec import extract_magic_from_auth_payload
    from tankpit_bot.wire.helpers import pack16

    assert extract_magic_from_auth_payload(pack16(len(_AUTH)) + _AUTH) == _MAGIC


def test_a_join_confirm_reports_the_accounts_equipment_counts() -> None:
    """All four trailing counts ride the confirm."""
    account = SimAccountDict(
        join_date="Feb. 02, 2020", name="red-9", rank=3, equipment=(1, 2, 3, 4)
    )
    frame = SimLobby(account).handle_frame(b"*1")[0]

    assert frame.decode("utf-8") == "=1|Feb. 02, 2020|red-9|3|1|2|3|4"


@pytest.mark.parametrize("room", SIM_ROOMS)
def test_every_advertised_room_is_selectable_and_enterable(room: RoomInfo) -> None:
    """A room the lobby lists is a room the lobby will let you into."""
    lobby = _lobby()
    room_id = room["room_id"]

    assert lobby.handle_frame(f"*{room_id}".encode()) != []
    assert lobby.handle_frame(f"+{room_id}|2|128|128|x".encode()) != []
    assert lobby.entered_room_id == room_id
