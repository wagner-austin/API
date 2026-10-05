"""Reading a capture as server ticks: command/batch pairing and the joined field."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict

from tankpit_bot.analysis.scan import decode_session_frames
from tankpit_bot.protocol.command_builders import build_move_command, build_query_command
from tankpit_bot.protocol.commands import CMD_KEEPALIVE, CMD_RADAR, COMMAND_PREFIX
from tankpit_bot.sim.commands import ClientCommandKind
from tankpit_bot.types import decode_capture_session
from tankpit_bot.validate.conformance_wire import BURST_GAP_MS, ReplayCapture, read_replay
from tests.analysis._capture_fixtures import (
    OWN_TANK,
    _ciphered,
    _command,
    _payload,
    _radar_result,
    _received,
    _sent,
    _session_json,
    _tank_info,
)

_PRACTICE_LISTING = b"+1|Practice|1|0,0,0,0,0,0,0|-1|p|field01.gif|2026"
_DESERT_LISTING = b"+5|Desert|5|0,0,0,0,0,0,0|2|d|field05.gif|2026"


def _replay(messages: list[JSONObject]) -> ReplayCapture:
    """Read a capture built from these messages."""
    session = decode_capture_session(
        narrow_json_to_dict(load_json_str(_session_json(messages=messages)))
    )
    return read_replay(decode_session_frames(session))


def _kinds(capture: ReplayCapture) -> list[tuple[int, list[ClientCommandKind], list[int | str]]]:
    """Each tick as (time, command kinds, received message types)."""
    return [
        (
            tick.timestamp_ms,
            [c["kind"] for c in tick.commands],
            [m["msg_type"] for m in tick.received],
        )
        for tick in capture.ticks
    ]


def test_each_batch_is_one_tick_holding_the_commands_sent_before_it() -> None:
    """A run of received messages is one tick; a gap opens the next."""
    capture = _replay(
        [
            _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
            _sent(_payload(_command(build_query_command(CMD_RADAR))), timestamp_ms=1100),
            _sent(_payload(_command(build_query_command(CMD_KEEPALIVE))), timestamp_ms=1150),
            _received(_payload(_radar_result()), timestamp_ms=1600),
            _received(_payload(_radar_result(found=False)), timestamp_ms=1600 + BURST_GAP_MS),
            _received(_payload(_radar_result()), timestamp_ms=1601 + 2 * BURST_GAP_MS),
            _sent(_payload(_command(build_move_command(4, 4))), timestamp_ms=4000),
        ]
    )
    assert _kinds(capture) == [
        (1000, [], [0x21]),
        (
            1600 + BURST_GAP_MS,
            [ClientCommandKind.RADAR, ClientCommandKind.KEEPALIVE],
            [0x46, 0x46],
        ),
        (1601 + 2 * BURST_GAP_MS, [], [0x46]),
    ]


def test_a_command_closes_the_open_batch() -> None:
    """Messages either side of a sent command belong to different ticks."""
    capture = _replay(
        [
            _received(_payload(_tank_info(OWN_TANK)), timestamp_ms=1000),
            _sent(_payload(_command(build_query_command(CMD_RADAR))), timestamp_ms=1005),
            _received(_payload(_radar_result()), timestamp_ms=1010),
        ]
    )
    assert _kinds(capture) == [(1000, [], [0x21]), (1010, [ClientCommandKind.RADAR], [0x46])]


def test_frames_that_are_not_commands_are_passed_over() -> None:
    """A non-``!`` sent frame and an undecodable command open nothing."""
    capture = _replay(
        [
            _sent(_payload(_ciphered(bytes([0x2B, 0x01, 0x02]))), timestamp_ms=1000),
            _sent(_payload(_ciphered(bytes([COMMAND_PREFIX, 0x02]))), timestamp_ms=1001),
            _received(_payload(_radar_result()), timestamp_ms=1500),
        ]
    )
    assert _kinds(capture) == [(1500, [], [0x46])]


def test_the_last_confirmed_join_of_a_listed_room_names_the_field() -> None:
    """Listings map rooms to fields; a join of an unlisted room is ignored."""
    capture = _replay(
        [
            _received(_payload(_PRACTICE_LISTING, _DESERT_LISTING), timestamp_ms=900),
            _received(_payload(b"+not a room listing"), timestamp_ms=901),
            _received(_payload(b"=1|2026|red-9|1|0|0|0|0"), timestamp_ms=902),
            _received(_payload(b"=5|2026|red-9|1|0|0|0|0"), timestamp_ms=903),
            _received(_payload(b"=77|2026|red-9|1|0|0|0|0"), timestamp_ms=904),
        ]
    )
    assert capture.field_image == "field05.gif"


def test_a_capture_that_never_joins_names_no_field() -> None:
    """Without a confirmed join there is no field, and no tick is invented."""
    capture = _replay([_received(_payload(_PRACTICE_LISTING), timestamp_ms=900)])
    assert capture.field_image is None
    assert capture.ticks == ()
