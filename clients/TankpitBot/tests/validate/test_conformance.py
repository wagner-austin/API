"""Capture-replay conformance: the sim replays captures tick by tick against their wire."""

from __future__ import annotations

import base64
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.analysis.scan import decode_session_frames
from tankpit_bot.protocol.command_builders import build_query_command
from tankpit_bot.protocol.commands import (
    CMD_ACTIVE_FORCES,
    CMD_KEEPALIVE,
    CMD_RADAR,
    COMMAND_PREFIX,
    TYPE_QUERY,
)
from tankpit_bot.protocol.encoders import encode_envelope_body
from tankpit_bot.protocol.types import MovementResponseDict
from tankpit_bot.resources import data_directory
from tankpit_bot.sim.commands import ClientCommandKind, decode_client_command
from tankpit_bot.sim.run import run_sim_session
from tankpit_bot.sim.scenarios import SIM_FIELD
from tankpit_bot.types import decode_capture_session
from tankpit_bot.validate.conformance import (
    LATEST_CAPTURE_NAME,
    discover_captures,
    places_self,
    replay_capture,
    run_conformance,
)
from tankpit_bot.validate.conformance_types import DivergenceDict, ReplaySkipReason
from tankpit_bot.validate.conformance_wire import read_replay
from tests.analysis._capture_fixtures import (
    FOREIGN_TANK,
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
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

#: What the sim answers a radar with on the handmade field: the spent
#: radar's 0x49, the 0x4F scan, then the 0x46 verdict. The handmade
#: capture's real server answered with the bare 0x46, so the tick diverges.
SIM_RADAR_ANSWER = ["49", "4F", "46"]

_LISTING = b"+1|Practice|1|0,0,0,0,0,0,0|-1|p|field01.gif|2026"
_JOIN = b"=1|2026|red-9|1|0|0|0|0"


def _placed(tank_id: int, x: int, y: int) -> bytes:
    """A ciphered 0x3D placement, tunneled in its 0x2E envelope as on the wire.

    A bare ``=`` frame is the lobby's text join confirmation; the binary
    placement only ever arrives inside the envelope.
    """
    payload = encode_envelope_body(
        MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=tank_id,
            x=x,
            y=y,
            direction=8,
            damage_state=3,
            rank=1,
            lb_score=0,
            carrying=0,
        )
    )
    return _ciphered(bytes([0x2E]) + payload)


def _query(command_id: int) -> JSONObject:
    """A sent query command; its time is set by the caller's ordering."""
    return _sent(_payload(_command(build_query_command(command_id))), timestamp_ms=0)


def _at(message: JSONObject, timestamp_ms: int) -> JSONObject:
    """The message re-stamped."""
    stamped = dict(message)
    stamped["timestamp_ms"] = timestamp_ms
    return stamped


def _handmade(joined: bool = True) -> str:
    """A capture whose ticks exercise every branch of the replay.

    In order: a radar sent before the server named the client (not
    compared), the introduction and placement, a radar the real server
    answered with a bare 0x46 (the sim answers more, so it diverges), an
    active-forces query the sim has no law for (unmodelled) and a
    keepalive alone (not compared).
    """
    room: list[JSONObject] = (
        [_received(_payload(_LISTING), 100), _received(_payload(_JOIN), 101)] if joined else []
    )
    return _session_json(
        messages=[
            *room,
            _at(_query(CMD_RADAR), 150),
            _received(_payload(_radar_result()), 200),
            _received(_payload(_tank_info(OWN_TANK), _placed(OWN_TANK, 40, 40)), 3000),
            _at(_query(CMD_RADAR), 3100),
            _received(_payload(_radar_result()), 5000),
            _at(_query(CMD_ACTIVE_FORCES), 5100),
            _received(_payload(_radar_result()), 7000),
            _at(_query(CMD_KEEPALIVE), 7100),
            _received(_payload(_radar_result()), 9000),
        ]
    )


@pytest.fixture()
def terrain_loads() -> Generator[list[Path], None, None]:
    """Load every field as open ground, recording which GIF was asked for.

    Yields:
        The list every terrain load appends its path to.
    """
    original = _test_hooks.load_terrain_map
    loads: list[Path] = []

    def load_open_terrain(gif_path: Path) -> TerrainMapProtocol:
        """An all-open field."""
        loads.append(gif_path)
        return InMemoryTerrainMap()

    _test_hooks.load_terrain_map = load_open_terrain
    yield loads
    _test_hooks.load_terrain_map = original


def _replay_text(text: str) -> tuple[int, int, int, list[tuple[str, bool]]]:
    """Replay a capture text on open field01 and return its totals."""
    session = decode_capture_session(narrow_json_to_dict(load_json_str(text)))
    result = replay_capture(
        read_replay(decode_session_frames(session)),
        text,
        InMemoryTerrainMap(),
        session="s",
        field=SIM_FIELD,
    )
    totals = result.totals
    return totals["ticks"], totals["matched"], totals["unmodelled"], list(result.groups)


def test_the_sim_replays_its_own_capture_tick_for_tick(
    fake_fs: FakeFileSystem, terrain_loads: list[Path]
) -> None:
    """A capture the sim itself served replays with every commanded tick matching.

    This is the harness's own control: anchoring, batch pairing and the
    token reduction together must not invent a divergence where there is
    none, so a sim-served session must come back identical.
    """
    fake_fs.write_text(data_directory() / SIM_FIELD, "fake-gif-bytes")
    run = run_sim_session(20, archive_dir=Path("runs") / "sim", opponent=True, stamp="20261005-1")
    text = fake_fs.get_written_files()[run["capture_path"]]
    ticks, matched, unmodelled, _ = _replay_text(text)
    assert terrain_loads != []
    assert ticks > 10
    assert (matched, unmodelled) == (ticks, 0)


def test_a_handmade_capture_counts_mismatch_and_unmodelled_ticks() -> None:
    """Only commanded ticks after the introduction are compared."""
    ticks, matched, unmodelled, groups = _replay_text(_handmade())
    assert (ticks, matched, unmodelled) == (1, 0, 1)
    assert groups == [("radar", False)]


def test_active_forces_has_no_sim_law() -> None:
    """The unmodelled case above rests on this: the query decodes as OTHER."""
    command = decode_client_command(bytes([TYPE_QUERY, CMD_ACTIVE_FORCES]))
    assert command["kind"] is ClientCommandKind.OTHER
    assert ord("!") == COMMAND_PREFIX


def test_places_self_needs_an_introduction_then_a_placement() -> None:
    """A placement before the introduction, or of another tank, does not count."""

    def capture_of(*messages: JSONObject) -> bool:
        session = decode_capture_session(
            narrow_json_to_dict(load_json_str(_session_json(messages=list(messages))))
        )
        return places_self(read_replay(decode_session_frames(session)))

    assert not capture_of(_received(_payload(_placed(OWN_TANK, 1, 1)), 100))
    assert not capture_of(
        _received(_payload(_tank_info(OWN_TANK)), 100),
        _received(_payload(_placed(FOREIGN_TANK, 1, 1)), 101),
    )
    assert capture_of(
        _received(_payload(_tank_info(OWN_TANK)), 100),
        _received(_payload(_placed(OWN_TANK, 1, 1)), 101),
    )


def test_discover_finds_every_capture_once_without_latest_copies(tmp_path: Path) -> None:
    """Recursive, deduplicated across overlapping roots, ``latest`` left out."""
    nested = tmp_path / "bot" / "artax"
    nested.mkdir(parents=True)
    for path in (
        tmp_path / "bot" / "a.capture_session.json",
        nested / "b.capture_session.json",
        nested / LATEST_CAPTURE_NAME,
        tmp_path / "bot" / "a.world.json",
    ):
        path.write_text("{}", encoding="utf-8")
    found = discover_captures([tmp_path / "bot", nested, tmp_path / "absent"])
    assert found == [tmp_path / "bot" / "a.capture_session.json", nested / "b.capture_session.json"]


def _unframed() -> str:
    """A capture whose only payload claims a frame longer than it holds."""
    payload = base64.b64encode(bytes([0x40, 0x00, 0x21])).decode("ascii")
    return _session_json(messages=[_received(payload, 100)])


def _unlisted_field() -> str:
    """A capture that joined a room on a field this distribution lacks."""
    listing = b"+5|Desert|5|0,0,0,0,0,0,0|2|d|field99.gif|2026"
    return _session_json(
        messages=[
            _received(_payload(listing), 100),
            _received(_payload(b"=5|2026|red-9|1|0|0|0|0"), 101),
        ]
    )


def _never_placed() -> str:
    """A capture that joined but never placed its own tank."""
    return _session_json(
        messages=[_received(_payload(_LISTING), 100), _received(_payload(_JOIN), 101)]
    )


def test_run_replays_what_it_can_and_names_why_it_skips_the_rest(
    tmp_path: Path, terrain_loads: list[Path]
) -> None:
    """Every skip reason the run itself decides, and two replays sharing one terrain."""
    files = {
        "a-no-magic": _session_json(magic=None),
        "b-unframed": _unframed(),
        "c-no-room": _handmade(joined=False),
        "d-field-missing": _unlisted_field(),
        "e-no-self": _never_placed(),
        "f-replayed": _handmade(),
        "g-replayed": _handmade(),
    }
    paths = [tmp_path / f"{name}.capture_session.json" for name in files]
    for path, text in zip(paths, files.values(), strict=True):
        path.write_text(text, encoding="utf-8")
    report = run_conformance(paths)
    assert [(s["session"], s["reason"]) for s in report["skipped"]] == [
        (str(paths[0]), ReplaySkipReason.NO_MAGIC),
        (str(paths[1]), ReplaySkipReason.UNFRAMED_PAYLOAD),
        (str(paths[2]), ReplaySkipReason.NO_ROOM),
        (str(paths[3]), ReplaySkipReason.FIELD_MISSING),
        (str(paths[4]), ReplaySkipReason.NO_SELF),
    ]
    assert report["skipped"][3]["detail"] == "field99.gif"
    assert [s["session"] for s in report["sessions"]] == [str(paths[5]), str(paths[6])]
    assert report["sessions"][0]["field"] == SIM_FIELD
    assert terrain_loads == [data_directory() / SIM_FIELD]
    assert [(g["commands"], g["ticks"], g["matched"]) for g in report["groups"]] == [
        ("radar", 2, 0)
    ]
    assert report["divergences"] == [
        DivergenceDict(
            commands="radar",
            live=["46"],
            sim=SIM_RADAR_ANSWER,
            count=2,
            example_session=str(paths[5]),
            example_timestamp_ms=5000,
        )
    ]


def test_groups_and_divergences_are_ordered_most_frequent_first(
    tmp_path: Path, terrain_loads: list[Path]
) -> None:
    """Most frequent first; the single doubled tick sorts after the two radars."""
    capture = narrow_json_to_dict(load_json_str(_handmade()))
    messages = narrow_json_to_list(capture["messages"])
    extra_tick: list[JSONObject] = [
        _at(_query(CMD_RADAR), 9100),
        _at(_query(CMD_RADAR), 9101),
        _received(_payload(_radar_result()), 11000),
    ]
    capture["messages"] = [*messages, *extra_tick]
    path = tmp_path / "double.capture_session.json"
    path.write_text(dump_json_str(capture), encoding="utf-8")
    report = run_conformance([path])
    assert len(terrain_loads) == 1
    assert [g["commands"] for g in report["groups"]] == ["radar", "radar+radar"]
    assert [d["commands"] for d in report["divergences"]] == ["radar", "radar+radar"]


def test_an_empty_run_reports_nothing() -> None:
    """No paths, no sessions, no groups."""
    report = run_conformance([])
    assert report == {"sessions": [], "skipped": [], "groups": [], "divergences": []}
