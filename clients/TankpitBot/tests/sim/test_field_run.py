"""Several production bots on one sim field, end to end.

The bots are REAL — two ``Bot`` instances playing the unmodified
``_tick_once`` against one :class:`SimServer`, each over its own link —
and their artifacts land in the fake file system. The arena run is
deterministic on the all-ground test terrain: each bot hunts the other,
finds no landing it will commit to, and leaves through the production
``no_viable_targets`` exit, tank 10 at round 8 and tank 9 at round 9.
That is what lets the departure path be asserted rather than hoped for.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_list

from tankpit_bot import _test_hooks
from tankpit_bot.analysis.scan import decode_session_frames
from tankpit_bot.bot.session_exit import SessionExitReason
from tankpit_bot.protocol import decode_message
from tankpit_bot.protocol.types import BinaryMessage, ShootEventDict
from tankpit_bot.resources import data_directory
from tankpit_bot.sim.field_clients import RIVAL_TEAM, FieldSeatError
from tankpit_bot.sim.field_run import (
    DEFAULT_FIELD_RUNS_ROOT,
    FieldRunResultDict,
    make_field_world,
    run_field_session,
)
from tankpit_bot.sim.practice_room import PracticeRoomDriver
from tankpit_bot.sim.run import main
from tankpit_bot.sim.scenarios import SIM_CLIENT_ID, SIM_ENEMY_ID, SIM_FIELD
from tankpit_bot.sim.world import make_sim_tank, make_sim_world
from tankpit_bot.types import decode_capture_session
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

_ARCHIVE = Path("runs") / "field"
_ROOT = Path("runs") / "fieldroot"
_ENVELOPE = 0x2E


@pytest.fixture()
def _field_fs(fake_fs: FakeFileSystem) -> Generator[FakeFileSystem, None, None]:
    """A field GIF and all-ground terrain for the sessions to seed on.

    Yields:
        The installed fake file system.
    """
    fake_fs.write_text(data_directory() / SIM_FIELD, "fake-gif-bytes")
    real_terrain = _test_hooks.load_terrain_map

    def all_ground(gif_path: Path) -> InMemoryTerrainMap:
        """Open ground everywhere.

        Args:
            gif_path: Ignored.

        Returns:
            The terrain.
        """
        del gif_path
        return InMemoryTerrainMap()

    _test_hooks.load_terrain_map = all_ground
    try:
        yield fake_fs
    finally:
        _test_hooks.load_terrain_map = real_terrain


def _arena(rounds: int = 40) -> FieldRunResultDict:
    """Two bots in the arena, the world named."""
    return run_field_session(
        rounds,
        clients=2,
        archive_dir=_ARCHIVE,
        stamp="arena",
        layout="bot-20260706-223721",
        population_seed=7,
        runs_root=str(_ROOT),
    )


def _received(fake_fs: FakeFileSystem, capture: Path, msg_type: int) -> list[BinaryMessage]:
    """Every message of one type a connection's capture received, decoded.

    In-play messages ride the binary envelope, whose frame byte is 0x2E
    with the message's own type inside the ciphered body; the lobby's
    plaintext frames carry other bytes and are not messages.
    """
    session = decode_capture_session(narrow_json_to_dict(load_json_str(fake_fs.read_text(capture))))
    decoded = [
        decode_message(frame["msg_type"], frame["body"])
        for frame in decode_session_frames(session)
        if frame["direction"] == "received" and frame["msg_type"] == _ENVELOPE
    ]
    return [message for message in decoded if message["msg_type"] == msg_type]


def _shots(messages: list[BinaryMessage]) -> list[ShootEventDict]:
    """The 0x53 shot events among decoded messages."""
    return [message for message in messages if message["msg_type"] == 0x53]


def test_the_arena_field_holds_the_bots_and_not_the_scripted_opponent() -> None:
    """The rivals are the opposition, so the scripted opponent is removed."""
    world = make_field_world(practice=False)

    assert SIM_CLIENT_ID in world["tanks"]
    assert SIM_ENEMY_ID not in world["tanks"]
    assert make_field_world(practice=True)["tanks"] == {}


def test_two_bots_seat_on_opposite_sides_and_each_plays_its_own_tank(
    _field_fs: FakeFileSystem,
) -> None:
    """Tank 9 is the primary client, tank 10 its rival on RIVAL_TEAM."""
    result = _arena()

    assert result["stamp"] == "arena"
    assert [(c["tank_id"], c["name"], c["team"]) for c in result["clients"]] == [
        (9, "red-9", 2),
        (10, "red-10", RIVAL_TEAM),
    ]
    for client in result["clients"]:
        assert client["commands_sent"] >= client["rounds_played"]
        assert client["capture_path"] == str(
            _ARCHIVE / f"sim-arena-tank-{client['tank_id']}.capture_session.json"
        )
        assert client["events_path"] == str(
            _ROOT / f"tank-{client['tank_id']}" / "probe" / "latest.sim.events.jsonl"
        )
        assert client["events_path"] in _field_fs.get_written_files()


def test_a_bot_that_exits_leaves_the_field_and_its_rival_is_told(
    _field_fs: FakeFileSystem,
) -> None:
    """The production exit sends the quit, the tank leaves, the rival reads the 0x29.

    Tank 10 exits first, so tank 9 is still connected to receive its
    departure; tank 9 leaves a round later to an empty room, which ends
    the field before its round budget.
    """
    result = _arena()

    nine, ten = result["clients"]
    assert (ten["rounds_played"], nine["rounds_played"], result["rounds_played"]) == (8, 9, 10)
    for client in (nine, ten):
        assert client["departed"]
        assert not client["alive"]
        assert client["exit_reason"] == SessionExitReason.NO_VIABLE_TARGETS
        assert "no affordable enemy" in client["exit_detail"]
    assert _received(_field_fs, Path(nine["capture_path"]), 0x29) == [
        {
            "msg_type": 0x29,
            "team": RIVAL_TEAM,
            "tank_id": 10,
            "was_silent": False,
            "was_eliminated": False,
        }
    ]
    assert _received(_field_fs, Path(ten["capture_path"]), 0x29) == []
    world = narrow_json_to_dict(load_json_str(_field_fs.read_text(Path(result["world_path"]))))
    ids = [narrow_json_to_dict(t)["tank_id"] for t in narrow_json_to_list(world["tanks"])]
    assert 9 not in ids
    assert 10 not in ids


def test_the_two_bots_fight_each_other(_field_fs: FakeFileSystem) -> None:
    """Multi-tank combat on demand: bot 9's own shots at bot 10 cross the wire.

    Every 0x53 tank 9 fires is a shot at tank 10, the only other tank on
    the arena field, and tank 10's connection sees them too.
    """
    result = _arena()

    nine, ten = result["clients"]
    shots = _shots(_received(_field_fs, Path(nine["capture_path"]), 0x53))
    assert shots
    assert {shot["shooter_id"] for shot in shots} == {9}
    assert _shots(_received(_field_fs, Path(ten["capture_path"]), 0x53)) == shots


def test_a_round_budget_ends_a_field_whose_bots_stay(_field_fs: FakeFileSystem) -> None:
    """Bots still seated when the rounds run out are not counted as departed."""
    result = _arena(rounds=3)

    assert result["rounds_played"] == 3
    for client in result["clients"]:
        assert client["rounds_played"] == 3
        assert client["exit_reason"] == "rounds_exhausted"
        assert not client["departed"]
        assert client["alive"]


def test_the_practice_field_seats_the_bots_among_the_roster(_field_fs: FakeFileSystem) -> None:
    """The bot_policy roster plays beside the two connected bots."""
    result = run_field_session(
        3,
        clients=2,
        archive_dir=_ARCHIVE,
        practice=True,
        stamp="practice",
        layout="bot-20260706-223721",
        population_seed=7,
    )

    assert [c["tank_id"] for c in result["clients"]] == [9, 10]
    world = narrow_json_to_dict(load_json_str(_field_fs.read_text(Path(result["world_path"]))))
    assert len(narrow_json_to_list(world["tanks"])) > 2
    assert result["clients"][0]["events_path"] == str(
        DEFAULT_FIELD_RUNS_ROOT / "tank-9" / "probe" / "latest.sim.events.jsonl"
    )


def test_a_field_of_one_is_refused(_field_fs: FakeFileSystem) -> None:
    """One bot is run_sim_session's job."""
    with pytest.raises(FieldSeatError, match="SIM_FIELD_SEATS: a field session seats 2 or more"):
        run_field_session(3, clients=1, archive_dir=_ARCHIVE)


def _shot(shooter_id: int, target_x: int, weapon: int = 0) -> ShootEventDict:
    """A 0x53 from one shooter at a tile on row 50."""
    return ShootEventDict(
        msg_type=0x53,
        team=1,
        shooter_id=shooter_id,
        source_x=40,
        source_y=50,
        target_x=target_x,
        target_y=50,
        aim_x=target_x,
        aim_y=50,
        weapon=weapon,
    )


def test_one_shot_seen_by_two_connections_is_one_hit() -> None:
    """Merged by greatest count: a shot narrated twice is noted once, a dual twice."""
    world = make_sim_world(SIM_FIELD)
    world["tanks"][7] = make_sim_tank(7, 1, 1, 40, 50, 1000)
    world["tanks"][500] = make_sim_tank(500, 2, 1, 45, 50, 1000)
    world["tanks"][501] = make_sim_tank(501, 2, 1, 46, 50, 1000)
    driver = PracticeRoomDriver(frozenset({500, 501}))
    single = _shot(7, 45)
    dual = _shot(7, 46, weapon=1)

    driver.note_field_batches(world, [[single, dual, dual], [single, dual]])

    assert driver.states[500]["hits_taken"] == 1
    assert driver.states[501]["hits_taken"] == 2


def test_the_cli_runs_a_field_with_clients(
    _field_fs: FakeFileSystem, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--clients 2`` plays the field and reports each bot."""
    assert (
        main(
            [
                "--clients",
                "2",
                "--rounds",
                "3",
                "--stamp",
                "cli",
                "--layout",
                "bot-20260706-223721",
                "--population-seed",
                "7",
                "--out",
                str(_ARCHIVE),
                "--runs-root",
                str(_ROOT),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "sim field cli: 3/3 rounds, 2 bots\n" in out
    assert "  tank 10 red-10 team 1: 3 rounds, 0 kills, 0 deaths, exit=rounds_exhausted\n" in out
    assert f"  world:   {_ARCHIVE / 'sim-cli.world.json'}\n" in out


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (["--ghost", "runs/x.capture_session.json"], "--ghost"),
        (["--ferry"], "--ferry"),
        (["--larder", "--from-atlas", "a.json"], "--larder, --from-atlas"),
        (["--human-opponent", "guest"], "--human-opponent"),
    ],
)
def test_the_cli_refuses_one_client_scenarios_on_a_field(
    flags: list[str], named: str, _field_fs: FakeFileSystem
) -> None:
    """Those worlds are written around one client, so nothing plays."""
    with pytest.raises(FieldSeatError, match=f"SIM_FIELD_SCENARIO: {named} play one client"):
        main(["--clients", "2", *flags])
    assert not any(path.startswith("runs") for path in _field_fs.get_written_files())
