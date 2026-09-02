"""Tests for the AI-operated fleet manager's domain layer.

Spawn, bounds, accounts, rooms, lifecycle, stats, and the port/entry
wiring. The aiohttp surface over this registry is exercised in
``test_fleet_http.py``, split out 2026-08-28 when the combined module
crossed the 600-line ceiling.
"""

from __future__ import annotations

from pathlib import Path

import psutil
import pytest

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import PathExistsProtocol
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.service._test_hooks import _real_spawn_bot_process
from tankpit_bot.service.fleet_config import (
    FLEET_PORT_DEFAULT,
    configured_accounts,
    lobby_rooms,
    resolve_fleet_port,
    tank_registry,
    troop_colors,
)
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.types.constants import TEAM_BLUE
from tankpit_bot.types.rooms import DEFAULT_LOBBY_ROOM
from tests.conftest import FakeEnv
from tests.service._artifact_fixtures import FakeArtifact
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_configured_accounts,
)


def _without_accounts() -> PathExistsProtocol:
    """Make accounts.json absent so tests never read the real file.

    Returns:
        The original ``path_exists`` hook to restore.
    """

    def fake_missing(path: Path) -> bool:
        _ = path
        return False

    original = top_hooks.path_exists
    top_hooks.path_exists = fake_missing
    return original


def test_spawn_builds_the_instance_environment(spawner: _FakeSpawner) -> None:
    """The child receives instance, bounds, account, and role via env."""
    originals = _with_configured_accounts()
    try:
        manager = FleetManager()
        row = manager.spawn(instance="alpha", account="second", kills=30, seconds=2700)
    finally:
        _restore_account_hooks(originals)

    assert spawner.envs == [
        {
            "TANKPIT_BOT_INSTANCE": "alpha",
            "TANKPIT_BOT_SESSION_KILLS": "30",
            "TANKPIT_BOT_SESSION_SECONDS": "2700",
            "TANKPIT_ROLE": "fighter",
            "TANKPIT_ACCOUNT": "second",
        }
    ]
    assert row["instance"] == "alpha"
    assert row["role"] == "fighter"
    assert row["alive"] is True
    assert row["pid"] == 1001


def test_spawn_role_is_explicit_validated_and_carried_by_restart(
    spawner: _FakeSpawner,
) -> None:
    """A gatherer spawn sets TANKPIT_ROLE explicitly and restart keeps it.

    The env var is ALWAYS set (never inherited): a TANKPIT_ROLE
    lingering in the manager's own environment must not silently
    re-role the fleet. An unknown role is refused before any process
    exists.
    """
    original = _without_accounts()
    try:
        manager = FleetManager()
        row = manager.spawn(instance="alpha", account="", kills=0, seconds=0, role="gatherer")

        with pytest.raises(FleetError, match="not a fleet role"):
            manager.spawn(instance="bravo", account="", kills=0, seconds=0, role="scout")

        spawner.processes[0].returncode = 0
        restarted = manager.restart("alpha")
    finally:
        top_hooks.path_exists = original

    assert row["role"] == "gatherer"
    assert spawner.envs[0]["TANKPIT_ROLE"] == "gatherer"
    assert restarted["role"] == "gatherer"
    assert spawner.envs == [spawner.envs[0], spawner.envs[0]]


def test_accounts_lists_configured_usernames_only(spawner: _FakeSpawner) -> None:
    """The account surface is accounts.json usernames — never passwords."""
    _ = spawner
    originals = _with_configured_accounts()
    try:
        manager = FleetManager()
        names = configured_accounts()
        row = manager.spawn(instance="alpha", account="second", kills=0, seconds=0)
        with pytest.raises(FleetError, match=r"not in accounts\.json"):
            manager.spawn(instance="bravo", account="intruder", kills=0, seconds=0)
    finally:
        _restore_account_hooks(originals)

    assert names == ["artax", "second"]
    assert row["account"] == "second"


def test_accounts_without_a_file_is_empty_and_default_still_spawns(
    spawner: _FakeSpawner,
) -> None:
    """No accounts.json: the list is empty and only default spawns."""

    def fake_exists(path: Path) -> bool:
        _ = path
        return False

    original_exists = top_hooks.path_exists
    top_hooks.path_exists = fake_exists
    try:
        manager = FleetManager()
        names = configured_accounts()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        with pytest.raises(FleetError, match="none configured"):
            manager.spawn(instance="bravo", account="anyone", kills=0, seconds=0)
    finally:
        top_hooks.path_exists = original_exists

    assert names == []
    assert len(spawner.envs) == 1


def test_spawn_without_account_omits_the_selector(spawner: _FakeSpawner) -> None:
    """An empty account means the accounts.json default, not an empty var."""
    manager = FleetManager()

    manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    assert "TANKPIT_ACCOUNT" not in spawner.envs[0]


def test_spawn_with_a_room_sets_the_selector(spawner: _FakeSpawner) -> None:
    """A named room reaches the child as TANKPIT_ROOM and rides the row.

    The 2026-08-26 Desert recon was hand-spawned because the manager
    had no room parameter; cross-room fleets are safe because the
    knowledge exchange merges same-room reports only.
    """
    manager = FleetManager()

    row = manager.spawn(instance="alpha", account="", kills=0, seconds=0, room="World (Desert)")

    assert spawner.envs[0]["TANKPIT_ROOM"] == "World (Desert)"
    assert row["room"] == "World (Desert)"


def test_rooms_offers_world_first_but_the_env_fallback_stays_practice(
    spawner: _FakeSpawner,
) -> None:
    """The page leads with World; a room-less run still gets Practice.

    The dropdown offers prefixes, not the map-stamped display names —
    the world's name carries the current map ("World (Desert)"), so
    ``World`` is what stays true across a rotation. Order is
    presentation: World leads because that is where the fleet plays.
    The no-config fallback deliberately does NOT follow it, so
    ``make run`` and the probes cannot wander into the live world
    where a deactivation costs a rank.
    """
    _ = spawner

    names = lobby_rooms()

    assert names == ["World", "Practice"]
    assert names[0] != DEFAULT_LOBBY_ROOM
    assert DEFAULT_LOBBY_ROOM == "Practice"


def test_tanks_serves_the_measured_registry(spawner: _FakeSpawner) -> None:
    """The per-colour ranks are READ, never derived.

    Nothing on the wire reports the rank of a colour an account is not
    currently playing — the lobby names only the last-played one — so
    the registry is measured state and the page can only show what
    somebody entered and recorded.
    """
    _ = spawner

    def fake_read(path: Path) -> str:
        _ = path
        return '{"accounts": {"Artax": {"World": {"orange": 6}}}}'

    original = top_hooks.read_text
    top_hooks.read_text = fake_read
    try:
        registry = tank_registry()
    finally:
        top_hooks.read_text = original

    assert registry == {"accounts": {"Artax": {"World": {"orange": 6}}}}


def test_tanks_without_a_registry_is_empty_not_an_error(spawner: _FakeSpawner) -> None:
    """An operator who never ran the census gets an empty panel.

    A missing registry is the ordinary first-run state, not a fault:
    the colour dropdown still works, it just has no reading to show.
    """
    _ = spawner

    def fake_read(path: Path) -> str:
        raise OSError(f"no registry at {path}")

    original = top_hooks.read_text
    top_hooks.read_text = fake_read
    try:
        registry = tank_registry()
    finally:
        top_hooks.read_text = original

    assert registry == {}


def test_troops_are_team_id_ordered_and_reach_the_child_as_the_wire_id(
    spawner: _FakeSpawner,
) -> None:
    """Color is picked by NAME and sent as the team id it indexes.

    An account holds four tanks per world, one per color, each with
    its own rank and inventory ([[game-rules]]) — so this selector
    picks WHICH TANK plays. The wire wants the team id, and
    ``TROOP_COLOR_NAMES`` is ordered so the index IS that id — a
    re-sort of that tuple would silently send the wrong tank.
    """
    manager = FleetManager()

    names = troop_colors()
    row = manager.spawn(instance="alpha", account="", kills=0, seconds=0, troop="orange")
    with pytest.raises(FleetError, match="not a tank color"):
        manager.spawn(instance="bravo", account="", kills=0, seconds=0, troop="chartreuse")

    assert names == ["red", "purple", "blue", "orange"]
    assert names[TEAM_BLUE] == "blue"
    assert row["troop"] == "orange"
    assert spawner.envs[0]["TANKPIT_TROOP"] == "3"


def test_doctrine_reaches_the_child_and_an_unknown_one_is_refused_at_spawn(
    spawner: _FakeSpawner,
) -> None:
    """The doctrine rides TANKPIT_DOCTRINE, and a typo never spawns.

    Validated at the manager as well as in the child's own resolver:
    caught here it is a 409 the operator reads, caught in the child it
    is a process that starts, raises and dies with no tank in the
    world.
    """
    manager = FleetManager()

    row = manager.spawn(instance="alpha", account="", kills=0, seconds=0, doctrine="duelist")
    with pytest.raises(FleetError, match="not an engagement doctrine"):
        manager.spawn(instance="bravo", account="", kills=0, seconds=0, doctrine="berserk")

    assert spawner.envs[0]["TANKPIT_DOCTRINE"] == "duelist"
    assert row["instance"] == "alpha"


def test_spawn_without_a_doctrine_omits_the_selector(spawner: _FakeSpawner) -> None:
    """An empty doctrine keeps the child's default rather than naming one.

    The child resolves an unset TANKPIT_DOCTRINE to skirmish; setting
    it here to that same word would look like an operator choice in
    the trail when it was not.
    """
    manager = FleetManager()

    manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    assert "TANKPIT_DOCTRINE" not in spawner.envs[0]


def test_spawn_without_a_troop_omits_the_selector(spawner: _FakeSpawner) -> None:
    """An empty color keeps the account's own tank for that map."""
    manager = FleetManager()

    row = manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    assert "TANKPIT_TROOP" not in spawner.envs[0]
    assert row["troop"] == ""


def test_spawn_without_a_room_omits_the_selector(spawner: _FakeSpawner) -> None:
    """An empty room keeps the child's default (Practice)."""
    manager = FleetManager()

    row = manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    assert "TANKPIT_ROOM" not in spawner.envs[0]
    assert row["room"] == ""


def test_spawn_rejects_invalid_instance_names(spawner: _FakeSpawner) -> None:
    """Path characters and uppercase never reach the filesystem layer."""
    manager = FleetManager()

    with pytest.raises(FleetError, match="not a valid instance name"):
        manager.spawn(instance="../escape", account="", kills=0, seconds=0)
    with pytest.raises(FleetError, match="not a valid instance name"):
        manager.spawn(instance="UPPER", account="", kills=0, seconds=0)
    assert spawner.envs == []


def test_spawn_rejects_negative_bounds(spawner: _FakeSpawner) -> None:
    """Negative bounds are a loud error, not a weird session."""
    manager = FleetManager()

    with pytest.raises(FleetError, match="non-negative"):
        manager.spawn(instance="alpha", account="", kills=-1, seconds=0)
    assert spawner.envs == []


def test_spawn_refuses_a_live_duplicate_but_replaces_a_dead_one(
    spawner: _FakeSpawner,
) -> None:
    """One live process per instance; a finished one may be respawned."""
    original = _without_accounts()
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)

        with pytest.raises(FleetError, match="already running"):
            manager.spawn(instance="alpha", account="", kills=0, seconds=0)

        spawner.processes[0].returncode = 0
        row = manager.spawn(instance="alpha", account="", kills=0, seconds=0)
    finally:
        top_hooks.path_exists = original
    assert row["pid"] == 1002


def test_report_sorts_and_reflects_liveness(spawner: _FakeSpawner) -> None:
    """The report row set is sorted and tracks process exit."""
    original = _without_accounts()
    try:
        manager = FleetManager()
        manager.spawn(instance="bravo", account="", kills=0, seconds=0)
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        spawner.processes[0].returncode = 7
        rows = manager.report()
    finally:
        top_hooks.path_exists = original

    assert [row["instance"] for row in rows] == ["alpha", "bravo"]
    assert rows[1]["alive"] is False
    assert rows[1]["returncode"] == 7


def test_stop_writes_the_instance_sentinel(spawner: _FakeSpawner) -> None:
    """A graceful stop is the instance's STOP file, nothing more."""
    written: list[tuple[Path, str]] = []

    def fake_write(path: Path, content: str) -> None:
        written.append((Path(path), content))

    original_write = top_hooks.write_text
    top_hooks.write_text = fake_write
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        manager.stop("alpha")
    finally:
        top_hooks.write_text = original_write

    assert written == [(Path("runs/bot/alpha/STOP"), "")]


def test_stop_unknown_instance_is_a_fleet_error(spawner: _FakeSpawner) -> None:
    """Stopping a name that was never spawned names the problem."""
    manager = FleetManager()

    with pytest.raises(FleetError, match="unknown instance"):
        manager.stop("ghost")


def test_remove_refuses_a_live_instance_and_drops_a_dead_one(
    spawner: _FakeSpawner,
) -> None:
    """The fleet never silently kills — stop first, then remove."""
    manager = FleetManager()
    manager.spawn(instance="alpha", account="", kills=0, seconds=0)

    with pytest.raises(FleetError, match="still running"):
        manager.remove("alpha")

    spawner.processes[0].returncode = 0
    row = manager.remove("alpha")
    assert row["alive"] is False
    assert manager.report() == []


def test_restart_respawns_a_dead_instance_with_its_parameters(
    spawner: _FakeSpawner,
) -> None:
    """Restart reuses account, bounds AND room, refusing while alive.

    The room is spawned here rather than left default because that is
    exactly how the omission hid: with no room on either spawn, the
    env-equality assertion below held while ``restart`` silently
    dropped the selector and relocated the bot to Practice (live,
    2026-08-28 — the row read ``World``, the child joined Practice).
    """
    originals = _with_configured_accounts()
    try:
        manager = FleetManager()
        manager.spawn(
            instance="alpha",
            account="second",
            kills=30,
            seconds=2700,
            room="World",
            troop="purple",
        )

        with pytest.raises(FleetError, match="still running"):
            manager.restart("alpha")
        with pytest.raises(FleetError, match="unknown instance"):
            manager.restart("ghost")

        spawner.processes[0].returncode = 0
        row = manager.restart("alpha")
    finally:
        _restore_account_hooks(originals)
    assert row["pid"] == 1002
    assert row["room"] == "World"
    assert row["troop"] == "purple"
    assert spawner.envs[0]["TANKPIT_ROOM"] == "World"
    assert spawner.envs[0]["TANKPIT_TROOP"] == "1"
    assert spawner.envs[1] == spawner.envs[0]


def test_stats_summarizes_the_instance_events(
    spawner: _FakeSpawner,
    artifact: FakeArtifact,
) -> None:
    """The stats summary folds the instance's own events artifact."""
    artifact.start_run(
        [
            '{"timestamp":"2026-08-06T10:00:00","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"STATE","message":"INITIALIZING"}',
            '{"timestamp":"2026-08-06T10:00:01","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"DIAGNOSTIC","message":"diagnostic_kind=tank_identity",'
            '"diagnostic_kind":"tank_identity","tank_id":601}',
            '{"timestamp":"2026-08-06T10:05:00","level":"INFO","logger":"l",'
            '"mode":"bot","channel":"DIAGNOSTIC","message":"diagnostic_kind=tank_deactivated",'
            '"diagnostic_kind":"tank_deactivated","victim_id":529,"killer_id":601}',
        ]
    )

    original_exists = _without_accounts()
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        summary = manager.stats("alpha")
    finally:
        top_hooks.path_exists = original_exists

    assert artifact.read_offsets == [0]
    assert summary["available"] is True
    assert summary["kills"] == 1
    assert summary["deaths"] == 0
    assert summary["duration_s"] == 300
    assert summary["clean_exit"] is False


def test_stats_without_events_is_unavailable_not_an_error(
    spawner: _FakeSpawner,
) -> None:
    """A just-spawned bot with no events yet reports available=False."""

    def fake_read(path: Path) -> str:
        raise OSError(f"no such file {path}")

    original_exists = _without_accounts()
    original_read = top_hooks.read_text
    top_hooks.read_text = fake_read
    try:
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        summary = manager.stats("alpha")
        with pytest.raises(FleetError, match="unknown instance"):
            manager.stats("ghost")
    finally:
        top_hooks.read_text = original_read
        top_hooks.path_exists = original_exists

    assert summary == {"available": False}


def test_resolve_fleet_port_contract() -> None:
    """Default, override, and loud rejection."""
    original_get_env = top_hooks.get_env
    try:
        top_hooks.get_env = FakeEnv({})
        assert resolve_fleet_port() == FLEET_PORT_DEFAULT
        top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": "27301"})
        assert resolve_fleet_port() == 27301
        top_hooks.get_env = FakeEnv({"TANKPIT_FLEET_PORT": "80"})
        with pytest.raises(ValueError, match="outside"):
            resolve_fleet_port()
    finally:
        top_hooks.get_env = original_get_env


def test_real_spawn_bot_process_launches_a_live_python_child() -> None:
    """The production spawner starts a real child; killed before it acts.

    The child is terminated immediately — interpreter startup takes far
    longer than the kill lands, so it never reaches the bot entry point
    (which would open a browser).

    Its console goes to the instance's own file, never to the
    manager's terminal: inheriting the console put every bot's tick
    lines in the ``make fleet`` window (2026-08-28). The file is
    opened even for a child that dies instantly, because the stream
    the interpreter prints a fatal traceback to IS this one.
    """
    console = bot_run_dir("covspawn") / "console.log"
    process = _real_spawn_bot_process({"TANKPIT_BOT_INSTANCE": "covspawn"})
    # Killed through the pid, because the registry's surface is
    # deliberately read-only: it can ask whether a bot is running,
    # never end one. Stopping is the sentinel's job.
    handle = psutil.Process(process.pid)
    try:
        assert process.pid > 0
        assert console.exists()
    finally:
        handle.kill()
        handle.wait(timeout=30)
        console.unlink()
        console.parent.rmdir()

    assert process.is_running() is False
    assert process.exit_code() == handle.wait(timeout=30)
