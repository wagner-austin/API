"""Tests for the fleet's CONFIGURATION surface.

What an operator may ASK FOR -- accounts, rooms, colours, the measured
tank registry, and the human-rank floor a room implies -- as opposed to
what is actually RUNNING, which is ``test_fleet.py``. The cut mirrors
the source split of ``fleet_manager`` into ``fleet_config`` plus the
registry, made when the module crossed the 600-line ceiling; this file
was carved out when its test module crossed the same line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import PathExistsProtocol
from tankpit_bot.service.fleet_config import (
    configured_accounts,
    lobby_rooms,
    tank_registry,
)
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.types.rooms import DEFAULT_LOBBY_ROOM
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


def test_the_rank_floor_follows_the_room_a_bot_joins(
    spawner: _FakeSpawner,
) -> None:
    """World spares recruit..sergeant; Practice spares recruits only.

    Operator ruling 2026-09-02, after a five-bot World fleet converged
    on one low-rank human. The two rooms differ in what a fight costs
    the person on the other side, so the floor is a property of the
    room rather than of the operator's environment.
    """
    original = _without_accounts()
    try:
        manager = FleetManager()
        manager.spawn(instance="w", account="", kills=0, seconds=0, room="World")
        manager.spawn(instance="p", account="", kills=0, seconds=0, room="Practice")
        manager.spawn(instance="d", account="", kills=0, seconds=0)
    finally:
        top_hooks.path_exists = original

    assert spawner.envs[0]["TANKPIT_BOT_HUMAN_MIN_RANK"] == "4"
    assert spawner.envs[1]["TANKPIT_BOT_HUMAN_MIN_RANK"] == "1"
    # No room named means the child's own default, which is Practice.
    assert spawner.envs[2]["TANKPIT_BOT_HUMAN_MIN_RANK"] == "1"


def test_the_rank_floor_survives_the_worlds_map_rotation(
    spawner: _FakeSpawner,
) -> None:
    """A map-stamped world name still resolves to the World floor.

    The world's display name carries the current map and rotates, so
    matching the whole string would silently drop the fleet back to
    the Practice floor the first time the map changed.
    """
    original = _without_accounts()
    try:
        FleetManager().spawn(instance="w", account="", kills=0, seconds=0, room="World (Desert)")
    finally:
        top_hooks.path_exists = original

    assert spawner.envs[0]["TANKPIT_BOT_HUMAN_MIN_RANK"] == "4"


def test_the_rank_floor_is_always_stated_never_inherited(
    spawner: _FakeSpawner,
) -> None:
    """Every spawn names the floor, so a stale global cannot lower it.

    Same reasoning as TANKPIT_ROLE: the child inherits the manager's
    whole environment, and one global cannot express "lieutenant on
    World, recruit on Practice" — so the fleet states it every time.
    """
    original = _without_accounts()
    try:
        FleetManager().spawn(instance="p", account="", kills=0, seconds=0)
    finally:
        top_hooks.path_exists = original

    assert "TANKPIT_BOT_HUMAN_MIN_RANK" in spawner.envs[0]


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
