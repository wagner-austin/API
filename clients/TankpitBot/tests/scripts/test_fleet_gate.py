"""Tests for the bring-up gate.

Every request goes through ``scripts._test_hooks.run_command``; here a fake
sedona answers each fleet path from a scripted queue, the last answer
repeating, and records the order paths were asked in. The payloads are
built with the fleet's own encoders, and the spawn body the gate sends is
read back with the manager's own ``parse_spawn_request``.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.fleet_gate import (
    FLEET_URL,
    GATE_ATTEMPTS,
    GATE_INSTANCE,
    GATE_PAUSE_SECONDS,
    GATE_ROOM,
    GATE_SECONDS,
    await_first_tick,
    free_account,
    gate,
    spawn,
)
from scripts.fleet_remote import SEDONA_SSH, FleetHostError

from scripts import _test_hooks as script_hooks
from tankpit_bot.service.fleet_routes import parse_spawn_request
from tankpit_bot.service.fleet_wire import SpawnRequestDict, encode_fleet_bot
from tests.scripts._fleet_answers import (
    SPAWNED_AT,
    accounts,
    activity,
    answer,
    bot_row,
    fleet,
    stats,
)

#: The key :class:`Sedona` files the PowerShell spawn line under.
SPAWN = "spawn"


class Sedona:
    """The manager on sedona's loopback, answering each path from a queue."""

    def __init__(self, answers: dict[str, list[script_hooks.CommandResult]]) -> None:
        """Hold the scripted answers.

        Args:
            answers: Keyed by request path, or :data:`SPAWN` for the POST;
                each queue is answered in order, its last answer repeating.
        """
        self.answers = answers
        self.asked: list[str] = []
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], cwd: Path) -> script_hooks.CommandResult:
        """Answer one command.

        Args:
            argv: The command.
            cwd: Ignored.

        Returns:
            The next scripted answer for its path.
        """
        self.calls.append(argv)
        key = SPAWN if argv[2] == "powershell" else argv[-1].removeprefix(FLEET_URL)
        self.asked.append(key)
        queue = self.answers[key]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _fail(stderr: str) -> script_hooks.CommandResult:
    return script_hooks.CommandResult(returncode=22, stdout="", stderr=stderr)


def _spawned(account: str) -> script_hooks.CommandResult:
    return answer(encode_fleet_bot(bot_row(GATE_INSTANCE, account, alive=True)))


@pytest.fixture(autouse=True)
def pauses() -> Generator[list[float], None, None]:
    """Record sleeps instead of sleeping, and put the hooks back after.

    Yields:
        The pauses the gate asked for.
    """
    run_command = script_hooks.run_command
    sleep_seconds = script_hooks.sleep_seconds
    pauses: list[float] = []

    def _pause(seconds: float) -> None:
        pauses.append(seconds)

    script_hooks.sleep_seconds = _pause
    yield pauses
    script_hooks.run_command = run_command
    script_hooks.sleep_seconds = sleep_seconds


def test_free_account_skips_every_account_with_a_live_bot(tmp_path: Path) -> None:
    """The first account whose bot is not alive is chosen; a dead bot's account is free."""
    sedona = Sedona(
        {
            "/accounts": [accounts("Artax", "Yuppler", "Arterial")],
            "/bots": [
                fleet(bot_row("a1a", "Artax", alive=True), bot_row("old", "Yuppler", alive=False))
            ],
        }
    )
    script_hooks.run_command = sedona
    assert free_account(tmp_path) == "Yuppler"
    assert sedona.calls[0] == ["ssh", SEDONA_SSH, "curl.exe", "-s", "-f", f"{FLEET_URL}/accounts"]


def test_free_account_refuses_when_every_account_plays(tmp_path: Path) -> None:
    """A fleet whose every account is playing is refused by name."""
    script_hooks.run_command = Sedona(
        {"/accounts": [accounts("Artax")], "/bots": [fleet(bot_row("a1a", "Artax", alive=True))]}
    )
    with pytest.raises(FleetHostError) as raised:
        free_account(tmp_path)
    assert str(raised.value) == "FLEET_GATE_NO_ACCOUNT: every account has a live bot (Artax)"


def test_spawn_sends_the_request_the_manager_parses(tmp_path: Path) -> None:
    """The base64 body decodes, through the manager's parser, to the bounded Practice gatherer."""
    sedona = Sedona({SPAWN: [_spawned("Arterial")]})
    script_hooks.run_command = sedona
    row = spawn(tmp_path, "Arterial")
    assert row == bot_row(GATE_INSTANCE, "Arterial", alive=True)
    argv = sedona.calls[0]
    assert argv[:5] == ["ssh", SEDONA_SSH, "powershell", "-NoProfile", "-Command"]
    found = re.search(r"FromBase64String\('([A-Za-z0-9+/=]+)'\)", argv[5])
    if found is None:
        pytest.fail(f"the spawn line carries no base64 body: {argv[5]}")
    assert parse_spawn_request(base64.b64decode(found.group(1))) == SpawnRequestDict(
        instance=GATE_INSTANCE,
        account="Arterial",
        kills=0,
        seconds=GATE_SECONDS,
        role="gatherer",
        room=GATE_ROOM,
        troop="",
        doctrine="",
    )
    assert f"Invoke-RestMethod -Method Post -Uri {FLEET_URL}/bots" in argv[5]
    assert '"' not in argv[5]


def test_spawn_refused_by_the_manager_is_refused_by_name(tmp_path: Path) -> None:
    """A refused spawn names its code and the manager's words."""
    script_hooks.run_command = Sedona({SPAWN: [_fail("instance 'gate' is already running")]})
    with pytest.raises(FleetHostError) as raised:
        spawn(tmp_path, "Arterial")
    assert str(raised.value).startswith("FLEET_GATE_SPAWN_FAILED: ssh ")
    assert str(raised.value).endswith("exited 22: instance 'gate' is already running")


def test_the_tail_is_read_only_a_poll_after_the_new_run_is_seen(
    tmp_path: Path, pauses: list[float]
) -> None:
    """A stale run's tick never passes: the tail is asked only after started_at caught up."""
    sedona = Sedona(
        {
            f"/bots/{GATE_INSTANCE}/stats": [
                answer({"available": False}),
                stats("2026-09-29T04:30:00"),
                stats(SPAWNED_AT),
            ],
            f"/bots/{GATE_INSTANCE}/activity": [activity(1, -1), activity(2, 942)],
            "/bots": [fleet(bot_row(GATE_INSTANCE, "Arterial", alive=True))],
        }
    )
    script_hooks.run_command = sedona
    line = await_first_tick(tmp_path, bot_row(GATE_INSTANCE, "Arterial", alive=True))
    assert line == "gate bot 'gate' on Arterial reached tick 2 at fuel 942 in Practice"
    digest, tail = f"/bots/{GATE_INSTANCE}/stats", f"/bots/{GATE_INSTANCE}/activity"
    assert sedona.asked == [digest, "/bots", digest, "/bots", digest, "/bots", tail, "/bots", tail]
    assert pauses == [GATE_PAUSE_SECONDS] * 4


def test_a_bot_that_exits_before_its_first_tick_is_refused(tmp_path: Path) -> None:
    """The gate stops waiting the moment the manager reports the bot gone, naming its exit."""
    script_hooks.run_command = Sedona(
        {
            f"/bots/{GATE_INSTANCE}/stats": [stats(SPAWNED_AT)],
            "/bots": [fleet(bot_row(GATE_INSTANCE, "Arterial", alive=False))],
        }
    )
    with pytest.raises(FleetHostError) as raised:
        await_first_tick(tmp_path, bot_row(GATE_INSTANCE, "Arterial", alive=True))
    assert str(raised.value) == "FLEET_GATE_BOT_EXITED: 'gate' exited with 1 before its first tick"


def test_a_gate_the_manager_no_longer_lists_is_refused(tmp_path: Path) -> None:
    """A manager that lost the instance is refused by name, never waited out."""
    script_hooks.run_command = Sedona(
        {f"/bots/{GATE_INSTANCE}/stats": [stats(SPAWNED_AT)], "/bots": [fleet()]}
    )
    with pytest.raises(FleetHostError) as raised:
        await_first_tick(tmp_path, bot_row(GATE_INSTANCE, "Arterial", alive=True))
    assert str(raised.value) == "FLEET_GATE_BOT_MISSING: the manager lists no 'gate'"


def test_a_bot_that_never_ticks_is_refused_with_the_last_reading(
    tmp_path: Path, pauses: list[float]
) -> None:
    """Out of attempts, the gate names the last tail it read."""
    script_hooks.run_command = Sedona(
        {
            f"/bots/{GATE_INSTANCE}/stats": [stats(SPAWNED_AT)],
            f"/bots/{GATE_INSTANCE}/activity": [activity(1, -1)],
            "/bots": [fleet(bot_row(GATE_INSTANCE, "Arterial", alive=True))],
        }
    )
    with pytest.raises(FleetHostError) as raised:
        await_first_tick(tmp_path, bot_row(GATE_INSTANCE, "Arterial", alive=True))
    assert str(raised.value) == (
        f"FLEET_GATE_NO_TICK: 'gate' after {GATE_ATTEMPTS} polls: tick 1, fuel -1"
    )
    assert len(pauses) == GATE_ATTEMPTS - 1


def test_a_run_never_seen_is_refused_naming_where_it_stood(tmp_path: Path) -> None:
    """A digest that never reaches the spawn is named with both instants."""
    script_hooks.run_command = Sedona(
        {
            f"/bots/{GATE_INSTANCE}/stats": [answer({"available": False})],
            "/bots": [fleet(bot_row(GATE_INSTANCE, "Arterial", alive=True))],
        }
    )
    with pytest.raises(FleetHostError) as raised:
        await_first_tick(tmp_path, bot_row(GATE_INSTANCE, "Arterial", alive=True))
    assert str(raised.value) == (
        f"FLEET_GATE_NO_TICK: 'gate' after {GATE_ATTEMPTS} polls: "
        f"run started nowhere yet, spawned {SPAWNED_AT}"
    )


def test_an_unreachable_manager_is_refused_by_name(tmp_path: Path) -> None:
    """A failed request names FLEET_GATE_UNREACHABLE and curl's words."""
    script_hooks.run_command = Sedona({"/accounts": [_fail("curl: (7) Failed to connect")]})
    with pytest.raises(FleetHostError) as raised:
        free_account(tmp_path)
    assert str(raised.value) == (
        f"FLEET_GATE_UNREACHABLE: ssh {SEDONA_SSH} curl.exe -s -f {FLEET_URL}/accounts "
        "exited 22: curl: (7) Failed to connect"
    )


def test_gate_picks_spawns_and_waits(tmp_path: Path) -> None:
    """The whole gate: a free account, one spawn on it, and its first tick."""
    sedona = Sedona(
        {
            "/accounts": [accounts("Artax", "Arterial")],
            "/bots": [
                fleet(bot_row("a1a", "Artax", alive=True)),
                fleet(
                    bot_row("a1a", "Artax", alive=True),
                    bot_row(GATE_INSTANCE, "Arterial", alive=True),
                ),
            ],
            SPAWN: [_spawned("Arterial")],
            f"/bots/{GATE_INSTANCE}/stats": [stats(SPAWNED_AT)],
            f"/bots/{GATE_INSTANCE}/activity": [activity(3, 1100)],
        }
    )
    script_hooks.run_command = sedona
    assert gate(tmp_path) == "gate bot 'gate' on Arterial reached tick 3 at fuel 1100 in Practice"
    assert sedona.asked[:3] == ["/accounts", "/bots", SPAWN]
