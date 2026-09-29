"""The bring-up gate: one bounded Practice bot must reach its first tick.

:func:`scripts.fleet_host.smoke` proves the edge answers and the manager
serves its roster. Neither proves a bot can play, and a fleet that serves
a roster but cannot log a tank in has been shipped twice (board task
4184b3f7, A5). So ``make up`` ends here: spawn one bot, bounded to
:data:`GATE_SECONDS` in :data:`GATE_ROOM`, on the first account with no
live bot, and wait until it has ticked in the game.

WHAT COUNTS AS A FIRST TICK. The manager's ``/bots/<instance>/activity``
tail carries the last ``tick_n`` and the last fuel total the run logged.
A tick number alone is not enough: the bot stamps tick 1 while still
initializing, with fuel ``-1``. Fuel is only logged once the tank's own
state has come off the game wire. So the gate asks for ``tick >= 1``
and ``fuel >= 0``.

WHY THE TAIL IS READ ONE POLL LATER THAN THE RUN IS SEEN. The gate
re-spawns the same instance name every time, and a re-spawned child
does not replace ``latest.events.jsonl`` until it writes its first
event. Until then the manager folds the PREVIOUS gate run's file, whose
tail already shows a tick and a fuel total. The run digest behind
``/stats`` names the run's ``started_at``, so the gate first waits for a
``started_at`` no earlier than the spawn. The manager also caches each
answer for ``TELEMETRY_CACHE_TTL_MS`` (2 s), so the activity tail is read
only on a later poll, :data:`GATE_PAUSE_SECONDS` (3 s) after the run was
seen. Every answer it can see by then was folded from the new run.

The manager listens on sedona's loopback only, so every request is a
``curl.exe`` or PowerShell line run over ssh, through
:func:`scripts.fleet_remote.run_checked`. A request that fails is refused
by name, and the gate never retries around it.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_int,
    require_str,
    require_str_list,
)

from scripts import _test_hooks as script_hooks
from scripts.fleet_remote import SEDONA_SSH, FleetHostError, on_sedona, run_checked
from tankpit_bot.fleetshare.types import FleetRole
from tankpit_bot.service.fleet_bot import FleetBotDict
from tankpit_bot.service.fleet_wire import (
    SpawnRequestDict,
    decode_fleet_bot,
    decode_fleet_snapshot,
    encode_spawn_request,
)

#: The fleet manager, as sedona itself reaches it.
FLEET_URL: Final[str] = "http://127.0.0.1:27300"

#: The gate bot's instance name, reused by every ``up``.
GATE_INSTANCE: Final[str] = "gate"

#: Where the gate bot plays: the room nobody's rank rides on.
GATE_ROOM: Final[str] = "Practice"

#: How long the gate bot plays. It ends itself with ``session_complete``
#: after this, so the gate never has to stop it.
GATE_SECONDS: Final[int] = 30

#: How often, and how far apart, the gate polls before giving up: a
#: minute. The pause outlasts the manager's 2 s answer cache (see the
#: module docstring).
GATE_ATTEMPTS: Final[int] = 20
GATE_PAUSE_SECONDS: Final[float] = 3.0


def _fleet_get(project_root: Path, path: str) -> JSONObject:
    """Ask the manager one GET on sedona's loopback.

    Args:
        project_root: Where to run ssh from.
        path: The request path.

    Returns:
        The decoded JSON object.

    Raises:
        FleetHostError: FLEET_GATE_UNREACHABLE when the request fails;
            ``curl -f`` makes an error status a failure.
        JSONTypeError: When the answer is not a JSON object.
    """
    body = run_checked(
        ["ssh", SEDONA_SSH, "curl.exe", "-s", "-f", f"{FLEET_URL}{path}"],
        project_root,
        "FLEET_GATE_UNREACHABLE",
    )
    return narrow_json_to_dict(load_json_str(body))


def free_account(project_root: Path) -> str:
    """The first configured account with no live bot, since the game refuses a second login.

    Args:
        project_root: Where to run ssh from.

    Returns:
        The account name.

    Raises:
        FleetHostError: FLEET_GATE_NO_ACCOUNT when every account is playing.
    """
    accounts = require_str_list(_fleet_get(project_root, "/accounts"), "accounts")
    snapshot = decode_fleet_snapshot(_fleet_get(project_root, "/bots"))
    playing = {bot["account"] for bot in snapshot["bots"] if bot["alive"]}
    for account in accounts:
        if account not in playing:
            return account
    raise FleetHostError(
        f"FLEET_GATE_NO_ACCOUNT: every account has a live bot ({', '.join(accounts)})"
    )


def spawn(project_root: Path, account: str) -> FleetBotDict:
    """Spawn the gate bot and return the manager's row for it.

    The body travels base64-encoded because the line passes through
    sedona's cmd.exe before PowerShell reads it, and cmd.exe would take
    the JSON's double quotes for its own.

    Args:
        project_root: Where to run ssh from.
        account: The account to play on.

    Returns:
        The spawned instance's report row.

    Raises:
        FleetHostError: FLEET_GATE_SPAWN_FAILED when the manager refuses.
    """
    body = dump_json_str(
        encode_spawn_request(
            SpawnRequestDict(
                instance=GATE_INSTANCE,
                account=account,
                kills=0,
                seconds=GATE_SECONDS,
                role=FleetRole.GATHERER.value,
                room=GATE_ROOM,
                troop="",
                doctrine="",
            )
        )
    )
    encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")
    command = (
        "$ErrorActionPreference = 'Stop'; "
        f"$body = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}')); "
        "ConvertTo-Json -Compress -InputObject (Invoke-RestMethod -Method Post "
        f"-Uri {FLEET_URL}/bots -ContentType application/json -Body $body)"
    )
    answer = run_checked(on_sedona(command), project_root, "FLEET_GATE_SPAWN_FAILED")
    return decode_fleet_bot(narrow_json_to_dict(load_json_str(answer)))


def _started_second(started_ms: int) -> str:
    """The spawn instant in the run digest's ``started_at`` form.

    Args:
        started_ms: The manager's spawn time, in epoch milliseconds.

    Returns:
        ``YYYY-MM-DDTHH:MM:SS`` in UTC, the container's clock.
    """
    return datetime.fromtimestamp(started_ms // 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")


def _still_alive(project_root: Path) -> None:
    """Refuse a gate bot that has exited, or that the manager no longer lists.

    Args:
        project_root: Where to run ssh from.

    Raises:
        FleetHostError: FLEET_GATE_BOT_MISSING when the manager lists no
            gate instance; FLEET_GATE_BOT_EXITED when it exited.
    """
    rows = [
        bot
        for bot in decode_fleet_snapshot(_fleet_get(project_root, "/bots"))["bots"]
        if bot["instance"] == GATE_INSTANCE
    ]
    if not rows:
        raise FleetHostError(f"FLEET_GATE_BOT_MISSING: the manager lists no {GATE_INSTANCE!r}")
    if not rows[0]["alive"]:
        raise FleetHostError(
            f"FLEET_GATE_BOT_EXITED: {GATE_INSTANCE!r} exited with {rows[0]['returncode']} "
            "before its first tick"
        )


def await_first_tick(project_root: Path, row: FleetBotDict) -> str:
    """Wait until the gate bot's own run has ticked in the game.

    Args:
        project_root: Where to run ssh from.
        row: The manager's row for the spawned gate bot.

    Returns:
        The line reporting the tick.

    Raises:
        FleetHostError: As :func:`_still_alive`; FLEET_GATE_NO_TICK naming
            the last reading when :data:`GATE_ATTEMPTS` polls never saw it.
    """
    since = _started_second(row["started_ms"])
    run_seen = False
    last = "no poll answered"
    for attempt in range(GATE_ATTEMPTS):
        if attempt > 0:
            script_hooks.sleep_seconds(GATE_PAUSE_SECONDS)
        if run_seen:
            activity = _fleet_get(project_root, f"/bots/{GATE_INSTANCE}/activity")
            tick = require_int(activity, "tick")
            fuel = require_int(activity, "fuel")
            if tick >= 1 and fuel >= 0:
                return (
                    f"gate bot {GATE_INSTANCE!r} on {row['account']} reached tick {tick} "
                    f"at fuel {fuel} in {GATE_ROOM}"
                )
            last = f"tick {tick}, fuel {fuel}"
        else:
            stats = _fleet_get(project_root, f"/bots/{GATE_INSTANCE}/stats")
            started = require_str(stats, "started_at") if require_bool(stats, "available") else ""
            run_seen = started >= since
            last = f"run started {started or 'nowhere yet'}, spawned {since}"
        _still_alive(project_root)
    raise FleetHostError(
        f"FLEET_GATE_NO_TICK: {GATE_INSTANCE!r} after {GATE_ATTEMPTS} polls: {last}"
    )


def gate(project_root: Path) -> str:
    """Spawn one bounded Practice bot and see it tick.

    Args:
        project_root: Where to run ssh from.

    Returns:
        The line reporting the tick.

    Raises:
        FleetHostError: As :func:`free_account`, :func:`spawn` and
            :func:`await_first_tick`.
    """
    return await_first_tick(project_root, spawn(project_root, free_account(project_root)))


__all__ = [
    "FLEET_URL",
    "GATE_ATTEMPTS",
    "GATE_INSTANCE",
    "GATE_PAUSE_SECONDS",
    "GATE_ROOM",
    "GATE_SECONDS",
    "await_first_tick",
    "free_account",
    "gate",
    "spawn",
]
