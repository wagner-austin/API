"""The fleet manager's answers, as sedona's loopback gives them, for the fleet script tests.

Built with the fleet's own encoders so a test's answer is exactly what the
manager would send, and shared by the gate's tests and ``up``'s.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from scripts.fleet_gate import FLEET_URL, GATE_INSTANCE, GATE_ROOM, GATE_SECONDS

from scripts import _test_hooks as script_hooks
from tankpit_bot.fleetshare.types import FleetRole
from tankpit_bot.service.fleet_bot import FleetBotDict
from tankpit_bot.service.fleet_wire import (
    FleetSnapshotDict,
    encode_fleet_bot,
    encode_fleet_snapshot,
)

#: 2026-09-29T04:36:49.210Z, the spawn instant every row carries.
SPAWNED_MS = 1_790_656_609_210

#: The same instant as the run digest's ``started_at`` states it.
SPAWNED_AT = "2026-09-29T04:36:49"


def answer(payload: JSONObject) -> script_hooks.CommandResult:
    """A successful request whose standard output is ``payload``.

    Args:
        payload: The JSON object the manager answered.

    Returns:
        The command result.
    """
    return script_hooks.CommandResult(returncode=0, stdout=dump_json_str(payload), stderr="")


def bot_row(instance: str, account: str, *, alive: bool) -> FleetBotDict:
    """One manager row for a bounded Practice gatherer spawned at :data:`SPAWNED_MS`.

    Args:
        instance: Instance name.
        account: Account it plays on.
        alive: Whether it is still running; a dead row exited 1.

    Returns:
        The row.
    """
    return FleetBotDict(
        instance=instance,
        account=account,
        role=FleetRole.GATHERER,
        room=GATE_ROOM,
        troop="",
        doctrine="",
        pid=716,
        alive=alive,
        returncode=None if alive else 1,
        kills=0,
        seconds=GATE_SECONDS,
        started_ms=SPAWNED_MS,
        service_port=27101,
    )


def fleet(*rows: FleetBotDict) -> script_hooks.CommandResult:
    """The ``GET /bots`` answer listing ``rows``.

    Args:
        rows: The instances the manager reports.

    Returns:
        The command result.
    """
    snapshot = FleetSnapshotDict(boot="1", draining=False, bots=list(rows))
    return answer(encode_fleet_snapshot(snapshot))


def accounts(*names: str) -> script_hooks.CommandResult:
    """The ``GET /accounts`` answer listing ``names``.

    Args:
        names: The configured accounts, in accounts.json order.

    Returns:
        The command result.
    """
    listed: list[JSONValue] = list(names)
    return answer({"accounts": listed})


def stats(started_at: str) -> script_hooks.CommandResult:
    """A ``/stats`` answer whose run digest started at ``started_at``.

    Args:
        started_at: The digest's ``started_at``.

    Returns:
        The command result.
    """
    return answer({"available": True, "started_at": started_at})


def activity(tick: int, fuel: int) -> script_hooks.CommandResult:
    """An ``/activity`` answer whose tail stands at ``tick`` and ``fuel``.

    Args:
        tick: The last ``tick_n``.
        fuel: The last fuel total, ``-1`` before any was logged.

    Returns:
        The command result.
    """
    return answer({"available": True, "state": "COLLECT/SEARCH", "tick": tick, "fuel": fuel})


def passing_gate() -> dict[str, script_hooks.CommandResult | list[script_hooks.CommandResult]]:
    """The answers to one passing gate, keyed by a word of each request, most specific first.

    Returns:
        An empty fleet before the spawn, the live gate bot on Arterial
        after it, a digest of the new run, and a tail that has ticked.
    """
    row = bot_row(GATE_INSTANCE, "Arterial", alive=True)
    return {
        f"/bots/{GATE_INSTANCE}/stats": stats(SPAWNED_AT),
        f"/bots/{GATE_INSTANCE}/activity": activity(2, 942),
        "/accounts": accounts("Arterial"),
        "Invoke-RestMethod": answer(encode_fleet_bot(row)),
        f"{FLEET_URL}/bots": [fleet(), fleet(row)],
    }
