"""The observe pass's shared fixtures: a node's records, its roster entry, the
script's document, and the board identity every observe suite sends.

Split out of ``test_observe`` when the record decoding moved to
:mod:`fleet.core.observe_records` with a suite of its own (MCPs board task
56fee128), so both suites, and the native observe and wake suites that read
a node through the same pass, build the same fixtures from one place. The
record fixture is the field set measured on austinpc on 2026-09-12
(2.1.270, with ``pidDomain``) with the identifiers and paths changed to a
Linux node's.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from platform_core.mcp_client import McpCredentials

from fleet.contracts.machine import LedgerObserver, Machine, MachineRole

BOARD = McpCredentials(url="http://127.0.0.1:8033/mcp", api_key="k", tenant_id="t")
IDENTITY: JSONObject = {
    "agent": "fleet-node-colossus",
    "sessionId": "33333333-cccc-4ccc-8ccc-333333333333",
    "cwd": "/home/corvis/PROJECTS/API",
}
#: A native evidence lifetime no workspace declares, so a case reading it
#: back proves the argument reached the observation.
EVIDENCE = 450
SESSION = "d31e5228-3269-4a22-a27f-e535cbef1894"
MACHINE = "linux:colossus"
STARTED_MS = 1_789_000_000_000
UPDATED_MS = 1_789_000_030_500

#: Where the observe script lands under the fleet's account on every node.
SCRIPT_PATH = "/home/corvis/.fleet/observe-sessions.sh"
EMPTY_IDENTITIES = dump_json_str(
    {
        "rows": [],
        "pagination": {"offset": 0, "limit": 200, "hasMore": False},
    }
)

#: The pidDomain Claude Code 2.1.294 writes on diphtheria: its
#: ``/etc/machine-id`` and pid namespace, measured 2026-10-16 (board task
#: 5cee2f65). Its ledger machine key stays ``linux:diphtheria``.
DIPHTHERIA_PID_DOMAIN = "linux:212dd22057a34d6587f019c387979599:pid:[4026531836]"


def harness_record(*, without: tuple[str, ...] = (), **overrides: JSONValue) -> JSONObject:
    """Build one registration document as the 2.1.270 harness writes it.

    Args:
        without: Fields to leave out, so a test can exercise absence without
            a second fixture.
        **overrides: Fields to vary.

    Returns:
        The record, whose ``pidDomain`` is the ``<platform>:<hostname>``
        spelling every harness wrote before 2.1.294.
    """
    record: JSONObject = {
        "sessionId": SESSION,
        "pid": 4242,
        "startedAt": STARTED_MS,
        "updatedAt": UPDATED_MS,
        "name": "api-7e",
        "nameSource": "derived",
        "cwd": "/home/corvis/PROJECTS/API",
        "messagingSocketPath": "/run/user/1000/claude-cc-msg-abc",
        "version": "2.1.270",
        "status": "idle",
        "pidDomain": MACHINE,
        "permissionMode": "default",
    }
    for key in without:
        del record[key]
    record.update(overrides)
    return record


def linux_record() -> JSONObject:
    """One registration document as a node's 2.1.294 harness writes it.

    Returns:
        The record, with the machine-id pidDomain.
    """
    return harness_record(pidDomain=DIPHTHERIA_PID_DOMAIN)


def node(
    name: str,
    role: str,
    *,
    enabled: bool = True,
    user: str | None = "corvis",
    observer: LedgerObserver = LedgerObserver.FLEET_ARM,
) -> Machine:
    """Build one roster machine that declares no dispatch capacity.

    Args:
        name: The machine name.
        role: hub, worker, vpn-jump, session-host or client.
        enabled: Whether the roster says it should answer.
        user: The provisioned account, or None.
        observer: The arm the roster says records its sessions.

    Returns:
        The machine.
    """
    return Machine(
        name=name,
        role=MachineRole(role),
        user=user,
        enabled=enabled,
        ledger_observer=observer,
        dispatch=None,
        held="observed, never dispatched to",
    )


def document(
    *records: JSONObject,
    hostname: str = "colossus",
    platform: str = "linux",
    pid_domain: str | None = None,
) -> str:
    """Render what the observe script prints for these records.

    Args:
        *records: The registration documents.
        hostname: The lowercased hostname the script reports.
        platform: The harness platform tag the script reports.
        pid_domain: The pidDomain the script reports; None spells it
            ``<platform>:<hostname>``, as every harness before 2.1.294 did.

    Returns:
        The compact JSON the script emits.
    """
    return dump_json_str(
        {
            "platform": platform,
            "hostname": hostname,
            "pidDomain": f"{platform}:{hostname}" if pid_domain is None else pid_domain,
            "records": list(records),
        }
    )
