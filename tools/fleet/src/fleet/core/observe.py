"""Observing the fleet's Claude Code sessions into the board's session ledger.

THE HUB CAN MOUNT ITS OWN SESSION RECORDS AND NOBODY ELSE'S. ``pcsession-mcp``
(repo MCPs) reads austinpc's ``~/.claude/sessions/<pid>.json`` through a
read-only bind mount and records every process there in ``agent_session_ledger``
(MCPs board task 5a3865bf). serendipity's directory is on serendipity, and no
container on the compose network has a route to it. This runner does: it is
the process on the hub that already reaches every node over ssh with the keys
the server deliberately does not hold, so it reads the same records there and
hands them to ``task_session_observe``, which records them through the same
canonical writer, keyed by the machine they came from.

THE SCRIPT IS SENT AND RUN BY PATH, like every remote command in this package
-- :mod:`fleet.core.remote` carries the measured reason. It emits ONE JSON
document: the node's platform and lowercased hostname (which together spell
the harness's own ``pidDomain``, ``win32:serendipity``) and every record in
the directory, verbatim. An absent directory is a node that has never run
Claude Code, reported as zero records rather than as a fault.

WHAT IS COPIED AND WHAT IS CHECKED. Every field the ledger stores is copied
from the record; nothing is inferred. Two things are refused by name: a record
that is not the shape the harness writes (``SESSION_RECORD_UNREADABLE``), and
a record whose own ``pidDomain`` names a machine other than the node it was
read from (``SESSION_MACHINE_MISMATCH``) -- the refusal ``pcsession-mcp``
makes for the hub's own directory, made here for the same reason.

WHICH FAILURES ARE VALUES. A node that is off, or refuses ssh, is an OUTCOME
of the pass -- reported per node and the pass continues, exactly as
:func:`fleet.core.probe.attempt_probe` treats an unreachable node -- because
one laptop asleep must not blind the ledger to the rest of the fleet. A record
that cannot be read is a CONTRACT fault and raises: the harness changed shape,
and recording around it would fill the ledger with rows nobody can trace.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final, TypedDict

from platform_core.error_codes_tooling import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, JSONValue, load_json_str
from platform_core.mcp_client import McpCredentials

from fleet.core import queue, remote
from fleet.core.registry import RegistryNode

#: What the script is called on the node, under the provisioned account's
#: ``.fleet`` directory -- the same place-by-path convention the probe and
#: dispatch scripts use, and a name that cannot collide with either.
SCRIPT_NAME: Final = "observe-sessions.ps1"

#: Roles whose sessions this pass records. The hub is observed by
#: ``pcsession-mcp`` on its own mount; a client (the phone) is never
#: provisioned and has no account of ours to ssh in as.
OBSERVABLE_ROLES: Final = frozenset({"worker", "vpn-jump"})

#: The script, rendered verbatim onto the node and run by path.
#:
#: ``platform`` is the literal ``win32`` because the runner that executes this
#: is PowerShell on Windows by construction (``POWERSHELL_INVOCATION``); a node
#: that cannot run it reports that as an unreachable outcome rather than as a
#: platform. ``hostname`` is lowercased because that is how the harness spells
#: ``pidDomain`` (measured on austinpc: ``$env:COMPUTERNAME`` is ``AUSTINPC``,
#: the record says ``win32:austinpc``). Records are emitted UNTOUCHED so the
#: decode on this side sees exactly the bytes the harness wrote.
OBSERVE_SCRIPT: Final = """\
$ErrorActionPreference = 'Stop'
$dir = Join-Path $HOME '.claude\\sessions'
$records = @()
if (Test-Path -LiteralPath $dir) {
    foreach ($file in Get-ChildItem -LiteralPath $dir -Filter '*.json' -File) {
        $records += , (Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json)
    }
}
$document = [pscustomobject]@{
    platform = 'win32'
    hostname = $env:COMPUTERNAME.ToLowerInvariant()
    records  = @($records)
}
$document | ConvertTo-Json -Depth 8 -Compress
"""


class NodeSessions(TypedDict):
    """What the script reported from one node.

    Attributes:
        platform: The harness platform tag, ``win32``.
        hostname: The node's hostname, lowercased.
        records: Every registration document in the directory, untouched.
    """

    platform: str
    hostname: str
    records: tuple[JSONObject, ...]


class SessionObservation(TypedDict):
    """One record in ``task_session_observe``'s wire shape.

    Every field is copied from the harness record. The two instants are
    ISO-8601 with an explicit offset, converted from the record's epoch
    milliseconds; the tool refuses anything else.
    """

    sessionId: str
    pid: int
    processStartedAt: str
    name: str
    nameSource: str
    cwd: str
    socketPath: str
    harnessVersion: str
    status: str
    heartbeatAt: str


class NodeOutcome(TypedDict):
    """What one node's pass came to.

    Attributes:
        node: The registry name.
        recorded: Whether the board accepted a pass for it. False means the
            node did not answer, and ``detail`` says why in ssh's own words.
        detail: The board's rendered line, or the transport failure.
    """

    node: str
    recorded: bool
    detail: str


def script_path(user: str) -> str:
    """Where the script lands on a node.

    Args:
        user: The provisioned account.

    Returns:
        An absolute, literal Windows path -- the write command on the far
        side expands nothing, so this must be spelled out.
    """
    return f"C:/Users/{user}/.fleet/{SCRIPT_NAME}"


def _unreadable(host: str, detail: str) -> AppError[FleetErrorCode]:
    """Build the contract-fault error for a record this pass cannot read.

    Args:
        host: The node it came from.
        detail: What was wrong with it.

    Returns:
        The error, ready to raise.
    """
    return AppError(
        code=FleetErrorCode.SESSION_RECORD_UNREADABLE,
        message=f"{host} reported a session record this runner cannot read: {detail}",
    )


def _field_str(host: str, record: JSONObject, key: str) -> str:
    """Read one required string field of a record.

    Args:
        host: The node, for the error.
        record: The record.
        key: The field.

    Returns:
        The value.

    Raises:
        AppError: ``SESSION_RECORD_UNREADABLE`` when absent or not a string.
    """
    value = record.get(key)
    if not isinstance(value, str) or value == "":
        raise _unreadable(host, f"{key!r} is {type(value).__name__}, not a non-empty string")
    return value


def _field_int(host: str, record: JSONObject, key: str) -> int:
    """Read one required integer field of a record.

    Args:
        host: The node, for the error.
        record: The record.
        key: The field.

    Returns:
        The value.

    Raises:
        AppError: ``SESSION_RECORD_UNREADABLE`` when absent or not an
            integer. ``bool`` is refused too: it is an ``int`` to Python and
            a pid to nobody.
    """
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise _unreadable(host, f"{key!r} is {type(value).__name__}, not an integer")
    return value


def iso_from_millis(millis: int) -> str:
    """Render epoch milliseconds as ISO-8601 with an explicit UTC offset.

    Args:
        millis: Milliseconds since the epoch, as the harness writes
            ``startedAt`` and ``updatedAt``.

    Returns:
        ``2026-09-13T00:10:00.000+00:00``.
    """
    return datetime.fromtimestamp(millis / 1000, tz=UTC).isoformat(timespec="milliseconds")


def decode_node_sessions(host: str, output: str) -> NodeSessions:
    """Read the script's document.

    Args:
        host: The node it came from.
        output: The script's standard output.

    Returns:
        The decoded document.

    Raises:
        AppError: ``SESSION_RECORD_UNREADABLE`` when the output is not the
            document the script emits -- not JSON, not an object, or with
            ``records`` that is not a list of objects.
    """
    document: JSONValue = load_json_str(output)
    if not isinstance(document, dict):
        raise _unreadable(host, f"the document is {type(document).__name__}, not an object")
    records_raw = document.get("records")
    if not isinstance(records_raw, list):
        raise _unreadable(host, f"'records' is {type(records_raw).__name__}, not an array")
    records: list[JSONObject] = []
    for entry in records_raw:
        if not isinstance(entry, dict):
            raise _unreadable(host, f"a record is {type(entry).__name__}, not an object")
        records.append(entry)
    return NodeSessions(
        platform=_field_str(host, document, "platform"),
        hostname=_field_str(host, document, "hostname"),
        records=tuple(records),
    )


def to_observation(host: str, machine: str, record: JSONObject) -> SessionObservation:
    """Copy one harness record into the tool's wire shape.

    Args:
        host: The node it came from, for errors.
        machine: The machine id this pass attributes records to.
        record: The registration document, as the harness wrote it.

    Returns:
        The observation.

    Raises:
        AppError: ``SESSION_RECORD_UNREADABLE`` for a missing or mistyped
            field; ``SESSION_MACHINE_MISMATCH`` when the record's own
            ``pidDomain`` names a different machine than ``machine``.
    """
    pid_domain = record.get("pidDomain")
    if pid_domain is not None:
        if not isinstance(pid_domain, str):
            raise _unreadable(host, f"'pidDomain' is {type(pid_domain).__name__}, not a string")
        if pid_domain != machine:
            raise AppError(
                code=FleetErrorCode.SESSION_MACHINE_MISMATCH,
                message=(
                    f"{host}: record for session {_field_str(host, record, 'sessionId')} "
                    f"names pidDomain {pid_domain!r} but was read from {machine!r}"
                ),
            )
    return SessionObservation(
        sessionId=_field_str(host, record, "sessionId"),
        pid=_field_int(host, record, "pid"),
        processStartedAt=iso_from_millis(_field_int(host, record, "startedAt")),
        name=_field_str(host, record, "name"),
        nameSource=_field_str(host, record, "nameSource"),
        cwd=_field_str(host, record, "cwd"),
        socketPath=_field_str(host, record, "messagingSocketPath"),
        harnessVersion=_field_str(host, record, "version"),
        status=_field_str(host, record, "status"),
        heartbeatAt=iso_from_millis(_field_int(host, record, "updatedAt")),
    )


def observation_json(observation: SessionObservation) -> JSONObject:
    """Render one observation as the JSON object the tool takes.

    Args:
        observation: The observation.

    Returns:
        A plain JSON object with the same keys.
    """
    return {
        "sessionId": observation["sessionId"],
        "pid": observation["pid"],
        "processStartedAt": observation["processStartedAt"],
        "name": observation["name"],
        "nameSource": observation["nameSource"],
        "cwd": observation["cwd"],
        "socketPath": observation["socketPath"],
        "harnessVersion": observation["harnessVersion"],
        "status": observation["status"],
        "heartbeatAt": observation["heartbeatAt"],
    }


def machine_of(sessions: NodeSessions) -> str:
    """The machine id a node's document spells.

    Args:
        sessions: The decoded document.

    Returns:
        ``<platform>:<hostname>``, the harness's ``pidDomain`` form.
    """
    return f"{sessions['platform']}:{sessions['hostname']}"


def observe_node(board: McpCredentials, node: RegistryNode, identity: JSONObject) -> NodeOutcome:
    """Read one node's records and record them on the board.

    Args:
        board: The taskboard's endpoint and headers.
        node: The node, from the identity registry.
        identity: This runner's identity arguments.

    Returns:
        The outcome -- recorded, or not reached and why.

    Raises:
        AppError: A contract fault in what the node reported
            (``SESSION_RECORD_UNREADABLE``, ``SESSION_MACHINE_MISMATCH``),
            or any transport or contract failure from the board call.
            Never for the node being unreachable: that is the outcome.
    """
    user = node["user"]
    if user is None:
        return NodeOutcome(
            node=node["name"], recorded=False, detail="no provisioned user in the registry"
        )
    outcome = remote.attempt_script(node["name"], script_path(user), OBSERVE_SCRIPT)
    failure = outcome["failure"]
    if failure is not None:
        return NodeOutcome(node=node["name"], recorded=False, detail=failure["message"])
    sessions = decode_node_sessions(node["name"], outcome["output"])
    machine = machine_of(sessions)
    observations: list[JSONValue] = [
        observation_json(to_observation(node["name"], machine, record))
        for record in sessions["records"]
    ]
    detail = queue.observe_sessions(
        board, machine=machine, observations=observations, identity=identity
    )
    return NodeOutcome(node=node["name"], recorded=True, detail=detail)


def observe_pass(
    board: McpCredentials, registry: dict[str, RegistryNode], identity: JSONObject
) -> tuple[NodeOutcome, ...]:
    """Observe every node the registry says is live and observable.

    Args:
        board: The taskboard's endpoint and headers.
        registry: The identity registry's nodes, from
            :func:`fleet.core.registry.decode_registry_nodes`.
        identity: This runner's identity arguments.

    Returns:
        One outcome per node visited, in name order. Disabled nodes, the
        hub and clients are not visited and produce no outcome -- they were
        never asked, which is a different fact from not answering.

    Raises:
        AppError: See :func:`observe_node`.
    """
    outcomes: list[NodeOutcome] = []
    for name in sorted(registry):
        node = registry[name]
        if not node["enabled"] or node["role"] not in OBSERVABLE_ROLES:
            continue
        outcomes.append(observe_node(board, node, identity))
    return tuple(outcomes)


def render_outcome(outcome: NodeOutcome) -> str:
    """One log line per node.

    Args:
        outcome: The node's outcome.

    Returns:
        ``<node>: <detail>``, prefixed ``not observed`` when the node did not
        answer, so a quiet node and a recorded one never read alike.
    """
    if outcome["recorded"]:
        return f"{outcome['node']}: {outcome['detail']}"
    return f"{outcome['node']}: not observed -- {outcome['detail']}"


__all__ = [
    "OBSERVABLE_ROLES",
    "OBSERVE_SCRIPT",
    "SCRIPT_NAME",
    "NodeOutcome",
    "NodeSessions",
    "SessionObservation",
    "decode_node_sessions",
    "iso_from_millis",
    "machine_of",
    "observation_json",
    "observe_node",
    "observe_pass",
    "render_outcome",
    "script_path",
    "to_observation",
]
