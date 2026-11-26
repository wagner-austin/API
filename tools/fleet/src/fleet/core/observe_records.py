"""Reading what one node's observe script reported, record by record.

The node pass (:mod:`fleet.core.observe`) sends the script and records the
result; this module reads the document the script prints and copies each
harness record into ``task_session_observe``'s wire shape.

THE MACHINE KEY AND THE pidDomain ARE TWO SPELLINGS (board task 5cee2f65).
Until Claude Code 2.1.294 the harness wrote ``pidDomain`` as
``<platform>:<hostname>`` on both platforms, so one string served as both.
2.1.294 writes ``linux:<machine-id>:pid:[<pid namespace inode>]`` on Linux
(measured on ziggurat 2026-10-16:
``linux:f191c99155c14a719632bdc88e77f914:pid:[4026531836]``, its
``/etc/machine-id`` and ``readlink /proc/self/ns/pid``) and still
``win32:<hostname>`` on Windows (all 8 records on austinpc), and every hub
tick from 06:30Z failed on the comparison. So the script computes the
node's own ``pidDomain`` the way its harness does, and a record is
compared against THAT, while the ledger keeps its ``<platform>:<hostname>``
key.

WHAT IS COPIED AND WHAT IS CHECKED. Every field the ledger stores is copied
from the record; nothing is inferred. Two things are refused by name: a record
that is not the shape the harness writes (``SESSION_RECORD_UNREADABLE``), and
a record whose own ``pidDomain`` is not the one its node reports
(``SESSION_MACHINE_MISMATCH``) -- the refusal ``pcsession-mcp`` makes for the
hub's own directory, made here for the same reason.

A REFUSAL IS A VALUE (MCPs board task 56fee128). Each decode returns its
result or a :class:`RecordRefusal` naming every fault it found, and the node
pass makes that the node's outcome. Until 2026-11-25 a refusal raised, and
one demure record whose ``messagingSocketPath`` was null ended every hub
tick from 01:09Z, before any later node was observed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

from platform_core.error_codes_fleet import FleetErrorCode
from platform_core.json_utils import JSONObject, JSONValue, load_json_str
from typing_extensions import TypeIs


class NodeSessions(TypedDict):
    """What the script reported from one node.

    Attributes:
        platform: The harness platform tag the node's own interpreter
            reports, ``linux`` on every node since board task e705f08c.
        hostname: The node's hostname, lowercased.
        pid_domain: The ``pidDomain`` the node's harness writes, computed on
            the node the way the harness computes it.
        records: Every registration document in the directory, untouched.
    """

    platform: str
    hostname: str
    pid_domain: str
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


class RecordRefusal(TypedDict):
    """Why this pass cannot read what one node reported.

    A decode returns this or its result; :func:`is_refusal` tells them apart.

    Attributes:
        code: ``SESSION_RECORD_UNREADABLE`` for a document or record not in
            the shape the harness writes, ``SESSION_MACHINE_MISMATCH`` for a
            record whose own ``pidDomain`` is not the one its node reports.
        message: The node, the session when its id was readable, and every
            field at fault, in one sentence.
    """

    code: FleetErrorCode
    message: str


def is_refusal(
    result: NodeSessions | SessionObservation | RecordRefusal,
) -> TypeIs[RecordRefusal]:
    """Whether a decode refused, narrowing its result either way.

    ``code`` is a key of :class:`RecordRefusal` alone, so its presence is
    the refusal. A ``TypeIs`` predicate rather than ``@final`` TypedDicts
    narrowed by ``in``: that decorator is read by mypy alone, so a mutant
    removing it changes nothing a test can observe.

    Args:
        result: What :func:`decode_node_sessions` or :func:`to_observation`
            returned.

    Returns:
        True for a refusal; the type checker reads the result as a
        :class:`RecordRefusal` when True and as the decode's own shape when
        False.
    """
    return "code" in result


def _unreadable(host: str, problems: list[str], session: str | None) -> RecordRefusal:
    """Build the refusal for a document or record this pass cannot read.

    Args:
        host: The node it came from.
        problems: Every fault found in it, in reading order.
        session: The record's ``sessionId`` when that much was readable.

    Returns:
        ``SESSION_RECORD_UNREADABLE``, naming the session and every fault.
    """
    named = "" if session is None else f" for session {session}"
    return RecordRefusal(
        code=FleetErrorCode.SESSION_RECORD_UNREADABLE,
        message=(
            f"{host} reported a session record this runner cannot read{named}: "
            f"{'; '.join(problems)}"
        ),
    )


def _text(record: JSONObject, key: str, problems: list[str]) -> str | None:
    """Read one required string field, noting the fault when it is not one.

    Every field of a record is read before any is judged, so one refusal
    names every fault rather than the first.

    Args:
        record: The record.
        key: The field.
        problems: The record's faults so far; a fault here is appended.

    Returns:
        The value, or None when it is absent, empty or not a string.
    """
    value = record.get(key)
    if isinstance(value, str) and value:
        return value
    problems.append(f"{key!r} is {type(value).__name__}, not a non-empty string")
    return None


def _whole(record: JSONObject, key: str, problems: list[str]) -> int | None:
    """Read one required integer field, noting the fault when it is not one.

    Args:
        record: The record.
        key: The field.
        problems: The record's faults so far; a fault here is appended.

    Returns:
        The value, or None when it is absent or not an integer. ``bool`` is
        a fault too: it is an ``int`` to Python and a pid to nobody.
    """
    value = record.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    problems.append(f"{key!r} is {type(value).__name__}, not an integer")
    return None


def iso_from_millis(millis: int) -> str:
    """Render epoch milliseconds as ISO-8601 with an explicit UTC offset.

    Args:
        millis: Milliseconds since the epoch, as the harness writes
            ``startedAt`` and ``updatedAt``.

    Returns:
        ``2026-09-13T00:10:00.000+00:00``.
    """
    return datetime.fromtimestamp(millis / 1000, tz=UTC).isoformat(timespec="milliseconds")


def decode_node_sessions(host: str, output: str) -> NodeSessions | RecordRefusal:
    """Read the script's document.

    Args:
        host: The node it came from.
        output: The script's standard output.

    Returns:
        The decoded document, or ``SESSION_RECORD_UNREADABLE`` when it is not
        the document the script emits: not an object, ``records`` not an
        array, a record that is not an object, or a missing ``platform``,
        ``hostname`` or ``pidDomain`` -- every fault named, in one refusal.

    Raises:
        InvalidJsonError: When the output is not JSON at all. The script
            prints one ``json.dumps`` document and exits nonzero on any
            error, which :func:`fleet.core.remote.attempt_script` already
            returns as the node's failure; output that is not JSON is this
            runner's own script broken, not a record a node holds.
    """
    document: JSONValue = load_json_str(output)
    if not isinstance(document, dict):
        return _unreadable(
            host, [f"the document is {type(document).__name__}, not an object"], None
        )
    records_raw = document.get("records")
    if not isinstance(records_raw, list):
        return _unreadable(host, [f"'records' is {type(records_raw).__name__}, not an array"], None)
    records = [entry for entry in records_raw if isinstance(entry, dict)]
    strays = [entry for entry in records_raw if not isinstance(entry, dict)]
    problems = [f"a record is {type(entry).__name__}, not an object" for entry in strays]
    platform = _text(document, "platform", problems)
    hostname = _text(document, "hostname", problems)
    pid_domain = _text(document, "pidDomain", problems)
    if strays or platform is None or hostname is None or pid_domain is None:
        return _unreadable(host, problems, None)
    return NodeSessions(
        platform=platform,
        hostname=hostname,
        pid_domain=pid_domain,
        records=tuple(records),
    )


def to_observation(
    host: str, pid_domain: str, record: JSONObject
) -> SessionObservation | RecordRefusal:
    """Copy one harness record into the tool's wire shape.

    Args:
        host: The node it came from, for the refusal.
        pid_domain: The ``pidDomain`` that node reports for its own harness.
        record: The registration document, as the harness wrote it.

    Returns:
        The observation; or ``SESSION_RECORD_UNREADABLE`` naming every
        missing or mistyped field; or ``SESSION_MACHINE_MISMATCH`` when the
        record is readable but its own ``pidDomain`` is not ``pid_domain``.
    """
    problems: list[str] = []
    session_id = _text(record, "sessionId", problems)
    recorded = record.get("pidDomain")
    # Optional (2.1.236 writes none), so absence is no fault and only a
    # value that is not a string is.
    domain_unreadable = recorded is not None and not isinstance(recorded, str)
    if domain_unreadable:
        problems.append(f"'pidDomain' is {type(recorded).__name__}, not a string")
    pid = _whole(record, "pid", problems)
    started = _whole(record, "startedAt", problems)
    name = _text(record, "name", problems)
    name_source = _text(record, "nameSource", problems)
    cwd = _text(record, "cwd", problems)
    socket_path = _text(record, "messagingSocketPath", problems)
    version = _text(record, "version", problems)
    status = _text(record, "status", problems)
    updated = _whole(record, "updatedAt", problems)
    if (
        domain_unreadable
        or session_id is None
        or pid is None
        or started is None
        or name is None
        or name_source is None
        or cwd is None
        or socket_path is None
        or version is None
        or status is None
        or updated is None
    ):
        return _unreadable(host, problems, session_id)
    if isinstance(recorded, str) and recorded != pid_domain:
        return RecordRefusal(
            code=FleetErrorCode.SESSION_MACHINE_MISMATCH,
            message=(
                f"{host}: record for session {session_id} "
                f"names pidDomain {recorded!r} but its node reports {pid_domain!r}"
            ),
        )
    return SessionObservation(
        sessionId=session_id,
        pid=pid,
        processStartedAt=iso_from_millis(started),
        name=name,
        nameSource=name_source,
        cwd=cwd,
        socketPath=socket_path,
        harnessVersion=version,
        status=status,
        heartbeatAt=iso_from_millis(updated),
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
    """The machine key a node's document spells.

    Args:
        sessions: The decoded document.

    Returns:
        ``<platform>:<hostname>``, the ledger's machine key, which since
        Claude Code 2.1.294 is no longer the harness's ``pidDomain``.
    """
    return f"{sessions['platform']}:{sessions['hostname']}"


__all__ = [
    "NodeSessions",
    "RecordRefusal",
    "SessionObservation",
    "decode_node_sessions",
    "is_refusal",
    "iso_from_millis",
    "machine_of",
    "observation_json",
    "to_observation",
]
