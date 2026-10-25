"""Asking a node what it has free right now.

THE SCRIPT IS SENT AND RUN BY PATH, never interpolated into a command line --
see :mod:`fleet.core.remote` for the two failed attempts that established the
rule. The body is a CONSTANT (:data:`fleet.core.dialect.CAPACITY_PROBE_SCRIPT`):
nothing is substituted into it by this package, so there is no value that
could carry a quote into a shell.

WHAT IT REPORTS AND WHAT IT DOES NOT. Free memory and free disk, and nothing
about processes. It would be easy to count the node's ``python`` instances
and call that its load; that number is unattributable -- some are the owner's
editor, some are another tool's -- and acting on it would make the fleet's
behaviour depend on software it does not manage. Live fleet runs come from the
ledger, where they are ours by construction because we wrote them. Work the
fleet does not run is charged only when it declares itself in a reservation
file, which names its holder (:mod:`fleet.core.reservations`, MCPs board task
84a2b690), so what the probe reports is still attributable.

THE OUTPUT FORMAT IS KEY=VALUE LINES, one per field, parsed strictly. Not JSON,
because the node renders it with shell string formatting and a malformed
number would produce malformed JSON that fails with a parser message instead of
a field name. Not positional, because a reordered script would silently swap
two numbers.

EVERY READ EXISTS TWICE, a value form and a raising boundary over it, for the
reason :mod:`fleet.core.remote` gives at length: a node being unreadable is a
refusal to weigh when choosing AMONG nodes, and an error when a caller named
one.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.action_cache import PROBE_FIELDS, ActionCacheCounts, inconsistency
from fleet.contracts.node import LiveLoad, NodeConfig, NodeState, SliceMemory
from fleet.core import dialect, names, remote, reservations

#: The share of the last minute a runnable task waited for a CPU (MCPs board
#: task 2ce5dc13).
PRESSURE_FIELD = "cpu_pressure_pct"

#: The share of the last minute a task stalled on I/O (MCPs board task
#: 814c4555).
IO_PRESSURE_FIELD = "io_pressure_pct"

#: The numeric fields a node must report before anything can be decided about
#: it; ``reserved_cores`` since MCPs board task 84a2b690, the cores its
#: reservations hold (:mod:`fleet.core.reservations`), and the two pressure
#: readings, which every probe has printed since the fleet's last Windows
#: node, whose kernel kept no such account, left it (board task e705f08c).
REQUIRED_FIELDS = (
    "free_ram_gb",
    "free_disk_gb",
    reservations.RESERVED_CORES_FIELD,
    PRESSURE_FIELD,
    IO_PRESSURE_FIELD,
)

#: The pair a node reports only when it runs a ``runners.slice`` (MCPs board
#: task 5d6e57e7): both or neither, since one script line prints both.
SLICE_FIELDS = ("ci_slice_current_gb", "ci_slice_high_gb")


class ProbeOutcome(TypedDict):
    """What a node reported, or why nothing could be read from it.

    THE VALUE FORM EXISTS FOR AUTO-SELECT. Choosing among several nodes is a
    question where "this one is off" belongs beside "this one is full" -- a
    refusal to weigh, not a reason to stop weighing. Choosing a NAMED node is a
    question where the same fact is an error. Both are served from here:
    :func:`attempt_probe` answers the first, :func:`probe_node` raises on top
    of it for the second.

    Attributes:
        state: What the node reported, or ``None`` when it could not be read.
        reason: Empty when ``state`` is present; otherwise the full
            explanation, already naming the node.
    """

    state: NodeState | None
    reason: str


def read_state(host: str, output: str, *, live: LiveLoad) -> ProbeOutcome:
    """Read a probe's output into a node's live state, or say why not.

    Args:
        host: The node that produced it, carried into the state so a reading
            cannot later be attributed to the wrong machine.
        output: The probe script's standard output.
        live: What the node's live fleet runs were granted, read from the
            ledger rather than probed -- see the module docstring.

    Returns:
        What the node reported, or the reason its answer could not be read.
    """
    fields = _read_fields(output)
    unreadable_file = fields.get(reservations.UNREADABLE_FIELD)
    if unreadable_file is not None:
        return ProbeOutcome(
            state=None,
            reason=(
                f"{host} holds a reservation file its probe cannot read, {unreadable_file}: "
                "it needs whole-number cores= and until_unix= lines "
                "(fleet.core.reservations), and a reservation read as nothing would let a "
                "suite onto the cores it holds"
            ),
        )
    readings: dict[str, float] = {}
    sliced = any(key in fields for key in SLICE_FIELDS)
    for key in REQUIRED_FIELDS + (SLICE_FIELDS if sliced else ()):
        value = _read_number(fields, key)
        if value is None:
            return ProbeOutcome(state=None, reason=_unreadable(host, fields, key, output))
        readings[key] = value
    reserved_cores = readings[reservations.RESERVED_CORES_FIELD]
    if not reserved_cores.is_integer() or reserved_cores < 0:
        return ProbeOutcome(
            state=None,
            reason=(
                f"{host} reported {reservations.RESERVED_CORES_FIELD}={reserved_cores!r}, "
                "which is not a whole number of cores"
            ),
        )
    ci_slice = (
        SliceMemory(
            current_gb=readings["ci_slice_current_gb"], high_gb=readings["ci_slice_high_gb"]
        )
        if sliced
        else None
    )
    action_cache, cache_problem = _read_action_cache(host, fields, output)
    if cache_problem:
        return ProbeOutcome(state=None, reason=cache_problem)
    return ProbeOutcome(
        state=NodeState(
            host=host,
            free_ram_gb=readings["free_ram_gb"],
            free_disk_gb=readings["free_disk_gb"],
            live=live,
            ci_slice=ci_slice,
            reserved_cores=int(reserved_cores),
            action_cache=action_cache,
            cpu_pressure_pct=readings[PRESSURE_FIELD],
            io_pressure_pct=readings[IO_PRESSURE_FIELD],
        ),
        reason="",
    )


def _read_action_cache(
    host: str, fields: dict[str, str], output: str
) -> tuple[ActionCacheCounts | None, str]:
    """Read a runner host's action cache counts, all four or none.

    A node prints them only when it publishes fresh counts
    (:data:`fleet.core.dialect.CAPACITY_PROBE_SCRIPT`), and its root
    timer writes all four in one file, so one without the others means the
    file or the probe did not run as written.

    Args:
        host: The node, for the message.
        fields: The parsed key=value pairs.
        output: The whole output, for the message.

    Returns:
        The counts or None, and an empty string; or None and the reason the
        node's answer cannot be read.
    """
    if not any(key in fields for key in PROBE_FIELDS):
        return None, ""
    counts: list[int] = []
    for key in PROBE_FIELDS:
        value = _read_number(fields, key)
        if value is None:
            return None, _unreadable(host, fields, key, output)
        if not value.is_integer():
            return None, f"{host} reported {key}={value!r}, which is not a whole number of runners"
        counts.append(int(value))
    runners, reading, served, missed = counts
    read = ActionCacheCounts(runners=runners, reading=reading, served=served, missed=missed)
    problem = inconsistency(read)
    if problem is not None:
        return None, f"{host} reported {problem}"
    return read, ""


def parse_probe(host: str, output: str, *, live: LiveLoad) -> NodeState:
    """Read a probe's output into a node's live state.

    The raising boundary over :func:`read_state`.

    Args:
        host: The node that produced it.
        output: The probe script's standard output.
        live: What the node's live fleet runs were granted.

    Returns:
        What the node reported.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if a required field is absent or
            does not parse as a number. Named "unreachable" rather than
            "unparsable" because that is what it means operationally: the
            node answered, but not with its state, so nothing can be decided
            about it. The message names the field and shows the output, since
            the usual cause is a shell error printed where a number was
            expected.
    """
    outcome = read_state(host, output, live=live)
    state = outcome["state"]
    if state is None:
        raise AppError(FleetErrorCode.NODE_UNREACHABLE, outcome["reason"])
    return state


def _read_fields(output: str) -> dict[str, str]:
    """Split key=value lines into a mapping.

    A line without an ``=`` is skipped rather than refused: a login banner or
    a warning can reach the same stream, and one that does not displace a
    field has not broken anything. A line whose key is absent IS refused, by
    :func:`read_state`, which is where the failure belongs.

    Args:
        output: The probe's standard output.

    Returns:
        Every key=value pair, later lines winning.
    """
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _read_number(fields: dict[str, str], key: str) -> float | None:
    """Read one numeric field, or report that it cannot be read.

    Args:
        fields: The parsed key=value pairs.
        key: The field to read.

    Returns:
        The value, or None when the field is missing or is not a number.
    """
    raw = fields.get(key)
    if raw is None or not _is_number(raw):
        return None
    return float(raw)


def _unreadable(host: str, fields: dict[str, str], key: str, output: str) -> str:
    """Explain why one field could not be read.

    Two messages rather than one, because they send a reader different ways: a
    MISSING field means the script did not run as written, while a field
    holding something that is not a number usually means a command printed an
    error where the value should have been. The node's whole answer is quoted
    either way, since that is what tells them apart.

    Args:
        host: The node, for the message.
        fields: The parsed key=value pairs.
        key: The field that could not be read.
        output: The whole output, for the message.

    Returns:
        The explanation.
    """
    raw = fields.get(key)
    if raw is None:
        return f"{host} answered without a {key!r} field; it reported: {output.strip()!r}"
    return f"{host} reported {key}={raw!r}, which is not a number; it reported: {output.strip()!r}"


def _is_number(value: str) -> bool:
    """Whether a string is a plain decimal number.

    Written rather than reached for via an exception, because the codebase
    does not use ``try``/``except`` to ask a question. ``str.isdigit`` alone
    is not enough: the probe formats with three decimal places and a
    thousands separator, and the separator is stripped before this is called.

    Args:
        value: The candidate, already stripped of separators.

    Returns:
        True when it is one optional sign, then digits, with at most one
        decimal point and at least one digit.
    """
    body = value[1:] if value[:1] in {"-", "+"} else value
    if body.count(".") > 1:
        return False
    digits = body.replace(".", "")
    return digits.isdigit()


def attempt_probe(node: NodeConfig, *, live: LiveLoad, writer: str) -> ProbeOutcome:
    """Ask a node what it has free, reporting failure as a value.

    What auto-select uses. A node that is powered off, that refuses ssh, or
    that answers with something other than its state all come back the same
    way: as a reason, to be weighed beside the nodes that did answer.

    Args:
        node: The node to probe.
        live: What its live fleet runs were granted, from the ledger.
        writer: Who is asking, which names the script's path on the node
            (:func:`fleet.core.names.capacity_probe_stem`) so two callers
            asking at once never write one file.

    Returns:
        Its live state, or why nothing could be read from it.
    """
    # The probe lives under the stage root, which the writer creates, so
    # nothing has to exist before the first probe. The script is a constant;
    # nothing is substituted into it, so nothing can be injected.
    outcome = remote.attempt_script(
        node["host"],
        dialect.script_path(node["stage_root"], names.capacity_probe_stem(writer)),
        dialect.CAPACITY_PROBE_SCRIPT,
    )
    failure = outcome["failure"]
    if failure is not None:
        return ProbeOutcome(state=None, reason=failure["message"])
    return read_state(node["host"], outcome["output"], live=live)


def probe_node(node: NodeConfig, *, live: LiveLoad, writer: str) -> NodeState:
    """Ask a node what it has free.

    The raising boundary over :func:`attempt_probe`, for a caller that has
    already committed to this node -- ``fleet-run --node lavender``, where
    lavender being down is an error rather than a preference to weigh.

    Args:
        node: The node to probe.
        live: What its live fleet runs were granted, from the ledger.
        writer: Who is asking, as :func:`attempt_probe` takes it.

    Returns:
        Its live state.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if ssh cannot reach it or its
            answer cannot be read, or ``DISPATCH_FAILED`` if the probe script
            itself exits non-zero.
    """
    outcome = attempt_probe(node, live=live, writer=writer)
    state = outcome["state"]
    if state is None:
        raise AppError(FleetErrorCode.NODE_UNREACHABLE, outcome["reason"])
    return state


__all__ = [
    "REQUIRED_FIELDS",
    "ProbeOutcome",
    "attempt_probe",
    "parse_probe",
    "probe_node",
    "read_state",
]
