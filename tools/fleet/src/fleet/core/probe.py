"""Asking a node what it has free right now.

THE SCRIPT IS SENT AND RUN BY PATH, never interpolated into a command line --
see :mod:`fleet.core.remote` for the two failed attempts that established the
rule. The body below is a CONSTANT: nothing is substituted into it by this
package, so there is no value that could carry a quote into a shell. The
braces it does contain are PowerShell's own format operator, evaluated on the
far side after the bytes have arrived, and are never seen by Python.

WHAT IT REPORTS AND WHAT IT DOES NOT. Free memory and free disk, and nothing
about processes. It would be easy to count the node's ``python.exe`` instances
and call that its load; that number is unattributable -- some are the owner's
editor, some are another tool's -- and acting on it would make the fleet's
behaviour depend on software it does not manage. Live fleet runs come from the
ledger, where they are ours by construction because we wrote them.

THE OUTPUT FORMAT IS KEY=VALUE LINES, one per field, parsed strictly. Not JSON,
because the node renders it with PowerShell string formatting and a malformed
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

from fleet.contracts.node import NodeConfig, NodeState
from fleet.core import remote

#: What the capacity probe is called under a node's stage root.
#:
#: A NAME, not a path, and under the stage root rather than the node's TEMP.
#: ``$env:TEMP`` cannot be used: :mod:`fleet.core.remote` writes through a
#: single-quoted PowerShell literal, which does not expand it, so the node
#: would grow a directory called ``$env:TEMP`` instead of resolving one. The
#: writer creates the parent, so nothing has to exist first.
PROBE_SCRIPT_NAME = "fleet-capacity.ps1"

#: The probe, verbatim. No substitution, so nothing can be injected into it.
#:
#: ``$ErrorActionPreference`` is left at its default: a failure to read the
#: machine's own memory is not something to soften, and the non-zero exit that
#: results is what :func:`~fleet.core.remote.run_ssh` turns into a typed error.
PROBE_SCRIPT = """\
$os = Get-CimInstance Win32_OperatingSystem
$drive = Get-PSDrive C
"free_ram_gb={0:N3}" -f ($os.FreePhysicalMemory / 1MB)
"free_disk_gb={0:N3}" -f ($drive.Free / 1GB)
"logical_cores={0}" -f (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
"""

#: The numeric fields a node must report before anything can be decided about it.
REQUIRED_FIELDS = ("free_ram_gb", "free_disk_gb")


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


def read_state(host: str, output: str, *, live_runs: int) -> ProbeOutcome:
    """Read a probe's output into a node's live state, or say why not.

    Args:
        host: The node that produced it, carried into the state so a reading
            cannot later be attributed to the wrong machine.
        output: The probe script's standard output.
        live_runs: Fleet dispatches currently live on the node, counted from
            the ledger rather than probed -- see the module docstring.

    Returns:
        What the node reported, or the reason its answer could not be read.
    """
    fields = _read_fields(output)
    readings: dict[str, float] = {}
    for key in REQUIRED_FIELDS:
        value = _read_number(fields, key)
        if value is None:
            return ProbeOutcome(state=None, reason=_unreadable(host, fields, key, output))
        readings[key] = value
    return ProbeOutcome(
        state=NodeState(
            host=host,
            free_ram_gb=readings["free_ram_gb"],
            free_disk_gb=readings["free_disk_gb"],
            live_runs=live_runs,
        ),
        reason="",
    )


def parse_probe(host: str, output: str, *, live_runs: int) -> NodeState:
    """Read a probe's output into a node's live state.

    The raising boundary over :func:`read_state`.

    Args:
        host: The node that produced it.
        output: The probe script's standard output.
        live_runs: Fleet dispatches currently live on the node.

    Returns:
        What the node reported.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if a required field is absent or
            does not parse as a number. Named "unreachable" rather than
            "unparsable" because that is what it means operationally: the
            node answered, but not with its state, so nothing can be decided
            about it. The message names the field and shows the output, since
            the usual cause is a PowerShell error printed where a number was
            expected.
    """
    outcome = read_state(host, output, live_runs=live_runs)
    state = outcome["state"]
    if state is None:
        raise AppError(FleetErrorCode.NODE_UNREACHABLE, outcome["reason"])
    return state


def _read_fields(output: str) -> dict[str, str]:
    """Split key=value lines into a mapping.

    A line without an ``=`` is skipped rather than refused: PowerShell writes
    warnings to the same stream, and a warning that does not displace a field
    has not broken anything. A line whose key is absent IS refused, by
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
    if raw is None:
        return None
    cleaned = raw.replace(",", "")
    if not _is_number(cleaned):
        return None
    return float(cleaned)


def _unreadable(host: str, fields: dict[str, str], key: str, output: str) -> str:
    """Explain why one field could not be read.

    Two messages rather than one, because they send a reader different ways: a
    MISSING field means the script did not run as written, while a field
    holding something that is not a number usually means PowerShell printed an
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


def attempt_probe(node: NodeConfig, *, live_runs: int) -> ProbeOutcome:
    """Ask a node what it has free, reporting failure as a value.

    What auto-select uses. A node that is powered off, that refuses ssh, or
    that answers with something other than its state all come back the same
    way: as a reason, to be weighed beside the nodes that did answer.

    Args:
        node: The node to probe.
        live_runs: Fleet dispatches currently live on it, from the ledger.

    Returns:
        Its live state, or why nothing could be read from it.
    """
    outcome = remote.attempt_script(
        node["host"], f"{node['stage_root']}/{PROBE_SCRIPT_NAME}", PROBE_SCRIPT
    )
    failure = outcome["failure"]
    if failure is not None:
        return ProbeOutcome(state=None, reason=failure["message"])
    return read_state(node["host"], outcome["output"], live_runs=live_runs)


def probe_node(node: NodeConfig, *, live_runs: int) -> NodeState:
    """Ask a node what it has free.

    The raising boundary over :func:`attempt_probe`, for a caller that has
    already committed to this node -- ``fleet-run --node lavender``, where
    lavender being down is an error rather than a preference to weigh.

    Args:
        node: The node to probe.
        live_runs: Fleet dispatches currently live on it, from the ledger.

    Returns:
        Its live state.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if ssh cannot reach it or its
            answer cannot be read, or ``DISPATCH_FAILED`` if the probe script
            itself exits non-zero.
    """
    outcome = attempt_probe(node, live_runs=live_runs)
    state = outcome["state"]
    if state is None:
        raise AppError(FleetErrorCode.NODE_UNREACHABLE, outcome["reason"])
    return state


__all__ = [
    "PROBE_SCRIPT",
    "PROBE_SCRIPT_NAME",
    "REQUIRED_FIELDS",
    "ProbeOutcome",
    "attempt_probe",
    "parse_probe",
    "probe_node",
    "read_state",
]
