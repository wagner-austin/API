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
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.node import NodeConfig, NodeState
from fleet.core import remote

#: Where the probe script is written on the node.
#:
#: Under the node's own temp directory rather than the stage root, because the
#: probe answers whether the stage root can be used and must not need it first.
PROBE_REMOTE_PATH = "$env:TEMP\\fleet-probe.ps1"

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


def parse_probe(host: str, output: str, *, live_runs: int) -> NodeState:
    """Read a probe's output into a node's live state.

    Args:
        host: The node that produced it, carried into the state so a reading
            cannot later be attributed to the wrong machine.
        output: The probe script's standard output.
        live_runs: Fleet dispatches currently live on the node, counted from
            the ledger rather than probed -- see the module docstring.

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
    fields = _read_fields(output)
    return NodeState(
        host=host,
        free_ram_gb=_require_number(host, fields, "free_ram_gb", output),
        free_disk_gb=_require_number(host, fields, "free_disk_gb", output),
        live_runs=live_runs,
    )


def _read_fields(output: str) -> dict[str, str]:
    """Split key=value lines into a mapping.

    A line without an ``=`` is skipped rather than refused: PowerShell writes
    warnings to the same stream, and a warning that does not displace a field
    has not broken anything. A line whose key is absent IS refused, by
    :func:`_require_number`, which is where the failure belongs.

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


def _require_number(host: str, fields: dict[str, str], key: str, output: str) -> float:
    """Read one numeric field, refusing an absent or unparsable one.

    Args:
        host: The node, for the message.
        fields: The parsed key=value pairs.
        key: The field to read.
        output: The whole output, for the message.

    Returns:
        The value.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` when the field is missing or is
            not a number.
    """
    raw = fields.get(key)
    if raw is None:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"{host} answered without a {key!r} field; it reported: {output.strip()!r}",
        )
    cleaned = raw.replace(",", "")
    if not _is_number(cleaned):
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"{host} reported {key}={raw!r}, which is not a number; it reported: "
            f"{output.strip()!r}",
        )
    return float(cleaned)


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


def probe_node(node: NodeConfig, *, live_runs: int) -> NodeState:
    """Ask a node what it has free.

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
    output = remote.run_script(node["host"], PROBE_REMOTE_PATH, PROBE_SCRIPT)
    return parse_probe(node["host"], output, live_runs=live_runs)


__all__ = ["PROBE_REMOTE_PATH", "PROBE_SCRIPT", "parse_probe", "probe_node"]
