"""Asking a node what it has installed; :mod:`fleet.core.toolchain_install` installs.

THE PROBE IS A CONSTANT SCRIPT, sent and run by path like every other remote
act here -- see :mod:`fleet.core.remote` for the two failed attempts that made
that a rule. It emits ``name=present=version`` lines, one per tool, parsed
strictly.

WHY IT ASKS FOR A VERSION AND NOT JUST PRESENCE. ``loki`` has poetry installed
under Python 3.12 while every project here pins 3.11. A presence check would
call that node ready, it would stage, and the build would fail resolving a
lockfile against the wrong interpreter -- which reads as a broken project
rather than a misconfigured node.

INSTALLING IS A SEPARATE MODULE AND A SEPARATE FLAG, for the reason
:mod:`fleet.core.toolchain_install` gives: these are other people's machines.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.capability import PROBE_NAME, Capability, measured
from fleet.contracts.detection import GPU_PROBE, TESTDB_PROBE
from fleet.contracts.elevation import INTEGRITY_PROBE
from fleet.contracts.node import NodeConfig
from fleet.contracts.tagged_tools import TAGGED_TOOLS
from fleet.contracts.toolchain import (
    PACKAGE_MANAGERS,
    REQUIRED_NODE_MAJOR,
    REQUIRED_PYTHON,
    REQUIRED_TOOLS,
    ToolReport,
    available_managers,
    install_command,
    missing,
    node_is_right,
    python_is_right,
    reported_version,
    version_number,
)
from fleet.core import dialect, names, remote


def read_reports(output: str) -> tuple[ToolReport, ...]:
    """Read a toolchain probe's output into one report per recognised line.

    Args:
        output: The probe script's standard output.

    Returns:
        One report per line naming a required tool, a tagged tool
        (``ffmpeg``), a package manager, a toolchain's probe line
        (``cargo``, ``cxx``, ``docker``, ``stack``), the CUDA device or test
        database line (``gpu``, ``testdb``, :mod:`fleet.contracts.detection`)
        or the session's token (``integrity``, :mod:`fleet.contracts.elevation`),
        in the order the node emitted them. Empty when no line did.
    """
    reports: list[ToolReport] = []
    wanted = (
        {tool["name"] for tool in REQUIRED_TOOLS + TAGGED_TOOLS}
        | set(PACKAGE_MANAGERS)
        | set(PROBE_NAME.values())
        | {GPU_PROBE, TESTDB_PROBE, INTEGRITY_PROBE}
    )
    for line in output.splitlines():
        parts = line.strip().split("=", 2)
        if len(parts) != 3 or parts[0] not in wanted:
            continue
        reports.append(
            ToolReport(name=parts[0], present=parts[1] == "yes", version=parts[2].strip())
        )
    return tuple(reports)


def _unrecognised(output: str) -> str:
    """Explain an answer that names no tool.

    Args:
        output: The probe script's standard output.

    Returns:
        The explanation, quoting the whole answer.
    """
    return (
        f"a toolchain probe returned nothing recognisable, so the node was never asked: "
        f"{output.strip()!r}"
    )


def parse_probe(output: str) -> tuple[ToolReport, ...]:
    """Read a toolchain probe's output into one report per tool.

    The raising boundary over :func:`read_reports`.

    Args:
        output: The probe script's standard output.

    Returns:
        One report per line that parsed, in the order the node emitted them.

    Raises:
        AppError: With ``NODE_TOOL_MISSING`` when the output names none of
            the required tools. That is not a node without tools -- it is a
            probe that did not run, and reporting every tool as absent would
            send the reader to install five things that are already there.
    """
    reports = read_reports(output)
    if not reports:
        raise AppError(FleetErrorCode.NODE_TOOL_MISSING, _unrecognised(output))
    return reports


def attempt_toolchain(
    node: NodeConfig, *, writer: str
) -> tuple[ToolReport, ...] | remote.RemoteFailure:
    """Ask a node what it has installed, reporting failure as a value.

    THE VALUE FORM EXISTS FOR THE NODE RUNNER. A runner deciding whether to
    claim this tick treats a node that did not answer the way it treats one
    that answered "python absent": it claims nothing and says why, and a
    tick that raised on either would stop on the condition it exists to
    report. :func:`probe_toolchain` raises on top of this for a caller that
    named the node, ``fleet-bootstrap``.

    Args:
        node: The node to probe.
        writer: Who is asking, which names the script's path on the node
            (:func:`fleet.core.names.toolchain_probe_stem`), so a node's two
            runners probing on the same minute never write one file.

    Returns:
        Its reports, never empty; or the typed reason there are none: the
        transport's ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED``, or
        ``NODE_TOOL_MISSING`` when it answered with no line this package
        recognises.
    """
    # Under the stage root rather than the node's TEMP: ``$env:TEMP`` was
    # tried first and is wrong, because the writer's single-quoted literal
    # does not expand it and a node would grow a directory of that name. The
    # writer creates the parent, so nothing needs to exist before the very
    # first probe.
    spoken = dialect.for_platform(node["platform"])
    outcome = remote.attempt_script(
        node["host"],
        spoken.script_path(node["stage_root"], names.toolchain_probe_stem(writer)),
        spoken.toolchain_probe_script(),
        platform=node["platform"],
    )
    failure = outcome["failure"]
    if failure is not None:
        return failure
    reports = read_reports(outcome["output"])
    if not reports:
        return remote.RemoteFailure(
            code=FleetErrorCode.NODE_TOOL_MISSING,
            message=f"{node['host']}: {_unrecognised(outcome['output'])}",
        )
    return reports


def probe_toolchain(node: NodeConfig, *, writer: str) -> tuple[ToolReport, ...]:
    """Ask a node what it has installed.

    The raising boundary over :func:`attempt_toolchain`.

    Args:
        node: The node to probe.
        writer: Who is asking, as :func:`attempt_toolchain` takes it.

    Returns:
        One report per required tool.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if ssh cannot reach it,
            ``DISPATCH_FAILED`` if the probe exits non-zero, or
            ``NODE_TOOL_MISSING`` if its answer cannot be read.
    """
    outcome = attempt_toolchain(node, writer=writer)
    if isinstance(outcome, tuple):
        return outcome
    raise AppError(outcome["code"], outcome["message"])


def readiness_gap(
    node_name: str, node: NodeConfig, reports: tuple[ToolReport, ...]
) -> AppError[FleetErrorCode] | None:
    """What stands between a node and a build, as a value.

    Args:
        node_name: The node's workspace name.
        node: Its declaration, for the host in the message.
        reports: What it answered.

    Returns:
        ``None`` when the node can build. Otherwise the refusal, not
        raised: ``NODE_TOOL_MISSING`` naming every absent tool, why a build
        needs it and what would install it on THIS node, or
        ``NODE_PYTHON_MISMATCH`` when everything is present but the
        interpreter is the wrong minor version, or ``NODE_NODEJS_MISMATCH``
        when Node.js is older than :data:`REQUIRED_NODE_MAJOR`. Separate
        codes because the fixes differ: one is a package manager, the others
        a decision about which runtime that machine should carry. A
        toolchain that differs from the node's declaration closes no gate:
        the runner claims with what the probe found
        (:mod:`fleet.contracts.detection`).
    """
    absent = missing(reports)
    if absent:
        managers = available_managers(reports)
        wanted = {tool["name"]: tool for tool in REQUIRED_TOOLS}
        detail = "; ".join(
            f"{name} -- {wanted[name]['reason']} -- "
            f"{install_command(name, managers) or 'no automatic install on this node'}"
            for name in absent
            if name in wanted
        )
        return AppError(
            FleetErrorCode.NODE_TOOL_MISSING,
            f"{node_name} ({node['host']}) cannot run a build: {detail}",
        )
    if not python_is_right(reports):
        return AppError(
            FleetErrorCode.NODE_PYTHON_MISMATCH,
            f"{node_name} ({node['host']}) reports Python "
            f"{reported_version(reports, 'python')!r} where {REQUIRED_PYTHON} is required; "
            "every project resolves its lockfile against that minor version, so a build here "
            "would fail resolving rather than testing",
        )
    if not node_is_right(reports):
        return AppError(
            FleetErrorCode.NODE_NODEJS_MISMATCH,
            f"{node_name} ({node['host']}) reports Node.js "
            f"{reported_version(reports, 'node')!r} where {REQUIRED_NODE_MAJOR} or newer is "
            "required; the TypeScript projects declare that engine, and native modules they "
            "install fail to build under an older one",
        )
    return None


def ready_summary(reports: tuple[ToolReport, ...]) -> str:
    """Say what a ready node's toolchain was judged on.

    A gate that passes says how much it examined: "ready" alone reads the
    same for a node that reported six tools and for one whose probe named
    one, and only the first is a pass.

    Only python's and node's versions are printed because only theirs are
    judged; the trailing-token rule that reads them would call bsdtar's
    banner ``libb2/bundled``, so the other tools are named, not versioned.

    Args:
        reports: What a node answered, already judged ready.

    Returns:
        ``python <number>; node <number>; <tool>, <tool> present``, the other
        required tools in the contract's order, e.g. ``python 3.11.9; node
        v24.20.0; poetry, git, make, tar present``, followed by ``; cargo
        <answer>`` and ``; cxx <answer>`` for each declared toolchain the
        probe found, so a node that has one and declares none is visible on
        every tick, and ``; ffmpeg present`` or ``; ffmpeg absent`` for each
        tagged tool, so the tag a runner claims with is visible beside it.
    """
    judged = {"python", "node"}
    present = [
        tool["name"]
        for tool in REQUIRED_TOOLS
        if tool["name"] not in judged
        and any(report["name"] == tool["name"] and report["present"] for report in reports)
    ]
    found = "".join(
        f"; {PROBE_NAME[capability]} {version}"
        for capability in Capability
        if (version := measured(capability, reports)) is not None
    )
    lacking = absent_tagged(reports)
    tagged = "".join(
        f"; {tool['name']} {'absent' if tool['name'] in lacking else 'present'}"
        for tool in TAGGED_TOOLS
    )
    return (
        f"python {version_number(reported_version(reports, 'python'))}; "
        f"node {version_number(reported_version(reports, 'node'))}; "
        f"{', '.join(present)} present{found}{tagged}"
    )


def absent_tagged(reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Name the tagged tools a node's probe did not find.

    Args:
        reports: What the node answered.

    Returns:
        Each :data:`fleet.contracts.tagged_tools.TAGGED_TOOLS` name the probe
        reported absent or did not report at all, in the contract's order.
    """
    found = {report["name"] for report in reports if report["present"]}
    return tuple(tool["name"] for tool in TAGGED_TOOLS if tool["name"] not in found)


def tagged_gap(node_name: str, reports: tuple[ToolReport, ...]) -> str | None:
    """Say which tagged tools a ready node lacks, and what that costs it.

    Not a refusal: the node's runners claim without those tags, so the
    queue offers the jobs that need them to another node and this one takes
    every other job (MCPs board task 939ec5c7).

    Args:
        node_name: The node's workspace name.
        reports: What it answered.

    Returns:
        None when the probe found every tagged tool; otherwise one line per
        tool joined by ``; ``, naming the tool, why a project needs it and
        the command that would install it on THIS node.
    """
    absent = absent_tagged(reports)
    if not absent:
        return None
    managers = available_managers(reports)
    reasons = {tool["name"]: tool["reason"] for tool in TAGGED_TOOLS}
    return f"{node_name} claims without the tag of every tool it lacks: " + "; ".join(
        f"{name} -- {reasons[name]}, so those jobs go to a node that has it -- "
        f"{install_command(name, managers) or 'no automatic install on this node'}"
        for name in absent
    )


def require_ready(node_name: str, node: NodeConfig, reports: tuple[ToolReport, ...]) -> None:
    """Refuse a node that cannot run a build.

    The raising boundary over :func:`readiness_gap`.

    Args:
        node_name: The node's workspace name.
        node: Its declaration, for the host in the message.
        reports: What it answered.

    Raises:
        AppError: The refusal :func:`readiness_gap` names, with its code.
    """
    gap = readiness_gap(node_name, node, reports)
    if gap is not None:
        raise gap


__all__ = [
    "absent_tagged",
    "attempt_toolchain",
    "parse_probe",
    "probe_toolchain",
    "read_reports",
    "readiness_gap",
    "ready_summary",
    "require_ready",
    "tagged_gap",
]
