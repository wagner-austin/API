"""CLI: score the CI hosts against the runner roster, register, sample, report.

Usage:
    fleet-runners --spec runners.json
    fleet-runners --spec runners.json --host lavender
    fleet-runners --spec runners.json --host lavender --sample load.jsonl
    fleet-runners --spec runners.json --report load.jsonl --cores 16
    fleet-runners --spec runners.json --host lavender --onboard owner/repo
    fleet-runners --spec runners.json --host lavender --register yes
    fleet-runners --spec runners.json --prune yes

This module parses the flags and refuses what it cannot honour; each mode's
act is :mod:`fleet.cli._runner_acts`.

A runner host is a NixOS machine whose generation declares every runner
unit, script and asset from the roster (MCPs board task 9aba868d), so no
mode here installs anything: ``make render-documents`` renders the roster
for MCPs' ``fleet-mcp/nixos``, and the netboot or a switch lays it.

``--prune yes`` deletes each registration of the roster's repositories that
no roster host declares and GitHub reads offline and idle, the remains of a
machine that was reimaged or retired, and names any online one it leaves
(:mod:`fleet.core.runner_registrations`). The audit holds the same
registrations to the roster, one row per repository.

``--register yes`` is GitHub's half of a freshly installed or switched
runner host: a token per repository minted with gh, each unregistered
runner restarted to register with it, then the audit
(:mod:`fleet.core.runner_register`). A registered runner is skipped, so a
cut-short registration is finished by running it again. The value is a
confirmation, because the act registers runners with GitHub.

``--onboard`` is the operator's 2026-09-09 mandate made runnable: ONE
command adds a repository's CI to the self-hosted fleet's roster, rewritten
in the same act, and prints the steps that lay and register it. See
:mod:`fleet.core.runner_onboard` for every refusal it can make.

The ``fleet-nodes`` of GitHub Actions serving: one line per declared check,
drift printed with the roster's own reason beside it, exit 0 only when every
audited check passed. A host that cannot be reached is a LINE, not a crash --
the audit of three machines must not refuse to describe two because a third
is off -- and the exit status is non-zero so a script cannot read a partial
fleet as a whole one.

``--sample`` appends one load reading per selected host to a JSONL record --
per install, the count of live processes UNDER its Runner.Worker, descendants
and all, which is the number the worker literals in any one repo cannot see
(:mod:`fleet.core.runner_load` carries the spec and the measured incident).
A loop belongs in the shell or a schedule, never here, so one invocation is
one sample. ``--report`` reads that record back and prints the distribution
against ``--cores`` -- and judges nothing.

THE PATH IS PASSED, NEVER SEARCHED FOR -- same rule as every document this
package reads. A command that hunted for the roster would report a healthy
fleet on any machine where it simply failed to find it.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import load_json_str
from platform_core.logging import LogFormat, LogLevel, setup_logging
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli import _runner_acts
from fleet.contracts.machine import decode_machine
from fleet.contracts.runners import HostRunnerSpec, RunnerSpec, parse_runner_spec
from fleet.core import _test_hooks
from fleet.core.workspace_files import MACHINES_DIRECTORY

SPEC_FLAG = "--spec"

HOST_FLAG = "--host"

SAMPLE_FLAG = "--sample"

REPORT_FLAG = "--report"

CORES_FLAG = "--cores"

ONBOARD_FLAG = "--onboard"

REGISTER_FLAG = "--register"

PRUNE_FLAG = "--prune"

_FLAGS = (
    SPEC_FLAG,
    HOST_FLAG,
    SAMPLE_FLAG,
    REPORT_FLAG,
    CORES_FLAG,
    ONBOARD_FLAG,
    REGISTER_FLAG,
    PRUNE_FLAG,
)


def load_reserved_cores(registry: str, host_name: str) -> int:
    """The logical cores a host's roster file reserves for its runners.

    The fleet's roster (``registry/machines/<name>.json``) is the one place
    a machine's split between its owner and dispatch is written; dispatch
    takes ``logical_cores - reserved_cores`` and on a runner host the rest
    is the runners'. The host's boot script holds the CI slice to that many
    CPUs (:mod:`fleet.core.runner_cpuset_render`, MCPs board task 349e25ee),
    and :mod:`fleet.cli.render_documents` reads it for that script.

    Args:
        registry: The registry directory ``--config`` names.
        host_name: The runner host, a machine of that roster.

    Returns:
        Its ``dispatch.budget.reserved_cores``.

    Raises:
        ValueError: When the machine's file declares no dispatch budget, so
            it reserves its runners nothing to hold them to.
        JSONTypeError: When the file does not decode as a machine.
    """
    path = pathlib.Path(registry) / MACHINES_DIRECTORY / f"{host_name}.json"
    machine = decode_machine(host_name, load_json_str(_test_hooks.read_text(path)))
    dispatch = machine["dispatch"]
    if dispatch is None:
        raise ValueError(
            f"{path} declares no dispatch budget, so it reserves its runners no cores; a runner "
            "host's roster file must give dispatch.budget.reserved_cores"
        )
    return dispatch["budget"]["reserved_cores"]


def load_runner_spec(path: str) -> RunnerSpec:
    """Read and validate the roster.

    Args:
        path: Path to ``runners.json``.

    Returns:
        The validated roster.

    Raises:
        AppError: ``RUNNER_SPEC_UNREADABLE`` when the document does not
            decode as a roster. The path itself not existing raises from the
            reader instead -- an audit pointed at nothing has established
            nothing, and must not report it as anything.
    """
    return parse_runner_spec(_test_hooks.read_text(pathlib.Path(path)), source=path)


def select_hosts(spec: RunnerSpec, host_name: str | None) -> list[HostRunnerSpec]:
    """The hosts one invocation covers.

    Args:
        spec: The roster.
        host_name: What ``--host`` was given, or None for every host.

    Returns:
        Every host in roster order, or the one named.

    Raises:
        AppError: ``RUNNER_HOST_UNKNOWN`` when a named host is not in the
            roster. Auditing nothing while exiting 0 would read exactly like
            a healthy host.
    """
    if host_name is None:
        return list(spec["hosts"])
    named = [host for host in spec["hosts"] if host["name"] == host_name]
    if not named:
        known = ", ".join(host["name"] for host in spec["hosts"])
        raise AppError(
            FleetErrorCode.RUNNER_HOST_UNKNOWN,
            f"the roster declares no host named {host_name!r}; it declares: {known}",
        )
    return named


def main(argv: Sequence[str] | None = None) -> int:
    """Audit, sample, report, onboard, register or prune, per the flags.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 when the audit found every check passing on every reachable host
        and every registration declared, the prune left no undeclared
        registration, or the onboard, sample or report completed. 1 when any
        audited check drifted, any audited host did not answer, or the prune
        kept a registration GitHub reads online or busy.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value,
            ``--spec`` is absent, more than one mode flag is given (each
            names a different act and a combined invocation would do one
            silently), ``--onboard`` or ``--register`` was given without
            ``--host``, ``--prune`` with it, ``--register`` or ``--prune``
            with any value but ``yes``, ``--report`` without ``--cores``,
            or ``--cores`` is not a positive integer.
        AppError: ``RUNNER_SPEC_UNREADABLE``, ``RUNNER_HOST_UNKNOWN``,
            ``RUNNER_AUDIT_UNPARSABLE``, ``RUNNER_TOKEN_UNAVAILABLE``,
            ``RUNNER_GITHUB_UNREADABLE``, ``RUNNER_DEREGISTRATION_REFUSED``,
            ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` as the acts describe.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    spec_path = parsed.get(SPEC_FLAG)
    if spec_path is None:
        raise ValueError(
            f"{SPEC_FLAG} is required: the roster's path is passed, never searched for"
        )
    modes = [
        flag
        for flag in (SAMPLE_FLAG, REPORT_FLAG, ONBOARD_FLAG, REGISTER_FLAG, PRUNE_FLAG)
        if flag in parsed
    ]
    if len(modes) > 1:
        raise ValueError(
            f"{' and '.join(modes)} are different acts; give one, or neither for the audit"
        )
    spec = load_runner_spec(spec_path)
    hosts = select_hosts(spec, parsed.get(HOST_FLAG))
    if ONBOARD_FLAG in parsed or REGISTER_FLAG in parsed or PRUNE_FLAG in parsed:
        return _change(parsed, spec, spec_path, hosts)
    return _read(parsed, spec, hosts)


def _change(
    parsed: dict[str, str], spec: RunnerSpec, spec_path: str, hosts: list[HostRunnerSpec]
) -> int:
    """Run the one mode that changes the roster, a host or GitHub: onboard, register or prune.

    Args:
        parsed: The parsed flags, naming exactly one of the three.
        spec: The roster.
        spec_path: Where it lives, which an onboard rewrites.
        hosts: The hosts ``--host`` selected.

    Returns:
        The act's exit status.

    Raises:
        ValueError: As :func:`main` describes for these modes.
    """
    onboard_repo = parsed.get(ONBOARD_FLAG)
    if onboard_repo is not None:
        _require_host(parsed, ONBOARD_FLAG, "onboarding adds a runner to one machine")
        return _runner_acts.onboard(spec, spec_path, hosts[0], onboard_repo)
    if REGISTER_FLAG in parsed:
        _require_confirmation(parsed, REGISTER_FLAG, "it registers runners with GitHub")
        _require_host(parsed, REGISTER_FLAG, "a registration is of one machine's runners")
        return _runner_acts.register(hosts[0])
    _require_confirmation(parsed, PRUNE_FLAG, "it deletes registrations from GitHub")
    if HOST_FLAG in parsed:
        raise ValueError(
            f"{PRUNE_FLAG} takes no {HOST_FLAG}: it holds every repository's registrations "
            "to the whole roster, which no one host's installs describe"
        )
    return _runner_acts.prune(spec)


def _read(parsed: dict[str, str], spec: RunnerSpec, hosts: list[HostRunnerSpec]) -> int:
    """Run the one mode that only reads or writes locally: sample, report or the audit.

    Args:
        parsed: The parsed flags, naming at most one of the first two.
        spec: The roster.
        hosts: The hosts ``--host`` selected.

    Returns:
        The act's exit status.

    Raises:
        ValueError: As :func:`main` describes for these modes.
    """
    sample_path = parsed.get(SAMPLE_FLAG)
    if sample_path is not None:
        return _runner_acts.sample(hosts, sample_path)
    report_path = parsed.get(REPORT_FLAG)
    if report_path is not None:
        return _runner_acts.report(
            report_path, cores=_parse_cores(parsed), host_name=parsed.get(HOST_FLAG)
        )
    return _runner_acts.audit(spec, hosts)


def _require_host(parsed: dict[str, str], flag: str, why: str) -> None:
    """Refuse a per-machine mode invoked without a machine.

    Args:
        parsed: The parsed flags.
        flag: The mode flag needing a host.
        why: The clause naming why one machine must be chosen.

    Raises:
        ValueError: When ``--host`` is absent.
    """
    if parsed.get(HOST_FLAG) is None:
        raise ValueError(f"{flag} requires {HOST_FLAG}: {why}, and defaulting one would be a guess")


def _require_confirmation(parsed: dict[str, str], flag: str, act: str) -> None:
    """Refuse a confirming mode whose value is not the literal ``yes``.

    Args:
        parsed: The parsed flags, ``flag`` among them.
        flag: ``--register`` or ``--prune``.
        act: The clause naming what the mode does that needs confirming.

    Raises:
        ValueError: When the value is anything else: the value is the act's
            confirmation.
    """
    if parsed[flag] != "yes":
        raise ValueError(
            f"{flag} takes the literal value yes, got {parsed[flag]!r}: {act}, and a value is "
            "its confirmation"
        )


def _parse_cores(parsed: dict[str, str]) -> int:
    """The validated ``--cores`` value for a report.

    Args:
        parsed: The parsed flags.

    Returns:
        The core count.

    Raises:
        ValueError: When absent or not a positive integer -- the
            distribution is read against the host's core count, and
            guessing one would judge with a number nobody chose.
    """
    raw_cores = parsed.get(CORES_FLAG)
    if raw_cores is None:
        raise ValueError(
            f"{REPORT_FLAG} requires {CORES_FLAG}: the distribution is read against "
            "the host's core count, and guessing one would judge with a number "
            "nobody chose"
        )
    if not raw_cores.isdigit() or int(raw_cores) <= 0:
        raise ValueError(f"{CORES_FLAG} must be a positive integer, got {raw_cores!r}")
    return int(raw_cores)


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-runners",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = [
    "CORES_FLAG",
    "HOST_FLAG",
    "ONBOARD_FLAG",
    "PRUNE_FLAG",
    "REGISTER_FLAG",
    "REPORT_FLAG",
    "SAMPLE_FLAG",
    "SPEC_FLAG",
    "entrypoint",
    "load_reserved_cores",
    "load_runner_spec",
    "main",
    "select_hosts",
]


if __name__ == "__main__":
    entrypoint()
