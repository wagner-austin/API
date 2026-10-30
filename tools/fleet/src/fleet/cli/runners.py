"""CLI: score the CI hosts against the runner roster, render, sample, report.

Usage:
    fleet-runners --spec runners.json
    fleet-runners --spec runners.json --host lavender
    fleet-runners --spec runners.json --host lavender --render /tmp/provision --config registry
    fleet-runners --spec runners.json --host lavender --sample load.jsonl
    fleet-runners --spec runners.json --report load.jsonl --cores 16
    fleet-runners --spec runners.json --host lavender --onboard owner/repo --config registry
    fleet-runners --spec runners.json --host lavender --rebuild yes --config registry
    fleet-runners --spec runners.json --prune yes

``--config`` names the fleet's registry directory, as it does for every fleet
command, for the three modes that run the provision: the provision holds the
host's runners to the cores its roster file reserves for them
(:func:`load_reserved_cores`, MCPs board task 349e25ee).

This module parses the flags and refuses what it cannot honour; each mode's
act is :mod:`fleet.cli._runner_acts`.

``--prune yes`` deletes each registration of the roster's repositories that
no roster host declares and GitHub reads offline and idle, the remains of a
machine that was reimaged or retired, and names any online one it leaves
(:mod:`fleet.core.runner_registrations`). The audit holds the same
registrations to the roster, one row per repository.

``--rebuild yes`` takes a fleet node a CORVIS stick or a netboot installed
and enrolled to an audited runner host: the base and the reboot it asks
for, fresh tokens, the roster's provision, then the audit
(:mod:`fleet.core.runner_rebuild`). Every stage is idempotent, so a
cut-short rebuild is finished by running it again. The value is a
confirmation, because the act lays a whole machine.

``--onboard`` is the operator's 2026-09-09 mandate made runnable: ONE
command registers a repository's CI onto the self-hosted fleet -- token
minted via gh, the host's provision re-run with the new install in it,
the roster rewritten in the same act, and the host audited as proof. See
:mod:`fleet.core.runner_onboard` for every refusal it can make.

The ``fleet-nodes`` of GitHub Actions serving: one line per declared check,
drift printed with the roster's own reason beside it, exit 0 only when every
audited check passed. A host that cannot be reached is a LINE, not a crash --
the audit of three machines must not refuse to describe two because a third
is off -- and the exit status is non-zero so a script cannot read a partial
fleet as a whole one.

``--render`` writes the converge scripts for one host into a directory and
prints the run order plus every step no script can perform (the licensed game
tree). It touches no network: rendering is a local act, and the audit
afterwards is the proof the scripts were run.

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
from fleet.cli._config import CONFIG_FLAG
from fleet.contracts.machine import decode_machine
from fleet.contracts.runners import HostRunnerSpec, RunnerSpec, parse_runner_spec
from fleet.core import _test_hooks
from fleet.core.workspace_files import MACHINES_DIRECTORY

SPEC_FLAG = "--spec"

HOST_FLAG = "--host"

RENDER_FLAG = "--render"

SAMPLE_FLAG = "--sample"

REPORT_FLAG = "--report"

CORES_FLAG = "--cores"

ONBOARD_FLAG = "--onboard"

REBUILD_FLAG = "--rebuild"

PRUNE_FLAG = "--prune"

_FLAGS = (
    SPEC_FLAG,
    HOST_FLAG,
    RENDER_FLAG,
    SAMPLE_FLAG,
    REPORT_FLAG,
    CORES_FLAG,
    ONBOARD_FLAG,
    REBUILD_FLAG,
    PRUNE_FLAG,
    CONFIG_FLAG,
)


def load_reserved_cores(registry: str, host_name: str) -> int:
    """The logical cores a host's roster file reserves for its runners.

    The fleet's roster (``registry/machines/<name>.json``) is the one place
    a machine's split between its owner and dispatch is written; dispatch
    takes ``logical_cores - reserved_cores`` and on a runner host the rest
    is the runners'. The provision holds the CI slice to that many CPUs
    (:mod:`fleet.core.runner_cpuset_render`, MCPs board task 349e25ee).

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
    """Audit, render, sample, report, onboard, rebuild or prune, per the flags.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 when the audit found every check passing on every reachable host
        and every registration declared, the prune left no undeclared
        registration, or the render, sample or report completed. 1 when any
        audited check drifted, any audited host did not answer, or the prune
        kept a registration GitHub reads online or busy.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value,
            ``--spec`` is absent, more than one mode flag is given (each
            names a different act and a combined invocation would do one
            silently), ``--render`` or ``--rebuild`` was given without
            ``--host``, ``--prune`` with it, ``--rebuild`` or ``--prune``
            with any value but ``yes``, ``--report`` without ``--cores``,
            ``--cores`` is not a positive integer, ``--render``,
            ``--onboard`` or ``--rebuild`` was given without ``--config``,
            or the host's roster file declares no dispatch budget.
        JSONTypeError: When the host's roster file does not decode.
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
        for flag in (RENDER_FLAG, SAMPLE_FLAG, REPORT_FLAG, ONBOARD_FLAG, REBUILD_FLAG, PRUNE_FLAG)
        if flag in parsed
    ]
    if len(modes) > 1:
        raise ValueError(
            f"{' and '.join(modes)} are different acts; give one, or neither for the audit"
        )
    spec = load_runner_spec(spec_path)
    hosts = select_hosts(spec, parsed.get(HOST_FLAG))
    if ONBOARD_FLAG in parsed or REBUILD_FLAG in parsed or PRUNE_FLAG in parsed:
        return _change(parsed, spec, spec_path, hosts)
    return _read(parsed, spec, hosts)


def _change(
    parsed: dict[str, str], spec: RunnerSpec, spec_path: str, hosts: list[HostRunnerSpec]
) -> int:
    """Run the one mode that changes a host or GitHub: onboard, rebuild or prune.

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
        _require_host(parsed, ONBOARD_FLAG, "onboarding registers a runner on one machine")
        reserved = _reserved_cores(parsed, ONBOARD_FLAG, hosts[0])
        return _runner_acts.onboard(spec, spec_path, hosts[0], onboard_repo, reserved)
    if REBUILD_FLAG in parsed:
        _require_confirmation(parsed, REBUILD_FLAG, "it lays a machine's whole runner host")
        _require_host(parsed, REBUILD_FLAG, "a rebuild lays one machine")
        return _runner_acts.rebuild(hosts[0], _reserved_cores(parsed, REBUILD_FLAG, hosts[0]))
    _require_confirmation(parsed, PRUNE_FLAG, "it deletes registrations from GitHub")
    if HOST_FLAG in parsed:
        raise ValueError(
            f"{PRUNE_FLAG} takes no {HOST_FLAG}: it holds every repository's registrations "
            "to the whole roster, which no one host's installs describe"
        )
    return _runner_acts.prune(spec)


def _read(parsed: dict[str, str], spec: RunnerSpec, hosts: list[HostRunnerSpec]) -> int:
    """Run the one mode that only reads or writes locally: render, sample, report or the audit.

    Args:
        parsed: The parsed flags, naming at most one of the first three.
        spec: The roster.
        hosts: The hosts ``--host`` selected.

    Returns:
        The act's exit status.

    Raises:
        ValueError: As :func:`main` describes for these modes.
    """
    render_dir = parsed.get(RENDER_FLAG)
    if render_dir is not None:
        _require_host(parsed, RENDER_FLAG, "a render is for one machine")
        reserved = _reserved_cores(parsed, RENDER_FLAG, hosts[0])
        return _runner_acts.render(hosts[0], render_dir, reserved)
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


def _reserved_cores(parsed: dict[str, str], flag: str, host: HostRunnerSpec) -> int:
    """The cores a mode that runs the provision holds the host's runners to.

    Args:
        parsed: The parsed flags.
        flag: ``--render``, ``--onboard`` or ``--rebuild``.
        host: The host the mode provisions.

    Returns:
        Its roster file's ``dispatch.budget.reserved_cores``
        (:func:`load_reserved_cores`).

    Raises:
        ValueError: When ``--config`` is absent: the provision's CPU set is
            the roster's reservation, and the registry's path is passed,
            never searched for; and as :func:`load_reserved_cores` describes.
    """
    registry = parsed.get(CONFIG_FLAG)
    if registry is None:
        raise ValueError(
            f"{flag} requires {CONFIG_FLAG}, the fleet's registry directory: the provision "
            "holds the host's runners to the cores its roster file reserves for them"
        )
    return load_reserved_cores(registry, host["name"])


def _require_confirmation(parsed: dict[str, str], flag: str, act: str) -> None:
    """Refuse a confirming mode whose value is not the literal ``yes``.

    Args:
        parsed: The parsed flags, ``flag`` among them.
        flag: ``--rebuild`` or ``--prune``.
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
    "REBUILD_FLAG",
    "RENDER_FLAG",
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
