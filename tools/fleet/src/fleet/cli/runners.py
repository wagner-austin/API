"""CLI: score the CI hosts against the runner roster, or render a new one.

Usage:
    fleet-runners --spec runners.json
    fleet-runners --spec runners.json --host lavender
    fleet-runners --spec runners.json --host lavender --render C:/provision

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

THE PATH IS PASSED, NEVER SEARCHED FOR -- same rule as every document this
package reads. A command that hunted for the roster would report a healthy
fleet on any machine where it simply failed to find it.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, load_json_str
from platform_core.logging import get_logger, setup_logging

from fleet.contracts.runners import HostRunnerSpec, RunnerSpec, decode_runner_spec
from fleet.core import _test_hooks, runner_audit, runner_render

_log = get_logger(__name__)

SPEC_FLAG = "--spec"

HOST_FLAG = "--host"

RENDER_FLAG = "--render"

_FLAGS = (SPEC_FLAG, HOST_FLAG, RENDER_FLAG)


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
    raw = _test_hooks.read_text(pathlib.Path(path))
    try:
        return decode_runner_spec(load_json_str(raw))
    except JSONTypeError as fault:
        raise AppError(
            FleetErrorCode.RUNNER_SPEC_UNREADABLE,
            f"the runner roster at {path} cannot be read: {fault}",
        ) from fault


def select_hosts(spec: RunnerSpec, host_name: str | None) -> list[HostRunnerSpec]:
    """The hosts one invocation covers.

    Args:
        spec: The roster.
        host_name: What ``--host`` was given, or None for every host.

    Returns:
        The matching hosts, in roster order.

    Raises:
        AppError: ``RUNNER_HOST_UNKNOWN`` when a named host is not in the
            roster. Auditing nothing while exiting 0 would read exactly like
            a healthy host.
    """
    if host_name is None:
        return list(spec["hosts"])
    matches = [host for host in spec["hosts"] if host["name"] == host_name]
    if not matches:
        known = ", ".join(host["name"] for host in spec["hosts"])
        raise AppError(
            FleetErrorCode.RUNNER_HOST_UNKNOWN,
            f"the roster declares no host named {host_name!r}; it declares: {known}",
        )
    return matches


def _audit(hosts: Sequence[HostRunnerSpec]) -> int:
    """Audit every selected host and print the verdicts.

    Args:
        hosts: The hosts to audit.

    Returns:
        0 when every check on every reachable host passed and every host was
        reachable; 1 otherwise.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` from the scorer -- a transcript
            this tooling cannot score is a fault in the tooling, not a fleet
            condition, and must stop the audit rather than print as a line.
    """
    faults = 0
    for host in hosts:
        outcome = runner_audit.attempt_audit_host(host)
        findings = outcome["findings"]
        if findings is None:
            _log.info("%s UNREACHABLE %s", host["name"], outcome["reason"])
            faults += 1
            continue
        for finding in findings:
            if finding["ok"]:
                _log.info("%s OK %s", host["name"], finding["check_id"])
            else:
                _log.info(
                    "%s DRIFT %s -- %s (%s)",
                    host["name"],
                    finding["check_id"],
                    finding["detail"],
                    finding["reason"],
                )
                faults += 1
    if faults:
        _log.info("%d fault(s); the fleet does not match its roster", faults)
        return 1
    return 0


def _render(host: HostRunnerSpec, out_dir: str) -> int:
    """Write the converge scripts for one host and print the run order.

    Args:
        host: The host to render for.
        out_dir: Directory receiving ``provision.ps1`` and ``provision.sh``.

    Returns:
        0 always: rendering is a local act that either completes or raises.
    """
    rendered = runner_render.render_provision(host)
    windows_path = pathlib.Path(out_dir) / "provision.ps1"
    linux_path = pathlib.Path(out_dir) / "provision.sh"
    _test_hooks.write_text(windows_path, rendered["windows_script"])
    _test_hooks.write_text(linux_path, rendered["linux_script"])
    _log.info("wrote %s", windows_path)
    _log.info("wrote %s", linux_path)
    _log.info("RUN ORDER for %s:", host["name"])
    _log.info("  1. provision.ps1 on the Windows host (as the interactive user)")
    _log.info("  2. wsl --shutdown, then re-run the keepalive task, if a memory floor changed")
    _log.info("  3. provision.sh inside the distro as root, with each RUNNER_TOKEN_* exported")
    for step in rendered["manual_steps"]:
        _log.info("  %s", step)
    _log.info("  4. fleet-runners audit --spec <roster> --host %s", host["name"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Audit the roster's hosts, or render one host's provision.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 when the audit found every check passing on every reachable host,
        or the render completed. 1 when any check drifted or any host did
        not answer.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value,
            ``--spec`` is absent, or ``--render`` was given without
            ``--host`` -- a render must name the machine it is for, because
            emitting scripts for every host into one directory would
            overwrite each with the next.
        AppError: ``RUNNER_SPEC_UNREADABLE``, ``RUNNER_HOST_UNKNOWN`` or
            ``RUNNER_AUDIT_UNPARSABLE`` as the helpers describe.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    spec_path = parsed.get(SPEC_FLAG)
    if spec_path is None:
        raise ValueError(
            f"{SPEC_FLAG} is required: the roster's path is passed, never searched for"
        )
    spec = load_runner_spec(spec_path)
    hosts = select_hosts(spec, parsed.get(HOST_FLAG))
    render_dir = parsed.get(RENDER_FLAG)
    if render_dir is not None:
        if parsed.get(HOST_FLAG) is None:
            raise ValueError(
                f"{RENDER_FLAG} requires {HOST_FLAG}: a render is for one machine, and "
                "rendering every host into one directory would overwrite each with the next"
            )
        return _render(hosts[0], render_dir)
    return _audit(hosts)


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="fleet-runners",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "HOST_FLAG",
    "RENDER_FLAG",
    "SPEC_FLAG",
    "entrypoint",
    "load_runner_spec",
    "main",
    "select_hosts",
]


if __name__ == "__main__":
    entrypoint()
