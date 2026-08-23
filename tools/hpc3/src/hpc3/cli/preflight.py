"""CLI: ask the scheduler whether a run would start, without starting it.

Usage:
    hpc3-preflight --config hpc3.json --run runs/arm-b-42.json
    hpc3-preflight --config hpc3.json --sweep runs/scale-rung.json

This is the step between "the tests pass" and "it is running on the cluster".
The unit suite proves this package builds the script it meant to; only the
scheduler can say whether the account, QOS, partition and resources will be
admitted, because that state lives on the cluster and nowhere else.

Nothing is queued and no allocation is made. Exactly one of ``--run`` and
``--sweep`` is required: defaulting to either would silently preflight
something other than what the caller named.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.json_utils import load_json_str

from hpc3.cli import _argv, _config, _test_hooks
from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import log_dir, qualified_name, script_dir
from hpc3.contracts.run import resolve_run, resolve_sweep
from hpc3.contracts.sweep import expand_sweep
from hpc3.contracts.workspace import Workspace, workspace_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.preflight import preflight

_FLAGS = (_config.CONFIG_FLAG, "--run", "--sweep")


def _load_specs(workspace: Workspace, parsed: dict[str, str]) -> list[JobSpec]:
    """Resolve either a single run or a sweep, whichever was named.

    Args:
        workspace: The decoded workspace supplying the defaults.
        parsed: Flags already read from the command line.

    Returns:
        Every spec to preflight, in declaration order.

    Raises:
        ValueError: If neither ``--run`` nor ``--sweep`` was given, or both
            were. Preflighting the wrong document is worse than refusing.
        JSONTypeError: If the document is not valid.
        AppError: If the document names an undeclared project or an unknown
            field, breaks a submission rule, or a sweep exceeds the QOS --
            caught here, before the cluster is contacted at all.
    """
    has_run = "--run" in parsed
    has_sweep = "--sweep" in parsed
    if has_run == has_sweep:
        raise ValueError("exactly one of --run or --sweep is required")

    flag = "--run" if has_run else "--sweep"
    raw = core_hooks.read_bytes(pathlib.Path(parsed[flag])).decode("utf-8")
    document = load_json_str(raw)
    if has_run:
        return [resolve_run(workspace, document)]
    return expand_sweep(resolve_sweep(workspace, document))


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight one run or every member of a sweep.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when the scheduler would admit every spec.

    Raises:
        ValueError: If a required flag is missing, an argument is unknown, or
            neither/both of ``--run`` and ``--sweep`` were given.
        AppError: On the first spec the scheduler refuses, the first missing
            environment, or an unreadable verdict. Nothing is caught: a job
            that would not run must not preflight clean.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = _argv.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    root = workspace["root"]

    specs = _load_specs(workspace, parsed)
    for spec in specs:
        project = spec["project"]
        result = preflight(
            spec,
            host=workspace["host"],
            script_dir=script_dir(root, project),
            log_dir=log_dir(root, project),
            cluster=cluster,
        )
        _test_hooks.emit(
            f"OK {qualified_name(project, spec['name'])}: "
            f"would start {result['start_estimate']} "
            f"on {result['node_list']} ({result['processors']} cpu, "
            f"{result['partition']})"
        )
    _test_hooks.emit(f"{len(specs)} spec(s) would be admitted; nothing was queued")
    _test_hooks.emit("start estimates are a snapshot of the queue, not a reservation")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main(None))


__all__ = ["entrypoint", "main"]
