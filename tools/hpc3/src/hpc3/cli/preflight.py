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

from platform_core import cli_args
from platform_core.json_utils import load_json_str

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import log_dir, qualified_name, script_dir
from hpc3.contracts.run import resolve_run, resolve_sweep
from hpc3.contracts.sweep import expand_sweep
from hpc3.contracts.workspace import Workspace, require_project_config, workspace_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.budget import check_projection
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
        AppError: On a projection that exceeds the declared budget, the
            first spec the scheduler refuses, the first missing environment,
            or an unreadable verdict. Nothing is caught: a job that would not
            run, or could not be paid for, must not preflight clean.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    root = workspace["root"]

    specs = _load_specs(workspace, parsed)

    # Preflight answers "would the scheduler admit this". Until 2026-08-26 it
    # did not also answer "and can we afford it", which was survivable only
    # while every admissible partition was free. A workspace may now declare a
    # service-unit budget, so a spec can be schedulable and unaffordable at
    # once -- and preflight is the step whose whole purpose is to find that out
    # before anything is queued.
    # One document resolves to one project, so every spec here shares a cap.
    budget = require_project_config(workspace, specs[0]["project"])["budget"]
    projected = check_projection(budget, specs, cluster)

    for spec in specs:
        project = spec["project"]
        result = preflight(
            spec,
            host=workspace["host"],
            script_dir=script_dir(root, project),
            log_dir=log_dir(root, project),
            cluster=cluster,
            charge_account=budget["charge_account"],
        )
        _test_hooks.emit(
            f"OK {qualified_name(project, spec['name'])}: "
            f"would start {result['start_estimate']} "
            f"on {result['node_list']} ({result['processors']} cpu, "
            f"{result['partition']})"
        )
    _test_hooks.emit(f"{len(specs)} spec(s) would be admitted; nothing was queued")
    # GPU-hours only, and the service-unit figure is deliberately not printed
    # beside it: it is structurally zero before submission (Slurm computes the
    # billing number from TRESBillingWeights and reports it only in
    # accounting), so showing "0.0 service units" for a job about to charge
    # would be the most misleading line on the screen.
    _test_hooks.emit(
        f"projected {projected['gpu_hours']:.2f} GPU-hours against a declared cap of "
        f"{budget['self_imposed_gpu_hours']:.2f}; "
        f"spend is measured after the fact, not projected"
    )
    _test_hooks.emit("start estimates are a snapshot of the queue, not a reservation")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
