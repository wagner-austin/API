"""CLI: resolve a sweep against the workspace and submit every member.

Usage:
    hpc3-sweep --config hpc3.json --run runs/scale-rung.json

The sweep is resolved and decoded before anything is sent, so a set larger
than the partition's per-user QOS fails locally in milliseconds. Submitting it
and watching the excess pend would look like a busy cluster and would not be.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.json_utils import load_json_str

from hpc3.cli import _argv, _config, _test_hooks
from hpc3.contracts.cluster import partition_bills
from hpc3.contracts.layout import log_dir, script_dir
from hpc3.contracts.run import resolve_sweep
from hpc3.contracts.sweep import expand_sweep
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.budget import check_projection
from hpc3.core.sweep import submit_sweep

_FLAGS = (_config.CONFIG_FLAG, "--run")


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve and submit one sweep.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when every member was submitted.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the workspace or the sweep document is malformed.
        AppError: If the sweep names an undeclared project or an unknown
            field, the template breaks a submission rule, the sweep exceeds
            the QOS or the budget, or a member could not be submitted.
            Members submitted before a failure stay running and are recorded
            in the ledger rather than rolled back.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = _argv.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    run_path = pathlib.Path(_argv.require_flag(parsed, "--run"))

    raw = core_hooks.read_bytes(run_path).decode("utf-8")
    spec = resolve_sweep(workspace, load_json_str(raw))

    # Checked before submission, because a flood that has started is no longer
    # a budget question. The QOS bounds what runs at once; nothing but this
    # bounds the total, and on the free partitions nothing bills either.
    budget = workspace["budget"]
    projected = check_projection(budget, expand_sweep(spec), cluster)
    _test_hooks.emit(
        f"budget OK: projected {projected['gpu_hours']:.1f} GPU-hours, "
        f"{projected['service_units']:.1f} SU "
        f"(caps {budget['max_gpu_hours']:.1f} / {budget['max_service_units']:.1f})"
    )

    # Derived from the project, so every member of every sweep lands in the
    # same place and two projects cannot scatter into each other.
    base = spec["base"]
    project = base["project"]
    root = workspace["root"]
    submitted = submit_sweep(
        spec,
        host=workspace["host"],
        script_dir=script_dir(root, project),
        log_dir=log_dir(root, project),
        ledger_path=pathlib.Path(workspace["ledger"]),
        submitted_at=_test_hooks.now_iso(),
        cluster=cluster,
    )

    cost = "BILLS service units" if partition_bills(cluster, base["partition"]) else "free"
    for member in submitted:
        _test_hooks.emit(f"submitted {member.job_id} {member.name}")
    _test_hooks.emit(
        f"{len(submitted)} member(s) on {base['partition']} ({cost}), "
        f"{base['gpu']}x{base['gpu_count']} each, {base['minutes']} min"
    )
    _test_hooks.emit(f"logs {log_dir(root, project)}")
    _test_hooks.emit(
        f"watch: hpc3-watch --config {parsed[_config.CONFIG_FLAG]} --job "
        + ",".join(member.job_id for member in submitted)
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main(None))


__all__ = ["entrypoint", "main"]
