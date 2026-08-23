"""CLI: resolve a pipeline against the workspace and submit it in order.

Usage:
    hpc3-chain --config hpc3.json --run runs/sirius-zodiac.json

Every stage is resolved and validated before the first one is sent. That is
the whole point of doing it here rather than stage by stage: the dependency
ids do not exist until a stage has been queued, so the naive shape discovers a
misspelled partition in stage three an hour after stage one started running.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.json_utils import load_json_str

from hpc3.cli import _argv, _config, _fatal, _test_hooks
from hpc3.contracts.cluster import describe_gpu_request
from hpc3.contracts.layout import log_dir, script_dir
from hpc3.contracts.run import resolve_chain
from hpc3.contracts.workspace import workspace_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.budget import check_projection
from hpc3.core.chain import submit_chain

_FLAGS = (_config.CONFIG_FLAG, "--run")


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve and submit one chain.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when every stage was submitted.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the workspace or the chain document is malformed.
        AppError: If the chain names an undeclared project or an unknown
            field, a stage breaks a submission rule, the pipeline exceeds the
            budget, or a stage could not be submitted. Stages submitted before
            a failure stay queued and are recorded in the ledger.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = _argv.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    run_path = pathlib.Path(_argv.require_flag(parsed, "--run"))

    raw = core_hooks.read_bytes(run_path).decode("utf-8")
    spec = resolve_chain(workspace, load_json_str(raw))
    stages = spec["stages"]

    # The whole pipeline is projected, not just the stage that runs first.
    # Stages are sequential in TIME and simultaneous in COMMITMENT: submitting
    # the chain commits every hour of it, and a budget consulted one stage at
    # a time would approve a pipeline it would have refused whole.
    budget = workspace["budget"]
    projected = check_projection(budget, stages, cluster)
    _test_hooks.emit(
        f"budget OK: projected {projected['gpu_hours']:.1f} GPU-hours, "
        f"{projected['service_units']:.1f} SU "
        f"(caps {budget['max_gpu_hours']:.1f} / {budget['max_service_units']:.1f})"
    )

    project = stages[0]["project"]
    root = workspace["root"]
    submitted = submit_chain(
        spec,
        host=workspace["host"],
        script_dir=script_dir(root, project),
        log_dir=log_dir(root, project),
        ledger_path=pathlib.Path(workspace["ledger"]),
        submitted_at=_test_hooks.now_iso(),
        cluster=cluster,
    )

    for position, (member, stage) in enumerate(zip(submitted, stages, strict=True)):
        waits = "starts when ready" if position == 0 else f"after {submitted[position - 1].job_id}"
        _test_hooks.emit(
            f"submitted {member.job_id} {member.name} "
            f"[{describe_gpu_request(stage['gpu'])}, {stage['minutes']} min] {waits}"
        )
    _test_hooks.emit(f"{len(submitted)} stage(s) on {stages[0]['partition']} (free)")
    # Stated rather than left implied: a stage whose predecessor fails is
    # cancelled, not left pending, and someone reading a chain that stopped
    # halfway needs to know the difference between "blocked" and "gone".
    _test_hooks.emit("a failed stage cancels the ones after it (--kill-on-invalid-dep)")
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
    raise SystemExit(_fatal.run(main))


__all__ = ["entrypoint", "main"]
