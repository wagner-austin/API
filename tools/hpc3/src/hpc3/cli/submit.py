"""CLI: resolve a run against the workspace and submit it.

Usage:
    hpc3-submit --config hpc3.json --run runs/arm-b-42.json

The run document names the project, the job and the command; everything else
comes from the workspace's entry for that project. The merged result is
decoded before anything is sent, so a job that bills without consent, names no
GPU model, or leaves a long preemptible run unprotected fails locally in
milliseconds rather than on the cluster in hours.

The host, the root, the ledger and the budget are not flags. The first three
come from the workspace and the budget from the project's own entry in it, so
this command and ``hpc3-triage`` cannot be pointed at different ledgers --
which is how a submitted job stops being watched by the thing that was
supposed to watch it -- and ``hpc3-watch`` enforces the same cap this
projected against, because both reach it through the job's project.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import load_json_str

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.cluster import ClusterFacts, describe_gpu_request, partition_bills
from hpc3.contracts.layout import log_dir, qualified_name, script_dir
from hpc3.contracts.run import resolve_run
from hpc3.contracts.workspace import require_project_config, workspace_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.budget import check_projection
from hpc3.core.submit import submit

_FLAGS = (_config.CONFIG_FLAG, "--run")


def _cost_label(cluster: ClusterFacts, partition: str, charge_account: str) -> str:
    """Describe what a submission to this partition costs.

    Args:
        cluster: The cluster whose measured usage factors apply.
        partition: The partition being submitted to.
        charge_account: The account that would be billed.

    Returns:
        ``"free"`` on a zero-usage-factor partition, otherwise the account the
        charge lands on. The summary line used to say "free" unconditionally,
        which was true of everything this package could submit until a
        declared budget admitted billed partitions.
    """
    if not partition_bills(cluster, partition):
        return "free"
    return f"BILLED to {charge_account}"


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve and submit one job.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when the job was submitted and Slurm returned an id.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the workspace or the run document is malformed.
        AppError: If the run names an undeclared project or an unknown field,
            the resolved spec violates a submission rule, the projection
            exceeds the budget, or a remote command fails. Nothing is caught:
            an unsubmitted job must not report an exit code of zero.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    run_path = pathlib.Path(cli_args.require_flag(parsed, "--run"))

    raw = core_hooks.read_bytes(run_path).decode("utf-8")
    spec = resolve_run(workspace, load_json_str(raw))

    # The cap is the project's, not this invocation's. A per-command budget is
    # a budget that is whatever the last person typed; a per-project one is a
    # declaration, and it is the same one `hpc3-watch` will enforce against
    # this job later because both find it the same way.
    budget = require_project_config(workspace, spec["project"])["budget"]
    check_projection(budget, [spec], cluster)

    # Derived, never passed in: a caller who can choose a log directory is a
    # caller who will eventually choose the wrong one, and that job's output is
    # then findable only by whoever remembers what was typed.
    project = spec["project"]
    host = workspace["host"]
    root = workspace["root"]
    job_id = submit(
        spec,
        host=host,
        script_dir=script_dir(root, project),
        log_dir=log_dir(root, project),
        ledger_path=pathlib.Path(workspace["ledger"]),
        submitted_at=_test_hooks.now_iso(),
        cluster=cluster,
        charge_account=budget["charge_account"],
    )

    _test_hooks.emit(f"submitted {job_id} {qualified_name(project, spec['name'])}")
    _test_hooks.emit(
        # Was the literal "(free)". That was true of everything this package
        # could submit until a declared budget admitted billed partitions, and
        # a summary that calls a charged job free is the last line an operator
        # reads before it starts costing.
        f"  {describe_gpu_request(spec['gpu'])} on {spec['partition']} "
        f"({_cost_label(cluster, spec['partition'], budget['charge_account'])}), "
        f"{spec['cpus']} cpu, {spec['mem_gb']}G, {spec['minutes']} min"
    )
    _test_hooks.emit(f"  logs {log_dir(root, project)}")
    _test_hooks.emit(f"watch: hpc3-watch --config {parsed[_config.CONFIG_FLAG]} --job {job_id}")
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
