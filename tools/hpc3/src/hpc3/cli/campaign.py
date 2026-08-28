"""CLI: submit whatever a set of runs is still missing, and nothing else.

Usage:
    hpc3-campaign --config runs/hpc3.json --run runs/sweep-turkic-bases.json

Takes the SWEEP DOCUMENT THAT ALREADY EXISTS. A campaign is not a new shape to
learn; it is the same seven members, run again, with the difference that it
asks the cluster what is already done before submitting anything.

    $ hpc3-campaign --config runs/hpc3-turkic-lstm.json \\
          --run runs/sweep-turkic-bases.json
    done      turkic-lstm.bases-tr
    done      turkic-lstm.bases-az
    in flight turkic-lstm.bases-ky <- turkic-lstm.bases-r1-ky
    submitted 55646901 turkic-lstm.bases-uz
    3 done, 1 in flight, 1 submitted, 0 remaining

Run it again after a preemption wave and it submits exactly the members that
were preempted. Run it twice in a row and the second run submits nothing.
There is no state kept between runs -- the artifacts that exist and the jobs
that are live are both facts the cluster will tell you, and neither goes stale
the way the four hand-written resume documents of 2026-08-28 did.

WHAT IT WILL NOT DO. It does not cancel, it does not delete, and it does not
overwrite: a member whose artifact exists is left alone, and a member a live
job is writing is left alone. Converging DOWN -- deciding a finished run
should be redone -- is a decision about the experiment, so it is made by
removing the artifact, not by a flag here that would eventually be passed by
someone who meant something else.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import load_json_str

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.contracts.layout import log_dir, script_dir
from hpc3.contracts.run import resolve_sweep
from hpc3.contracts.sweep import expand_sweep
from hpc3.contracts.workspace import require_project_config, workspace_cluster
from hpc3.core import _test_hooks as core_hooks
from hpc3.core import ledger, submit
from hpc3.core.budget import check_projection
from hpc3.core.campaign import (
    existence_command,
    parse_existence,
    plan_campaign,
    require_every_member_declares_an_artifact,
)
from hpc3.core.inflight import claimed_artifacts
from hpc3.core.remote import run_remote
from hpc3.core.squeue import account_command, parse_account_output

_RUN_FLAG = "--run"
_FLAGS = (_config.CONFIG_FLAG, _RUN_FLAG)


def main(argv: Sequence[str] | None = None) -> int:
    """Submit the members of a sweep that are neither done nor running.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0. A campaign with nothing left to do is the SUCCESS case,
        not an error -- it is what convergence looks like, and a command that
        exited non-zero once finished could not be run on a schedule.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the workspace or the sweep document is malformed.
        AppError: If a member declares no artifact, the projection exceeds the
            budget, or a remote command fails. Nothing is caught.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    workspace = _config.load_workspace(parsed)
    cluster = workspace_cluster(workspace)
    run_path = pathlib.Path(cli_args.require_flag(parsed, _RUN_FLAG))

    raw = core_hooks.read_bytes(run_path).decode("utf-8")
    spec = resolve_sweep(workspace, load_json_str(raw))
    specs = expand_sweep(spec)

    # Refused before any query: a campaign that cannot measure its own
    # progress should fail on the document, not after two round trips.
    artifacts = require_every_member_declares_an_artifact(specs)

    host = workspace["host"]
    present = parse_existence(run_remote(host, existence_command(artifacts)))
    claimed = claimed_artifacts(
        ledger.read(pathlib.Path(workspace["ledger"]), cluster),
        parse_account_output(run_remote(host, account_command())),
    )
    plan = plan_campaign(specs, present=present, claimed=claimed)

    for label in plan["done"]:
        _test_hooks.emit(f"done      {label}")
    for label, holder in plan["in_flight"].items():
        _test_hooks.emit(f"in flight {label} <- {holder}")

    # Projected over the GAP, not the whole campaign. Budgeting for members
    # that already finished would refuse the last member of a long experiment
    # for the cost of the ones that paid for themselves weeks ago.
    project = spec["base"]["project"]
    budget = require_project_config(workspace, project)["budget"]
    check_projection(budget, plan["missing"], cluster)

    root = workspace["root"]
    for member in plan["missing"]:
        job_id = submit.submit(
            member,
            host=host,
            script_dir=script_dir(root, project),
            log_dir=log_dir(root, project),
            ledger_path=pathlib.Path(workspace["ledger"]),
            submitted_at=_test_hooks.now_iso(),
            cluster=cluster,
            charge_account=budget["charge_account"],
        )
        _test_hooks.emit(f"submitted {job_id} {project}.{member['name']}")

    _test_hooks.emit(
        f"{len(plan['done'])} done, {len(plan['in_flight'])} in flight, "
        f"{len(plan['missing'])} submitted, "
        f"{len(plan['in_flight']) + len(plan['missing'])} remaining"
    )
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
