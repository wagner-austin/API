"""CLI: submit a rendered image build, and record it.

Usage:
    hpc3-image-build --config runs/hpc3.json --project abl \\
        --name image-v22 --image-dir /pub/wagnera3/images/v22 \\
        --image-name abl.sif

The last step of :doc:`adopting an image <../README>`, and until 2026-08-28 it
was not a step of this package at all -- the README told you to run
``ssh hpc3 'cd <dir> && sbatch build.sbatch'`` yourself. That works, and it
leaves nothing behind: no ledger row, so ``hpc3-trace`` cannot say which job
built an image, ``hpc3-watch`` was never given the id, and ``hpc3-triage``
reports the build as ``unclaimed`` for the two hours it runs, correctly,
because from this machine's records it is a stranger holding eight cores.

Takes ``--project`` and ``--name`` rather than reading a name out of the
script, because the ledger's name is the QUALIFIED ``<project>.<name>`` and
the project half is the part a script cannot tell you: ``img.abl-sif-v22``
names a project no workspace declares. The rendered script's own directive is
then required to match, so the row and ``squeue`` cannot disagree.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args

from hpc3.cli import _config, _fatal, _test_hooks
from hpc3.clusters import require_cluster
from hpc3.contracts.layout import qualified_name
from hpc3.core.image_submit import submit_build

_PROJECT_FLAG = "--project"
_NAME_FLAG = "--name"
_IMAGE_DIR_FLAG = "--image-dir"
_IMAGE_NAME_FLAG = "--image-name"
_FLAGS = (
    _config.CONFIG_FLAG,
    _PROJECT_FLAG,
    _NAME_FLAG,
    _IMAGE_DIR_FLAG,
    _IMAGE_NAME_FLAG,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Submit one image build and record it in the ledger.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when the build was submitted and Slurm returned an id.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the workspace is malformed.
        AppError: If the project is not declared, the build script cannot be
            read or names a different job, Slurm would refuse it, or a remote
            command fails. Nothing is caught: an unsubmitted build must not
            report an exit code of zero.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    connection = _config.load_workspace_connection(parsed)
    cluster = require_cluster(connection["cluster"])

    # NO REGISTRY READ AT ALL, and the check that used to be here was
    # redundant rather than load-bearing. It looked the project up to stop a
    # ledger row naming one no workspace declares -- but `submit_build` calls
    # `check_name_agrees` FIRST, before preflight, against the rendered
    # script's own job name, so a mistyped project makes `label` disagree with
    # build.sbatch and the build is refused there. That check is strictly
    # stronger: it catches the typo AND a renderer and submitter that have
    # drifted apart, and it works while a project is being onboarded, which a
    # registry lookup cannot -- registration needs the digest this build is
    # about to produce.
    project = cli_args.require_flag(parsed, _PROJECT_FLAG)

    label = qualified_name(project, cli_args.require_flag(parsed, _NAME_FLAG))
    image_dir = cli_args.require_flag(parsed, _IMAGE_DIR_FLAG)
    image_name = cli_args.require_flag(parsed, _IMAGE_NAME_FLAG)
    host = connection["host"]

    job_id = submit_build(
        host=host,
        image_dir=image_dir,
        project=project,
        label=label,
        artifact=f"{image_dir}/{image_name}",
        ledger_path=pathlib.Path(connection["ledger"]),
        submitted_at=_test_hooks.now_iso(),
        cluster=cluster,
    )

    _test_hooks.emit(f"submitted {job_id} {label}")
    _test_hooks.emit(f"  building {image_dir}/{image_name}")
    _test_hooks.emit(f"  logs {image_dir}/build-{job_id}.out")
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
