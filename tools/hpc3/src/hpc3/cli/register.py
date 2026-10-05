"""CLI: register a built, imaged project, in one move.

Usage:
    hpc3-register --project newcomer \\
        --partition free --gpu none --cpus 2 --mem-gb 4 --minutes 60 \\
        --requeue yes --resumes no --deterministic yes --certified-inputs no \\
        --image /pub/wagnera3/newcomer/images/v1/newcomer.sif --env-path /opt/env \\
        --pins newcomer==0.1.0 --gpu-hours 0 --billing free \\
        --repo ../../clients/Newcomer

The LAST step of onboarding, after ``hpc3-bootstrap``, ``hpc3-image-capture``,
``hpc3-image`` and ``hpc3-image-build`` -- or, for a project whose workload is
another package's CLI, after rebuilding that package's image. Before it, the
project's section must already be written in ``docs/RESEARCH.md``: what it
measures and what its provenance does not cover is the one step nothing can
derive, so the command checks for it and refuses without it.

WHAT IT DOES, in order, cheapest first:

1. Reads every flag. Each is a decision about the project; none defaults, and
   a missing one is refused with every missing flag named.
2. Refuses, naming EVERY unmet precondition at once, if the project is already
   declared, its document exists, its section is not written, or its repo is
   not a directory.
3. Copies the cluster, host, root and ledger from the workspaces beside it,
   refusing if they disagree, and binds the workspace root into the image.
4. READS the image's digest on the cluster -- it takes no digest flag -- and
   proves ``--env-path`` and ``--pins`` inside that image with the probe
   ``hpc3-preflight`` runs. An image that is not built cannot be registered.
5. Writes ``tools/hpc3/runs/hpc3-<project>.json`` -- the filename
   ``test_committed_runs.py`` requires, derived rather than typed -- and
   regenerates the research index's table from the workspace documents.

So the four files the task that built this listed are covered: the document
and the table are written, the section is a refused precondition rather than
a red test met afterwards, and ``test_committed_runs.py`` has carried no
project list to edit since 2026-09-03.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence
from typing import Final

from platform_core import cli_args
from platform_core.cluster_layout import require_project

from hpc3.cli import _fatal, _register_flags, _test_hooks
from hpc3.cli._paths import index_path, runs_directory
from hpc3.clusters import require_cluster
from hpc3.contracts.cluster import require_partition
from hpc3.contracts.project import ProjectConfig
from hpc3.core import env_probe, register
from hpc3.core.registry import shared_connection, workspace_filename

PROJECT_FLAG: Final[str] = "--project"
PARTITION_FLAG: Final[str] = "--partition"
GPU_FLAG: Final[str] = "--gpu"
CPUS_FLAG: Final[str] = "--cpus"
MEM_FLAG: Final[str] = "--mem-gb"
MINUTES_FLAG: Final[str] = "--minutes"
REQUEUE_FLAG: Final[str] = "--requeue"
RESUMES_FLAG: Final[str] = "--resumes"
DETERMINISTIC_FLAG: Final[str] = "--deterministic"
CERTIFIED_FLAG: Final[str] = "--certified-inputs"
IMAGE_FLAG: Final[str] = "--image"
ENV_PATH_FLAG: Final[str] = "--env-path"
PINS_FLAG: Final[str] = "--pins"
GPU_HOURS_FLAG: Final[str] = "--gpu-hours"
BILLING_FLAG: Final[str] = "--billing"
REPO_FLAG: Final[str] = "--repo"

#: Every flag, each required, and the ONE place the set is written.
FLAGS: Final[tuple[str, ...]] = (
    PROJECT_FLAG,
    PARTITION_FLAG,
    GPU_FLAG,
    CPUS_FLAG,
    MEM_FLAG,
    MINUTES_FLAG,
    REQUEUE_FLAG,
    RESUMES_FLAG,
    DETERMINISTIC_FLAG,
    CERTIFIED_FLAG,
    IMAGE_FLAG,
    ENV_PATH_FLAG,
    PINS_FLAG,
    GPU_HOURS_FLAG,
    BILLING_FLAG,
    REPO_FLAG,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Register one project, writing its workspace document and the table.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when the document is written and the table regenerated.

    Raises:
        ValueError: If any flag is missing (all are named), unknown, or
            malformed.
        JSONTypeError: If the project name is not a valid one.
        AppError: With ``REGISTRATION_INCOMPLETE`` naming every unmet
            precondition, ``PARTITION_UNKNOWN`` / ``GPU_TYPE_UNPINNED`` for
            hardware the cluster lacks, ``REMOTE_COMMAND_FAILED`` if the
            image cannot be digested, or ``ENV_*`` if the environment inside
            it is absent, borrowed, or does not hold the pins. Nothing is
            caught: nothing is written until every check has passed.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, FLAGS)
    _register_flags.require_every_flag(parsed, FLAGS)
    project = require_project({"project": parsed[PROJECT_FLAG]}, "project")
    runs = runs_directory()
    index = index_path()
    repo = pathlib.Path(parsed[REPO_FLAG])

    register.require_registrable(runs=runs, index=index, project=project, repo=repo)
    connection = shared_connection(runs)
    cluster = require_cluster(connection["cluster"])
    partition = require_partition(cluster, {"partition": parsed[PARTITION_FLAG]}, "partition")
    gpu = _register_flags.gpu_request(cluster, GPU_FLAG, parsed[GPU_FLAG])
    pins = _register_flags.pinned_packages(PINS_FLAG, parsed[PINS_FLAG])
    caps = _register_flags.budget(
        _register_flags.non_negative_number(GPU_HOURS_FLAG, parsed[GPU_HOURS_FLAG]),
        BILLING_FLAG,
        parsed[BILLING_FLAG],
    )
    cpus = _register_flags.positive_int(CPUS_FLAG, parsed[CPUS_FLAG])
    mem_gb = _register_flags.positive_int(MEM_FLAG, parsed[MEM_FLAG])
    minutes = _register_flags.positive_int(MINUTES_FLAG, parsed[MINUTES_FLAG])
    requeue = _register_flags.yes_or_no(REQUEUE_FLAG, parsed[REQUEUE_FLAG])
    resumes = _register_flags.yes_or_no(RESUMES_FLAG, parsed[RESUMES_FLAG])
    deterministic = _register_flags.yes_or_no(DETERMINISTIC_FLAG, parsed[DETERMINISTIC_FLAG])
    certified = _register_flags.yes_or_no(CERTIFIED_FLAG, parsed[CERTIFIED_FLAG])

    host = connection["host"]
    image = register.read_image(host, parsed[IMAGE_FLAG], binds=[connection["root"]])
    env_path = parsed[ENV_PATH_FLAG]
    env_probe.verify_environment(host, env_path, pins, image=image)

    config = ProjectConfig(
        partition=partition,
        gpu=gpu,
        cpus=cpus,
        mem_gb=mem_gb,
        minutes=minutes,
        requeue=requeue,
        resumes_from_checkpoint=resumes,
        image=image,
        env_path=env_path,
        pinned_packages=pins,
        deterministic=deterministic,
        certified_inputs=certified,
        budget=caps,
        repo=register.relative_repo(runs, repo),
    )
    registry = register.write_registration(
        runs=runs,
        index=index,
        project=project,
        payload=register.workspace_payload(connection, project, config),
    )

    document = runs / workspace_filename(project)
    _test_hooks.emit(f"registered {project} in {document}")
    _test_hooks.emit(f"  image {image['path']} sha256 {image['sha256']}")
    _test_hooks.emit(f"  {env_path} verified inside it, {len(pins)} pin(s) held")
    _test_hooks.emit(f"  table in {index} regenerated from the registry: {len(registry)} projects")
    _test_hooks.emit(f"next: hpc3-preflight --config {document} --run <run document>")
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = ["FLAGS", "entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
