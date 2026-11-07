"""One run of the mutation step for the cases that fake its commands (MCPs board task e52e5148).

The cases bind the ``world`` fixture, whose ``run_answering`` answers each
command through ``world.answer``; ``test_mutation_real.py`` runs the same
calls against real git, coverage, pytest and cosmic-ray.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from maketools.bounded_spans import SPANS_ENVIRONMENT_KEY
from maketools.mutation.context import MutationRun

#: The suite's environment the step is handed.
ENVIRONMENT: Final[dict[str, str]] = {
    "PATH": "/bin",
    "COVERAGE_FILE": "/pkg/runs/.coverage-deadbeef",
    SPANS_ENVIRONMENT_KEY: "/pkg/runs/.check-spans-cafe",
}

#: What every test run gets instead: no span file, no bytecode.
RUN_ENVIRONMENT: Final[dict[str, str]] = {
    "PATH": "/bin",
    "COVERAGE_FILE": "/pkg/runs/.coverage-deadbeef",
    "PYTHONDONTWRITEBYTECODE": "1",
}

#: The package's interpreter and the cosmic-ray beside it.
PYTHON: Final[str] = "/pkg/.venv/bin/python"
COSMIC_RAY: Final[str] = "/pkg/.venv/bin/cosmic-ray"

#: The package and the scratch directory a case that touches no file uses.
PACKAGE: Final[Path] = Path("/pkg")
RUN_DIR: Final[Path] = Path("/scratch/run")


def mutation_run(package_dir: Path = PACKAGE, run_dir: Path = RUN_DIR) -> MutationRun:
    """The run the cases hand the step.

    Args:
        package_dir: The package; a case reading real files passes its own.
        run_dir: The scratch directory; a case writing files passes its own.

    Returns:
        The run.
    """
    return MutationRun(
        package_dir=package_dir, python=PYTHON, environment=ENVIRONMENT, run_dir=run_dir
    )
