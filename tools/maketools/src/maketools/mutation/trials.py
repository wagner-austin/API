"""Running a mutant's tests: unmutated first, then under the mutant (MCPs task e52e5148).

Every run is the package's own pytest with its own configuration, given
four things a mutant needs, as MCPs' step gives them: ``-x``, since one
failure kills it; no cache provider, so no run writes ``.pytest_cache``; a
base temporary directory in the step's scratch directory; and ``-W
error::SyntaxWarning``, so a mutant the interpreter refuses (``x is ""``,
which CPython warns of and which no test could tell from ``x == ""``) fails
to import instead of surviving. A run of named tests adds ``-n 0``: a few
tests in one process start faster than xdist's workers.

A mutant's run is bounded by coreutils ``timeout`` inside the command
cosmic-ray runs, since cosmic-ray splits that command with ``shlex`` and
runs it as given; a run past it counts as killed, as a hung mutant is.
"""

from __future__ import annotations

import math
import shlex
from typing import Final

from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks
from maketools.bounded_spans import SPANS_ENVIRONMENT_KEY
from maketools.mutation.context import MutationRun, run_command, run_or_refuse, tail
from maketools.mutation.cosmic import (
    RunOutcome,
    Site,
    WorkerOutcome,
    command,
    decode_result,
    mutated_line,
)

#: Seconds ``timeout`` waits after its signal before it kills the run.
KILL_AFTER_SECONDS: Final[int] = 10


def survived(worker: WorkerOutcome, tests: RunOutcome | None) -> bool | None:
    """Whether a mutant survived, from how cosmic-ray's worker ended.

    Args:
        worker: How the worker ended.
        tests: What the tests did to the mutant.

    Returns:
        True when the tests ran and passed, False when they failed, and
        None for any other ending, which judges nothing.
    """
    endings: dict[tuple[WorkerOutcome, RunOutcome | None], bool] = {
        (WorkerOutcome.NORMAL, RunOutcome.KILLED): False,
        (WorkerOutcome.NORMAL, RunOutcome.SURVIVED): True,
    }
    return endings.get((worker, tests))


def run_environment(run: MutationRun) -> dict[str, str]:
    """The environment every test run gets: the suite's, less the span file, writing no bytecode.

    NO RUN WRITES BYTECODE, the unmutated ones included. A ``.pyc`` is
    trusted while its source's mtime, to the second, and size are unchanged,
    and a mutant often keeps the size (``1`` to ``2``, ``-`` to ``%``): MCPs
    fleet job ad971633 reported a mutant surviving a test that kills it,
    because a baseline had written the restored source's bytecode and the
    next same-size mutant, written within the same second, ran that cached
    original instead.

    Args:
        run: The step's run.

    Returns:
        A copy of the suite's environment without
        :data:`maketools.bounded_spans.SPANS_ENVIRONMENT_KEY` and with
        ``PYTHONDONTWRITEBYTECODE`` set.
    """
    environment = {**run.environment, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop(SPANS_ENVIRONMENT_KEY, None)
    return environment


def pytest_argv(run: MutationRun, tests: tuple[str, ...]) -> list[str]:
    """The command that runs one set of tests.

    Args:
        run: The step's run.
        tests: The tests' node ids, or empty for the whole suite.

    Returns:
        The package's pytest, through its own interpreter.
    """
    argv = [run.python, "-W", "error::SyntaxWarning", "-m", "pytest", "-x", "-q"]
    argv += ["-p", "no:cacheprovider", f"--basetemp={run.run_dir / 'pytest'}"]
    if tests:
        argv += ["-n", "0", *tests]
    return argv


def mutant_timeout(baseline_seconds: float) -> int:
    """How long a mutant's run may take before it counts as killed.

    Three times the unmutated run, rounded up, plus 30 seconds for a node
    busier than it was then: MCPs' bound, so one mutant reads alike in both.

    Args:
        baseline_seconds: The unmutated run's wall time.

    Returns:
        Whole seconds.
    """
    return math.ceil(baseline_seconds * 3) + 30


def run_baseline(run: MutationRun, tests: tuple[str, ...]) -> int:
    """Run a set of tests unmutated, which must pass, and time it.

    Args:
        run: The step's run.
        tests: The tests' node ids, or empty for the whole suite.

    Returns:
        The timeout of a mutant these tests run.

    Raises:
        AppError: ``MUTATION_BASELINE_FAILED`` when they fail unmutated,
            since every mutant would then read as killed.
    """
    started = _test_hooks.now()
    result = run_command(run, pytest_argv(run, tests), run_environment(run))
    if result["returncode"]:
        named = " ".join(tests) or "the whole suite"
        raise AppError(
            MutationErrorCode.BASELINE_FAILED, f"{named} failed unmutated, {tail(result)}"
        )
    return mutant_timeout(_test_hooks.now() - started)


def run_mutant(run: MutationRun, site: Site, tests: tuple[str, ...], timeout: int) -> str | None:
    """Apply one mutant, run its tests under a timeout, and restore the file.

    Args:
        run: The step's run.
        site: The mutant.
        tests: Its tests' node ids, or empty for the whole suite.
        timeout: Seconds before its run is killed.

    Returns:
        The line the mutant put in place when it survived, else None.

    Raises:
        AppError: ``MUTATION_COSMIC_RAY_FAILED`` when cosmic-ray fails, its
            worker does not end normally, or the tests could not be started.
    """
    bounded = ["timeout", f"--kill-after={KILL_AFTER_SECONDS}", str(timeout)]
    tests_command = shlex.join([*bounded, *pytest_argv(run, tests)])
    argv = [command(run.python), "mutate-and-test", site.path, site.operator]
    argv += [str(site.occurrence), tests_command]
    failed = MutationErrorCode.COSMIC_RAY_FAILED
    result = decode_result(run_or_refuse(run, argv, run_environment(run), failed))
    alive = survived(result.worker, result.tests)
    if alive is None:
        raise AppError(
            failed,
            f"{site.path}:{site.line} {site.operator} ended {result.worker}, "
            f"tests {result.tests}: {result.output}",
        )
    return mutated_line(result.diff) if alive else None


__all__ = [
    "KILL_AFTER_SECONDS",
    "mutant_timeout",
    "pytest_argv",
    "run_baseline",
    "run_environment",
    "run_mutant",
    "survived",
]
