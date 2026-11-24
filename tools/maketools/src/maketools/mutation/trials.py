"""Running a mutant's tests, and the command every test run is (board task fec5851c).

Every run is the package's own pytest with its own configuration, given
three things a mutant needs: ``-x``, since one failure kills it; no cache
provider, so no run writes ``.pytest_cache``; and ``-W error::SyntaxWarning``,
so a mutant the interpreter refuses (``x is ""``, which CPython warns of and
which no test could tell from ``x == ""``) fails to import instead of
surviving. A run of named tests adds ``-n 0``: a few tests in one process
start faster than xdist's workers, and run in the order given. The
unmutated rounds before the first trial (:mod:`.baseline`, board task
3b386c0d) also write the JUnit report each share's cost is read from.

THE BASE TEMPORARY DIRECTORY IS THE PACKAGE'S (board task e6c9604f). The
step once moved it into its scratch directory, outside the checkout; but a
package's configuration may place it inside on purpose, and maketools' does
(``--basetemp=.pytest-tmp``): its tests read the repository a ``tmp_path``
sits in. Its whole-suite baseline then failed unmutated on
``test_a_sync_refuses_duties_no_node_may_hold``, which found no repository
above its ``tmp_path`` (fleet job ae15eceb), after the same suite had passed
5961 cases in the check.

The command is a shell script because a package's ``--prepare`` command
runs before pytest, in the same run; cosmic-ray splits the line it is given
with ``shlex``, so every argument is quoted by :func:`shlex.join`.
"""

from __future__ import annotations

import math
import shlex
from pathlib import Path

from .context import (
    JsonT,
    MutationErrorCode,
    MutationStepError,
    StepContext,
    run_or_refuse,
)
from .cosmic import RunOutcome, Site, WorkerOutcome, command, decode_result, mutated_line

#: The variable naming the check lock's span file. The tests the step runs
#: are not the step, so they run without it, and none can record into it.
SPANS_VARIABLE = "CORVIS_CHECK_SPANS"


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


def run_environment(context: StepContext[JsonT]) -> dict[str, str]:
    """The environment every test run gets: the step's, less the span file, writing no bytecode.

    NO RUN WRITES BYTECODE, the unmutated ones included. A ``.pyc`` is
    trusted while its source's mtime, to the second, and size are unchanged,
    and a mutant often keeps the size (``1`` to ``2``, ``-`` to ``%``).
    cosmic-ray already runs a mutant's tests without writing any; a baseline
    that wrote the restored sources' bytecode let the next same-size mutant,
    written within the same second, run that cached original instead:
    fleet job ad971633 reported ``rsplit("|", 2)`` surviving a test that
    kills it when run by hand.

    Args:
        context: The run.

    Returns:
        A copy of the step's environment without :data:`SPANS_VARIABLE` and
        with ``PYTHONDONTWRITEBYTECODE`` set.
    """
    environment = {**context.environment, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop(SPANS_VARIABLE, None)
    return environment


def pytest_script(context: StepContext[JsonT], tests: tuple[str, ...], report: Path | None) -> str:
    """The shell script that runs one set of tests.

    Args:
        context: The run.
        tests: The test files or node ids, or empty for the whole suite.
        report: Where a baseline round writes its xunit1 JUnit report, or
            None for a run that writes none.

    Returns:
        ``exec`` of pytest, after the package's prepare command when it has
        one.
    """
    pytest = [context.python, "-W", "error::SyntaxWarning", "-m", "pytest", "-x", "-q"]
    pytest += ["-p", "no:cacheprovider"]
    if report is not None:
        pytest += [f"--junitxml={report}", "-o", "junit_family=xunit1"]
        pytest += ["-o", "junit_duration_report=total"]
    if tests:
        pytest += ["-n", "0", *tests]
    script = f"exec {shlex.join(pytest)}"
    if context.prepare:
        return f"{context.prepare} && {script}"
    return script


def mutant_timeout(baseline_seconds: float) -> int:
    """How long a mutant's run may take before it counts as killed.

    Three times what its tests took unmutated, rounded up, plus 30 seconds
    for a node busier than it was then.

    Args:
        baseline_seconds: What the baseline measured for its tests: the
            whole suite's wall time, or its files' summed case times.

    Returns:
        Whole seconds.
    """
    return math.ceil(baseline_seconds * 3) + 30


def run_mutant(
    context: StepContext[JsonT], site: Site, tests: tuple[str, ...], timeout: int
) -> str | None:
    """Apply one mutant, run its tests under a timeout, and restore the file.

    Args:
        context: The run.
        site: The mutant.
        tests: Its tests' node ids, or empty for the whole suite.
        timeout: Seconds before its run is killed.

    Returns:
        The line the mutant put in place when it survived, else None.

    Raises:
        MutationStepError: ``MUTATION_COSMIC_RAY_FAILED`` when cosmic-ray
            fails, its worker does not end normally, or the tests could not
            be started.
    """
    tests_command = shlex.join(
        [
            "timeout",
            "--kill-after=10",
            str(timeout),
            "sh",
            "-c",
            pytest_script(context, tests, None),
        ]
    )
    argv = [command(context.python), "mutate-and-test", site.path, site.operator]
    argv += [str(site.occurrence), tests_command]
    failed = MutationErrorCode.COSMIC_RAY_FAILED
    printed = run_or_refuse(context, argv, run_environment(context), failed)
    result = decode_result(printed, context.codec)
    alive = survived(result.worker, result.tests)
    if alive is None:
        raise MutationStepError(
            failed,
            f"{site.path}:{site.line} {site.operator} ended {result.worker}, "
            f"tests {result.tests}: {result.output}",
        )
    return mutated_line(result.diff) if alive else None


__all__ = [
    "SPANS_VARIABLE",
    "mutant_timeout",
    "pytest_script",
    "run_environment",
    "run_mutant",
    "survived",
]
