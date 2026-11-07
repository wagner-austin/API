"""The mutation step as the test launcher runs it after a passing suite (MCPs task e52e5148).

``run.py test --mutation`` (:func:`maketools.test_run.run_tests`) runs the
suite with per-test coverage contexts, and when it passes hands this module
the package, its interpreter and the suite's environment. The step gets a
scratch directory under the package's ``runs/`` (gitignored, beside the
run's coverage file), removed after it, and its wall time is recorded as
the check's mutation span (:mod:`maketools.bounded_spans`), which
``check-budget`` holds to its own bound instead of the five minutes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from maketools import _test_hooks
from maketools.bounded_spans import MUTATION_SPAN, SPANS_ENVIRONMENT_KEY, record_span
from maketools.mutation.context import MutationRun
from maketools.mutation.step import run_mutation_step


def record_time(environment: Mapping[str, str], seconds: int) -> None:
    """Record the step's time for the check that started it, or say none did.

    Args:
        environment: The suite's environment.
        seconds: The step's wall time, in whole seconds.
    """
    named = environment.get(SPANS_ENVIRONMENT_KEY, "")
    if named:
        record_span(Path(named), MUTATION_SPAN, seconds, _test_hooks.now())
        _test_hooks.write_line(f"mutation: {seconds}s, recorded as the check's {MUTATION_SPAN}")
        return
    _test_hooks.write_line(
        f"mutation: {seconds}s; no check-budget named a span file in {SPANS_ENVIRONMENT_KEY}, "
        "so this run's time is held to no bound"
    )


def mutate_after_suite(
    project: Path, python: str, environment: Mapping[str, str], run_dir: Path
) -> int:
    """Run the step over a package whose suite just passed.

    Args:
        project: The package.
        python: Its environment's interpreter.
        environment: The suite's environment, naming its coverage file.
        run_dir: A directory that does not exist yet, for the step's files.

    Returns:
        The step's exit code: 1 when a mutant survived, else 0.
    """
    run = MutationRun(package_dir=project, python=python, environment=environment, run_dir=run_dir)
    run_dir.mkdir()
    started = _test_hooks.now()
    try:
        code = run_mutation_step(run)
    finally:
        _test_hooks.remove_tree(run_dir)
    record_time(environment, round(_test_hooks.now() - started))
    return code


__all__ = ["mutate_after_suite", "record_time"]
