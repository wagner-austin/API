"""Tests for :mod:`maketools.mutation.trials`, lifted from MCPs (MCPs task e52e5148).

MCPs' own cases for the module (``mcp-shared-py/tests/test_mutation_trials.py``)
with the import paths made this package's. Through
:mod:`tests._mutation_world`; cosmic-ray's answers are the JSON it printed on
sedona (``test_mutation_cosmic.py``), and ``test_mutation_real.py`` runs the
same calls against real pytest and cosmic-ray.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from maketools.mutation.context import MutationStepError
from maketools.mutation.cosmic import RunOutcome, Site, WorkerOutcome
from maketools.mutation.trials import (
    mutant_timeout,
    pytest_script,
    run_environment,
    run_mutant,
    survived,
)
from tests._mutation_world import COSMIC_RAY, ENVIRONMENT, RUN_ENVIRONMENT, World, failed, ok
from tests.test_mutation_cosmic import EXCEPTION, NO_TEST, SURVIVED

#: The package's own pytest: no base temporary directory of the step's.
PYTEST = "/pkg/.venv/bin/python -W error::SyntaxWarning -m pytest -x -q -p no:cacheprovider"
NAMED = ("tests/t.py::a", "tests/t.py::b[x y]")
SITE = Site("src/p.py", 7, 4, "core/ReplaceComparisonOperator_Eq_Lt", 3)
KILLED = SURVIVED.replace('"survived"', '"killed"')
INCOMPETENT = KILLED.replace('"killed"', '"incompetent"')
#: Where a baseline round's report goes, and the arguments that write it.
REPORT = Path("/scratch/run/baseline-1.junit.xml")
JUNIT = (
    "--junitxml=/scratch/run/baseline-1.junit.xml -o junit_family=xunit1 "
    "-o junit_duration_report=total"
)


def test_tests_run_without_the_check_locks_span_file_and_the_step_keeps_it() -> None:
    context = World(lambda argv: ok(), []).context()
    assert run_environment(context) == RUN_ENVIRONMENT
    assert context.environment == ENVIRONMENT


def test_named_tests_run_in_one_process_in_the_order_given() -> None:
    context = World(lambda argv: ok(), []).context()
    assert pytest_script(context, NAMED, None) == (
        f"exec {PYTEST} -n 0 tests/t.py::a 'tests/t.py::b[x y]'"
    )


def test_the_whole_suite_runs_with_the_packages_own_workers() -> None:
    context = World(lambda argv: ok(), []).context()
    assert pytest_script(context, (), None) == f"exec {PYTEST}"


def test_a_baseline_round_writes_an_xunit1_report_beside_its_named_tests() -> None:
    context = World(lambda argv: ok(), []).context()
    assert pytest_script(context, NAMED, REPORT) == (
        f"exec {PYTEST} {JUNIT} -n 0 tests/t.py::a 'tests/t.py::b[x y]'"
    )
    assert pytest_script(context, (), REPORT) == f"exec {PYTEST} {JUNIT}"


def test_a_prepare_command_runs_before_pytest() -> None:
    context = World(lambda argv: ok(), []).context("python scripts/gen.py")
    assert pytest_script(context, (), None) == f"python scripts/gen.py && exec {PYTEST}"


def test_a_mutants_timeout_is_three_baselines_rounded_up_plus_thirty_seconds() -> None:
    # 3 x 10.2 = 30.6, up to 31, plus 30.
    assert mutant_timeout(10.2) == 61
    assert mutant_timeout(0.0) == 30


def test_a_surviving_mutant_answers_the_line_it_put_in_place() -> None:
    world = World(lambda argv: ok(SURVIVED), [])
    assert run_mutant(world.context(), SITE, NAMED, 61) == 'if place < "":'
    (argv, env), *_ = world.calls
    assert argv[:5] == [
        COSMIC_RAY,
        "mutate-and-test",
        "src/p.py",
        "core/ReplaceComparisonOperator_Eq_Lt",
        "3",
    ]
    assert shlex.split(argv[5]) == [
        "timeout",
        "--kill-after=10",
        "61",
        "sh",
        "-c",
        f"exec {PYTEST} -n 0 tests/t.py::a 'tests/t.py::b[x y]'",
    ]
    assert len(argv) == 6
    assert env == RUN_ENVIRONMENT


def test_a_killed_mutant_answers_none() -> None:
    assert run_mutant(World(lambda argv: ok(KILLED), []).context(), SITE, (), 30) is None


@pytest.mark.parametrize(
    ("answer", "ending"),
    [
        (EXCEPTION, "ended exception, tests incompetent: Traceback"),
        (INCOMPETENT, "ended normal, tests incompetent: "),
        (NO_TEST, "ended no-test, tests None: "),
    ],
)
def test_a_worker_that_judged_nothing_is_refused(answer: str, ending: str) -> None:
    world = World(lambda argv: ok(answer), [])
    with pytest.raises(MutationStepError) as raised:
        run_mutant(world.context(), SITE, (), 30)
    assert str(raised.value).startswith(
        f"MUTATION_COSMIC_RAY_FAILED: src/p.py:7 core/ReplaceComparisonOperator_Eq_Lt {ending}"
    )


def test_cosmic_ray_failing_is_refused_naming_its_command() -> None:
    world = World(lambda argv: failed(2, "Error: no such session"), [])
    with pytest.raises(MutationStepError) as raised:
        run_mutant(world.context(), SITE, (), 30)
    assert str(raised.value).startswith(f"MUTATION_COSMIC_RAY_FAILED: {COSMIC_RAY} mutate-and-test")
    assert str(raised.value).endswith("exited 2: Error: no such session")


def test_only_a_normal_worker_that_killed_or_kept_a_mutant_judges_it() -> None:
    assert survived(WorkerOutcome.NORMAL, RunOutcome.KILLED) is False
    assert survived(WorkerOutcome.NORMAL, RunOutcome.SURVIVED) is True
    assert survived(WorkerOutcome.NORMAL, RunOutcome.INCOMPETENT) is None
    assert survived(WorkerOutcome.EXCEPTION, RunOutcome.KILLED) is None
    assert survived(WorkerOutcome.NO_TEST, None) is None
