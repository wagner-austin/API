"""Tests for :mod:`maketools.mutation.trials` (MCPs board task e52e5148).

cosmic-ray's answers are the JSON it printed on sedona
(``test_mutation_cosmic.py``); ``test_mutation_real.py`` runs the same
calls against real pytest and cosmic-ray.
"""

from __future__ import annotations

import shlex

import pytest
from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError

from maketools.commands import CommandResult
from maketools.mutation.context import CHILD_WALL_SECONDS
from maketools.mutation.cosmic import RunOutcome, Site, WorkerOutcome
from maketools.mutation.trials import (
    mutant_timeout,
    pytest_argv,
    run_baseline,
    run_environment,
    run_mutant,
    survived,
)
from tests._mutation_run import COSMIC_RAY, ENVIRONMENT, PACKAGE, RUN_ENVIRONMENT, mutation_run
from tests.conftest import AnsweringCall, World, failed, ok
from tests.test_mutation_cosmic import EXCEPTION, NO_TEST, SURVIVED

PYTEST = [
    "/pkg/.venv/bin/python",
    "-W",
    "error::SyntaxWarning",
    "-m",
    "pytest",
    "-x",
    "-q",
    "-p",
    "no:cacheprovider",
    "--basetemp=/scratch/run/pytest",
]
NAMED = ("tests/t.py::a", "tests/t.py::b[x y]")
SITE = Site("src/p.py", 7, 4, "core/ReplaceComparisonOperator_Eq_Lt", 3)
KILLED = SURVIVED.replace('"survived"', '"killed"')
INCOMPETENT = KILLED.replace('"killed"', '"incompetent"')


def test_tests_run_without_the_span_file_or_bytecode_and_the_step_keeps_its_environment() -> None:
    run = mutation_run()
    assert run_environment(run) == RUN_ENVIRONMENT
    assert run.environment == ENVIRONMENT


def test_named_tests_run_in_one_process_with_the_mutants_flags() -> None:
    assert pytest_argv(mutation_run(), NAMED) == [*PYTEST, "-n", "0", *NAMED]


def test_the_whole_suite_runs_with_the_packages_own_workers() -> None:
    assert pytest_argv(mutation_run(), ()) == PYTEST


def test_a_mutants_timeout_is_three_baselines_rounded_up_plus_thirty_seconds() -> None:
    assert mutant_timeout(10.2) == 61
    assert mutant_timeout(0.0) == 30


def test_a_passing_baseline_sets_the_timeout_from_its_wall_time(world: World) -> None:
    def answer(argv: tuple[str, ...]) -> CommandResult:
        world.now_value += 110.2
        return ok("")

    world.answer = answer
    assert run_baseline(mutation_run(), NAMED) == 361
    assert world.answering_calls == [
        AnsweringCall(
            argv=(*PYTEST, "-n", "0", *NAMED),
            cwd=PACKAGE,
            env=RUN_ENVIRONMENT,
            timeout_seconds=CHILD_WALL_SECONDS,
        )
    ]


def test_a_failing_baseline_is_refused_naming_its_tests(world: World) -> None:
    world.answer = lambda argv: failed(1, "E boom")
    with pytest.raises(AppError) as refused:
        run_baseline(mutation_run(), NAMED)
    assert refused.value.code is MutationErrorCode.BASELINE_FAILED
    assert refused.value.message == (
        "tests/t.py::a tests/t.py::b[x y] failed unmutated, exited 1: E boom"
    )


def test_a_failing_whole_suite_baseline_is_named_as_the_whole_suite(world: World) -> None:
    world.answer = lambda argv: failed(2, "E boom")
    with pytest.raises(AppError, match=r"^the whole suite failed unmutated, exited 2: E boom$"):
        run_baseline(mutation_run(), ())


def test_a_surviving_mutant_answers_the_line_it_put_in_place(world: World) -> None:
    world.answer = lambda argv: ok(SURVIVED)
    assert run_mutant(mutation_run(), SITE, NAMED, 61) == 'if place < "":'
    (call,) = world.answering_calls
    assert call["argv"][:5] == (
        COSMIC_RAY,
        "mutate-and-test",
        "src/p.py",
        "core/ReplaceComparisonOperator_Eq_Lt",
        "3",
    )
    assert shlex.split(call["argv"][5]) == [
        "timeout",
        "--kill-after=10",
        "61",
        *PYTEST,
        "-n",
        "0",
        *NAMED,
    ]
    assert len(call["argv"]) == 6
    assert call["env"] == RUN_ENVIRONMENT


def test_a_killed_mutant_answers_none(world: World) -> None:
    world.answer = lambda argv: ok(KILLED)
    assert run_mutant(mutation_run(), SITE, (), 30) is None


@pytest.mark.parametrize(
    ("answer", "ending"),
    [
        (EXCEPTION, "ended exception, tests incompetent: Traceback"),
        (INCOMPETENT, "ended normal, tests incompetent: "),
        (NO_TEST, "ended no-test, tests None: "),
    ],
)
def test_a_worker_that_judged_nothing_is_refused(world: World, answer: str, ending: str) -> None:
    world.answer = lambda argv: ok(answer)
    with pytest.raises(AppError) as refused:
        run_mutant(mutation_run(), SITE, (), 30)
    assert refused.value.code is MutationErrorCode.COSMIC_RAY_FAILED
    assert refused.value.message.startswith(
        f"src/p.py:7 core/ReplaceComparisonOperator_Eq_Lt {ending}"
    )


def test_cosmic_ray_failing_is_refused_naming_its_command(world: World) -> None:
    world.answer = lambda argv: failed(2, "Error: no such session")
    with pytest.raises(AppError) as refused:
        run_mutant(mutation_run(), SITE, (), 30)
    assert refused.value.code is MutationErrorCode.COSMIC_RAY_FAILED
    assert refused.value.message.startswith(f"{COSMIC_RAY} mutate-and-test src/p.py")
    assert refused.value.message.endswith("exited 2: Error: no such session")


def test_only_a_normal_worker_that_killed_or_kept_a_mutant_judges_it() -> None:
    assert survived(WorkerOutcome.NORMAL, RunOutcome.KILLED) is False
    assert survived(WorkerOutcome.NORMAL, RunOutcome.SURVIVED) is True
    assert survived(WorkerOutcome.NORMAL, RunOutcome.INCOMPETENT) is None
    assert survived(WorkerOutcome.EXCEPTION, RunOutcome.KILLED) is None
    assert survived(WorkerOutcome.NO_TEST, None) is None
