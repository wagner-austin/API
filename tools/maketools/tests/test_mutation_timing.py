"""Mutation costs are observable without changing which tests or mutants run.

MCPs' own cases for :mod:`maketools.mutation.timing`, lifted from MCPs (MCPs
board task aee4bf33; ``mcp-shared-py/tests/test_mutation_timing.py``), with
the import paths made this package's.
"""

from __future__ import annotations

from itertools import count

import pytest

from maketools.mutation.step import run_mutation_step
from maketools.mutation.timing import Phase, Scope, TrialTimings
from tests._mutation_scenario import scenario_world
from tests.test_mutation_step import TIMES


def test_timings_distinguish_baselines_from_trials_and_preserve_the_verdict() -> None:
    # One clock read per tick: the start (0) and the bound's read (1); the
    # whole suite's timing (2 to 5) around its own wall time (3 to 4), so 3 s;
    # the one round's timing (6 to 7), whose cost the JUnit report gives, so
    # 1 s; then each trial's timing spans one tick after the bound's read.
    world, _ = scenario_world([*TIMES])
    clock = count()
    context = world.context()
    context = context._replace(hooks=context.hooks._replace(now=lambda: float(next(clock))))
    assert run_mutation_step(context) == 1
    timings = [line for line in world.lines if line.startswith("mutation timing:")]
    assert timings == [
        "mutation timing: baseline selected runs=1 seconds=1.000",
        "mutation timing: baseline whole-suite runs=1 seconds=3.000",
        "mutation timing: mutant selected runs=2 seconds=2.000",
        "mutation timing: mutant whole-suite runs=1 seconds=1.000",
        "mutation timing: slow baseline whole-suite seconds=3.000 tests=0",
        "mutation timing: slow baseline selected seconds=1.000 tests=2",
        "mutation timing: slow mutant whole-suite seconds=1.000 "
        "src/p/m.py:1:9 core/NumberReplacer occurrence=0 tests=0",
        "mutation timing: slow mutant selected seconds=1.000 "
        "src/p/m.py:5:18 core/ReplaceComparisonOperator_GtE_Gt occurrence=0 tests=2",
        "mutation timing: slow mutant selected seconds=1.000 "
        "src/p/m.py:6:15 core/ReplaceComparisonOperator_Lt_LtE occurrence=0 tests=2",
    ]
    assert world.lines[-1].startswith("mutation: FAILED: 2 of 4 mutant(s)")


def test_an_empty_run_reports_zero_in_every_category() -> None:
    timings = TrialTimings(lambda: 0.0)
    assert timings.lines() == [
        "mutation timing: baseline selected runs=0 seconds=0.000",
        "mutation timing: baseline whole-suite runs=0 seconds=0.000",
        "mutation timing: mutant selected runs=0 seconds=0.000",
        "mutation timing: mutant whole-suite runs=0 seconds=0.000",
    ]


@pytest.mark.parametrize("phase", list(Phase))
@pytest.mark.parametrize("scope", list(Scope))
def test_costs_sum_and_only_the_five_slowest_are_printed(phase: Phase, scope: Scope) -> None:
    # Independent intervals total 21.5 seconds, including a zero and a fraction.
    durations = [1.0, 3.0, 0.0, 2.0, 7.0, 8.0, 0.5]
    clock = iter(value for duration in durations for value in (10.0, 10.0 + duration))
    timings = TrialTimings(lambda: next(clock))
    for index in range(len(durations)):
        with timings.measure(phase, scope, f"trial-{index}"):
            pass
    lines = timings.lines()
    assert f"mutation timing: {phase} {scope} runs=7 seconds=21.500" in lines[:4]
    assert lines[4:] == [
        f"mutation timing: slow {phase} {scope} seconds=8.000 trial-5",
        f"mutation timing: slow {phase} {scope} seconds=7.000 trial-4",
        f"mutation timing: slow {phase} {scope} seconds=3.000 trial-1",
        f"mutation timing: slow {phase} {scope} seconds=2.000 trial-3",
        f"mutation timing: slow {phase} {scope} seconds=1.000 trial-0",
    ]
    assert timings.lines() == lines


def test_a_failed_invocation_keeps_its_cost_and_propagates_the_same_exception() -> None:
    clock = iter([4.0, 6.25])
    timings = TrialTimings(lambda: next(clock))
    failure = ValueError("the actual command failed")
    with pytest.raises(ValueError) as raised, timings.measure(Phase.MUTANT, Scope.SELECTED, "site"):
        raise failure
    assert raised.value is failure
    assert timings.lines()[2:] == [
        "mutation timing: mutant selected runs=1 seconds=2.250",
        "mutation timing: mutant whole-suite runs=0 seconds=0.000",
        "mutation timing: slow mutant selected seconds=2.250 site",
    ]
