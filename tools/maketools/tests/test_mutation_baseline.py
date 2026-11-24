"""Tests for :mod:`maketools.mutation.baseline`, lifted from MCPs (MCPs board task 3b386c0d).

MCPs' own cases for the module (``mcp-shared-py/tests/test_mutation_baseline.py``)
with the import paths made this package's: a lifted file is covered here as
it is there. Through :mod:`tests._mutation_world`, whose files are a dict:
each baseline round's pytest run is answered by writing the report a real
pytest would. The report's shape is pytest's xunit1 JUnit output
(``testcase`` with ``classname``, ``name``, ``file``, ``line`` and
``time``); ``test_mutation_real.py`` reads the one real pytest writes. Every
cost here is a sum of binary fractions, so each expected value is exact.
"""

from __future__ import annotations

import itertools
import re
import xml.etree.ElementTree as ElementTree

import pytest

from maketools.mutation.baseline import (
    Baselines,
    Share,
    node_file,
    ordered,
    read_costs,
    rounds,
    run_baselines,
    shares,
)
from maketools.mutation.context import MutationErrorCode, MutationStepError, Ran
from maketools.mutation.timing import TrialTimings
from tests._mutation_world import RUN_DIR, RUN_ENVIRONMENT, World, failed, ok

PYTEST = "/pkg/.venv/bin/python -W error::SyntaxWarning -m pytest -x -q -p no:cacheprovider"


def _junit(index: int) -> str:
    """The arguments that make round ``index`` write its report.

    Args:
        index: The round, from 1.

    Returns:
        The arguments, as the script spells them.
    """
    return (
        f"--junitxml=/scratch/run/baseline-{index}.junit.xml -o junit_family=xunit1 "
        "-o junit_duration_report=total"
    )


def _case(file: str | None, seconds: str | None, name: str = "t") -> str:
    """One xunit1 ``testcase`` element, as pytest writes it.

    Args:
        file: Its ``file``, or None to leave it out.
        seconds: Its ``time``, or None to leave it out.
        name: Its ``name``.

    Returns:
        The element's text.
    """
    attributes = f'classname="tests.x" name="{name}" line="3"'
    attributes += "" if file is None else f' file="{file}"'
    attributes += "" if seconds is None else f' time="{seconds}"'
    return f"<testcase {attributes} />"


def _report(*cases: str) -> str:
    """A JUnit report holding the cases.

    Args:
        cases: Each ``testcase`` element.

    Returns:
        The report's text.
    """
    body = "".join(cases)
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
        f'errors="0" failures="0" skipped="0" tests="{len(cases)}">{body}</testsuite>'
        "</testsuites>"
    )


def test_a_node_ids_file_is_what_precedes_its_first_separator() -> None:
    assert node_file("tests/a.py::Case::test[x::y]") == "tests/a.py"
    assert node_file("tests/a.py") == "tests/a.py"


def test_a_mutants_tests_are_shared_out_file_by_file_by_path() -> None:
    assert shares(("tests/b.py::x", "tests/a.py::y", "tests/a.py::z")) == [
        Share(file="tests/a.py", tests=("tests/a.py::y", "tests/a.py::z")),
        Share(file="tests/b.py", tests=("tests/b.py::x",)),
    ]
    assert shares(()) == []


#: a.py has two distinct shares and b.py one, which a later mutant repeats.
SELECTIONS: list[tuple[str, ...] | None] = [
    None,
    ("tests/a.py::1", "tests/b.py::1"),
    ("tests/a.py::1", "tests/a.py::2"),
    ("tests/a.py::1",),
    ("tests/b.py::1",),
    (),
]


def test_each_distinct_share_runs_in_one_round_and_each_round_holds_each_file_once() -> None:
    assert rounds(SELECTIONS) == [
        ("tests/a.py::1", "tests/b.py::1"),
        ("tests/a.py::1", "tests/a.py::2"),
    ]


def test_no_named_test_makes_no_round() -> None:
    assert rounds([None, ()]) == []
    assert rounds([]) == []


def test_each_files_cost_is_the_sum_of_its_cases_times() -> None:
    report = _report(
        _case("tests/a.py", "0.5"), _case("tests/b.py", "2.0"), _case("tests/a.py", "1.25")
    )
    assert read_costs(report) == {"tests/a.py": 1.75, "tests/b.py": 2.0}
    assert read_costs(_report()) == {}


@pytest.mark.parametrize(("file", "seconds"), [(None, "0.5"), ("tests/a.py", None), (None, None)])
def test_a_case_naming_no_file_or_no_time_is_refused_naming_both(
    file: str | None, seconds: str | None
) -> None:
    with pytest.raises(MutationStepError) as raised:
        read_costs(_report(_case("tests/a.py", "1.0"), _case(file, seconds, "t[1]")))
    assert raised.value.code is MutationErrorCode.BASELINE_UNREPORTED
    assert str(raised.value) == (
        f"MUTATION_BASELINE_UNREPORTED: the baseline's JUnit case 't[1]' names file {file!r} "
        f"and time {seconds!r}; the xunit1 family names both"
    )


def test_a_report_that_is_not_xml_is_refused_by_the_parser() -> None:
    with pytest.raises(ElementTree.ParseError):
        read_costs("<testsuites>")


#: What each round's pytest reports: round 1 ran a.py's first share and
#: b.py's, round 2 a.py's second.
REPORTS = {
    1: _report(_case("tests/a.py", "0.5"), _case("tests/b.py", "2.0")),
    2: _report(_case("tests/a.py", "0.5"), _case("tests/a.py", "1.25")),
}


class Pytest:
    """Each baseline's pytest: writes its round's report, then ends as told."""

    def __init__(self, world: World, ending: Ran, reports: dict[int, str]) -> None:
        self.world = world
        self.ending = ending
        self.reports = reports

    def __call__(self, argv: list[str]) -> Ran:
        written = re.search(r"baseline-(\d+)\.junit\.xml", argv[2])
        if written is not None:
            index = int(written.group(1))
            self.world.files[RUN_DIR / f"baseline-{index}.junit.xml"] = self.reports[index]
        return self.ending


def _world(ending: Ran, times: list[float], reports: dict[int, str]) -> World:
    world = World(lambda argv: ok(), times)
    world.answer = Pytest(world, ending, reports)
    return world


def test_the_whole_suite_then_each_round_runs_once_and_each_share_gets_its_rounds_cost() -> None:
    # The whole suite's timing (10.0 to 70.5) around its wall time, the same;
    # then round 1 (71.0 to 72.0) and round 2 (72.0 to 74.0).
    world = _world(ok(), [10.0, 10.0, 70.5, 70.5, 71.0, 72.0, 72.0, 74.0], REPORTS)
    context = world.context()
    timings = TrialTimings(context.hooks.now)
    assert run_baselines(context, SELECTIONS, timings) == Baselines(
        whole_suite=60.5,
        costs={
            Share("tests/a.py", ("tests/a.py::1",)): 0.5,
            Share("tests/b.py", ("tests/b.py::1",)): 2.0,
            Share("tests/a.py", ("tests/a.py::1", "tests/a.py::2")): 1.75,
        },
    )
    assert world.calls == [
        (["sh", "-c", f"exec {PYTEST}"], RUN_ENVIRONMENT),
        (
            ["sh", "-c", f"exec {PYTEST} {_junit(1)} -n 0 tests/a.py::1 tests/b.py::1"],
            RUN_ENVIRONMENT,
        ),
        (
            ["sh", "-c", f"exec {PYTEST} {_junit(2)} -n 0 tests/a.py::1 tests/a.py::2"],
            RUN_ENVIRONMENT,
        ),
    ]
    assert timings.lines()[:2] == [
        "mutation timing: baseline selected runs=2 seconds=3.000",
        "mutation timing: baseline whole-suite runs=1 seconds=60.500",
    ]
    assert timings.lines()[4:] == [
        "mutation timing: slow baseline whole-suite seconds=60.500 tests=0",
        "mutation timing: slow baseline selected seconds=2.000 tests=2",
        "mutation timing: slow baseline selected seconds=1.000 tests=2",
    ]
    assert world.times == []


def test_no_mutant_needing_the_whole_suite_runs_no_whole_suite_baseline() -> None:
    world = _world(ok(), [0.0, 1.0], {1: _report(_case("tests/a.py", "0.25"))})
    context = world.context()
    baselines = run_baselines(context, [("tests/a.py::1",)], TrialTimings(context.hooks.now))
    assert baselines == Baselines(
        whole_suite=None, costs={Share("tests/a.py", ("tests/a.py::1",)): 0.25}
    )
    assert len(world.calls) == 1


def test_a_failing_whole_suite_baseline_is_named_as_the_whole_suite() -> None:
    # The timing's start, the run's start, then the timing's end as the refusal leaves.
    world = _world(failed(2, "E boom"), [0.0, 0.0, 0.0], REPORTS)
    context = world.context()
    timings = TrialTimings(context.hooks.now)
    with pytest.raises(MutationStepError) as raised:
        run_baselines(context, SELECTIONS, timings)
    assert raised.value.code is MutationErrorCode.BASELINE_FAILED
    assert str(raised.value) == (
        "MUTATION_BASELINE_FAILED: the whole suite failed unmutated, exited 2: E boom"
    )
    # A refused run keeps its cost.
    assert timings.lines()[1] == "mutation timing: baseline whole-suite runs=1 seconds=0.000"


def test_a_failing_round_is_refused_naming_its_tests() -> None:
    world = _world(failed(1, "E boom"), [0.0, 1.0], REPORTS)
    context = world.context()
    with pytest.raises(MutationStepError) as raised:
        run_baselines(context, SELECTIONS[1:2], TrialTimings(context.hooks.now))
    assert str(raised.value) == (
        "MUTATION_BASELINE_FAILED: tests/a.py::1 tests/b.py::1 failed unmutated, exited 1: E boom"
    )


def test_a_round_whose_report_names_no_case_in_a_file_it_ran_is_refused() -> None:
    world = _world(ok(), [0.0, 1.0], {1: _report(_case("tests/a.py", "0.5"))})
    context = world.context()
    with pytest.raises(MutationStepError) as raised:
        run_baselines(context, SELECTIONS[1:2], TrialTimings(context.hooks.now))
    assert raised.value.code is MutationErrorCode.BASELINE_UNREPORTED
    assert str(raised.value) == (
        "MUTATION_BASELINE_UNREPORTED: /scratch/run/baseline-1.junit.xml names no case in "
        "tests/b.py, which the round ran, so what its tests cost is unknown"
    )


TESTS = ("tests/a.py::1", "tests/a.py::2", "tests/b.py::1", "tests/c.py::1")


def _proved(a: float, b: float, c: float) -> Baselines:
    """The baselines of :data:`TESTS`' three shares at the given costs.

    Args:
        a: What a.py's share cost.
        b: What b.py's share cost.
        c: What c.py's share cost.

    Returns:
        The baselines.
    """
    return Baselines(
        whole_suite=None,
        costs={
            Share("tests/a.py", ("tests/a.py::1", "tests/a.py::2")): a,
            Share("tests/b.py", ("tests/b.py::1",)): b,
            Share("tests/c.py", ("tests/c.py::1",)): c,
        },
    )


def test_a_mutants_tests_run_the_cheapest_files_first_then_by_path() -> None:
    # b.py and c.py cost the same, so their paths order them.
    assert ordered(TESTS, _proved(2.0, 0.5, 0.5)) == (
        ("tests/b.py::1", "tests/c.py::1", "tests/a.py::1", "tests/a.py::2"),
        3.0,
    )


@pytest.mark.parametrize("costs", list(itertools.permutations([0.25, 1.0, 4.0])))
def test_whatever_the_costs_each_file_runs_whole_in_cost_order_and_the_costs_add_up(
    costs: tuple[float, float, float],
) -> None:
    nodes, cost = ordered(TESTS, _proved(*costs))
    by_file = {"tests/a.py": costs[0], "tests/b.py": costs[1], "tests/c.py": costs[2]}
    files = [node_file(node) for node in nodes]
    assert sorted(nodes) == sorted(TESTS)
    assert [by_file[file] for file in files] == sorted(by_file[file] for file in files)
    first = files.index("tests/a.py")
    assert files[first : first + 2] == ["tests/a.py", "tests/a.py"]
    assert cost == 5.25


def test_a_whole_suite_mutant_costs_the_suites_wall_time() -> None:
    assert ordered((), Baselines(whole_suite=50.0, costs={})) == ((), 50.0)


def test_a_whole_suite_mutant_with_no_whole_suite_baseline_is_refused() -> None:
    with pytest.raises(MutationStepError) as raised:
        ordered((), Baselines(whole_suite=None, costs={}))
    assert raised.value.code is MutationErrorCode.BASELINE_UNREPORTED
    assert str(raised.value) == (
        "MUTATION_BASELINE_UNREPORTED: a mutant runs the whole suite and no baseline of the "
        "whole suite ran"
    )
