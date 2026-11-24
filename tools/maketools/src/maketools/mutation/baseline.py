"""The unmutated runs before the first trial, and what each mutant's tests cost (task 3b386c0d).

WHY NOT ONE PER SET. The step used to run each distinct set of tests
unmutated before its first mutant, and session-audit's sets each held a
real-board module whose fixture takes 12 to 17 s: fleet job 48638505 spent
378 s of its 900 s bound on 28 such baselines and 98 s on the whole suite's,
and ran 100 of its 335 mutants. Sets differ mostly in which cheap tests they
hold beside the same slow modules.

WHAT IS PROVED, PER FILE. A mutant runs its tests file by file, so what must
pass unmutated is each file's share of its tests, run together: a share of a
module may lean on a test it leaves out (session-audit's real-board modules
share one board through a module fixture), so each distinct share is run
once, as it is, before any trial. The shares are run in rounds
(:func:`rounds`): round ``k`` holds every file's ``k``-th distinct share, so
each round is one pytest process in which every file runs once, and a slow
module is set up once per distinct share instead of once per set. A mutant's
files then run in another order than its round's, which assumes what the
package's own suite assumes when its workers take files in any order: that
no file leans on another having run. A whole-suite mutant's baseline is the
whole suite, once, under the package's own workers.

WHAT A SHARE COSTS. Each round writes pytest's JUnit report in the xunit1
family, whose every ``testcase`` names its ``file`` and its ``time``: setup,
call and teardown (``junit_duration_report=total``), so a module fixture's
setup counts in the file it serves. A mutant's files run cheapest first, so
under ``-x`` a fast test kills it before a slow module starts, and its
timeout is three times its shares' cost plus 30 s (:func:`~.trials.mutant_timeout`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable
from itertools import zip_longest
from typing import NamedTuple

from .context import JsonT, MutationErrorCode, MutationStepError, StepContext, tail
from .timing import Phase, Scope, TrialTimings
from .trials import pytest_script, run_environment


class Share(NamedTuple):
    """A file and the tests of it one mutant runs, which a round proves together.

    Attributes:
        file: The test file, as node ids spell it.
        tests: Its node ids the mutant runs, sorted.
    """

    file: str
    tests: tuple[str, ...]


class Baselines(NamedTuple):
    """What the unmutated runs measured.

    Attributes:
        whole_suite: The whole suite's wall time, which a whole-suite
            mutant's timeout is set from, or None when no mutant needs it.
        costs: Each share to the summed time of its cases in its round.
    """

    whole_suite: float | None
    costs: dict[Share, float]


def node_file(node: str) -> str:
    """The file a pytest node id names.

    Args:
        node: A node id, ``<file>::<test>``.

    Returns:
        The file, relative to the package as the node id spells it.
    """
    return node.partition("::")[0]


def shares(tests: tuple[str, ...]) -> list[Share]:
    """A mutant's tests, file by file.

    Args:
        tests: Node ids, sorted.

    Returns:
        Each file once, by path, with its tests in the order given.
    """
    by_file: dict[str, list[str]] = {}
    for node in tests:
        by_file.setdefault(node_file(node), []).append(node)
    return [Share(file=file, tests=tuple(by_file[file])) for file in sorted(by_file)]


def rounds(selections: Iterable[tuple[str, ...] | None]) -> list[tuple[str, ...]]:
    """The rounds that run every distinct share once, each file once a round.

    Args:
        selections: Each mutant's tests: None for no test, empty for the
            whole suite, else node ids, sorted.

    Returns:
        Each round's node ids: round ``k`` holds every file's ``k``-th
        distinct share, in the order the mutants first named it, its files
        by path.
    """
    distinct: dict[str, list[tuple[str, ...]]] = {}
    for tests in selections:
        for share in shares(tests or ()):
            known = distinct.setdefault(share.file, [])
            if share.tests not in known:
                known.append(share.tests)
    columns = zip_longest(*(distinct[file] for file in sorted(distinct)), fillvalue=())
    return [tuple(node for share in column for node in share) for column in columns]


def read_costs(report: str) -> dict[str, float]:
    """Each test file's summed case time, from an xunit1 JUnit report.

    Args:
        report: The report's text.

    Returns:
        Each file a case names to the sum of its cases' times.

    Raises:
        MutationStepError: ``MUTATION_BASELINE_UNREPORTED`` for a case that
            names no file or no time.
        xml.etree.ElementTree.ParseError: When the report is not XML; pytest
            wrote it, so a broken one is no baseline to judge by.
    """
    costs: dict[str, float] = {}
    for case in ElementTree.fromstring(report).iter("testcase"):
        file, seconds = case.get("file"), case.get("time")
        if file is None or seconds is None:
            raise MutationStepError(
                MutationErrorCode.BASELINE_UNREPORTED,
                f"the baseline's JUnit case {case.get('name')!r} names file {file!r} and time "
                f"{seconds!r}; the xunit1 family names both",
            )
        costs[file] = costs.get(file, 0.0) + float(seconds)
    return costs


def _must_pass(context: StepContext[JsonT], tests: tuple[str, ...], script: str) -> None:
    """Run one unmutated set of tests, which must pass.

    Args:
        context: The run.
        tests: Its node ids, or empty for the whole suite, to name it.
        script: The pytest script that runs it.

    Raises:
        MutationStepError: ``MUTATION_BASELINE_FAILED`` when it fails
            unmutated, since every mutant it runs would then read as killed.
    """
    ran = context.hooks.run(["sh", "-c", script], run_environment(context))
    if ran.exit_code:
        named = " ".join(tests) or "the whole suite"
        raise MutationStepError(
            MutationErrorCode.BASELINE_FAILED, f"{named} failed unmutated, {tail(ran)}"
        )


def run_baselines(
    context: StepContext[JsonT],
    selections: list[tuple[str, ...] | None],
    timings: TrialTimings,
) -> Baselines:
    """Run the whole suite if a mutant needs it, then every round, and read each share's cost.

    Args:
        context: The run.
        selections: Each chosen mutant's tests.
        timings: Where each run's cost is recorded.

    Returns:
        The whole suite's wall time and each share's cost.

    Raises:
        MutationStepError: ``MUTATION_BASELINE_FAILED`` for a run that
            fails unmutated, and ``MUTATION_BASELINE_UNREPORTED`` for a round
            whose report names no case in a file of it.
    """
    whole_suite: float | None = None
    if () in selections:
        with timings.measure(Phase.BASELINE, Scope.WHOLE_SUITE, "tests=0"):
            started = context.hooks.now()
            _must_pass(context, (), pytest_script(context, (), None))
            whole_suite = context.hooks.now() - started
    costs: dict[Share, float] = {}
    for index, nodes in enumerate(rounds(selections)):
        report = context.run_dir / f"baseline-{index + 1}.junit.xml"
        with timings.measure(Phase.BASELINE, Scope.SELECTED, f"tests={len(nodes)}"):
            _must_pass(context, nodes, pytest_script(context, nodes, report))
        measured = read_costs(context.hooks.read_text(report))
        for share in shares(nodes):
            if share.file not in measured:
                raise MutationStepError(
                    MutationErrorCode.BASELINE_UNREPORTED,
                    f"{report} names no case in {share.file}, which the round ran, so what its "
                    "tests cost is unknown",
                )
            costs[share] = measured[share.file]
    return Baselines(whole_suite=whole_suite, costs=costs)


def ordered(tests: tuple[str, ...], baselines: Baselines) -> tuple[tuple[str, ...], float]:
    """A mutant's tests, the cheapest file's first, and what they cost unmutated.

    Args:
        tests: The node ids of the tests that ran its line, sorted, or empty
            for the whole suite.
        baselines: The runs that proved each of its shares.

    Returns:
        Its node ids, file by file by cost and then by path, and their
        cost; for the whole suite, no node ids and the suite's wall time.

    Raises:
        MutationStepError: ``MUTATION_BASELINE_UNREPORTED`` for the whole
            suite when no baseline of it ran.
    """
    if not tests:
        if baselines.whole_suite is None:
            raise MutationStepError(
                MutationErrorCode.BASELINE_UNREPORTED,
                "a mutant runs the whole suite and no baseline of the whole suite ran",
            )
        return (), baselines.whole_suite
    ranked = sorted((baselines.costs[share], share) for share in shares(tests))
    ordered_nodes = tuple(node for _, share in ranked for node in share.tests)
    return ordered_nodes, sum(cost for cost, _ in ranked)


__all__ = [
    "Baselines",
    "Share",
    "node_file",
    "ordered",
    "read_costs",
    "rounds",
    "run_baselines",
    "shares",
]
