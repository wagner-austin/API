"""The figures a check's transcript tail prints, read once as typed values.

MCPs board task 601eef87. The verdict line (:mod:`fleet.core.verdict`)
renders the test counts, the coverage and the budget as ``key=value`` text,
and the queue's close now also carries them as a typed result that MCPs
stores beside the job (MCPs migration 957, ``recordDispatchResult``). Both
come from the readers here, so a pattern is matched once and the line and
the stored result cannot disagree. A figure the tail does not print is
``None`` or absent, never zero: a verdict that invents a number is worse
than one that says it could not read it.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from typing_extensions import TypedDict

#: ``N passed`` as both vitest (``Tests  887 passed (887)``) and pytest
#: (``1166 passed in 12.3s``) print it; the LAST occurrence is the suite's
#: own total, after any per-file lines.
PASSED: Final[re.Pattern[str]] = re.compile(r"(\d+) passed")

#: ``N failed``, the same two families, read only off the line that carries
#: the passed total: both runners print the two counts on that one line
#: (``Tests  2 failed | 885 passed (887)``, ``3 failed, 1166 passed in
#: 12.3s``), and anywhere else in the tail a service's own log can say it.
#: Read off the whole tail, MCPs/watcher's ``attempt 1/3 failed, retrying``
#: made a run of 2312 passed and none failed read ``2312p/3f`` (FLEET-CHECK
#: 0bce943c, colossus, 2026-10-11).
FAILED: Final[re.Pattern[str]] = re.compile(r"(\d+) failed")

#: pytest-cov's total row: ``TOTAL  5756  0  1526  0  100%`` (branches on)
#: or ``TOTAL  5756  0  100%``; the last percentage on the row is the total.
PYTEST_TOTAL: Final[re.Pattern[str]] = re.compile(r"^TOTAL\s.*?(\d+(?:\.\d+)?)%\s*$", re.MULTILINE)

#: vitest's v8 text reporter row for the whole tree: ``All files | 100 |
#: 100 | 100 | 100 |``, statements then branches. Printed only when a file
#: is under threshold in the default configuration, which is why its
#: absence on a passing run is not a defect but ``unread``.
VITEST_ALL_FILES: Final[re.Pattern[str]] = re.compile(
    r"^All files\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|", re.MULTILINE
)

#: osrs-lab's Gradle total row: ``COVERAGE TOTAL: lines 592/592 branches
#: 378/378``, covered over total summed from each module's JaCoCo XML report
#: by its check-summary init script (board task fd8e5f9b). Gradle and
#: JaCoCo print no count of their own (a refused gate prints a rounded
#: ratio, ``lines covered ratio is 0.7``), so the counts are the summary's.
#: Its tests are ``TESTS TOTAL: 68 passed, 0 failed, 0 skipped of 68``, which
#: :data:`PASSED` and :data:`FAILED` already read.
GRADLE_COVERAGE_TOTAL: Final[re.Pattern[str]] = re.compile(
    r"^COVERAGE TOTAL: lines (\d+)/(\d+) branches (\d+)/(\d+)$", re.MULTILINE
)

#: The sign node's built-in runner opens each summary line with, named rather
#: than written, since the character reads as a Latin ``i`` (ruff RUF001).
NODE_TEST_SIGN: Final = "\N{INFORMATION SOURCE}"

#: node's built-in runner (``node --test``, coax-drone/site) closes with its
#: own totals, one per line, each behind U+2139 INFORMATION SOURCE and a
#: space: ``<i> tests 60``, ``<i> pass 60``, ``<i> fail 0``, ``<i>`` standing
#: for :data:`NODE_TEST_SIGN` here. It prints no
#: ``N passed``, so until MCPs board task ed8e76a7 every coax-drone/site
#: verdict read ``tests=unread`` over 49 to 60 green tests (FLEET-CHECK
#: d533d7b3, 7a3bb896, 602098fd, 5c236243). The LAST of each is the run's: a
#: nested ``node --test`` inside a case would print its own first.
NODE_TEST_PASS: Final[re.Pattern[str]] = re.compile(rf"^{NODE_TEST_SIGN} pass (\d+)$", re.MULTILINE)

#: node's failed total, ``<i> fail 2``, printed beside the passed one on
#: every run, a green one included.
NODE_TEST_FAIL: Final[re.Pattern[str]] = re.compile(rf"^{NODE_TEST_SIGN} fail (\d+)$", re.MULTILINE)

#: node's ``--experimental-test-coverage`` row for the whole tree:
#: ``<i> all files     | 100.00 |   100.00 |  100.00 |``, lines, branches
#: and functions in that order, over the modules ``--test-coverage-include``
#: names.
NODE_TEST_ALL_FILES: Final[re.Pattern[str]] = re.compile(
    rf"^{NODE_TEST_SIGN} all files\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|", re.MULTILINE
)

#: vitest's threshold refusal, which names the figure that missed.
VITEST_THRESHOLD: Final[re.Pattern[str]] = re.compile(
    r"Coverage for (statements|branches|functions|lines) \((\d+(?:\.\d+)?)%\) does not meet"
)

#: maketools' check budget verdict for a check that stayed within it:
#: ``CHECK BUDGET: <package> took 37s of 300s (<split>).`` (MCPs board task
#: 1b152218), printed by check-lock in MCPs and by check-budget here.
BUDGET_WITHIN: Final[re.Pattern[str]] = re.compile(r"CHECK BUDGET: \S+ took (\d+)s of (\d+)s")

#: The same rule's verdict for a check past it, which exits non-zero after
#: every suite passed: ``CHECK OVER BUDGET: <package> took 1261s, over the
#: 300s budget by 961s.``
BUDGET_OVER: Final[re.Pattern[str]] = re.compile(
    r"CHECK OVER BUDGET: \S+ took (\d+)s, over the (\d+)s budget"
)


class CoverageMeasure(StrEnum):
    """Which coverage total a runner printed, as MCPs migration 957 names it."""

    STATEMENTS = "statements"
    BRANCHES = "branches"
    LINES = "lines"
    #: pytest-cov's one total, statements and branches together.
    TOTAL = "total"


class BudgetOutcome(StrEnum):
    """Whether a check kept to its time budget, as MCPs migration 957 names it."""

    WITHIN = "within"
    EXCEEDED = "exceeded"


class SuiteCounts(TypedDict):
    """A suite's totals as its runner printed them.

    Attributes:
        passed: The passed total.
        failed: The failed total; zero beside ``N passed`` when the runner
            printed none, as vitest and pytest print it only when non-zero.
    """

    passed: int
    failed: int


class Coverage(TypedDict):
    """One coverage total.

    Attributes:
        measure: Which total.
        percent: Its percentage, 0 to 100.
    """

    measure: CoverageMeasure
    percent: float


class CoverageReading(TypedDict):
    """The coverage a tail printed, rendered and typed from one match.

    Attributes:
        rendered: The verdict line's text, as the runner printed the figures.
        measures: Each total in MCPs' vocabulary; a figure outside it
            (vitest's ``functions``) or a Gradle total over nothing has none.
    """

    rendered: str
    measures: list[Coverage]


class BudgetReading(TypedDict):
    """A check budget verdict.

    Attributes:
        outcome: Within or past the bound.
        seconds: The seconds the check took.
        bound: The budget in seconds.
    """

    outcome: BudgetOutcome
    seconds: int
    bound: int


def read_test_counts(tail: str) -> SuiteCounts | None:
    """Read the test counts off a transcript tail.

    node's built-in runner's own totals are read first, the last
    :data:`NODE_TEST_PASS` and :data:`NODE_TEST_FAIL` lines, when the tail
    holds both: they are the run's, while an ``N passed`` in such a
    transcript could be a case's own title. Otherwise the passed total is
    the last ``N passed`` in the tail, and the failed count is read off that
    same line only (:data:`FAILED` says why).

    Args:
        tail: The transcript's last lines.

    Returns:
        The totals when a passed count was read, else None.
    """
    node_passed: str | None = None
    for match in NODE_TEST_PASS.finditer(tail):
        node_passed = match.group(1)
    node_failed: str | None = None
    for match in NODE_TEST_FAIL.finditer(tail):
        node_failed = match.group(1)
    if node_passed is not None and node_failed is not None:
        return SuiteCounts(passed=int(node_passed), failed=int(node_failed))
    found: tuple[re.Match[str], str] | None = None
    for line in tail.split("\n"):
        for match in PASSED.finditer(line):
            found = (match, line)
    if found is None:
        return None
    totals, totals_line = found
    failed = FAILED.search(totals_line)
    return SuiteCounts(
        passed=int(totals.group(1)), failed=0 if failed is None else int(failed.group(1))
    )


def _fraction(covered: str, total: str) -> list[float]:
    """A covered-over-total pair as a percentage, or none for a total of nothing.

    Args:
        covered: The covered count.
        total: The total count.

    Returns:
        One percentage rounded to two places, or none when the total is 0.
    """
    whole = int(total)
    return [round(int(covered) * 100 / whole, 2)] if whole else []


def read_coverage_reading(tail: str) -> CoverageReading | None:
    """Read the coverage figures off a transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``statements=<n>% branches=<n>%`` from vitest's tree row,
        ``total=<n>%`` from pytest-cov's, ``lines=<c>/<t> branches=<c>/<t>``
        from the Gradle summary's, ``lines=<n>% branches=<n>%`` from node's
        built-in runner's tree row, ``<figure>=<n>% below threshold`` from
        vitest's refusal, each with its typed totals; else None.
    """
    vitest = VITEST_ALL_FILES.search(tail)
    if vitest is not None:
        return CoverageReading(
            rendered=f"statements={vitest.group(1)}% branches={vitest.group(2)}%",
            measures=[
                Coverage(measure=CoverageMeasure.STATEMENTS, percent=float(vitest.group(1))),
                Coverage(measure=CoverageMeasure.BRANCHES, percent=float(vitest.group(2))),
            ],
        )
    pytest_total: str | None = None
    for match in PYTEST_TOTAL.finditer(tail):
        pytest_total = match.group(1)
    if pytest_total is not None:
        return CoverageReading(
            rendered=f"total={pytest_total}%",
            measures=[Coverage(measure=CoverageMeasure.TOTAL, percent=float(pytest_total))],
        )
    gradle = GRADLE_COVERAGE_TOTAL.search(tail)
    if gradle is not None:
        lines, line_total, branches, branch_total = gradle.groups()
        return CoverageReading(
            rendered=f"lines={lines}/{line_total} branches={branches}/{branch_total}",
            measures=[
                *(
                    Coverage(measure=CoverageMeasure.LINES, percent=p)
                    for p in _fraction(lines, line_total)
                ),
                *(
                    Coverage(measure=CoverageMeasure.BRANCHES, percent=p)
                    for p in _fraction(branches, branch_total)
                ),
            ],
        )
    node_total: re.Match[str] | None = None
    for match in NODE_TEST_ALL_FILES.finditer(tail):
        node_total = match
    if node_total is not None:
        return CoverageReading(
            rendered=f"lines={node_total.group(1)}% branches={node_total.group(2)}%",
            measures=[
                Coverage(measure=CoverageMeasure.LINES, percent=float(node_total.group(1))),
                Coverage(measure=CoverageMeasure.BRANCHES, percent=float(node_total.group(2))),
            ],
        )
    threshold = VITEST_THRESHOLD.search(tail)
    if threshold is not None:
        figure, percent = threshold.groups()
        measured = [measure for measure in CoverageMeasure if measure.value == figure]
        return CoverageReading(
            rendered=f"{figure}={percent}% below threshold",
            measures=[Coverage(measure=measure, percent=float(percent)) for measure in measured],
        )
    return None


def read_budget_reading(tail: str) -> BudgetReading | None:
    """Read the check budget verdict off a transcript tail.

    A check over its budget exits non-zero with every suite green, so its
    row reads like any other red one unless the line says which failure it
    was; this field is how a fleet row reads as the budget's loud failure.

    Args:
        tail: The transcript's last lines.

    Returns:
        The over-budget verdict, else the within-budget one, else None (a
        package whose check predates the budget, or a run that stopped
        before the verdict printed).
    """
    over = BUDGET_OVER.search(tail)
    if over is not None:
        return BudgetReading(
            outcome=BudgetOutcome.EXCEEDED, seconds=int(over.group(1)), bound=int(over.group(2))
        )
    within = BUDGET_WITHIN.search(tail)
    if within is not None:
        return BudgetReading(
            outcome=BudgetOutcome.WITHIN, seconds=int(within.group(1)), bound=int(within.group(2))
        )
    return None


__all__ = [
    "BUDGET_OVER",
    "BUDGET_WITHIN",
    "FAILED",
    "GRADLE_COVERAGE_TOTAL",
    "NODE_TEST_ALL_FILES",
    "NODE_TEST_FAIL",
    "NODE_TEST_PASS",
    "NODE_TEST_SIGN",
    "PASSED",
    "PYTEST_TOTAL",
    "VITEST_ALL_FILES",
    "VITEST_THRESHOLD",
    "BudgetOutcome",
    "BudgetReading",
    "Coverage",
    "CoverageMeasure",
    "CoverageReading",
    "SuiteCounts",
    "read_budget_reading",
    "read_coverage_reading",
    "read_test_counts",
]
