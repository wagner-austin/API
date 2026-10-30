"""The one line a check's outcome becomes on the submitter's task thread.

WHY A LINE ON THE THREAD AND NOT A DISPATCH DETAIL. Until MCPs board task
fd5cabfa a finished dispatch was a ``detail`` string on the queue row, read by
whoever remembered to call ``dispatch_get``; a closure that wanted to cite a
check had nothing on its own thread to point at. The runner now posts one line
where the closure will be read, carrying everything the review needs without
a second lookup: the commit, the node, the exit status, whether the banner was
read, the test and coverage counts read off the transcript, and the path of
the whole transcript on the node.

THE BANNER IS THE VERDICT; THE COUNTS ARE THE EVIDENCE. ``=== ALL CHECKS
PASSED ===`` is the last thing every package's ``make check`` prints and the
only thing that means it passed (the operator's rule; an exit code through a
pipe has lied before, ``feedback_out_file_pipe_clobbers_lastexitcode``). The
counts are parsed from the two families of runner output this fleet has,
vitest's and pytest's, and a count that is not in the tail is written as
``unread`` rather than guessed: a verdict that invents a number is worse than
one that says it could not read it.
"""

from __future__ import annotations

import datetime
import re
from typing import Final

from typing_extensions import TypedDict

from fleet.core.ci_load import read_ci_load
from fleet.core.linux_unit_end import UNIT_ENDED_LINE

#: The line every package's ``make check`` prints last when it passed.
CHECK_BANNER: Final = "=== ALL CHECKS PASSED ==="

#: The line's opening token, so a thread can be grepped for verdicts.
VERDICT_PREFIX: Final = "FLEET-CHECK"

#: How many lines of the transcript the collector reads for the counts: the
#: summary tables sit at the end, but vitest prints its per-file coverage
#: table AFTER the test totals, one row per source file, so the distance
#: grows with the package. Two hundred lines, the bound until MCPs board task
#: 93a1c2c0, left every verdict of four MCPs packages 'unread': measured on
#: colossus's 323 logs on 2026-10-11, the last 'N passed' line sat 252 lines
#: from the end for MCPs/watcher, 284 for MCPs/mcp-proxy, 460 for
#: MCPs/packages/db and 826 for MCPs/mcp-shared. Fifteen hundred holds the
#: largest with 1.8 times its length to grow into.
LOG_TAIL_LINES: Final = 1500

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

#: A terminal's colour codes, which vitest writes into the transcript too.
ANSI: Final[re.Pattern[str]] = re.compile(r"\x1b\[[0-9;]*m")

#: vitest's line for one finished test file, colours stripped: a check mark,
#: then ``host  tests/passkey-twin.test.ts (1 test) 139101ms``, its
#: project's label optional, and a failed file marked with a cross or an
#: arrow instead (U+2713, U+00D7, U+276F). A single test's line is indented
#: further and never matches.
VITEST_FILE_MARKS: Final = "".join(chr(point) for point in (0x2713, 0x00D7, 0x276F))
VITEST_FILE: Final[re.Pattern[str]] = re.compile(
    rf"^ [{VITEST_FILE_MARKS}] +(?:\S+ +)?(\S+\.test\.[a-z]+) \(\d+ tests?[^)]*\) (\d+)ms$",
    re.MULTILINE,
)

#: How many of the slowest files a verdict names.
SLOWEST_FILES: Final = 3


class Verdict(TypedDict):
    """One check's outcome, as the thread line carries it.

    Attributes:
        job_id: The queue row.
        project: The project key.
        sha: The commit checked, forty hex.
        node: Where it ran.
        exit_code: The recipe's status (or the install step's, when one
            failed first; the transcript says which).
        banner: Whether :data:`CHECK_BANNER` was read in the tail.
        tests: ``<passed>p/<failed>f`` read off the tail, or ``unread``.
        coverage: The coverage figures read off the tail, or ``unread``.
        budget: The check budget verdict read off the tail, or ``unread``.
        ended: When the check ended, ``YYYY-MM-DDTHH:MM:SSZ``: the time
            the node's result records, or when the runner stopped it. The
            queue row's ``closed_at`` minus this is how long a finished
            check waited to be closed (MCPs board task c1d48330), which no
            other record held.
        log_path: The transcript's absolute path on the node.
        run_id: The fleet ledger's run id.
        workers: The test workers the capacity check granted the run, from
            its ledger row. A suite's parallelism is the one input a
            total must not depend on, and until MCPs board task 022efecf
            no record said what it was: packages/db's total changed with
            it, and nothing on the thread could show two runs differed.
        slowest: The slowest test files and their seconds read off the
            tail (:func:`read_slowest`), or ``unread``: where a total's
            time went, which the total alone does not say (MCPs board
            task 349e25ee).
        ci: The host's CI load over the check
            (:func:`fleet.core.ci_load.read_ci_load`): what else the
            machine ran while the total was taken.
    """

    job_id: str
    project: str
    sha: str
    node: str
    exit_code: int
    banner: bool
    tests: str
    coverage: str
    budget: str
    ended: str
    log_path: str
    run_id: str
    workers: int
    slowest: str
    ci: str


def read_slowest(tail: str) -> str:
    """Read the slowest test files off a transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<file>:<s>s,...`` for the :data:`SLOWEST_FILES` slowest files
        vitest timed, slowest first and seconds rounded, else ``unread``
        (a pytest suite, which times no file, or a run that stopped before
        any file finished).
    """
    # Each file as (-milliseconds, name), so the natural order is slowest
    # first and a tie is broken by name.
    ranked: list[tuple[int, str]] = sorted(
        (-int(match.group(2)), str(match.group(1)))
        for match in VITEST_FILE.finditer(ANSI.sub("", tail))
    )
    if not ranked:
        return "unread"
    return ",".join(f"{name}:{round(-ms / 1000)}s" for ms, name in ranked[:SLOWEST_FILES])


def read_tests(tail: str) -> str:
    """Read the test counts off a transcript tail.

    The passed total is the last ``N passed`` in the tail; the failed count
    is read off that same line only (:data:`FAILED` says why).

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<passed>p/<failed>f`` when a passed count was read (a missing
        failed count is zero: both runners print it only when non-zero),
        else ``unread``.
    """
    totals: re.Match[str] | None = None
    for match in PASSED.finditer(tail):
        totals = match
    if totals is None:
        return "unread"
    line_start = tail.rfind("\n", 0, totals.start()) + 1
    line_end = tail.find("\n", totals.end())
    failed = FAILED.search(tail, line_start, len(tail) if line_end == -1 else line_end)
    return f"{totals.group(1)}p/{'0' if failed is None else failed.group(1)}f"


def read_coverage(tail: str) -> str:
    """Read the coverage figures off a transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``statements=<n>% branches=<n>%`` from vitest's tree row,
        ``total=<n>%`` from pytest-cov's, ``<figure>=<n>% below threshold``
        from vitest's refusal, else ``unread``.
    """
    vitest = VITEST_ALL_FILES.search(tail)
    if vitest is not None:
        return f"statements={vitest.group(1)}% branches={vitest.group(2)}%"
    pytest_total: str | None = None
    for match in PYTEST_TOTAL.finditer(tail):
        pytest_total = match.group(1)
    if pytest_total is not None:
        return f"total={pytest_total}%"
    threshold = VITEST_THRESHOLD.search(tail)
    if threshold is not None:
        return f"{threshold.group(1)}={threshold.group(2)}% below threshold"
    return "unread"


def read_budget(tail: str) -> str:
    """Read the check budget verdict off a transcript tail.

    A check over its budget exits non-zero with every suite green, so its
    row reads like any other red one unless the line says which failure it
    was; this field is how a fleet row reads as the budget's loud failure.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``over:<took>s/<budget>s`` from the over-budget verdict,
        ``within:<took>s/<budget>s`` from the within-budget one, else
        ``unread`` (a package whose check predates the budget, or a run
        that stopped before the verdict printed).
    """
    over = BUDGET_OVER.search(tail)
    if over is not None:
        return f"over:{over.group(1)}s/{over.group(2)}s"
    within = BUDGET_WITHIN.search(tail)
    if within is not None:
        return f"within:{within.group(1)}s/{within.group(2)}s"
    return "unread"


def read_unit_end(tail: str) -> str | None:
    """Read how a build's unit or task ended, when the build itself could not say.

    Args:
        tail: The transcript's last lines.

    Returns:
        The last :data:`fleet.core.linux_unit_end.UNIT_ENDED_MARKER` line,
        marker included and line ending excluded (``FLEET_UNIT_ENDED: unit
        ... ended with systemd result oom-kill (killed KILL) ...``), or
        None for a build that wrote its own status, which is every build
        that was not killed.
    """
    last: str | None = None
    for match in UNIT_ENDED_LINE.finditer(tail):
        last = match.group(0)
    return last


def judge(
    *,
    job_id: str,
    project: str,
    sha: str,
    node: str,
    exit_code: int,
    tail: str,
    ended_unix: int,
    log_path: str,
    run_id: str,
    workers: int,
) -> Verdict:
    """Compose a verdict from a finished run and its transcript tail.

    Args:
        job_id: The queue row.
        project: The project key.
        sha: The commit checked.
        node: Where it ran.
        exit_code: The status the node recorded.
        tail: The transcript's last lines, as the node printed them.
        ended_unix: When the check ended, whole seconds since the epoch.
        log_path: The transcript's absolute path on the node.
        run_id: The fleet ledger's run id.
        workers: The test workers its ledger row records it was granted.

    Returns:
        The verdict, with the counts read or marked unread.
    """
    ended = datetime.datetime.fromtimestamp(ended_unix, tz=datetime.UTC)
    return Verdict(
        job_id=job_id,
        project=project,
        sha=sha,
        node=node,
        exit_code=exit_code,
        banner=CHECK_BANNER in tail,
        tests=read_tests(tail),
        coverage=read_coverage(tail),
        budget=read_budget(tail),
        ended=ended.strftime("%Y-%m-%dT%H:%M:%SZ"),
        log_path=log_path,
        run_id=run_id,
        workers=workers,
        slowest=read_slowest(tail),
        ci=read_ci_load(tail),
    )


def render_verdict(verdict: Verdict) -> str:
    """Render a verdict as the one thread line.

    Args:
        verdict: The verdict.

    Returns:
        ``FLEET-CHECK <job8> <project> sha=<sha> node=<node> exit=<n>
        banner=<yes|no> tests=<...> coverage=<...> budget=<...>
        ended=<time> log=<node>:<path> run=<run id> workers=<n>
        slowest=<...> ci=<...>``, one line, so a closure can quote it and a
        reader can grep a thread for the prefix. ``workers``, ``slowest``
        and ``ci`` come after every field the readers in MCPs' maketools
        and session-audit match, each of which they match one by one.
    """
    return (
        f"{VERDICT_PREFIX} {verdict['job_id'][:8]} {verdict['project']} "
        f"sha={verdict['sha']} node={verdict['node']} exit={verdict['exit_code']} "
        f"banner={'yes' if verdict['banner'] else 'no'} tests={verdict['tests']} "
        f"coverage={verdict['coverage']} budget={verdict['budget']} "
        f"ended={verdict['ended']} log={verdict['node']}:{verdict['log_path']} "
        f"run={verdict['run_id']} workers={verdict['workers']} "
        f"slowest={verdict['slowest']} ci={verdict['ci']}"
    )


__all__ = [
    "ANSI",
    "BUDGET_OVER",
    "BUDGET_WITHIN",
    "CHECK_BANNER",
    "FAILED",
    "LOG_TAIL_LINES",
    "PASSED",
    "PYTEST_TOTAL",
    "SLOWEST_FILES",
    "VERDICT_PREFIX",
    "VITEST_ALL_FILES",
    "VITEST_FILE",
    "VITEST_FILE_MARKS",
    "VITEST_THRESHOLD",
    "Verdict",
    "judge",
    "read_budget",
    "read_coverage",
    "read_slowest",
    "read_tests",
    "read_unit_end",
    "render_verdict",
]
