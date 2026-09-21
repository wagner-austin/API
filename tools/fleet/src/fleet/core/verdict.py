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

import re
from typing import Final

from typing_extensions import TypedDict

#: The line every package's ``make check`` prints last when it passed.
CHECK_BANNER: Final = "=== ALL CHECKS PASSED ==="

#: The line's opening token, so a thread can be grepped for verdicts.
VERDICT_PREFIX: Final = "FLEET-CHECK"

#: How many lines of the transcript the collector reads for the counts: the
#: summary tables sit at the end, and two hundred lines holds vitest's
#: per-file coverage table for the largest package here with room over.
LOG_TAIL_LINES: Final = 200

#: ``N passed`` as both vitest (``Tests  887 passed (887)``) and pytest
#: (``1166 passed in 12.3s``) print it; the LAST occurrence is the suite's
#: own total, after any per-file lines.
PASSED: Final[re.Pattern[str]] = re.compile(r"(\d+) passed")

#: ``N failed``, the same two families.
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
        log_path: The transcript's absolute path on the node.
        run_id: The fleet ledger's run id.
    """

    job_id: str
    project: str
    sha: str
    node: str
    exit_code: int
    banner: bool
    tests: str
    coverage: str
    log_path: str
    run_id: str


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    """The integer of the pattern's last match, or None.

    Args:
        pattern: A pattern with one integer group.
        text: The transcript tail.

    Returns:
        The last match's integer, or None when the pattern is absent.
    """
    last: str | None = None
    for match in pattern.finditer(text):
        last = match.group(1)
    if last is None:
        return None
    return int(last)


def read_tests(tail: str) -> str:
    """Read the test counts off a transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<passed>p/<failed>f`` when a passed count was read (a missing
        failed count is zero: both runners print it only when non-zero),
        else ``unread``.
    """
    passed = _last_int(PASSED, tail)
    if passed is None:
        return "unread"
    failed = _last_int(FAILED, tail)
    return f"{passed}p/{0 if failed is None else failed}f"


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


def judge(
    *,
    job_id: str,
    project: str,
    sha: str,
    node: str,
    exit_code: int,
    tail: str,
    log_path: str,
    run_id: str,
) -> Verdict:
    """Compose a verdict from a finished run and its transcript tail.

    Args:
        job_id: The queue row.
        project: The project key.
        sha: The commit checked.
        node: Where it ran.
        exit_code: The status the node recorded.
        tail: The transcript's last lines, as the node printed them.
        log_path: The transcript's absolute path on the node.
        run_id: The fleet ledger's run id.

    Returns:
        The verdict, with the counts read or marked unread.
    """
    return Verdict(
        job_id=job_id,
        project=project,
        sha=sha,
        node=node,
        exit_code=exit_code,
        banner=CHECK_BANNER in tail,
        tests=read_tests(tail),
        coverage=read_coverage(tail),
        log_path=log_path,
        run_id=run_id,
    )


def render_verdict(verdict: Verdict) -> str:
    """Render a verdict as the one thread line.

    Args:
        verdict: The verdict.

    Returns:
        ``FLEET-CHECK <job8> <project> sha=<sha> node=<node> exit=<n>
        banner=<yes|no> tests=<...> coverage=<...> log=<node>:<path>
        run=<run id>``, one line, so a closure can quote it and a reader can
        grep a thread for the prefix.
    """
    return (
        f"{VERDICT_PREFIX} {verdict['job_id'][:8]} {verdict['project']} "
        f"sha={verdict['sha']} node={verdict['node']} exit={verdict['exit_code']} "
        f"banner={'yes' if verdict['banner'] else 'no'} tests={verdict['tests']} "
        f"coverage={verdict['coverage']} log={verdict['node']}:{verdict['log_path']} "
        f"run={verdict['run_id']}"
    )


__all__ = [
    "CHECK_BANNER",
    "FAILED",
    "LOG_TAIL_LINES",
    "PASSED",
    "PYTEST_TOTAL",
    "VERDICT_PREFIX",
    "VITEST_ALL_FILES",
    "VITEST_THRESHOLD",
    "Verdict",
    "judge",
    "read_coverage",
    "read_tests",
    "render_verdict",
]
