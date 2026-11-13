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
counts are parsed from the families of runner output this fleet has,
vitest's, pytest's, osrs-lab's Gradle summary and node's built-in runner's
totals, and a count that is not in
the tail is written as ``unread`` rather than guessed: a verdict that invents
a number is worse than one that says it could not read it.
"""

from __future__ import annotations

import datetime
import re
from typing import Final

from typing_extensions import TypedDict

from fleet.core.build_steps import read_build_steps
from fleet.core.check_figures import (
    BudgetOutcome,
    read_budget_reading,
    read_coverage_reading,
    read_test_counts,
)
from fleet.core.linux_unit_end import UNIT_ENDED_LINE
from fleet.core.mutation_verdict import read_mutation
from fleet.core.release_artifact import read_artifact
from fleet.core.run_registry import read_registry
from fleet.core.run_summary import read_run_summary
from fleet.core.slice_load import CI_SLICE, DISPATCH_SLICE, read_slice_load
from fleet.core.vm_tests import read_vm_tests

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

#: How the verdict line spells each budget outcome: ``over:347s/300s``.
BUDGET_WORDS: Final[dict[BudgetOutcome, str]] = {
    BudgetOutcome.EXCEEDED: "over",
    BudgetOutcome.WITHIN: "within",
}

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

#: pytest's ``--durations`` line for one test's call phase, as
#: ``653.99s call     tests/test_service_entry.py::test_...[grandma-api]``.
#: Its setup and teardown lines name the same test and never match, so a
#: pytest suite's slowest entries are its slowest tests (MCPs board task
#: eb6586e4: tools/maketools-execution's verdict read ``slowest=unread``
#: while its transcript timed each deploy case).
PYTEST_CALL: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.(\d\d)s call +(\S+)$", re.MULTILINE)

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
            (:func:`fleet.core.slice_load.read_slice_load`): what else the
            machine ran while the total was taken.
        dispatch: The dispatch share's load over the check, read the same
            way: the CPU-seconds the check, and on a NixOS node its nix
            daemon's builds, took in the share it was placed in, beside
            whatever else ran there (MCPs board task 51e3eac5).
        vmtests: The NixOS VM tests the check built and ran, each with its
            test-script seconds (:func:`fleet.core.vm_tests.read_vm_tests`):
            which tests a run paid for and what each cost, read from the
            whole transcript, not its tail (MCPs board task d6b03ba3).
        generation: The cache generation the build read, and
        began: whether this build began it, and
        phases: each phase's seconds, all three read off the build's last
            line (:mod:`fleet.core.run_summary`): a check that began its
            generation fetched every package it installs, and the phases
            say whether that fetch fell before the budget's clock or inside
            it (MCPs board tasks abb0a10e and 74d10d85).
        steps: The slowest docker build steps the tail timed
            (:func:`fleet.core.build_steps.read_build_steps`), or
            ``unread``: which step inside a build a phase's seconds went to
            (MCPs board task eb6586e4).
        registry: The rolled commit whose registry gave the build its
            install steps, as its launch recorded it beside the run
            (:mod:`fleet.core.run_registry`), ``working-tree`` for a
            ``fleet-run`` from a checkout, or ``unread`` for a run launched
            before the record. A serve on an older roll ran job 0fa14fab's
            install steps and nothing said so (MCPs board task 1fd01f09).
        artifact: The release archive the recipe built, by content
            (:func:`fleet.core.release_artifact.read_artifact`):
            ``<path>:<entries>:<digest>``, ``none`` for a project that
            declares none, an error code with the path for one the recipe
            did not leave readable, or ``unread``. Which commit built a
            deployed jar was otherwise a session's rebuild (MCPs board task
            0c8ace8d).
        mutation: How the package's mutation step ended and its span
            against its own bound
            (:func:`fleet.core.mutation_verdict.read_mutation`), for
            example ``failed:8/453,801s/900s``: a check whose suite passed
            and whose step found surviving mutants otherwise read like any
            other red check (MCPs board task 637f648f).
        results_path: Where the node keeps the check's structured results,
            its runners' JUnit XML and mypy JSON
            (:func:`fleet.core.check_results.retained_results_path`), which
            MCPs reads a run's cases and findings from instead of the
            transcript's text (MCPs board task 25ab85cf); absent on the node
            when the recipe wrote none.
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
    dispatch: str
    vmtests: str
    generation: str
    began: str
    phases: str
    steps: str
    registry: str
    artifact: str
    mutation: str
    results_path: str


def read_slowest(tail: str) -> str:
    """Read the slowest test files off a transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<name>:<s>s,...`` for the :data:`SLOWEST_FILES` slowest entries,
        slowest first and seconds rounded: the files vitest timed and the
        tests pytest's ``--durations`` timed (:data:`PYTEST_CALL`), else
        ``unread`` (a pytest suite run without ``--durations``, or a run
        that stopped before anything finished).
    """
    plain = ANSI.sub("", tail)
    # Each entry as (-milliseconds, name), so the natural order is slowest
    # first and a tie is broken by name.
    ranked: list[tuple[int, str]] = sorted(
        [
            *((-int(match.group(2)), str(match.group(1))) for match in VITEST_FILE.finditer(plain)),
            *(
                (-(int(match.group(1)) * 1000 + int(match.group(2)) * 10), str(match.group(3)))
                for match in PYTEST_CALL.finditer(plain)
            ),
        ]
    )
    if not ranked:
        return "unread"
    return ",".join(f"{name}:{round(-ms / 1000)}s" for ms, name in ranked[:SLOWEST_FILES])


def read_tests(tail: str) -> str:
    """Render the test counts off a transcript tail (:func:`read_test_counts`).

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<passed>p/<failed>f`` when a passed count was read, else
        ``unread``.
    """
    counts = read_test_counts(tail)
    return "unread" if counts is None else f"{counts['passed']}p/{counts['failed']}f"


def read_coverage(tail: str) -> str:
    """Render the coverage figures off a transcript tail (:func:`read_coverage_reading`).

    Args:
        tail: The transcript's last lines.

    Returns:
        The figures as the runner printed them, else ``unread``.
    """
    reading = read_coverage_reading(tail)
    return "unread" if reading is None else reading["rendered"]


def read_budget(tail: str) -> str:
    """Render the check budget verdict off a transcript tail (:func:`read_budget_reading`).

    Args:
        tail: The transcript's last lines.

    Returns:
        ``over:<took>s/<budget>s`` past the budget, ``within:<took>s/<budget>s``
        inside it, else ``unread``.
    """
    reading = read_budget_reading(tail)
    if reading is None:
        return "unread"
    return f"{BUDGET_WORDS[reading['outcome']]}:{reading['seconds']}s/{reading['bound']}s"


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
    results_path: str,
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
        results_path: Where the node keeps the check's structured results.
        run_id: The fleet ledger's run id.
        workers: The test workers its ledger row records it was granted.

    Returns:
        The verdict, with the counts read or marked unread.
    """
    ended = datetime.datetime.fromtimestamp(ended_unix, tz=datetime.UTC)
    summary = read_run_summary(tail)
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
        ci=read_slice_load(tail, CI_SLICE),
        dispatch=read_slice_load(tail, DISPATCH_SLICE),
        vmtests=read_vm_tests(tail),
        generation=summary["generation"],
        began=summary["began"],
        phases=summary["phases"],
        steps=read_build_steps(ANSI.sub("", tail)),
        registry=read_registry(tail),
        artifact=read_artifact(tail),
        mutation=read_mutation(ANSI.sub("", tail)),
        results_path=results_path,
    )


def render_verdict(verdict: Verdict) -> str:
    """Render a verdict as the one thread line.

    Args:
        verdict: The verdict.

    Returns:
        ``FLEET-CHECK <job8> <project> sha=<sha> node=<node> exit=<n>
        banner=<yes|no> tests=<...> coverage=<...> budget=<...>
        ended=<time> log=<node>:<path> run=<run id> workers=<n>
        slowest=<...> ci=<...> dispatch=<...> vmtests=<...> generation=<...> began=<...>
        phases=<...>
        steps=<...> registry=<commit> artifact=<...> mutation=<...>
        results=<node>:<path>``,
        one line, so a closure can quote it and a reader can grep a thread
        for the prefix. ``workers`` and every field after it come after every
        field the readers in MCPs' maketools and session-audit match, each
        of which they match one by one.
    """
    return (
        f"{VERDICT_PREFIX} {verdict['job_id'][:8]} {verdict['project']} "
        f"sha={verdict['sha']} node={verdict['node']} exit={verdict['exit_code']} "
        f"banner={'yes' if verdict['banner'] else 'no'} tests={verdict['tests']} "
        f"coverage={verdict['coverage']} budget={verdict['budget']} "
        f"ended={verdict['ended']} log={verdict['node']}:{verdict['log_path']} "
        f"run={verdict['run_id']} workers={verdict['workers']} "
        f"slowest={verdict['slowest']} ci={verdict['ci']} dispatch={verdict['dispatch']} "
        f"vmtests={verdict['vmtests']} "
        f"generation={verdict['generation']} began={verdict['began']} "
        f"phases={verdict['phases']} steps={verdict['steps']} "
        f"registry={verdict['registry']} artifact={verdict['artifact']} "
        f"mutation={verdict['mutation']} "
        f"results={verdict['node']}:{verdict['results_path']}"
    )


__all__ = [
    "ANSI",
    "BUDGET_WORDS",
    "CHECK_BANNER",
    "LOG_TAIL_LINES",
    "PYTEST_CALL",
    "SLOWEST_FILES",
    "VERDICT_PREFIX",
    "VITEST_FILE",
    "VITEST_FILE_MARKS",
    "Verdict",
    "judge",
    "read_budget",
    "read_coverage",
    "read_slowest",
    "read_tests",
    "read_unit_end",
    "render_verdict",
]
