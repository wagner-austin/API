"""Reap pytest/xdist processes that a test run left behind.

WHY THIS EXISTS. ``pytest -n auto`` spawns one execnet worker per core, and
in the torch projects every worker imports torch, which RESERVES ~1.1 GB of
address space on import. Working set stays tiny, so a wedged run looks
harmless while it holds tens of GB of commit.

The wedge: pytest-timeout has no SIGALRM on Windows, so it uses the
``thread`` method, whose expiry path is literally ``os._exit(1)``. That
kills a worker with no cleanup and no protocol shutdown, so the xdist
controller is left waiting on a channel that will never answer, and the
remaining workers block forever in ``sys.stdin.readline()`` -- which is
their entire command line::

    python -u -c "import sys;exec(eval(sys.stdin.readline()))"

2026-08-19 incident: three ``make check`` runs on 08-18 left 101 live
processes holding ~112 GB of commit for 23 hours. The box threw an "Out of
Virtual Memory" popup that night, and a 22,906-feature SIRIUS annotation run
died at 67% the next morning.

TWO MODES. Descendants of a pid are ours by construction, so no age or idle
gate is applied. A standalone sweep of a PREVIOUS run's wreckage is gated
twice: by age, and by an idle check.

THE IDLE GATE IS AGGREGATE. Age alone cannot distinguish a 90-minute suite
from a 90-minute wedge, so the sweep samples cumulative CPU across the
candidate set twice and reaps only if the WHOLE SET is idle. Aggregate, not
per-process, because a live run legitimately contains idle workers waiting
for work; killing those would break it. Measured separation on a controlled
pair: 4.984 s (busy) vs 0.000 s (blocked).

Ported from ``scripts/reap-test-processes.ps1`` on 2026-09-20 (board task
33bb86ce); the logic is platform-neutral over :class:`ProcessRow`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

from maketools import _test_hooks
from maketools.processes import ProcessRow

#: How far up or down the tree the walks go; a cycle in a stale snapshot
#: (a pid reused by a descendant) would otherwise never end.
MAX_DEPTH: Final[int] = 15

#: The names a run's processes carry, without a Windows ``.exe``.
TEST_PROCESS_NAMES: Final[frozenset[str]] = frozenset({"pytest", "poetry"})

#: Sweep defaults: minimum age, sample gap, and the aggregate CPU that still
#: reads as idle.
DEFAULT_OLDER_THAN_MINUTES: Final[int] = 60
DEFAULT_IDLE_SAMPLE_SECONDS: Final[int] = 5
DEFAULT_IDLE_THRESHOLD_SECONDS: Final[float] = 0.10


class ReapReport(TypedDict):
    """What a reap did.

    Attributes:
        targets: How many processes were selected.
        killed: How many ended, including ones already gone.
        failed: How many are still alive after the attempt.
        commit_mb: The commit the targets held, summed.
    """

    targets: int
    killed: int
    failed: int
    commit_mb: float


def is_test_process(row: ProcessRow) -> bool:
    """Whether a row is one of the three shapes a run leaves behind.

    The poetry launcher, the pytest controller and the execnet workers: a
    python, pytest or poetry executable whose command line mentions pytest
    or carries the worker bootstrap.

    Args:
        row: The process.

    Returns:
        True for a candidate.
    """
    name = row["name"].lower().removesuffix(".exe")
    if not (name.startswith("python") or name in TEST_PROCESS_NAMES):
        return False
    return "pytest" in row["command_line"] or "exec(eval" in row["command_line"]


def descendants(rows: Sequence[ProcessRow], root_pid: int, depth: int = 0) -> list[ProcessRow]:
    """Every process under a pid, deepest first.

    Deepest first because killing a parent first can leave a child
    reparented and missed.

    Args:
        rows: The snapshot.
        root_pid: The ancestor.
        depth: Recursion depth so far.

    Returns:
        The descendants, each preceded by its own descendants.
    """
    if depth > MAX_DEPTH:
        return []
    found: list[ProcessRow] = []
    for row in rows:
        if row["parent_pid"] == root_pid and row["pid"] != root_pid:
            found.extend(descendants(rows, row["pid"], depth + 1))
            found.append(row)
    return found


def ancestor_depth(by_pid: dict[int, ProcessRow], row: ProcessRow) -> int:
    """How many ancestors a process has in the snapshot.

    Args:
        by_pid: The snapshot indexed by pid.
        row: The process.

    Returns:
        The count, capped at :data:`MAX_DEPTH`.
    """
    depth = 0
    current: ProcessRow | None = row
    while current is not None and depth < MAX_DEPTH:
        current = by_pid.get(current["parent_pid"])
        depth += 1
    return depth


def in_project(by_pid: dict[int, ProcessRow], row: ProcessRow, needle: str) -> bool:
    """Whether a process or one of its ancestors names the project.

    An execnet worker's own command line does NOT name the project; it is
    the generic stdin bootstrap. Membership is resolved through the
    ancestry instead.

    Args:
        by_pid: The snapshot indexed by pid.
        row: The process.
        needle: The project directory's leaf name.

    Returns:
        True when the needle appears in a command line or executable path
        within :data:`MAX_DEPTH` ancestors.
    """
    current: ProcessRow | None = row
    for _ in range(MAX_DEPTH):
        if current is None:
            return False
        if needle in current["command_line"] or needle in current["executable"]:
            return True
        current = by_pid.get(current["parent_pid"])
    return False


def kill_targets(targets: Sequence[ProcessRow]) -> ReapReport:
    """Terminate every target, counting the outcomes.

    Already gone (a parent's death took it) is success, not failure. A
    single stubborn pid must not strand the rest of the sweep, so the loop
    continues past a refusal and the report carries the count.

    Args:
        targets: The processes, deepest first.

    Returns:
        The report.
    """
    commit_mb = sum(row["commit_mb"] for row in targets)
    killed = 0
    failed = 0
    for row in targets:
        try:
            _test_hooks.kill(row["pid"])
            killed += 1
        except OSError as error:
            if _test_hooks.process_alive(row["pid"]):
                _test_hooks.write_error(
                    f"reap: could not kill pid={row['pid']} {row['name']}: {error}"
                )
                failed += 1
            else:
                killed += 1
    _test_hooks.write_line(
        f"reap: killed {killed}, failed {failed}, ~{commit_mb:,.0f} MB commit reclaimed"
    )
    return ReapReport(targets=len(targets), killed=killed, failed=failed, commit_mb=commit_mb)


def reap_descendants(root_pid: int) -> ReapReport:
    """Kill every test process under a pid; they are ours by construction.

    Args:
        root_pid: The launcher.

    Returns:
        The report; ``targets`` is 0 when nothing was left.
    """
    _test_hooks.write_line(f"reap: descendants of pid {root_pid}")
    rows = _test_hooks.process_table()
    targets = [row for row in descendants(rows, root_pid) if is_test_process(row)]
    if not targets:
        _test_hooks.write_line("reap: nothing to reap.")
        return ReapReport(targets=0, killed=0, failed=0, commit_mb=0.0)
    return kill_targets(targets)


def sweep_stale(
    project: Path,
    *,
    older_than_minutes: int = DEFAULT_OLDER_THAN_MINUTES,
    idle_sample_seconds: int = DEFAULT_IDLE_SAMPLE_SECONDS,
    idle_threshold_seconds: float = DEFAULT_IDLE_THRESHOLD_SECONDS,
) -> ReapReport:
    """Reap a previous run's wreckage, if the whole candidate set is idle.

    Args:
        project: The project directory; membership matches on its leaf name,
            because full paths vary in case and separator between the
            Makefile's cwd, poetry's argv and the venv's executable path.
        older_than_minutes: Minimum age before a process is a candidate.
        idle_sample_seconds: Gap between the two CPU samples.
        idle_threshold_seconds: Aggregate CPU over the gap that still reads
            as idle; anything above aborts the sweep as a live run.

    Returns:
        The report; ``targets`` is 0 when nothing was stale or the set was
        busy.
    """
    needle = project.resolve().name
    _test_hooks.write_line(
        f"reap: sweep stale (project '{needle}', older than {older_than_minutes} min)"
    )
    rows = _test_hooks.process_table()
    by_pid = {row["pid"]: row for row in rows}
    cutoff = _test_hooks.now() - older_than_minutes * 60
    candidates = [
        row
        for row in rows
        if is_test_process(row) and row["created_unix"] < cutoff and in_project(by_pid, row, needle)
    ]
    if not candidates:
        _test_hooks.write_line("reap: nothing stale.")
        return ReapReport(targets=0, killed=0, failed=0, commit_mb=0.0)
    before = {row["pid"]: row["cpu_seconds"] for row in candidates}
    _test_hooks.sleep(idle_sample_seconds)
    after = _test_hooks.process_table()
    delta = sum(row["cpu_seconds"] - before[row["pid"]] for row in after if row["pid"] in before)
    _test_hooks.write_line(
        f"reap: {len(candidates)} candidate(s), CPU delta over {idle_sample_seconds}s = "
        f"{delta:.3f}s (idle threshold {idle_threshold_seconds}s)"
    )
    if delta > idle_threshold_seconds:
        _test_hooks.write_line(
            "reap: ABORTING - these processes are doing work, so this is a LIVE run, not wreckage."
        )
        return ReapReport(targets=0, killed=0, failed=0, commit_mb=0.0)

    def depth(row: ProcessRow) -> int:
        return ancestor_depth(by_pid, row)

    targets = sorted(candidates, key=depth, reverse=True)
    _test_hooks.write_line(
        f"reap: {len(targets)} process(es) holding "
        f"{sum(row['commit_mb'] for row in targets):,.0f} MB commit"
    )
    return kill_targets(targets)


__all__ = [
    "DEFAULT_IDLE_SAMPLE_SECONDS",
    "DEFAULT_IDLE_THRESHOLD_SECONDS",
    "DEFAULT_OLDER_THAN_MINUTES",
    "MAX_DEPTH",
    "TEST_PROCESS_NAMES",
    "ReapReport",
    "ancestor_depth",
    "descendants",
    "in_project",
    "is_test_process",
    "kill_targets",
    "reap_descendants",
    "sweep_stale",
]
