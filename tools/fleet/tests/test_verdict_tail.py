"""The verdict's tail reaches the totals of the largest package (MCPs board task 93a1c2c0).

vitest prints its per-file coverage table after the test totals, one row per
source file, so the totals sit further from the transcript's end the larger
the package. On colossus on 2026-10-11 they sat 826 lines from the end of
MCPs/mcp-shared's log, and the 200-line tail of the time posted
``tests=unread coverage=unread`` for a run of 10401 passing tests. The
transcript here has that shape, row for row; the dialect prints the last
:data:`fleet.core.verdict.LOG_TAIL_LINES` lines with ``tail -n``, which these
cases take the same way.
"""

from __future__ import annotations

from typing import Final

from fleet.core import dialect, verdict

#: One file's row of vitest's coverage table.
ROW: Final = "  file-{index:04d}.ts     |     100 |      100 |     100 |     100 |"

#: mcp-shared's run of 2026-10-11, its totals 826 lines from the end: the
#: Tests line, the tree row, 822 file rows, the budget and the banner.
TRANSCRIPT: Final = "\n".join(
    [
        " Test Files  731 passed (731)",
        "      Tests  10401 passed | 74 skipped (10475)",
        "All files          |     100 |      100 |     100 |     100 |",
        *(ROW.format(index=index) for index in range(822)),
        "CHECK BUDGET: mcp-shared took 73s of 300s (lint and setup 26s, suite 47s).",
        "=== ALL CHECKS PASSED ===",
    ]
)


def _tail(lines: int) -> str:
    """The transcript as ``tail -n <lines>`` prints it."""
    return "\n".join(TRANSCRIPT.splitlines()[-lines:])


def _judged(tail: str) -> verdict.Verdict:
    """The verdict of a passing run with this tail."""
    return verdict.judge(
        job_id="94558d87-a7da-42c5-aa63-de95043a4bb5",
        project="MCPs/mcp-shared",
        sha="58b7d70c4913f4216f86100ae53867b677854f23",
        node="colossus",
        exit_code=0,
        tail=tail,
        ended_unix=1791706310,
        log_path="/home/user/fleet/stage/logs/MCPs-mcp-shared-colossus-1791706189.log",
        run_id="MCPs-mcp-shared-colossus-1791706189",
    )


def test_the_totals_sit_826_lines_from_the_end_as_measured() -> None:
    lines = TRANSCRIPT.splitlines()
    assert len(lines) - lines.index("      Tests  10401 passed | 74 skipped (10475)") == 826


def test_the_collectors_tail_reads_the_counts_of_the_largest_package() -> None:
    judged = _judged(_tail(verdict.LOG_TAIL_LINES))

    assert judged["tests"] == "10401p/0f"
    assert judged["coverage"] == "statements=100% branches=100%"
    assert judged["budget"] == "within:73s/300s"
    assert judged["banner"] is True


def test_the_two_hundred_line_tail_it_replaces_read_neither_count() -> None:
    """The failure this exists for, so the case above is known to separate."""
    judged = _judged(_tail(200))

    assert (judged["tests"], judged["coverage"]) == ("unread", "unread")


def test_the_linux_dialect_prints_that_many_lines() -> None:
    script = dialect.log_tail_script("/s/run-1", verdict.LOG_TAIL_LINES)

    assert f"tail -n {verdict.LOG_TAIL_LINES} '/s/run-1/" in script
    assert verdict.LOG_TAIL_LINES == 1500
