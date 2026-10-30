"""The CI load a check's build prints and the verdict reads (MCPs board task 349e25ee).

The lines run here under ``sh -eu``, as the builds run them, against a
``cpu.stat`` this case writes in place of the CI slice's, between a start
and an end that stand where the recipe's phase does. The tails are lavender's
own: MCPs/execution-deploy's per-file lines from its 308 s run (fleet job
5d96ec2e), colours and all.
"""

from __future__ import annotations

import pathlib
import subprocess

from fleet.contracts.runner_slice import CI_SLICE_NAME
from fleet.core import verdict
from fleet.core.ci_load import (
    CI_CPU_STAT,
    CI_LOAD_GONE,
    CI_LOAD_NONE,
    read_ci_load,
    sh_ci_load_end_lines,
    sh_ci_load_start_lines,
)

#: vitest's marks: a file passed, failed by a test, failed, and skipped.
PASSED, FAILED, CROSSED, SKIPPED = (chr(point) for point in (0x2713, 0x276F, 0x00D7, 0x2193))

#: Two of the run's file lines and one of a single test's, as the log holds them.
LAVENDER_TAIL = (
    f" \x1b[32m{PASSED}\x1b[39m \x1b[30m\x1b[43m host \x1b[49m\x1b[39m "
    "tests/cascade-dependencies.test.ts "
    "\x1b[2m(\x1b[22m\x1b[2m2 tests\x1b[22m\x1b[2m)\x1b[22m"
    "\x1b[33m 55946\x1b[2mms\x1b[22m\x1b[39m\n"
    f"     {PASSED} registers the first passkey of an account with none 16045ms\n"
    f" \x1b[32m{PASSED}\x1b[39m \x1b[30m\x1b[43m host \x1b[49m\x1b[39m tests/passkey-twin.test.ts "
    "\x1b[2m(\x1b[22m\x1b[2m1 test\x1b[22m\x1b[2m)\x1b[22m"
    "\x1b[33m 139101\x1b[2mms\x1b[22m\x1b[39m\n"
    f" {FAILED} tests/served/proxy.test.ts (12 tests | 1 failed) 7480ms\n"
    f" {PASSED} tests/served/endpoints.test.ts (10 tests) 16ms\n"
    f" {SKIPPED} tests/served/skipped.test.ts (1 test | 1 skipped)\n"
)


#: The 308 s run's ledger id.
RUN = "MCPs-execution-deploy-lavender-1793327005"


def _run(tmp_path: pathlib.Path, before: str | None, after: str | None) -> str:
    """Run the start lines, change the stat file, run the end lines, and read the transcript.

    Args:
        tmp_path: The case's directory.
        before: The stat file's text at the start, or None for no file.
        after: Its text at the end, or None for no file.

    Returns:
        What the lines appended to the transcript.
    """
    stat = tmp_path / "cpu.stat"
    log = tmp_path / "check.log"
    log.write_text("", encoding="utf-8")
    if before is not None:
        stat.write_text(before, encoding="utf-8")
    change = f"rm -f '{stat}'" if after is None else f"printf '%s' '{after}' > '{stat}'"
    script = "\n".join(
        [*sh_ci_load_start_lines(str(stat)), change, *sh_ci_load_end_lines(str(stat), str(log))]
    )
    subprocess.run(["sh", "-eu", "-c", script], check=True, timeout=30)
    return log.read_text(encoding="utf-8")


class TestTheLinesOnTheNode:
    def test_a_slice_that_ran_through_the_check_prints_its_cpu_seconds_and_the_wall(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed = _run(
            tmp_path,
            "usage_usec 1000000000\nuser_usec 7\n",
            "usage_usec 1412500000\nuser_usec 9\n",
        )
        assert printed == (
            f"FLEET_CI_LOAD: {CI_SLICE_NAME} used 412 CPU-seconds in the 0 s the check ran\n"
        )
        assert read_ci_load(printed) == "412s/0s"

    def test_a_node_with_no_slice_says_so(self, tmp_path: pathlib.Path) -> None:
        printed = _run(tmp_path, None, None)
        assert printed == f"{CI_LOAD_NONE}\n"
        assert read_ci_load(printed) == "none"

    def test_a_slice_gone_by_the_end_says_so_and_the_build_goes_on(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed = _run(tmp_path, "usage_usec 5\n", None)
        assert printed == f"{CI_LOAD_GONE}\n"
        assert read_ci_load(printed) == "gone"

    def test_the_node_s_lines_read_the_ci_slice(self) -> None:
        expected = f"/sys/fs/cgroup/{CI_SLICE_NAME}/cpu.stat"
        assert expected == CI_CPU_STAT


class TestTheReading:
    def test_a_tail_without_the_line_is_unread_and_the_last_line_counts(self) -> None:
        assert read_ci_load("make: *** [check] Error 2\n") == "unread"
        two = (
            f"FLEET_CI_LOAD: {CI_SLICE_NAME} used 1 CPU-seconds in the 2 s the check ran\n"
            f"FLEET_CI_LOAD: {CI_SLICE_NAME} used 30 CPU-seconds in the 40 s the check ran\n"
        )
        assert read_ci_load(two) == "30s/40s"

    def test_the_slowest_files_are_read_through_the_colours_slowest_first(self) -> None:
        assert verdict.read_slowest(LAVENDER_TAIL) == (
            "tests/passkey-twin.test.ts:139s,tests/cascade-dependencies.test.ts:56s,"
            "tests/served/proxy.test.ts:7s"
        )

    def test_a_tie_is_broken_by_name_and_a_tail_timing_no_file_is_unread(self) -> None:
        tied = (
            f" {PASSED} tests/b.test.ts (1 test) 2000ms\n"
            f" {CROSSED} tests/a.test.ts (1 test) 2000ms\n"
        )
        assert verdict.read_slowest(tied) == "tests/a.test.ts:2s,tests/b.test.ts:2s"
        assert verdict.read_slowest("771 passed, 4 skipped in 20.06s\n") == "unread"

    def test_both_reach_the_line_a_closure_cites(self) -> None:
        tail = (
            LAVENDER_TAIL
            + f"FLEET_CI_LOAD: {CI_SLICE_NAME} used 2111 CPU-seconds in the 664 s the check ran\n"
            + "CHECK OVER BUDGET: execution took 308s, over the 300s budget by 8s.\n"
        )
        judged = verdict.judge(
            job_id="5d96ec2e-6d19-4f4c-a12f-e5b73464e73c",
            project="MCPs/execution-deploy",
            sha="ed5aed5be0000000000000000000000000000000",
            node="lavender",
            exit_code=2,
            tail=tail,
            ended_unix=1793328000,
            log_path=f"/home/corvis/fleet/stage/logs/{RUN}.log",
            run_id=RUN,
            workers=3,
        )
        assert verdict.render_verdict(judged).endswith(
            "budget=over:308s/300s ended=2026-10-30T02:40:00Z "
            f"log=lavender:/home/corvis/fleet/stage/logs/{RUN}.log run={RUN} workers=3 "
            "slowest=tests/passkey-twin.test.ts:139s,tests/cascade-dependencies.test.ts:56s,"
            "tests/served/proxy.test.ts:7s ci=2111s/664s"
        )
