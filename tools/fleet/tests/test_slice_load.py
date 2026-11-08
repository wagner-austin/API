"""The slice loads a check's build prints and the verdict reads.

MCPs board tasks 349e25ee (the CI slice) and 51e3eac5 (the dispatch share).
The lines run here under ``sh -eu``, as the builds run them, against
``cpu.stat`` files this case writes in place of each slice's, between a
start and an end that stand where the recipe's phase does. The tails are
lavender's own: MCPs/execution-deploy's per-file lines from its 308 s run
(fleet job 5d96ec2e), colours and all.
"""

from __future__ import annotations

import pathlib
import subprocess
import time

from fleet.contracts.check_unit import DISPATCH_SLICE_NAME
from fleet.contracts.runner_slice import CI_SLICE_NAME
from fleet.core import verdict
from fleet.core.slice_load import (
    CI_SLICE,
    DISPATCH_SLICE,
    MEASURED_SLICES,
    SLICE_LOAD_MARKER,
    MeasuredSlice,
    read_slice_load,
    sh_slice_load_end_lines,
    sh_slice_load_start_lines,
    slice_gone_line,
    slice_none_line,
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


def _run(
    tmp_path: pathlib.Path, stats: dict[str, tuple[str | None, str | None]]
) -> tuple[str, range]:
    """Run the start lines, change each stat file, run the end lines, and read the transcript.

    THE WALL IS WHAT THE CLOCK SAID, NOT 0 (MCPs board task 5280a25d). The
    lines read ``date +%s`` at the start and again at each slice's end, so a
    run of a few milliseconds that crosses a second's boundary prints 1 s:
    fleet job c79448c2 on colossus at API bb5b75ef1 failed the case that
    expected 0 s on exactly that. The clock is read here, in whole seconds,
    before the script starts and after it ends, and every wall the script's
    own reads could have made lies between them.

    Args:
        tmp_path: The case's directory.
        stats: Each measured slice's field, with its stat file's text at the
            start and at the end, None for no file.

    Returns:
        What the lines appended to the transcript, and every whole-second
        wall a line of it can carry.
    """
    log = tmp_path / "check.log"
    log.write_text("", encoding="utf-8")
    slices: list[MeasuredSlice] = []
    changes: list[str] = []
    for measured in MEASURED_SLICES:
        before, after = stats[measured["field"]]
        stat = tmp_path / f"{measured['field']}.cpu.stat"
        if before is not None:
            stat.write_text(before, encoding="utf-8")
        changes.append(f"rm -f '{stat}'" if after is None else f"printf '%s' '{after}' > '{stat}'")
        slices.append(
            MeasuredSlice(name=measured["name"], cpu_stat=str(stat), field=measured["field"])
        )
    script = "\n".join(
        [
            *sh_slice_load_start_lines(tuple(slices)),
            *changes,
            *sh_slice_load_end_lines(tuple(slices), str(log)),
        ]
    )
    started = int(time.time())
    subprocess.run(["sh", "-eu", "-c", script], check=True, timeout=30)
    return log.read_text(encoding="utf-8"), range(int(time.time()) - started + 1)


def _used(name: str, cpu: int, wall: int) -> str:
    """One slice's line as the end lines print it.

    Args:
        name: The slice's unit name.
        cpu: Its CPU-seconds.
        wall: The check's wall seconds.

    Returns:
        The line, with its newline.
    """
    return f"FLEET_SLICE_LOAD: {name} used {cpu} CPU-seconds in the {wall} s the check ran\n"


class TestTheLinesOnTheNode:
    def test_each_slice_that_ran_through_the_check_prints_its_cpu_seconds_and_the_wall(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed, walls = _run(
            tmp_path,
            {
                "ci": ("usage_usec 1000000000\nuser_usec 7\n", "usage_usec 1412500000\n"),
                "dispatch": ("usage_usec 5000000\n", "usage_usec 1468800000\nuser_usec 1\n"),
            },
        )

        # The CI slice's end is read first, so its wall is never the later.
        read = [
            (ci, dispatch)
            for ci in walls
            for dispatch in walls
            if ci <= dispatch
            and printed
            == _used(CI_SLICE_NAME, 412, ci) + _used(DISPATCH_SLICE_NAME, 1463, dispatch)
        ]
        assert len(read) == 1
        ci, dispatch = read[0]
        assert read_slice_load(printed, CI_SLICE) == f"412s/{ci}s"
        assert read_slice_load(printed, DISPATCH_SLICE) == f"1463s/{dispatch}s"

    def test_a_node_without_a_slice_says_so_for_that_slice_alone(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed, walls = _run(
            tmp_path, {"ci": (None, None), "dispatch": ("usage_usec 0\n", "usage_usec 0\n")}
        )

        read = [
            wall
            for wall in walls
            if printed == f"{slice_none_line(CI_SLICE)}\n" + _used(DISPATCH_SLICE_NAME, 0, wall)
        ]
        assert len(read) == 1
        assert read_slice_load(printed, CI_SLICE) == "none"
        assert read_slice_load(printed, DISPATCH_SLICE) == f"0s/{read[0]}s"

    def test_a_slice_gone_by_the_end_says_so_and_the_build_goes_on(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed, _ = _run(tmp_path, {"ci": (None, None), "dispatch": ("usage_usec 5\n", None)})

        assert printed == f"{slice_none_line(CI_SLICE)}\n{slice_gone_line(DISPATCH_SLICE)}\n"
        assert read_slice_load(printed, DISPATCH_SLICE) == "gone"
        assert read_slice_load(printed, CI_SLICE) == "none"

    def test_the_node_s_lines_read_the_ci_slice_and_dispatch_s_share(self) -> None:
        assert [measured["cpu_stat"] for measured in MEASURED_SLICES] == [
            f"/sys/fs/cgroup/{CI_SLICE_NAME}/cpu.stat",
            f"/sys/fs/cgroup/system.slice/{DISPATCH_SLICE_NAME}/cpu.stat",
        ]
        assert [measured["field"] for measured in MEASURED_SLICES] == ["ci", "dispatch"]


class TestTheReading:
    def test_a_tail_without_the_line_is_unread_and_the_last_line_counts(self) -> None:
        assert read_slice_load("make: *** [check] Error 2\n", CI_SLICE) == "unread"
        two = (
            f"FLEET_SLICE_LOAD: {CI_SLICE_NAME} used 1 CPU-seconds in the 2 s the check ran\n"
            f"FLEET_SLICE_LOAD: {CI_SLICE_NAME} used 30 CPU-seconds in the 40 s the check ran\n"
        )
        assert read_slice_load(two, CI_SLICE) == "30s/40s"
        assert read_slice_load(two, DISPATCH_SLICE) == "unread"

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

    def test_a_pytest_suite_run_with_durations_names_its_slowest_tests(self) -> None:
        """The durations block of fleet job 5844f96c, tools/maketools-execution
        on loki with its build cache emptied (MCPs board task eb6586e4): only
        each test's call line counts, never its setup or teardown."""
        case = (
            "tests/test_service_entry.py::test_the_service_deploys_and_passes_its_own_healthcheck"
        )
        tail = (
            "============================= slowest durations ==============================\n"
            f"653.99s call     {case}[grandma-api]\n"
            f"584.83s call     {case}[turkic-api]\n"
            f"581.73s call     {case}[transcript-api]\n"
            f"512.52s call     {case}[opportunity-radar-api]\n"
            f"1.14s teardown {case}[grandma-api]\n"
            f"900.00s setup    {case}[qr-api]\n"
            "======================== 8 passed in 664.10s (0:11:04) =========================\n"
        )
        assert verdict.read_slowest(tail) == (
            f"{case}[grandma-api]:654s,{case}[turkic-api]:585s,{case}[transcript-api]:582s"
        )

    def test_every_reading_reaches_the_line_a_closure_cites(self) -> None:
        tail = (
            LAVENDER_TAIL
            + f"{SLICE_LOAD_MARKER}: {CI_SLICE_NAME} used 2111 CPU-seconds in the 664 s the "
            + "check ran\n"
            + f"{SLICE_LOAD_MARKER}: {DISPATCH_SLICE_NAME} used 940 CPU-seconds in the 664 s the "
            + "check ran\n"
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
            results_path=f"/home/corvis/fleet/stage/logs/{RUN}.results",
            run_id=RUN,
            workers=3,
        )
        assert verdict.render_verdict(judged).endswith(
            "budget=over:308s/300s ended=2026-10-30T02:40:00Z "
            f"log=lavender:/home/corvis/fleet/stage/logs/{RUN}.log run={RUN} workers=3 "
            "slowest=tests/passkey-twin.test.ts:139s,tests/cascade-dependencies.test.ts:56s,"
            "tests/served/proxy.test.ts:7s ci=2111s/664s dispatch=940s/664s vmtests=unread "
            "generation=unread began=unread phases=unread steps=unread registry=unread "
            f"artifact=unread results=lavender:/home/corvis/fleet/stage/logs/{RUN}.results"
        )
