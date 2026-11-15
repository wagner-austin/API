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
from fleet.core.container_cpu import container_cpus_line
from fleet.core.linux_isolated_build import EXEC_LANE_CGROUP
from fleet.core.slice_load import (
    CGROUP_ROOT,
    CHECK_CGROUP,
    CI_SLICE,
    DISPATCH_SLICE,
    MEASURED_SLICES,
    OWN_CGROUP,
    PRESSURE_MARKER,
    PRESSURES,
    MeasuredSlice,
    PressureCgroup,
    pressure_line,
    pressure_none_line,
    read_run_pressure,
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

#: Where lavender's execdocker manager sat on 2026-11-15, as ``systemctl show`` printed it.
LANE_MANAGER = "/system.slice/system-dispatch.slice/system-dispatch-checks.slice/user@1001.service"

#: A ``systemctl`` answering only the lane cgroup's exact question, as lavender's did.
SYSTEMCTL = (
    "#!/bin/sh\n"
    '[ "$*" = "show --property=ControlGroup --value user@1001.service" ] || exit 1\n'
    f"printf '%s\\n' '{LANE_MANAGER}'\n"
)


def _psi(some: int) -> str:
    """One resource's pressure file as the kernel writes it, with this ``some`` total in µs."""
    return (
        f"some avg10=0.00 avg60=0.00 avg300=0.00 total={some}\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
    )


def _check_cgroup(tmp_path: pathlib.Path, root: pathlib.Path) -> tuple[PressureCgroup, str]:
    """:data:`CHECK_CGROUP` reading this case's stand-in for ``/proc/self/cgroup``.

    Returns:
        The cgroup, and its directory below ``root``.
    """
    own_cgroup = tmp_path / "self-cgroup"
    own_cgroup.write_text("0::/system.slice/fleet-check.service\n", encoding="utf-8")
    locate = CHECK_CGROUP["locate"].replace(OWN_CGROUP, str(own_cgroup))
    return (
        PressureCgroup(field=CHECK_CGROUP["field"], subject=CHECK_CGROUP["subject"], locate=locate),
        "system.slice/fleet-check.service",
    )


def _run(
    tmp_path: pathlib.Path,
    stats: dict[str, tuple[str | None, str | None]],
    pressure: dict[str, tuple[int | None, int | None]],
    lane: bool,
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
        pressure: Each resource of :data:`PRESSURES` the measured cgroup
            keeps a pressure file for, with its ``some`` total at the start
            and at the end, None for no file; empty for a cgroup without PSI.
        lane: Measure :data:`EXEC_LANE_CGROUP`, located by :data:`SYSTEMCTL`
            with ``exec_uid`` 1001, rather than the build's own cgroup.

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
    root = tmp_path / "cgroup"
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "systemctl").write_text(SYSTEMCTL, encoding="utf-8")
    (tools / "systemctl").chmod(0o755)
    if lane:
        pressure_of, directory = EXEC_LANE_CGROUP, LANE_MANAGER.removeprefix("/")
    else:
        pressure_of, directory = _check_cgroup(tmp_path, root)
    cgroup = root / directory
    cgroup.mkdir(parents=True)
    for resource, (before_total, after_total) in pressure.items():
        path = cgroup / f"{resource}.pressure"
        if before_total is not None:
            path.write_text(_psi(before_total), encoding="utf-8")
        changes.append(
            f"rm -f '{path}'"
            if after_total is None
            else f"printf '%s' '{_psi(after_total)}' > '{path}'"
        )
    script = "\n".join(
        [
            f"PATH='{tools}':\"$PATH\"",
            "exec_uid=1001",
            *sh_slice_load_start_lines(tuple(slices), str(root), pressure_of),
            *changes,
            *sh_slice_load_end_lines(tuple(slices), str(log), pressure_of),
        ]
    )
    started = int(time.time())
    subprocess.run(["sh", "-eu", "-c", script], check=True, timeout=30)
    return log.read_text(encoding="utf-8"), range(int(time.time()) - started + 1)


def _used(name: str, cpu: int, throttled: int, wall: int) -> str:
    """One slice's line as the end lines print it.

    Args:
        name: The slice's unit name.
        cpu: Its CPU-seconds.
        throttled: The seconds its quota held it back.
        wall: The check's wall seconds.

    Returns:
        The line, with its newline.
    """
    return (
        f"FLEET_SLICE_LOAD: {name} used {cpu} CPU-seconds and was throttled for {throttled} s "
        f"in the {wall} s the check ran\n"
    )


#: A slice's ``cpu.stat`` that counted nothing over the check, at its start and its end.
IDLE: tuple[str, str] = ("usage_usec 0\nthrottled_usec 0\n", "usage_usec 0\nthrottled_usec 0\n")


class TestTheLinesOnTheNode:
    def test_each_slice_that_ran_through_the_check_prints_its_cpu_seconds_and_the_wall(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed, walls = _run(
            tmp_path,
            {
                "ci": (
                    "usage_usec 1000000000\nthrottled_usec 0\nuser_usec 7\n",
                    "usage_usec 1412500000\nthrottled_usec 0\n",
                ),
                "dispatch": (
                    "usage_usec 5000000\nthrottled_usec 1000000\n",
                    "usage_usec 1468800000\nthrottled_usec 79600000\nuser_usec 1\n",
                ),
            },
            {},
            False,
        )

        # The CI slice's end is read first, so its wall is never the later.
        read = [
            (ci, dispatch)
            for ci in walls
            for dispatch in walls
            if ci <= dispatch
            and printed
            == _used(CI_SLICE_NAME, 412, 0, ci)
            + _used(DISPATCH_SLICE_NAME, 1463, 78, dispatch)
            + f"{pressure_none_line(CHECK_CGROUP)}\n"
        ]
        assert len(read) == 1
        ci, dispatch = read[0]
        assert read_slice_load(printed, CI_SLICE) == f"412s/{ci}s/throttled:0s"
        assert read_slice_load(printed, DISPATCH_SLICE) == f"1463s/{dispatch}s/throttled:78s"

    def test_a_node_without_a_slice_says_so_for_that_slice_alone(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed, walls = _run(
            tmp_path,
            {"ci": (None, None), "dispatch": IDLE},
            {},
            False,
        )

        read = [
            wall
            for wall in walls
            if printed
            == f"{slice_none_line(CI_SLICE)}\n"
            + _used(DISPATCH_SLICE_NAME, 0, 0, wall)
            + f"{pressure_none_line(CHECK_CGROUP)}\n"
        ]
        assert len(read) == 1
        assert read_slice_load(printed, CI_SLICE) == "none"
        assert read_slice_load(printed, DISPATCH_SLICE) == f"0s/{read[0]}s/throttled:0s"

    def test_a_slice_gone_by_the_end_says_so_and_the_build_goes_on(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed, _ = _run(
            tmp_path,
            {"ci": (None, None), "dispatch": ("usage_usec 5\nthrottled_usec 0\n", None)},
            {},
            False,
        )

        assert printed == (
            f"{slice_none_line(CI_SLICE)}\n{slice_gone_line(DISPATCH_SLICE)}\n"
            f"{pressure_none_line(CHECK_CGROUP)}\n"
        )
        assert read_slice_load(printed, DISPATCH_SLICE) == "gone"
        assert read_slice_load(printed, CI_SLICE) == "none"

    def test_the_build_s_own_cgroup_prints_the_seconds_its_tasks_stalled_on_each_resource(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The figure 86f3aa91 lacked (MCPs board task 2e2df92a): whether a
        check beside a docker-lane build waited on CPU, disk or memory."""
        printed, walls = _run(
            tmp_path,
            {"ci": IDLE, "dispatch": IDLE},
            {"cpu": (1_000_000, 4_900_000), "io": (0, 212_400_000), "memory": (5, 5)},
            False,
        )

        stalled = [
            wall
            for wall in walls
            if printed.endswith(f"{pressure_line(CHECK_CGROUP, ('3', '212', '0'), str(wall))}\n")
        ]
        assert len(stalled) == 1
        assert printed.count(PRESSURE_MARKER) == 1

    def test_a_docker_lane_build_prints_the_stalls_of_the_lane_manager_s_cgroup(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Measured on lavender (MCPs board task 2e2df92a): osrs-lab/observer's
        unit used 1 CPU-s while execdocker's manager used 8 cores, so the
        check's own cgroup says nothing of what a lane build waited on."""
        printed, walls = _run(
            tmp_path,
            {"ci": IDLE, "dispatch": IDLE},
            {"cpu": (0, 80_000_000), "io": (7, 2_000_007), "memory": (0, 0)},
            True,
        )

        stalled = [
            wall
            for wall in walls
            if printed.endswith(f"{pressure_line(EXEC_LANE_CGROUP, ('80', '2', '0'), str(wall))}\n")
        ]
        assert len(stalled) == 1
        assert (
            "the docker lane cgroup, which every check on the lane shares, stalled 80 s" in printed
        )

    def test_a_pressure_file_gone_by_the_end_says_so_and_the_build_goes_on(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed, _ = _run(
            tmp_path,
            {"ci": IDLE, "dispatch": IDLE},
            {"cpu": (1, 2), "io": (1, None), "memory": (1, 2)},
            True,
        )

        assert printed.endswith(f"\n{pressure_none_line(EXEC_LANE_CGROUP)}\n")

    def test_the_node_s_pressure_is_read_from_the_build_s_own_cgroup(self) -> None:
        assert (CGROUP_ROOT, OWN_CGROUP, PRESSURES) == (
            "/sys/fs/cgroup",
            "/proc/self/cgroup",
            ("cpu", "io", "memory"),
        )
        assert (CHECK_CGROUP["field"], CHECK_CGROUP["subject"], CHECK_CGROUP["locate"]) == (
            "check",
            "the cgroup of the check",
            "sed -n 's/^0:://p' '/proc/self/cgroup'",
        )

    def test_the_node_s_lines_read_the_ci_slice_and_dispatch_s_share(self) -> None:
        assert [measured["cpu_stat"] for measured in MEASURED_SLICES] == [
            f"/sys/fs/cgroup/{CI_SLICE_NAME}/cpu.stat",
            f"/sys/fs/cgroup/system.slice/{DISPATCH_SLICE_NAME}/cpu.stat",
        ]
        assert [measured["field"] for measured in MEASURED_SLICES] == ["ci", "dispatch"]


class TestTheReading:
    def test_a_tail_without_the_line_is_unread_and_the_last_line_counts(self) -> None:
        assert read_slice_load("make: *** [check] Error 2\n", CI_SLICE) == "unread"
        two = _used(CI_SLICE_NAME, 1, 0, 2) + _used(CI_SLICE_NAME, 30, 5, 40)
        assert read_slice_load(two, CI_SLICE) == "30s/40s/throttled:5s"
        assert read_slice_load(two, DISPATCH_SLICE) == "unread"

    def test_the_last_pressure_line_names_its_cgroup_and_each_stall(self) -> None:
        cgroups = (CHECK_CGROUP, EXEC_LANE_CGROUP)
        assert read_run_pressure("make: *** [check] Error 2\n", cgroups) == "unread"
        both = (
            f"{pressure_line(CHECK_CGROUP, ('1', '2', '3'), '9')}\n"
            f"{pressure_line(EXEC_LANE_CGROUP, ('80', '2', '0'), '412')}\n"
        )
        assert read_run_pressure(both, cgroups) == "lane:80s-cpu/2s-io/0s-memory"
        assert read_run_pressure(both, (CHECK_CGROUP,)) == "unread"
        assert (
            read_run_pressure(f"{pressure_line(CHECK_CGROUP, ('13', '2', '0'), '237')}\n", cgroups)
            == "check:13s-cpu/2s-io/0s-memory"
        )
        assert read_run_pressure(f"{pressure_none_line(EXEC_LANE_CGROUP)}\n", cgroups) == (
            "lane:none"
        )

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
            + _used(CI_SLICE_NAME, 2111, 0, 664)
            + _used(DISPATCH_SLICE_NAME, 940, 212, 664)
            + f"{pressure_line(EXEC_LANE_CGROUP, ('80', '2', '0'), '664')}\n"
            + f"{container_cpus_line('400000 100000,max 100000')}\n"
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
            "tests/served/proxy.test.ts:7s ci=2111s/664s/throttled:0s "
            "dispatch=940s/664s/throttled:212s vmtests=unread "
            "generation=unread began=unread phases=unread steps=unread registry=unread "
            "artifact=unread mutation=unread stalls=lane:80s-cpu/2s-io/0s-memory "
            "containers=400000/100000,max/100000 "
            f"results=lavender:/home/corvis/fleet/stage/logs/{RUN}.results"
        )
