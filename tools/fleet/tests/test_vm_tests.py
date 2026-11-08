"""The VM tests a check's build prints and the verdict reads (MCPs board task d6b03ba3).

The lines run here under ``sh -eu``, as the builds run them, against a
transcript this case writes. Its test lines are fleet job 089ad006's own,
MCPs/fleet-nixos's cold run on lavender at a3f586595: each VM test's closing
line as nix prefixed it, and lines of the same tests that are not it.
"""

from __future__ import annotations

import pathlib
import subprocess

from fleet.core import verdict
from fleet.core.vm_tests import read_vm_tests, sh_vm_test_lines

#: The eleven closing lines of 089ad006's transcript, in its order.
FINISHED = (
    ("corvis-hub", "66.41"),
    ("corvis-duty-storage", "125.12"),
    ("corvis-runner-host", "127.58"),
    ("corvis-storage-exports", "48.83"),
    ("netboot-grub-start", "92.49"),
    ("netboot-unreachable", "31.44"),
    ("corvis-netboot-server", "34.31"),
    ("uci-vpn-agents", "29.56"),
    ("corvis-worker-vm", "113.55"),
    ("print-host", "10.31"),
    ("corvis-worker-secure-boot", "18.49"),
)

#: Lines of the same run that must not read as a finished test.
NOT_FINISHED = (
    "vm-test-run-corvis-runner-host> (finished: subtest: setup-python finds each pinned "
    "CPython in every install's tool cache, and each runs with its standard library, "
    "in 11.51 seconds)\n"
    "building '/nix/store/4bhv640a8liy5z1xmsx8zx8v3m2xfddb-vm-test-run-corvis-runner-host.drv'...\n"
    "check-step runner-host ended 2026-11-05T14:04:36Z after 167 s, exit 0\n"
    "corvis-worker-gpu> == lavender (NVIDIA GeForce GTX 1630)\n"
)

#: What the verdict reads off them: the eleven, by name.
READ = (
    "11:corvis-duty-storage=125.12s,corvis-hub=66.41s,corvis-netboot-server=34.31s,"
    "corvis-runner-host=127.58s,corvis-storage-exports=48.83s,corvis-worker-secure-boot=18.49s,"
    "corvis-worker-vm=113.55s,netboot-grub-start=92.49s,netboot-unreachable=31.44s,"
    "print-host=10.31s,uci-vpn-agents=29.56s"
)


def _transcript(tests: tuple[tuple[str, str], ...]) -> str:
    """A transcript holding these tests' closing lines among lines that are not.

    Args:
        tests: Each test's name and seconds.

    Returns:
        The text.
    """
    finished = "".join(
        f"vm-test-run-{name}> test script finished in {seconds}s\n" for name, seconds in tests
    )
    return NOT_FINISHED + finished + NOT_FINISHED


def _run(tmp_path: pathlib.Path, transcript: str) -> str:
    """Run the lines over a transcript and read what they appended.

    Args:
        tmp_path: The case's directory.
        transcript: The transcript's text before the lines run.

    Returns:
        What the lines appended.
    """
    log = tmp_path / "check.log"
    log.write_text(transcript, encoding="utf-8")
    subprocess.run(
        ["sh", "-eu", "-c", "\n".join(sh_vm_test_lines(str(log)))], check=True, timeout=30
    )
    return log.read_text(encoding="utf-8").removeprefix(transcript)


class TestTheLinesOnTheNode:
    def test_each_finished_test_is_printed_and_counted(self, tmp_path: pathlib.Path) -> None:
        printed = _run(tmp_path, _transcript(FINISHED))

        assert printed == (
            "".join(f"FLEET_VM_TEST: {name} {seconds}\n" for name, seconds in FINISHED)
            + "FLEET_VM_TESTS: 11 finished\n"
        )
        assert read_vm_tests(printed) == READ

    def test_a_check_that_ran_no_vm_test_counts_none(self, tmp_path: pathlib.Path) -> None:
        printed = _run(tmp_path, NOT_FINISHED)

        assert printed == "FLEET_VM_TESTS: 0 finished\n"
        assert read_vm_tests(printed) == "0"


class TestTheReading:
    def test_a_tail_without_the_count_is_unread(self) -> None:
        assert read_vm_tests(_transcript(FINISHED)) == "unread"
        assert read_vm_tests("FLEET_VM_TEST: corvis-hub 66.41\n") == "unread"

    def test_only_the_tests_since_the_count_before_the_last_are_read(self) -> None:
        tail = (
            "FLEET_VM_TEST: corvis-hub 66.41\n"
            "FLEET_VM_TESTS: 1 finished\n"
            "FLEET_VM_TEST: print-host 10.31\n"
            "FLEET_VM_TESTS: 1 finished\n"
        )

        assert read_vm_tests(tail) == "1:print-host=10.31s"

    def test_the_verdict_line_carries_the_reading(self) -> None:
        run = "MCPs-fleet-nixos-lavender-1793887262"
        tail = (
            "".join(f"FLEET_VM_TEST: {name} {seconds}\n" for name, seconds in FINISHED)
            + "FLEET_VM_TESTS: 11 finished\n"
        )
        judged = verdict.judge(
            job_id="089ad006-6b6f-4bc6-8ed6-c57ae2cdfc58",
            project="MCPs/fleet-nixos",
            sha="a3f586595bde83882250452c099251b43d85ae2a",
            node="lavender",
            exit_code=0,
            tail=tail,
            ended_unix=1793887550,
            log_path=f"/home/corvis/fleet/stage/logs/{run}.log",
            results_path=f"/home/corvis/fleet/stage/logs/{run}.results",
            run_id=run,
            workers=1,
        )

        assert judged["vmtests"] == READ
        assert f" dispatch=unread vmtests={READ} generation=unread " in (
            verdict.render_verdict(judged)
        )
