"""The VM tests a check's build prints and the verdict reads (MCPs board tasks d6b03ba3, 7b62e759).

The lines run here under ``sh -eu``, as the builds run them, against a
transcript this case writes. Its test lines are fleet job 089ad006's own,
MCPs/fleet-nixos's cold run on lavender at a3f586595: each VM test's closing
line as nix prefixed it, and lines of the same tests that are not it. Its
driver lines are fleet job 6742e77b's, the same project's cold run on
lavender at f221f3aeb (log lines 2665, 3076, 3122, 3140 and 3200), with the
cache's address after ``from`` replaced, since that part is matched as any
text.
"""

from __future__ import annotations

import pathlib
import random
import string
import subprocess

from fleet.core import verdict
from fleet.core.vm_tests import read_vm_drivers, read_vm_tests, sh_vm_test_lines

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

#: Lines of the same run that must not read as a finished test or a realised driver.
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

#: 6742e77b's lines for two drivers nix substituted and two it built, in its order.
REALISED = (
    "copying path '/nix/store/a8schxcv3xyd4ncff3phpa18c7qra3cw-nixos-test-driver-print-host' "
    "from 'http://cache'...\n"
    "copying path '/nix/store/a4x17gj8h5dpvhr4mi42f09w1g7wkrjs-nixos-test-driver-corvis-hub' "
    "from 'http://cache'...\n"
    "building '/nix/store/sic04nmj012ppj77w61xhn42zsdwj9j6-nixos-test-driver-corvis-duty-storage"
    ".drv'...\n"
    "building '/nix/store/51q2z8988nicf0rjpa0a3q7z1schsyz9-nixos-test-driver-corvis-worker-vm"
    ".drv'...\n"
)

#: Lines naming a driver that are not the run realising a test's driver: the
#: generic driver package (6742e77b's line 2665), a builder's own output
#: quoting one, and a path whose store hash is short.
NOT_REALISED = (
    "copying path '/nix/store/8hlnfxwkpfsnqplfnp1n92py9kszyw63-nixos-test-driver-1.1' "
    "from 'http://cache'...\n"
    "vm-test-run-corvis-hub> copying path "
    "'/nix/store/a4x17gj8h5dpvhr4mi42f09w1g7wkrjs-nixos-test-driver-corvis-hub' "
    "from 'http://cache'...\n"
    "building '/nix/store/sic04nmj012ppj77w61x-nixos-test-driver-corvis-duty-storage.drv'...\n"
)

#: What the verdict reads off them, worked from the four lines by hand: the
#: drivers sorted by name, each with how nix realised it.
DRIVERS = "4:corvis-duty-storage=built,corvis-hub=fetched,corvis-worker-vm=built,print-host=fetched"


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
            + "FLEET_VM_DRIVERS: 0 realised\n"
        )
        assert read_vm_tests(printed) == READ
        assert read_vm_drivers(printed) == "0"

    def test_a_check_that_ran_no_vm_test_counts_none(self, tmp_path: pathlib.Path) -> None:
        printed = _run(tmp_path, NOT_FINISHED)

        assert printed == "FLEET_VM_TESTS: 0 finished\nFLEET_VM_DRIVERS: 0 realised\n"
        assert read_vm_tests(printed) == "0"

    def test_each_driver_fetched_or_built_is_printed_and_counted(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed = _run(tmp_path, NOT_REALISED + REALISED + _transcript(FINISHED[:1]))

        assert printed == (
            "FLEET_VM_TEST: corvis-hub 66.41\n"
            "FLEET_VM_TESTS: 1 finished\n"
            "FLEET_VM_DRIVER: print-host fetched\n"
            "FLEET_VM_DRIVER: corvis-hub fetched\n"
            "FLEET_VM_DRIVER: corvis-duty-storage built\n"
            "FLEET_VM_DRIVER: corvis-worker-vm built\n"
            "FLEET_VM_DRIVERS: 4 realised\n"
        )
        assert read_vm_drivers(printed) == DRIVERS
        assert read_vm_tests(printed) == "1:corvis-hub=66.41s"

    def test_a_warm_store_s_run_realises_none(self, tmp_path: pathlib.Path) -> None:
        printed = _run(tmp_path, NOT_REALISED + _transcript(FINISHED))

        assert printed.endswith("FLEET_VM_TESTS: 11 finished\nFLEET_VM_DRIVERS: 0 realised\n")
        assert read_vm_drivers(printed) == "0"

    def test_every_generated_driver_line_reads_back_as_its_name_and_how(
        self, tmp_path: pathlib.Path
    ) -> None:
        # A property over generated names and hashes, from a fixed seed so a
        # failure names a reproducible transcript: what the lines print and
        # the verdict reads is each driver nix's lines named, worked out here
        # from the generated pairs and not by the code under test.
        rng = random.Random(7)
        alphabet = string.ascii_lowercase + string.digits
        drivers: dict[str, str] = {}
        lines: list[str] = []
        while len(drivers) < 40:
            name = rng.choice(string.ascii_lowercase) + "".join(
                rng.choice(alphabet + "-") for _ in range(rng.randrange(0, 24))
            )
            if name in drivers:
                continue
            how = rng.choice(("fetched", "built"))
            drivers[name] = how
            store = "/nix/store/" + "".join(rng.choice(alphabet) for _ in range(32))
            lines.append(
                f"copying path '{store}-nixos-test-driver-{name}' from 'http://cache'...\n"
                if how == "fetched"
                else f"building '{store}-nixos-test-driver-{name}.drv'...\n"
            )
            lines.append(rng.choice(NOT_REALISED.splitlines(keepends=True)))

        printed = _run(tmp_path, "".join(lines))

        assert read_vm_drivers(printed) == "40:" + ",".join(
            f"{name}={drivers[name]}" for name in sorted(drivers)
        )


class TestTheReading:
    def test_a_tail_without_the_count_is_unread(self) -> None:
        assert read_vm_tests(_transcript(FINISHED)) == "unread"
        assert read_vm_tests("FLEET_VM_TEST: corvis-hub 66.41\n") == "unread"
        assert read_vm_drivers(REALISED) == "unread"
        assert read_vm_drivers("FLEET_VM_DRIVER: corvis-hub fetched\n") == "unread"

    def test_only_the_tests_since_the_count_before_the_last_are_read(self) -> None:
        tail = (
            "FLEET_VM_TEST: corvis-hub 66.41\n"
            "FLEET_VM_TESTS: 1 finished\n"
            "FLEET_VM_TEST: print-host 10.31\n"
            "FLEET_VM_TESTS: 1 finished\n"
        )

        assert read_vm_tests(tail) == "1:print-host=10.31s"

    def test_only_the_drivers_since_the_count_before_the_last_are_read(self) -> None:
        tail = (
            "FLEET_VM_DRIVER: corvis-hub fetched\n"
            "FLEET_VM_DRIVERS: 1 realised\n"
            "FLEET_VM_DRIVER: print-host built\n"
            "FLEET_VM_DRIVERS: 1 realised\n"
        )

        assert read_vm_drivers(tail) == "1:print-host=built"

    def test_the_verdict_line_carries_the_reading(self) -> None:
        run = "MCPs-fleet-nixos-lavender-1793887262"
        tail = (
            "".join(f"FLEET_VM_TEST: {name} {seconds}\n" for name, seconds in FINISHED)
            + "FLEET_VM_TESTS: 11 finished\n"
            + "FLEET_VM_DRIVER: corvis-hub fetched\n"
            + "FLEET_VM_DRIVER: corvis-duty-storage built\n"
            + "FLEET_VM_DRIVERS: 2 realised\n"
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

        drivers = "2:corvis-duty-storage=built,corvis-hub=fetched"
        assert judged["vmtests"] == READ
        assert judged["drivers"] == drivers
        assert f" dispatch=unread vmtests={READ} drivers={drivers} generation=unread " in (
            verdict.render_verdict(judged)
        )
