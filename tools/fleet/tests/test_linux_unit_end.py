"""A Linux build's unit records its own ending (MCPs board task c8585623).

Fleet job 859e7ab3 was killed by systemd-oomd 2 min 8 s into its install on
diphtheria, wrote no result, and read as running until its lease lapsed 22
minutes later. These cases pin the launch to an ``ExecStopPost=`` that runs
the unit-end script, and EXECUTE that script under ``sh`` with each
environment systemd hands it. The values are the ones systemd 255 gave on
diphtheria on 2026-10-07: ``signal killed KILL`` for a SIGKILLed cgroup and
``exit-code exited 3`` for a build that exited 3, plus ``oom-kill`` as the
journal recorded it for 859e7ab3. The launch itself runs for real too, with
``loginctl`` and ``systemd-run`` answered by scripts ahead on PATH, so the
script it writes to the disk is read back byte for byte.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.config import config_test_hooks

from fleet.core import names
from fleet.core.dialect_linux import PROLOGUE, SH_INVOCATION, UNIT_END_DELIMITER, LinuxDialect
from fleet.core.linux_unit_end import (
    UNEXPLAINED_EXIT_CODE,
    UNIT_ENDED_MARKER,
    unit_end_path,
    unit_end_script,
)
from fleet.core.verdict import read_unit_end
from tests.conftest import DEMO_RUN_ID

DIALECT = LinuxDialect()

TARGET = "/home/corvis/fleet/stage/run-1"

UNIT = names.task_name(DEMO_RUN_ID)


class TestTheLaunchWiresTheUnitEnd:
    def test_the_script_is_written_beside_the_build(self) -> None:
        assert unit_end_path(TARGET) == f"{TARGET}/unit-end.sh"

    def test_the_launch_writes_the_script_then_names_it_as_exec_stop_post(self) -> None:
        body = DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID, elevated=False)
        script = unit_end_script(target=TARGET, unit=UNIT, prologue=PROLOGUE)
        heredoc = (
            f"cat > '{TARGET}/unit-end.sh' <<'{UNIT_END_DELIMITER}'\n{script}{UNIT_END_DELIMITER}\n"
        )

        assert heredoc in body
        assert body.index(heredoc) < body.index("systemd-run ")
        assert f"--property=ExecStopPost='/bin/sh {TARGET}/unit-end.sh' " in body
        assert UNIT_END_DELIMITER not in script.splitlines()

    def test_the_script_is_fail_fast_and_writes_the_result_last(self) -> None:
        script = unit_end_script(target=TARGET, unit=UNIT, prologue=PROLOGUE)
        lines = script.splitlines()

        assert script.startswith(PROLOGUE)
        assert lines[-1] == f"printf '%s\\n' \"$code\" > {TARGET}/{names.RESULT_NAME}"
        assert f">> {names.log_path(TARGET)}" in lines[-2]


class TestTheVerdictReadsTheEnding:
    def test_a_tail_without_the_line_reads_none(self) -> None:
        assert read_unit_end("Test Files  3 failed (3)\nmake: *** [check] Error 1\n") is None

    def test_the_last_line_is_read_with_its_marker(self) -> None:
        tail = (
            "npm warn deprecated boolean@3.2.0\n"
            f"{UNIT_ENDED_MARKER}: unit a ended with systemd result signal (killed TERM) "
            "before the build wrote its status; recorded exit 143\n"
            f"{UNIT_ENDED_MARKER}: unit b ended with systemd result oom-kill (killed KILL) "
            "before the build wrote its status; recorded exit 137\n"
        )

        assert read_unit_end(tail) == (
            f"{UNIT_ENDED_MARKER}: unit b ended with systemd result oom-kill (killed KILL) "
            "before the build wrote its status; recorded exit 137"
        )


def _environment(**variables: str) -> dict[str, str]:
    """The suite's environment with systemd's ending variables replaced.

    Args:
        **variables: The variables systemd sets, as it set them.

    Returns:
        The parent environment without any of the three, then these.
    """
    parent = {
        name: value
        for name, value in config_test_hooks.get_environment().items()
        if name not in {"SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS"}
    }
    return {**parent, **variables}


def _run_end(target: pathlib.Path, environment: dict[str, str]) -> None:
    """Write the unit-end script for ``target`` and run it as systemd would.

    Args:
        target: The dispatch directory.
        environment: The process environment.
    """
    script = target / "unit-end.sh"
    script.write_text(
        unit_end_script(target=target.as_posix(), unit=UNIT, prologue=PROLOGUE),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [*SH_INVOCATION, str(script)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def _line(result: str, exit_code: str, exit_status: str, code: int) -> str:
    """The transcript line the script appends.

    Args:
        result: ``SERVICE_RESULT``.
        exit_code: ``EXIT_CODE``.
        exit_status: ``EXIT_STATUS``.
        code: The status it records.

    Returns:
        The line, with its newline.
    """
    return (
        f"{UNIT_ENDED_MARKER}: unit {UNIT} ended with systemd result {result} "
        f"({exit_code} {exit_status}) before the build wrote its status; "
        f"recorded exit {code}\n"
    )


@pytest.mark.host_linux
class TestTheScriptUnderSh:
    """The unit-end script, executed by /bin/sh on this machine."""

    @pytest.mark.parametrize(
        ("result", "exit_code", "exit_status", "code"),
        [
            ("oom-kill", "killed", "KILL", 137),
            ("signal", "killed", "KILL", 137),
            ("signal", "killed", "TERM", 143),
            ("core-dump", "dumped", "SEGV", 139),
            ("exit-code", "exited", "3", 3),
        ],
    )
    def test_a_build_that_wrote_nothing_gets_its_ending_recorded(
        self, tmp_path: pathlib.Path, result: str, exit_code: str, exit_status: str, code: int
    ) -> None:
        log = tmp_path / f"{names.RESULT_NAME}.log"
        log.write_text("npm warn deprecated node-domexception@1.0.0\n", encoding="utf-8")

        _run_end(
            tmp_path,
            _environment(SERVICE_RESULT=result, EXIT_CODE=exit_code, EXIT_STATUS=exit_status),
        )

        assert (tmp_path / names.RESULT_NAME).read_text(encoding="utf-8") == f"{code}\n"
        assert log.read_text(encoding="utf-8") == (
            "npm warn deprecated node-domexception@1.0.0\n"
            + _line(result, exit_code, exit_status, code)
        )

    def test_a_build_that_wrote_its_status_is_left_alone(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / names.RESULT_NAME).write_text("0\n", encoding="utf-8")
        log = tmp_path / f"{names.RESULT_NAME}.log"
        log.write_text("=== ALL CHECKS PASSED ===\n", encoding="utf-8")

        _run_end(
            tmp_path, _environment(SERVICE_RESULT="success", EXIT_CODE="exited", EXIT_STATUS="0")
        )

        assert (tmp_path / names.RESULT_NAME).read_text(encoding="utf-8") == "0\n"
        assert log.read_text(encoding="utf-8") == "=== ALL CHECKS PASSED ===\n"

    def test_an_exit_0_without_a_status_is_never_recorded_as_a_pass(
        self, tmp_path: pathlib.Path
    ) -> None:
        _run_end(
            tmp_path, _environment(SERVICE_RESULT="success", EXIT_CODE="exited", EXIT_STATUS="0")
        )

        assert (tmp_path / names.RESULT_NAME).read_text(encoding="utf-8") == (
            f"{UNEXPLAINED_EXIT_CODE}\n"
        )
        assert (tmp_path / f"{names.RESULT_NAME}.log").read_text(encoding="utf-8") == _line(
            "success", "exited", "0", UNEXPLAINED_EXIT_CODE
        )

    def test_a_signal_name_no_number_has_records_the_unexplained_status(
        self, tmp_path: pathlib.Path
    ) -> None:
        _run_end(
            tmp_path, _environment(SERVICE_RESULT="signal", EXIT_CODE="killed", EXIT_STATUS="NOPE")
        )

        assert (tmp_path / names.RESULT_NAME).read_text(encoding="utf-8") == (
            f"{UNEXPLAINED_EXIT_CODE}\n"
        )
        assert (tmp_path / f"{names.RESULT_NAME}.log").read_text(encoding="utf-8") == _line(
            "signal", "killed", "NOPE", UNEXPLAINED_EXIT_CODE
        )

    def test_a_unit_whose_main_process_never_ran_records_the_unexplained_status(
        self, tmp_path: pathlib.Path
    ) -> None:
        _run_end(tmp_path, _environment(SERVICE_RESULT="resources"))

        assert (tmp_path / names.RESULT_NAME).read_text(encoding="utf-8") == (
            f"{UNEXPLAINED_EXIT_CODE}\n"
        )
        assert (tmp_path / f"{names.RESULT_NAME}.log").read_text(encoding="utf-8") == _line(
            "resources", "", "", UNEXPLAINED_EXIT_CODE
        )


@pytest.mark.host_linux
class TestTheLaunchUnderSh:
    """The real launch script, with the user manager answered from PATH."""

    def test_it_writes_the_unit_end_script_and_starts_the_unit_with_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        tools = tmp_path / "tools"
        tools.mkdir()
        argv_file = tmp_path / "systemd-run.argv"
        loginctl = tools / "loginctl"
        loginctl.write_bytes(b"#!/bin/sh\necho yes\n")
        loginctl.chmod(0o755)
        systemd_run = tools / "systemd-run"
        systemd_run.write_bytes(
            f"#!/bin/sh\nprintf '%s\\n' \"$@\" > '{argv_file.as_posix()}'\n".encode()
        )
        systemd_run.chmod(0o755)
        target = tmp_path / "run-1"
        target.mkdir()
        body = DIALECT.launch_script(
            target=target.as_posix(), run_id=DEMO_RUN_ID, elevated=False
        ).replace(PROLOGUE, PROLOGUE + f"PATH='{tools.as_posix()}':$PATH\n", 1)
        script = tmp_path / "launch.sh"
        script.write_text(body, encoding="utf-8")

        completed = subprocess.run(
            [*SH_INVOCATION, str(script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == "launched\n"
        assert (target / "unit-end.sh").read_text(encoding="utf-8") == unit_end_script(
            target=target.as_posix(), unit=UNIT, prologue=PROLOGUE
        )
        assert argv_file.read_text(encoding="utf-8").splitlines() == [
            "--user",
            f"--unit={UNIT}",
            "--collect",
            "--quiet",
            f"--property=WorkingDirectory={target.as_posix()}",
            f"--property=ExecStopPost=/bin/sh {target.as_posix()}/unit-end.sh",
            "/bin/sh",
            f"{target.as_posix()}/{names.BUILD_STEM}.sh",
        ]
