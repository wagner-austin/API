"""A Windows build that ends without its status is reported, not left running.

MCPs board task 4e3afe4f, the Windows half of board task c8585623. The result
script reads the build's scheduled task before the result file, and writes
the status of a build whose task has ended without one, with a
``FLEET_UNIT_ENDED`` line in the transcript (:mod:`fleet.core.windows_result`).
The text cases pin that order and the line. The host case EXECUTES it on a
Windows machine against a real build: the build render
(:func:`fleet.core.windows_build.build_script`) launched by the launch render
as a node launches it, its ``powershell.exe`` ended by pid mid-install, and
the result and tail renders run as the collector runs them. The committed
render's every arm runs under Pester in
``tests/pester/rendered-dialect-result.Tests.ps1``.
"""

from __future__ import annotations

import pathlib
import subprocess
import time
import uuid
from collections.abc import Generator

import pytest

from fleet.contracts.source import InstallStep
from fleet.core import names, windows_task
from fleet.core.dialect_windows import POWERSHELL_INVOCATION, WindowsDialect
from fleet.core.linux_unit_end import UNEXPLAINED_EXIT_CODE, UNIT_ENDED_MARKER
from fleet.core.phase_markers import PHASE_MARKER
from fleet.core.verdict import LOG_TAIL_LINES, read_unit_end
from fleet.core.windows_build import build_script
from fleet.core.windows_result import GOING_STATES, TASK_STATES, result_script
from tests.conftest import DEMO_RUN_ID

DIALECT = WindowsDialect()

TARGET = "C:/fleet/stage/run-1"

#: How long the host case waits for the build to reach its install, and for
#: the result script to see the build's task end, in seconds.
WAIT_SECONDS = 60


def _lines() -> list[str]:
    """The result script for the example dispatch, as lines.

    Returns:
        Its lines.
    """
    return result_script(target=TARGET, run_id=DEMO_RUN_ID).splitlines()


class TestTheScriptsText:
    def test_the_dialect_hands_its_result_script_to_this_module(self) -> None:
        assert DIALECT.result_script(target=TARGET, run_id=DEMO_RUN_ID) == result_script(
            target=TARGET, run_id=DEMO_RUN_ID
        )

    def test_it_names_the_dispatchs_own_task_result_and_transcript(self) -> None:
        lines = _lines()

        assert lines[:4] == [
            "param(",
            f"    [string]$Target = '{TARGET}',",
            f"    [string]$TaskName = '{names.task_name(DEMO_RUN_ID)}'",
            ")",
        ]
        assert f'$result = "$Target/{names.RESULT_NAME}"' in lines
        assert f'$log = "{names.log_path("$Target")}"' in lines

    def test_it_reads_the_task_before_the_result_file(self) -> None:
        """A build that writes its status and exits between the two reads
        would otherwise be recorded as ended without one."""
        lines = _lines()
        read = lines.index(
            "$task = @($scheduler.GetFolder('\\').GetTasks(1) | "
            "Where-Object { $_.Name -eq $TaskName })"
        )
        first_look = next(
            index for index, line in enumerate(lines) if "Test-Path -LiteralPath $result" in line
        )

        assert read < first_look
        assert "Get-ScheduledTask" not in "\n".join(lines)

    def test_only_a_running_or_queued_task_reads_as_going(self) -> None:
        assert {TASK_STATES[state] for state in GOING_STATES} == {"Queued", "Running"}
        assert (
            "$going = ($task.Count -gt 0) -and (@(2, 4) -contains [int]$task[0].State)" in _lines()
        )

    def test_an_ending_without_a_status_appends_the_line_then_writes_the_status(self) -> None:
        lines = _lines()
        appended = lines.index('    [System.IO.File]::AppendAllText($log, "$line`r`n")')
        written = lines.index("    $code | Set-Content -LiteralPath $result")

        assert f"    $code = {UNEXPLAINED_EXIT_CODE}" in lines
        assert (
            f'    $line = "{UNIT_ENDED_MARKER}: $how before the build wrote its status; '
            'recorded exit $code"'
        ) in lines
        assert appended < written < lines.index("if (Test-Path -LiteralPath $result) {")

    def test_it_reports_when_as_well_as_what(self) -> None:
        """Only the node knows when the build ended, and whether its lease
        covered it is asked against that; PowerShell 5.1's -UFormat %s
        converts from LOCAL time, so the epoch is computed from UTC."""
        body = result_script(target=TARGET, run_id=DEMO_RUN_ID)

        assert "LastWriteTimeUtc - [datetime]'1970-01-01'" in body
        assert "-UFormat" not in body

    def test_a_target_it_cannot_carry_verbatim_is_refused(self) -> None:
        with pytest.raises(ValueError, match="target"):
            result_script(target="C:/fleet/stage/it's", run_id=DEMO_RUN_ID)


class TestTheVerdictReadsAWindowsEnding:
    def test_a_crlf_tail_is_read_without_its_carriage_return(self) -> None:
        """A Windows tail crosses ssh with CRLF endings and is decoded as it
        came, so the line must end before the return."""
        line = (
            f"{UNIT_ENDED_MARKER}: task fleet-run-1 ended with Task Scheduler state 3 (Ready) "
            "and last result 0x00000001 before the build wrote its status; recorded exit 1"
        )
        tail = f"{PHASE_MARKER} hold started 2026-10-07T23:00:00Z\r\nReply from ::1\r\n{line}\r\n"

        assert read_unit_end(tail) == line


def _run(script: pathlib.Path, text: str) -> str:
    """Write a rendered script and run it as a node runs one.

    Args:
        script: Where to write it.
        text: The script.

    Returns:
        What it printed, decoded as :mod:`fleet.core._command` decodes ssh
        output, line endings kept.
    """
    script.write_text(text, encoding="utf-8")
    completed = subprocess.run(
        [*POWERSHELL_INVOCATION, str(script)],
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return completed.stdout.decode("utf-8")


def _holds(log: pathlib.Path, text: str) -> bool:
    """Whether the transcript exists and carries a text.

    Args:
        log: The transcript.
        text: What to look for.

    Returns:
        True when it does.
    """
    return log.exists() and text in log.read_text(encoding="utf-8", errors="replace")


def _wait_for_phase(log: pathlib.Path, phase: str) -> None:
    """Wait for the build to open a phase, so it is ended mid-step.

    Args:
        log: The transcript.
        phase: The phase's name.
    """
    opening = f"{PHASE_MARKER} {phase} started "
    deadline = time.monotonic() + WAIT_SECONDS
    while not _holds(log, opening) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert _holds(log, opening), f"the build never opened {phase} in {WAIT_SECONDS} s"


def _first_report(script: pathlib.Path, text: str) -> str:
    """Run the result script, as the collector's passes do, until it reports.

    Args:
        script: Where to write it.
        text: The result script.

    Returns:
        Its first non-empty answer.
    """
    deadline = time.monotonic() + WAIT_SECONDS
    reported = _run(script, text)
    while not reported and time.monotonic() < deadline:
        reported = _run(script, text)
    assert reported, f"the result script reported nothing {WAIT_SECONDS} s after the kill"
    return reported


@pytest.fixture(name="dispatch")
def _dispatch(tmp_path: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    """A dispatch directory named by its own run id, stopped afterwards.

    The stop render ends whatever of the build still runs and deletes its
    task, as a node's retire does, so no case leaves a task registered.

    Args:
        tmp_path: The case's directory, which holds the scripts.

    Yields:
        The dispatch directory; its name is the run id.
    """
    target = tmp_path / f"tests-windows-result-{uuid.uuid4().hex[:12]}"
    target.mkdir()
    yield target
    _run(
        tmp_path / "stop.ps1",
        windows_task.stop_script(target=target.as_posix(), run_id=target.name),
    )


@pytest.mark.host_windows
class TestABuildEndedOnThisMachine:
    """The real build, launched as a scheduled task and ended by pid."""

    def test_a_build_killed_mid_install_is_reported_with_how_it_ended(
        self, dispatch: pathlib.Path
    ) -> None:
        target = dispatch.as_posix()
        run_id = dispatch.name
        scripts = dispatch.parent
        (dispatch / f"{names.BUILD_STEM}.ps1").write_text(
            build_script(
                target=target,
                path="",
                workers=1,
                install=(InstallStep(phase="hold", argv=("ping", "-n", "300", "127.0.0.1")),),
                cache_root=(scripts / "cache").as_posix(),
                elevated=False,
                agent="opus-demo-0929",
            ),
            encoding="utf-8",
        )
        launch = windows_task.launch_script(target=target, run_id=run_id, elevated=False)
        assert _run(scripts / "launch.ps1", launch) == "launched\r\n"
        log = pathlib.Path(names.log_path(target))
        _wait_for_phase(log, "hold")
        collect = DIALECT.result_script(target=target, run_id=run_id)

        assert _run(scripts / "collect.ps1", collect) == ""

        build = (dispatch / names.PID_NAME).read_text(encoding="utf-8").strip()
        killed_at = int(time.time())
        subprocess.run(["taskkill", "/PID", build, "/F"], check=True, capture_output=True)
        reported = _first_report(scripts / "collect.ps1", collect)
        code, written = reported.split()
        line = (
            f"{UNIT_ENDED_MARKER}: task {names.task_name(run_id)} ended with Task Scheduler "
            "state 3 (Ready) and last result 0x00000001 before the build wrote its status; "
            "recorded exit 1"
        )

        assert code == "1"
        assert killed_at <= int(written) <= int(time.time())
        assert (dispatch / names.RESULT_NAME).read_text(encoding="utf-8").strip() == "1"
        tail = _run(scripts / "tail.ps1", DIALECT.log_tail_script(target, LOG_TAIL_LINES))
        assert tail.endswith(f"{line}\r\n")
        assert read_unit_end(tail) == line
        assert _run(scripts / "collect.ps1", collect) == reported
        assert log.read_text(encoding="utf-8").count(UNIT_ENDED_MARKER) == 1
