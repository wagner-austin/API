"""The PowerShell scripts a Windows node is handed, and the quoting that broke them.

EVERY TEST IN THE FIRST TWO CLASSES IS A REGRESSION FROM ONE DISPATCH. On
2026-09-04 the first ``fleet-run`` to reach a node registered a scheduled task
whose ``-Argument`` was the eleven characters ``-Command "cd`` and whose
WORKING DIRECTORY was the remaining two hundred, because the build was
interpolated into a single-quoted PowerShell string that contained single
quotes. The task could not be started. Nothing failed: PowerShell exited 0 and
the ledger recorded a run that did not exist.

These assertions are about the TEXT of the scripts rather than about running
them, and that is the honest limit of what a test on this machine can say. The
things they pin -- no inner quote in the argument, a wait for the task to
actually start, the battery settings -- are each the difference between a
dispatch that runs and one that silently does not, and each was found by
reading a task's XML off a node rather than by reasoning about the string.
The two probes ARE run for real, on this machine, because it is a Windows hub.
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import sys

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, load_json_str

from fleet.core import names
from fleet.core.dialect_windows import (
    LAUNCH_TIMEOUT_SECONDS,
    OBSERVE_SESSIONS_SCRIPT,
    POWERSHELL_INVOCATION,
    TASK_HAS_NOT_RUN,
    WindowsDialect,
)
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID

DIALECT = WindowsDialect()

#: One registration document in the shape the harness writes on a Windows
#: node: a drive-letter cwd and a named-pipe socket.
SESSION_RECORD: JSONObject = {
    "sessionId": "d31e5228-3269-4a22-a27f-e535cbef1894",
    "pid": 4242,
    "startedAt": 1_789_000_000_000,
    "updatedAt": 1_789_000_030_500,
    "name": "api-7e",
    "nameSource": "derived",
    "cwd": "C:\\Users\\serendipity\\PROJECTS\\API",
    "messagingSocketPath": "\\\\.\\pipe\\LOCAL\\cc-msg-abc",
    "version": "2.1.278",
    "status": "idle",
    "pidDomain": "win32:serendipity",
}


class TestBuildScript:
    def test_it_runs_the_recipe_in_the_project(self) -> None:
        body = DIALECT.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert f"Set-Location -LiteralPath 'C:/s/run-1/{DEMO_PROJECT}'" in body
        assert "make check" in body

    def test_it_pins_the_worker_count(self) -> None:
        body = DIALECT.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert "PYTEST_XDIST_AUTO_NUM_WORKERS = '6'" in body

    def test_it_records_the_status_last(self) -> None:
        """The result file's absence is how a run is known to be unfinished.

        Written after the recipe, so it can never exist while make is still
        going -- which is what lets `fleet-collect` treat absence as running.
        """
        body = DIALECT.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)
        lines = [line for line in body.splitlines() if line.strip()]

        assert lines[-1].startswith("$LASTEXITCODE")
        assert names.RESULT_NAME in lines[-1]

    def test_it_reads_the_exit_code_and_not_the_success_flag(self) -> None:
        """`make` writes to stderr on a passing run; under redirection that
        sets $? false in PS 5.1 while $LASTEXITCODE stays correct."""
        body = DIALECT.build_script(target="C:/s/run-1", project=DEMO_PROJECT, workers=6)

        assert "$LASTEXITCODE" in body
        assert "$?" not in body


class TestLaunchScript:
    def test_it_registers_and_starts_a_scheduled_task(self) -> None:
        """Not an ssh child. Windows OpenSSH puts that in a job object that
        dies with the connection, and this command returns immediately."""
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "Register-ScheduledTask" in body
        assert "Start-ScheduledTask" in body

    def test_it_sets_priority_four(self) -> None:
        """Priority 7 is the Register-ScheduledTask default and sets LOW I/O.

        A run that inherits it crawls, and the symptom reads as a slow node
        rather than a misconfigured launch.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "-Priority 4" in body
        assert "[TimeSpan]::Zero" in body
        assert "-LogonType S4U" in body

    def test_it_runs_the_build_by_path_and_never_inlines_it(self) -> None:
        """THE REGRESSION. Interpolating the build into -Argument split the
        task in two: PowerShell ended the single-quoted string at the first
        inner quote and bound the rest to -WorkingDirectory. Measured on
        sedona 2026-09-04; the task could not be started at all.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert f'-File "{DIALECT.script_path("C:/s/run-1", names.BUILD_STEM)}"' in body
        assert "-Command" not in body
        assert "make check" not in body

    def test_the_argument_string_contains_no_single_quotes(self) -> None:
        """The mechanical form of the same defect, asserted directly.

        -Argument is passed as a single-quoted PowerShell string, so ANY
        single quote inside it terminates the argument early.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)
        argument = body.split("-Argument '", 1)[1].split("'\n", 1)[0]

        assert "'" not in argument

    def test_it_survives_the_lid_being_shut(self) -> None:
        """Two of the three nodes are laptops and both battery settings
        default to refusing: without these a dispatch to an unplugged sedona
        registers a task that never runs, and reports nothing.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "-AllowStartIfOnBatteries" in body
        assert "-DontStopIfGoingOnBatteries" in body

    def test_it_waits_for_the_task_to_actually_start(self) -> None:
        """Start-ScheduledTask reports a refusal as a NON-terminating error.

        On 2026-09-04 it failed with 'Element not found', PowerShell exited 0,
        and the dispatch was recorded as running. The script now watches for
        the task to leave SCHED_S_TASK_HAS_NOT_RUN and throws if it does not.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert "$ErrorActionPreference = 'Stop'" in body
        assert str(TASK_HAS_NOT_RUN) in body
        assert str(LAUNCH_TIMEOUT_SECONDS) in body
        assert "throw" in body

    def test_the_task_name_is_the_one_the_stop_script_stops(self) -> None:
        """Both come from names.task_name, so a rename cannot make
        fleet-cancel report success having stopped nothing."""
        launched = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert names.task_name(DEMO_RUN_ID) in launched
        assert names.task_name(DEMO_RUN_ID) in DIALECT.stop_script(DEMO_RUN_ID)


class TestResultAndStopScripts:
    def test_it_reports_when_as_well_as_what(self) -> None:
        """Whether a run was PROTECTED is a question about whether its lease
        covered it, and only the node knows when the build ended. Asking for
        the status alone forced the reader to substitute "is a lease held now"
        -- a question about how promptly somebody collected -- which refused a
        run that finished three minutes inside its window."""
        body = DIALECT.result_script("C:/s/run-1")

        assert "LastWriteTimeUtc" in body
        assert "1970-01-01" in body

    def test_it_does_not_use_uformat_for_the_epoch(self) -> None:
        """PowerShell 5.1's -UFormat %s converts from LOCAL time, which would
        put every node's answer out by its own offset."""
        assert "-UFormat" not in DIALECT.result_script("C:/s/run-1")

    def test_the_result_script_prints_nothing_while_running(self) -> None:
        """Absence is the signal, so an unfinished run is not read as exit 0."""
        body = DIALECT.result_script("C:/s/run-1")

        assert "Test-Path" in body
        assert f"C:/s/run-1/{names.RESULT_NAME}" in body

    def test_the_stop_script_never_prompts(self) -> None:
        """There is nobody at the node to answer, and a prompt would hang."""
        assert "-Confirm:$false" in DIALECT.stop_script(DEMO_RUN_ID)


class TestTransportShape:
    def test_scripts_are_ps1_and_run_through_powershell_by_path(self) -> None:
        assert DIALECT.script_path("C:/s", "build") == "C:/s/build.ps1"
        assert DIALECT.invocation() == POWERSHELL_INVOCATION
        assert DIALECT.invocation()[-1] == "-File"

    def test_the_write_command_is_one_quoted_argument_for_cmd(self) -> None:
        """Windows OpenSSH hands the command to cmd.exe, which would take an
        unquoted pipe as its own. Measured 2026-09-04: 'Set-Content' is not
        recognized as an internal or external command."""
        command = DIALECT.write_command("C:/s/run-1/x.ps1")

        assert command.startswith('powershell -NoProfile -Command "')
        assert command.endswith('"')
        assert "Set-Content -LiteralPath 'C:/s/run-1/x.ps1'" in command
        assert "Split-Path -Parent 'C:/s/run-1/x.ps1'" in command

    def test_the_directory_and_reassembly_scripts_use_literal_paths(self) -> None:
        made = DIALECT.make_directory_script("C:/s/run-[1]")
        rebuilt = DIALECT.reassemble_script("C:/s/run-1")

        assert "-LiteralPath 'C:/s/run-[1]'" in made
        assert f"C:/s/run-1/{names.ENCODED_NAME}" in rebuilt
        assert f"C:/s/run-1/{names.ARCHIVE_NAME}" in rebuilt
        assert "Get-FileHash -Algorithm SHA256" in rebuilt
        assert "tar" not in rebuilt

    def test_echo_is_write_output_of_a_quoted_literal(self) -> None:
        assert DIALECT.echo_command("installing make") == "Write-Output 'installing make'"

    def test_the_fleet_directory_is_the_literal_profile_path(self) -> None:
        """The write command expands nothing, so ``$env:USERPROFILE`` would
        be a directory called that; the account's profile is spelled out."""
        assert DIALECT.fleet_directory("austi") == "C:/Users/austi/.fleet"
        assert DIALECT.script_path(DIALECT.fleet_directory("austi"), "observe-sessions") == (
            "C:/Users/austi/.fleet/observe-sessions.ps1"
        )

    def test_the_observe_script_reads_the_harness_directory_and_reports_zero_when_absent(
        self,
    ) -> None:
        """Moved here verbatim from the observe module when the Linux dialect
        got its own rendering (board task cd5010c4); these are the properties
        the observe pass was written against."""
        body = DIALECT.observe_sessions_script()

        assert body == OBSERVE_SESSIONS_SCRIPT
        assert body.startswith("$ErrorActionPreference = 'Stop'\n")
        assert "'.claude\\sessions'" in body
        assert "if (Test-Path -LiteralPath $dir)" in body
        assert "-Filter '*.json'" in body
        assert "$env:COMPUTERNAME.ToLowerInvariant()" in body
        assert "platform = 'win32'" in body
        assert "ConvertTo-Json -Depth 8 -Compress" in body


@pytest.mark.skipif(sys.platform != "win32", reason="the probes are PowerShell; run them here")
class TestProbesForReal:
    """The two constant probes, executed on this hub and parsed by shape.

    The one place the Windows dialect's text is run rather than read: this
    machine is one, so the assertions can be about what PowerShell 5.1
    actually prints. The ``pip`` line in particular was read as absent on a
    hub that has pip until the probe stopped piping ``python -m pip`` through
    ``Select-Object -First 1``.
    """

    def run_probe(self, tmp_path: pathlib.Path, body: str) -> dict[str, str]:
        """Write a probe under tmp_path, run it by path, and split its lines.

        Args:
            tmp_path: Where to write it.
            body: The probe's text.

        Returns:
            Every ``key=value`` line as a mapping.
        """
        script = tmp_path / "probe.ps1"
        script.write_text(body, encoding="utf-8")
        completed = subprocess.run(
            [*POWERSHELL_INVOCATION, str(script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        fields: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            key, _separator, value = line.partition("=")
            fields[key] = value
        return fields

    def test_the_capacity_probe_reports_the_three_numbers(self, tmp_path: pathlib.Path) -> None:
        fields = self.run_probe(tmp_path, DIALECT.capacity_probe_script())

        assert set(fields) == {"free_ram_gb", "free_disk_gb", "logical_cores"}
        assert float(fields["free_ram_gb"].replace(",", "")) > 0.0
        assert float(fields["free_disk_gb"].replace(",", "")) > 0.0
        assert int(fields["logical_cores"]) >= 1

    def test_the_toolchain_probe_reports_every_tool_and_pip_by_module(
        self, tmp_path: pathlib.Path
    ) -> None:
        fields = self.run_probe(tmp_path, DIALECT.toolchain_probe_script())

        assert set(fields) == {"python", "poetry", "git", "make", "tar", "winget", "choco", "pip"}
        assert fields["python"].startswith("yes=Python 3.")
        assert fields["pip"].startswith("yes=pip ")
        for value in fields.values():
            assert value.startswith(("yes=", "no="))


@pytest.mark.skipif(sys.platform != "win32", reason="the script is PowerShell; run it here")
class TestObserveScriptForReal:
    """The observe script, run by PowerShell 5.1 against a profile on disk.

    ``$HOME`` follows ``USERPROFILE`` (measured on the hub 2026-09-21), so
    the script reads a directory this test laid out rather than the
    operator's own sessions, and the document it prints is decoded the way
    the hub decodes it.
    """

    def run_observe(self, tmp_path: pathlib.Path, profile: pathlib.Path) -> JSONObject:
        """Run the script by path with ``profile`` as the home, decode its line.

        Args:
            tmp_path: Where the script is written.
            profile: The directory ``$HOME`` resolves to.

        Returns:
            The decoded document.
        """
        script = tmp_path / "observe-sessions.ps1"
        script.write_text(DIALECT.observe_sessions_script(), encoding="utf-8")
        parent = config_test_hooks.get_environment()
        completed = subprocess.run(
            [*POWERSHELL_INVOCATION, str(script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env={**parent, "USERPROFILE": str(profile)},
        )
        assert completed.returncode == 0, completed.stderr
        document: JSONValue = load_json_str(completed.stdout)
        if not isinstance(document, dict):
            raise AssertionError(f"the script printed {type(document).__name__}, not an object")
        return document

    def test_it_reports_every_record_verbatim_under_win32_and_the_lowercased_host(
        self, tmp_path: pathlib.Path
    ) -> None:
        profile = tmp_path / "profile"
        sessions = profile / ".claude" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "4242.json").write_text(dump_json_str(SESSION_RECORD), encoding="utf-8")

        document = self.run_observe(tmp_path, profile)

        assert document == {
            "platform": "win32",
            "hostname": socket.gethostname().lower(),
            "records": [SESSION_RECORD],
        }

    def test_it_reports_zero_records_for_a_profile_that_never_ran_claude_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        profile = tmp_path / "profile"
        profile.mkdir()

        document = self.run_observe(tmp_path, profile)

        assert document == {
            "platform": "win32",
            "hostname": socket.gethostname().lower(),
            "records": [],
        }
