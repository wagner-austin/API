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
The Python install guard IS run for real here, on this machine, because it
is a Windows hub. The scripts committed under tools/fleet/rendered (the
probes, the session observer, the directory, digest, result, launch, stop,
build and log-tail scripts) are run for real by the Pester suites under
tests/pester instead, under MCPs' PowerShell harness (MCPs board task
d69786fa).
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.source import InstallStep
from fleet.contracts.toolchain import (
    PINNED_PYTHON,
    PYTHON_REGISTERED_GUARD,
    PYTHON_REGISTERED_MESSAGE,
)
from fleet.core import names
from fleet.core.dialect_windows import (
    OBSERVE_SESSIONS_SCRIPT,
    POWERSHELL_INVOCATION,
    WindowsDialect,
)
from fleet.core.names import CACHE_VARIABLE
from fleet.core.windows_task import LAUNCH_TIMEOUT_SECONDS
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID

DIALECT = WindowsDialect()


def _python_registered_here(prefix: str) -> bool:
    """Whether an uninstall entry on this machine starts with ``prefix``.

    Read through ``reg.exe`` rather than PowerShell, so the guard under test is
    checked against an answer it did not compute. ``reg query /f /d`` matches
    the text anywhere in a value's data and exits 0 when it found one.

    Args:
        prefix: The start of the entry's DisplayName.

    Returns:
        True when any entry under HKLM or HKCU carries it.
    """
    uninstall = "\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall"
    for hive in ("HKLM", "HKCU"):
        found = subprocess.run(
            ["reg", "query", hive + uninstall, "/s", "/f", prefix, "/d"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if found.returncode == 0:
            return True
    return False


def _step(phase: str, *argv: str) -> InstallStep:
    """An install step of ``phase`` running ``argv``."""
    return InstallStep(phase=phase, argv=argv)


def _build(
    *,
    path: str = DEMO_PROJECT,
    install: tuple[InstallStep, ...] = (),
    workers: int = 6,
    elevated: bool = False,
) -> str:
    """Render the Windows build script for one run under the fixture's roots.

    Args:
        path: The recipe's directory inside the export.
        install: The install steps.
        workers: The worker count.
        elevated: Whether the build was launched at RunLevel Highest.

    Returns:
        The script's text.
    """
    return DIALECT.build_script(
        target="C:/s/run-1",
        path=path,
        workers=workers,
        install=install,
        cache_root="C:/s/cache",
        isolated_docker=False,
        elevated=elevated,
        agent="opus-demo-0929",
    )


class TestBuildScript:
    """The text's parameters and order. The Pester suite over the committed
    render (tests/pester/rendered-dialect-build.Tests.ps1) runs it against
    stand-in tools, which is where the transcript and the statuses are
    measured."""

    def test_the_locations_are_parameters_defaulting_to_the_dispatch(self) -> None:
        body = _build()

        assert "[string]$Target = 'C:/s/run-1'," in body
        assert f"[string]$Recipe = 'C:/s/run-1/{DEMO_PROJECT}'," in body
        assert "[string]$CacheRoot = 'C:/s/cache'," in body
        assert "[string]$Make = 'make'," in body
        assert '[string]$Cmd = "$env:SystemRoot\\System32\\cmd.exe"' in body

    def test_a_root_project_runs_its_recipe_at_the_export_root(self) -> None:
        assert "[string]$Recipe = 'C:/s/run-1'," in _build(path="")
        assert "$env:CORVIS_FLEET_WORKSPACE = '.'" in _build(path="")
        assert f"$env:CORVIS_FLEET_WORKSPACE = '{DEMO_PROJECT}'" in _build()

    def test_it_pins_the_worker_count(self) -> None:
        body = _build()

        assert "[int]$Workers = 6," in body
        assert '$env:PYTEST_XDIST_AUTO_NUM_WORKERS = "$Workers"' in body
        assert '$env:CORVIS_TEST_MAX_WORKERS = "$Workers"' in body

    def test_it_tells_the_suite_which_lane_launched_it(self) -> None:
        """So MCPs' execution suite can assert its token is the one the fleet
        meant it to have (MCPs board task a98d7083)."""
        assert "$env:CORVIS_FLEET_ELEVATED = '0'" in _build()
        assert "$env:CORVIS_FLEET_ELEVATED = '1'" in _build(elevated=True)

    def test_it_tells_the_suite_who_asked_for_it_and_refuses_an_unembeddable_label(
        self,
    ) -> None:
        """So a fleet hold the suite takes is attributed to the submitter
        (MCPs board task 6c4516af A4)."""
        assert "$env:BOARD_AGENT_LABEL = 'opus-demo-0929'" in _build()
        with pytest.raises(ValueError, match="\"o'brien-0929\" is outside the board's"):
            DIALECT.build_script(
                target="C:/s/run-1",
                path="",
                workers=1,
                install=(),
                cache_root="C:/s/cache",
                isolated_docker=False,
                elevated=False,
                agent="o'brien-0929",
            )

    def test_it_puts_git_for_windows_bash_first_on_the_path(self) -> None:
        """So ``bash scripts/testdb-setup.sh`` runs Git's bash, never WSL's
        System32 launcher (MCPs board task daae17f2)."""
        lines = _build().splitlines()

        assert '    [string]$GitBin = "$env:ProgramFiles\\Git\\bin",' in lines
        assert lines.index('$env:PATH = "$GitBin;$env:PATH"') < lines.index(
            "Set-Location -LiteralPath $Target"
        )

    def test_it_points_the_three_package_managers_at_the_node_cache(self) -> None:
        """A clean export carries no dependencies; the node's cache is where
        they are restored from, and every run on the node shares it."""
        body = _build()

        assert '$env:npm_config_cache = "$CacheRoot/npm"' in body
        assert '$env:POETRY_CACHE_DIR = "$CacheRoot/pypoetry"' in body
        assert '$env:PLAYWRIGHT_BROWSERS_PATH = "$CacheRoot/ms-playwright"' in body

    def test_the_install_steps_are_one_string_each_in_order_beside_their_phases(self) -> None:
        body = _build(
            install=(
                _step("install", "npm", "ci"),
                _step("workspace-build", "npx", "playwright", "install", "chromium"),
            )
        )

        assert "[string[]]$Install = @('npm ci', 'npx playwright install chromium')," in body
        assert "[string[]]$InstallPhases = @('install', 'workspace-build')," in body
        assert "[string[]]$Install = @()," in _build(install=())
        assert "[string[]]$InstallPhases = @()," in _build(install=())

    def test_an_install_token_that_cannot_be_embedded_is_refused(self) -> None:
        with pytest.raises(ValueError, match='install token "npm it\'s"'):
            _build(install=(_step("install", "npm", "it's"),))

    def test_each_step_and_the_recipe_are_timed_as_phases_and_the_cache_is_named(self) -> None:
        body = _build()

        assert (
            '    $status = Invoke-Phase -Shell $Cmd -Name \'check\' -Command "`"$Make`" check"'
        ) in body
        assert (
            "$status = Invoke-Phase -Shell $Cmd -Name $InstallPhases[$index] "
            "-Command $Install[$index]"
        ) in body
        assert f"$env:{CACHE_VARIABLE} = $CacheRoot" in body
        assert "fleet-phase $Name started $(Get-PhaseStamp)" in body
        assert "fleet-phase $Name ended $(Get-PhaseStamp) after $seconds s, exit $code" in body

    def test_every_native_run_is_cmd_exes_redirection_under_the_strict_header(self) -> None:
        """So a tool's stderr never reaches PowerShell, and ``Stop`` holds."""
        body = _build()

        assert "$ErrorActionPreference = 'Stop'" in body
        assert "Continue" not in body
        assert "*>>" not in body
        assert '    & $Shell /d /s /c "$Command >> `"$log`" 2>&1"' in body
        assert f'$log = "$Target/{names.RESULT_NAME}.log"' in body
        assert "    $code = Invoke-Logged $Shell $Command" in body

    def test_it_records_the_status_last_and_exits_0(self) -> None:
        """The result file's absence is how a run is known to be unfinished.

        Written after the recipe, so it can never exist while make is still
        going -- which is what lets `fleet-collect` treat absence as running.
        """
        lines = _build().splitlines()

        assert lines[-2] == "$status | Set-Content -LiteralPath $result"
        assert lines[-1] == "exit 0"

    def test_it_reads_the_exit_code_and_not_the_success_flag(self) -> None:
        body = _build()

        assert "    return $LASTEXITCODE" in body
        assert "$?" not in body


class TestLaunchScript:
    def test_it_registers_and_starts_a_scheduled_task(self) -> None:
        """Not an ssh child. Windows OpenSSH puts that in a job object that
        dies with the connection, and this command returns immediately."""
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=False)

        assert "Register-ScheduledTask" in body
        assert "Start-ScheduledTask" in body

    def test_it_sets_priority_four(self) -> None:
        """Priority 7 is the Register-ScheduledTask default and sets LOW I/O.

        A run that inherits it crawls, and the symptom reads as a slow node
        rather than a misconfigured launch.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=False)

        assert "-Priority 4" in body
        assert "[TimeSpan]::Zero" in body
        assert "-LogonType S4U" in body

    def test_an_elevated_build_registers_at_run_level_highest_and_an_ordinary_one_limited(
        self,
    ) -> None:
        """The one difference an elevated build makes (MCPs board task
        a98d7083): the same S4U principal with the account's full token, so
        the task still outlives the ssh connection."""
        elevated = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=True)
        ordinary = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=False)

        assert "-LogonType S4U -RunLevel Highest" in elevated
        assert "RunLevel" not in ordinary
        assert elevated.replace(" -RunLevel Highest", "") == ordinary

    def test_it_runs_the_build_by_path_and_never_inlines_it(self) -> None:
        """THE REGRESSION. Interpolating the build into -Argument split the
        task in two: PowerShell ended the single-quoted string at the first
        inner quote and bound the rest to -WorkingDirectory. Measured on
        sedona 2026-09-04; the task could not be started at all.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=False)

        assert "[string]$Target = 'C:/s/run-1'" in body
        assert f'$build = "$Target/{names.BUILD_STEM}.ps1"' in body
        assert "-Command" not in body
        assert "make check" not in body

    def test_the_argument_string_contains_no_single_quotes(self) -> None:
        """The mechanical form of the same defect, asserted directly.

        -Argument is one double-quoted PowerShell string whose only
        interpolation is the build's path, so no quote from the tree can end
        it early.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=False)
        argument = body.split('-Argument "', 1)[1].split('"\n', 1)[0]

        assert argument == '-NoProfile -ExecutionPolicy Bypass -File `"$build`"'
        assert "'" not in argument

    def test_it_survives_the_lid_being_shut(self) -> None:
        """Two of the three nodes are laptops and both battery settings
        default to refusing: without these a dispatch to an unplugged sedona
        registers a task that never runs, and reports nothing.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=False)

        assert "-AllowStartIfOnBatteries" in body
        assert "-DontStopIfGoingOnBatteries" in body

    def test_it_waits_for_the_build_to_record_itself(self) -> None:
        """Start-ScheduledTask reports a refusal as a NON-terminating error.

        On 2026-09-04 it failed with 'Element not found', PowerShell exited 0,
        and the dispatch was recorded as running. The script waits for the
        build's own process id and throws by name when it never appears; the
        Pester suite over its render runs both outcomes against real tasks.
        """
        body = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID, elevated=False)

        assert f"[int]$LaunchSeconds = {LAUNCH_TIMEOUT_SECONDS}" in body
        assert f'$recorded = "$Target/{names.PID_NAME}"' in body
        assert "FLEET_LAUNCH_NOT_STARTED" in body


class TestResultAndStopScripts:
    def test_it_reports_when_as_well_as_what(self) -> None:
        """Whether a run was PROTECTED is a question about whether its lease
        covered it, and only the node knows when the build ended. Asking for
        the status alone forced the reader to substitute "is a lease held now"
        -- a question about how promptly somebody collected -- which refused a
        run that finished three minutes inside its window. Pester runs it
        (tests/pester/rendered-dialect-state.Tests.ps1)."""
        body = DIALECT.result_script("C:/s/run-1")

        assert "LastWriteTimeUtc" in body
        assert "1970-01-01" in body

    def test_it_does_not_use_uformat_for_the_epoch(self) -> None:
        """PowerShell 5.1's -UFormat %s converts from LOCAL time, which would
        put every node's answer out by its own offset."""
        assert "-UFormat" not in DIALECT.result_script("C:/s/run-1")

    def test_the_result_script_reads_the_result_file_under_its_target(self) -> None:
        """Absence is the signal, so an unfinished run is not read as exit 0."""
        body = DIALECT.result_script("C:/s/run-1")

        assert "[string]$Target = 'C:/s/run-1'" in body
        assert f'$result = "$Target/{names.RESULT_NAME}"' in body
        assert "if (Test-Path -LiteralPath $result) {" in body


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
        """The directory is made by a .NET call, not by ``New-Item``: that
        cmdlet has no ``-LiteralPath`` (measured on sedona 2026-09-22, "A
        parameter cannot be found that matches parameter name
        'LiteralPath'") and its ``-Path`` reads brackets as wildcards, so
        the only spelling that is both valid and literal is this one."""
        made = DIALECT.make_directory_script("C:/s/run-[1]")
        digested = DIALECT.digest_script("C:/s/run-1")

        assert "[string]$Directory = 'C:/s/run-[1]'" in made
        assert "[IO.Directory]::CreateDirectory($Directory) | Out-Null" in made
        assert "New-Item" not in made
        assert "LASTEXITCODE" not in made
        assert "[string]$Target = 'C:/s/run-1'" in digested
        assert digested.endswith(
            f'(Get-FileHash -Algorithm SHA256 -LiteralPath "$Target/{names.ARCHIVE_NAME}")'
            ".Hash.ToLower()\n"
        )

    def test_a_location_that_cannot_be_embedded_is_refused(self) -> None:
        with pytest.raises(ValueError, match='directory "C:/s/it\'s" contains'):
            DIALECT.make_directory_script("C:/s/it's")

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
        assert body.startswith(
            'param(\n    [string]$SessionsDirectory = "$HOME\\.claude\\sessions"\n)'
        )
        assert "if (Test-Path -LiteralPath $SessionsDirectory)" in body
        assert "-Filter '*.json'" in body
        assert "$env:COMPUTERNAME.ToLowerInvariant()" in body
        assert "platform = 'win32'" in body
        assert "ConvertTo-Json -Depth 8 -Compress" in body


@pytest.mark.host_windows
class TestTheInstallGuardForReal:
    """The Python install guard, executed on this hub against its own
    registry. The toolchain probe moved to the Pester suite over
    tools/fleet/rendered, which lays out the PATH it reads."""

    def test_the_python_install_guard_stops_exactly_where_the_version_is_registered(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Run on this hub, against its own registry, read independently."""
        script = tmp_path / "guard.ps1"
        script.write_text(PYTHON_REGISTERED_GUARD + "Write-Output 'clear'\n", encoding="utf-8")

        completed = subprocess.run(
            [*POWERSHELL_INVOCATION, str(script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        if _python_registered_here(f"Python {PINNED_PYTHON} Core Interpreter"):
            assert completed.returncode == 1
            assert completed.stderr.strip() == PYTHON_REGISTERED_MESSAGE
            assert completed.stdout.strip() == ""
        else:
            assert completed.returncode == 0, completed.stderr
            assert completed.stdout.strip() == "clear"


class TestToolchainProbeText:
    """The text's contract. The Pester suite over the committed render
    (tests/pester/rendered-dialect-toolchain.Tests.ps1) runs it against a PATH
    of stand-in tools, a python under WindowsApps (lavender's runner,
    2026-09-22/23, board task 465689f5) and a pip that fails."""

    def test_it_runs_under_the_strict_header_and_suppresses_nothing(self) -> None:
        body = DIALECT.toolchain_probe_script()

        assert "Set-StrictMode -Version Latest\n$ErrorActionPreference = 'Stop'\n" in body
        assert "SilentlyContinue" not in body
        assert "Get-Command" not in body

    def test_every_tool_answers_through_cmd_exes_redirection(self) -> None:
        body = DIALECT.toolchain_probe_script()

        assert '[string]$Cmd = "$env:SystemRoot\\System32\\cmd.exe",' in body
        assert '    $lines = & $Shell /d /s /c "`"$Path`" $Arguments 2>&1"' in body
        assert body.count(" 2>&1") == 1
