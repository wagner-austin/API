"""Machine variables: rendered, and the audit row run for real.

The base's writes are executed in :mod:`tests.test_runner_base_render`, with
the rest of the Windows base. Here the audit driver runs under Windows
PowerShell 5.1 for a host with no installs, with functions standing in for
the host reads (``wsl``'s ``df``, the execution policy, the registry, git),
so the rows under test read the case's registry value and every other line
runs as rendered.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.runner_base import MachineVariable
from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_audit, runner_machine_env
from fleet.core.dialect_windows import POWERSHELL_INVOCATION
from tests._runner_fixtures import a_base

#: The host reads, answered as a laid host answers them except for the
#: machine environment, which reads ``$script:Held``.
_FAKES = """function wsl { $global:LASTEXITCODE = 0; return @('Used', '  46G') }
function Get-ExecutionPolicy { param([string]$Scope) return 'RemoteSigned' }
function Get-ItemProperty {
    param([string]$LiteralPath)
    if ($LiteralPath -like '*\\Session Manager\\Environment') {
        return [pscustomobject]$script:Held
    }
    [pscustomobject]@{ LongPathsEnabled = 1 }
}
function git { return 'true' }
"""


def _host(variables: list[MachineVariable]) -> HostRunnerSpec:
    """A host declaring only the base and the given variables.

    Args:
        variables: The machine variables.

    Returns:
        The spec, with no installs, assets, timers or GPU.
    """
    base = a_base()
    base["machine_environment"] = variables
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=None,
        wslconfig_min_memory_gb=None,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        installs=[],
        assets=[],
        base=base,
    )


_POETRY = MachineVariable(name="POETRY_CACHE_DIR", value="C:\\fleet\\poetry", reason="why")


class TestRendered:
    """What the base and the roster say without running."""

    def test_no_variables_render_no_lines_and_no_rows(self) -> None:
        spec = _host([])
        assert runner_machine_env.render_machine_environment_lines(spec) == []
        assert runner_machine_env.render_machine_environment_check_lines(spec) == []
        assert [check["check_id"] for check in runner_audit.expected_checks(spec)] == [
            "disk:/:ceiling-150gb:baseline-46gb@2026-09-26",
            "execution-policy:LocalMachine:RemoteSigned",
            runner_audit.LONG_PATHS_CHECK_ID,
        ]

    def test_each_variable_is_one_row_carrying_its_reason(self) -> None:
        other = MachineVariable(name="PIP_CACHE_DIR", value="C:\\fleet\\pip", reason="also")
        checks = runner_audit.expected_checks(_host([_POETRY, other]))
        assert checks[3:] == [
            runner_audit.ExpectedCheck(check_id="machine-env:POETRY_CACHE_DIR", reason="why"),
            runner_audit.ExpectedCheck(check_id="machine-env:PIP_CACHE_DIR", reason="also"),
        ]

    def test_the_base_writes_the_environment_key_and_asks_for_a_reboot(self) -> None:
        assert runner_machine_env.render_machine_environment_lines(_host([_POETRY])) == [
            "$EnvironmentKey = "
            "'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment'",
            "if ((Get-ItemProperty -LiteralPath $EnvironmentKey).'POETRY_CACHE_DIR' "
            "-cne 'C:\\fleet\\poetry') {",
            "    Set-ItemProperty -LiteralPath $EnvironmentKey -Name 'POETRY_CACHE_DIR' "
            "-Value 'C:\\fleet\\poetry'",
            "    $Reboot = $true",
            "    Write-Output 'set the machine variable POETRY_CACHE_DIR to C:\\fleet\\poetry'",
            "}",
        ]

    def test_a_value_the_script_cannot_carry_is_refused(self) -> None:
        bad = MachineVariable(name="X", value="C:\\it's", reason="why")
        with pytest.raises(ValueError) as refused:
            runner_machine_env.render_machine_environment_lines(_host([bad]))
        assert str(refused.value) == (
            'machine variable X "C:\\\\it\'s" contains "\'", which cannot be embedded in a '
            "rendered script verbatim; rename the item rather than escaping it"
        )


def _audit(tmp_path: pathlib.Path, held: str) -> list[str]:
    """Run the rendered audit for a host declaring POETRY_CACHE_DIR.

    Args:
        tmp_path: Where the script is written.
        held: What the environment key holds, a PowerShell hashtable.

    Returns:
        The CHECK lines the script emitted.
    """
    script = tmp_path / "audit.ps1"
    script.write_text(
        f"$script:Held = {held}\n" + _FAKES + runner_audit.render_audit_script(_host([_POETRY])),
        encoding="utf-8",
    )
    ran = subprocess.run(
        [*POWERSHELL_INVOCATION, str(script)], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    return [line for line in ran.stdout.splitlines() if line.startswith("CHECK machine-env:")]


@pytest.mark.host_windows
class TestTheRowRunsForReal:
    """The row executed: exact value passes, anything else drifts naming it."""

    @pytest.mark.parametrize(
        ("held", "verdict"),
        [
            ("@{ 'POETRY_CACHE_DIR' = 'C:\\fleet\\poetry' }", "OK"),
            (
                "@{ 'POETRY_CACHE_DIR' = 'C:\\FLEET\\POETRY' }",
                "DRIFT the machine environment holds POETRY_CACHE_DIR=C:\\FLEET\\POETRY; "
                "the roster says C:\\fleet\\poetry",
            ),
            (
                "@{}",
                "DRIFT the machine environment holds POETRY_CACHE_DIR=; "
                "the roster says C:\\fleet\\poetry",
            ),
        ],
    )
    def test_the_row_holds_the_exact_value(
        self, tmp_path: pathlib.Path, held: str, verdict: str
    ) -> None:
        assert _audit(tmp_path, held) == [f"CHECK machine-env:POETRY_CACHE_DIR {verdict}"]
