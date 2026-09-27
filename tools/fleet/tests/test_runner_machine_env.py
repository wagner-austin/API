"""Machine variables: what the base and the audit render for them.

Both renders are executed by their Pester suites over the committed files
under rendered/: the base's writes in rendered-base-windows.Tests.ps1, the
audit row in rendered-audit.Tests.ps1, each against a scratch HKCU key.
"""

from __future__ import annotations

import pytest

from fleet.contracts.runner_base import MachineVariable
from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_audit, runner_machine_env
from tests._runner_fixtures import a_base


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
        assert runner_machine_env.render_machine_environment_parameters(spec)[1] == (
            "    [string[]]$MachineVariables = @(),"
        )
        assert runner_machine_env.render_machine_environment_check_lines(spec) == []
        assert [check["check_id"] for check in runner_audit.expected_checks(spec)] == [
            "disk:/:ceiling-150gb:baseline-46gb@2026-09-26",
            "cache:/home/gharunner/.cache:ceiling-60gb",
            "execution-policy:LocalMachine:RemoteSigned",
            runner_audit.LONG_PATHS_CHECK_ID,
        ]

    def test_each_variable_is_one_row_carrying_its_reason(self) -> None:
        other = MachineVariable(name="PIP_CACHE_DIR", value="C:\\fleet\\pip", reason="also")
        checks = runner_audit.expected_checks(_host([_POETRY, other]))
        assert checks[4:] == [
            runner_audit.ExpectedCheck(check_id="machine-env:POETRY_CACHE_DIR", reason="why"),
            runner_audit.ExpectedCheck(check_id="machine-env:PIP_CACHE_DIR", reason="also"),
        ]

    def test_the_base_takes_the_key_and_the_variables_as_parameters(self) -> None:
        """The Pester suite over the committed base render passes a scratch
        HKCU key and reads each variable back from it."""
        other = MachineVariable(name="PIP_CACHE_DIR", value="C:\\fleet\\pip", reason="also")
        assert runner_machine_env.render_machine_environment_parameters(
            _host([_POETRY, other])
        ) == [
            "    [string]$EnvironmentKey = "
            "'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment',",
            "    [string[]]$MachineVariables = @('POETRY_CACHE_DIR=C:\\fleet\\poetry', "
            "'PIP_CACHE_DIR=C:\\fleet\\pip'),",
        ]

    def test_a_changed_variable_is_written_and_asks_for_a_reboot(self) -> None:
        lines = runner_machine_env.render_machine_environment_lines()

        assert "    $name, $value = $pair -split '=', 2" in lines
        write = "        Set-ItemProperty -LiteralPath $EnvironmentKey -Name $name -Value $value"
        assert write in lines
        assert "        $Reboot = $true" in lines

    def test_a_value_the_script_cannot_carry_is_refused(self) -> None:
        bad = MachineVariable(name="X", value="C:\\it's", reason="why")
        with pytest.raises(ValueError) as refused:
            runner_machine_env.render_machine_environment_parameters(_host([bad]))
        assert str(refused.value) == (
            'machine variable X "C:\\\\it\'s" contains "\'", which cannot be embedded in a '
            "rendered script verbatim; rename the item rather than escaping it"
        )

    def test_the_audit_row_reads_the_key_parameter_and_compares_case_sensitively(self) -> None:
        """The row executed is the Pester suite over the committed audit
        render (tests/pester/rendered-audit.Tests.ps1): exact value, a
        case-only difference, and an absent variable."""
        assert runner_machine_env.render_machine_environment_check_lines(_host([_POETRY])) == [
            "$Held = [string](Get-Item -LiteralPath $EnvironmentKey).GetValue('POETRY_CACHE_DIR')",
            "Write-Check 'machine-env:POETRY_CACHE_DIR' ($Held -ceq 'C:\\fleet\\poetry') "
            "('the machine environment holds POETRY_CACHE_DIR=' + $Held + "
            "'; the roster says C:\\fleet\\poetry')",
        ]
