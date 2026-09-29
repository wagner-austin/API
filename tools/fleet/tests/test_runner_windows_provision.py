"""The Windows provision's render: its parameters, its records and its sections.

The script itself is executed by tests/pester/rendered-provision-windows.Tests.ps1
over the committed provision-windows-<host> and onboard-windows-<host>
renders; these cases hold what the renderer decides before anything runs.
"""

from __future__ import annotations

import pytest

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_windows_provision
from tests._runner_fixtures import a_base, a_ci_slice


def _install(python: list[str]) -> RunnerInstall:
    """A windows-side install.

    Args:
        python: The versions it seeds.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/tree-bot",
        runner_name="lavender",
        side="windows",
        service="actions.runner.wagner-austin-tree-bot.lavender",
        workdir="C:/actions-runner-tree-bot/_work",
        labels=["lavender", "gpu"],
        python_toolcache=python,
    )


def _host(
    *, keepalive: str | None, floor: int | None, installs: list[RunnerInstall]
) -> HostRunnerSpec:
    """A host shaped per case.

    Args:
        keepalive: The keepalive task, or ``None``.
        floor: The memory floor in GB, or ``None``.
        installs: Its installs.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=keepalive,
        wslconfig_min_memory_gb=floor,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        installs=installs,
        assets=[],
        base=a_base(),
        ci_slice=a_ci_slice(),
    )


def _parameters(script: str) -> list[str]:
    """The param block's lines, between ``param(`` and its closing ``)``.

    Args:
        script: A rendered script.

    Returns:
        The lines inside the block.
    """
    lines = script.splitlines()
    start = lines.index("param(")
    return lines[start + 1 : lines.index(")", start)]


class TestInstallRecord:
    def test_an_install_is_one_hashtable_with_backslashed_paths_and_joined_labels(self) -> None:
        assert runner_windows_provision.render_install_record(_install(["3.11.9", "3.12.10"])) == (
            "@{ Repo = 'wagner-austin/tree-bot'; Name = 'lavender'; Labels = 'lavender,gpu'; "
            "Service = 'actions.runner.wagner-austin-tree-bot.lavender'; "
            "Directory = 'C:\\actions-runner-tree-bot'; "
            "Workdir = 'C:\\actions-runner-tree-bot\\_work'; "
            "TokenVariable = 'RUNNER_TOKEN_TREE_BOT'; Python = @('3.11.9', '3.12.10') }"
        )

    def test_an_install_seeding_nothing_carries_an_empty_python_list(self) -> None:
        assert runner_windows_provision.render_install_record(_install([])).endswith(
            "; Python = @() }"
        )

    def test_a_value_the_script_cannot_carry_is_refused(self) -> None:
        install = _install([])
        install["runner_name"] = "it's"
        with pytest.raises(ValueError, match="runner_name"):
            runner_windows_provision.render_install_record(install)


class TestProvisionScript:
    def test_a_full_host_declares_every_parameter_in_order(self) -> None:
        script = runner_windows_provision.render_windows_provision_script(
            _host(keepalive="wsl-keepalive", floor=26, installs=[_install(["3.11.9"])]),
            {"RUNNER_TOKEN_TREE_BOT": "abc123"},
        )
        assert _parameters(script) == [
            '    [string]$WslConfigPath = "$env:USERPROFILE\\.wslconfig",',
            "    [string]$KeepaliveTask = 'wsl-keepalive',",
            "    [string]$Distro = 'Ubuntu',",
            '    [string]$Wsl = "$env:SystemRoot\\System32\\wsl.exe",',
            "    [hashtable[]]$Installs = @(",
            "        " + runner_windows_provision.render_install_record(_install(["3.11.9"])),
            "    ),",
            "    [hashtable]$Tokens = @{'RUNNER_TOKEN_TREE_BOT' = 'abc123'},",
            f"    [string]$RunnerUrl = '{runner_windows_provision.RUNNER_URL}',",
            f"    [string]$PythonPackageUrl = '{runner_windows_provision.PYTHON_PACKAGE_URL}',",
            "    [string]$PythonName = 'python.exe',",
            "    [scriptblock]$GetService = { param([string]$Name) "
            "@(Get-CimInstance Win32_Service -Filter \"Name='$Name'\") },",
            "    [scriptblock]$StopService = { param([string]$Name) Stop-Service -Name $Name },",
            "    [scriptblock]$StartService = { param([string]$Name) Start-Service -Name $Name },",
            '    [string]$Sc = "$env:SystemRoot\\System32\\sc.exe"',
        ]
        assert "memory=26GB" in script
        assert "nothing declared" not in script

    def test_a_host_declaring_nothing_windows_side_says_so(self) -> None:
        script = runner_windows_provision.render_windows_provision_script(
            _host(keepalive=None, floor=None, installs=[]), {}
        )
        assert _parameters(script)[:3] == ["    [hashtable[]]$Installs = @(", "", "    ),"]
        assert "$WslConfig" not in script
        assert "Register-ScheduledTask" not in script
        assert script.endswith(
            "Write-Output 'nothing declared for the Windows side of this host'\n"
        )

    def test_a_host_with_only_installs_says_nothing_about_emptiness(self) -> None:
        script = runner_windows_provision.render_windows_provision_script(
            _host(keepalive=None, floor=None, installs=[_install([])]), {}
        )
        assert "nothing declared" not in script

    def test_a_token_the_script_cannot_carry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="token"):
            runner_windows_provision.render_windows_provision_script(
                _host(keepalive=None, floor=None, installs=[_install([])]),
                {"RUNNER_TOKEN_TREE_BOT": "a'b"},
            )


class TestOnboardScript:
    def test_onboarding_carries_the_install_loop_without_the_host_level_sections(self) -> None:
        script = runner_windows_provision.render_windows_onboard_script(
            _install([]), {"RUNNER_TOKEN_TREE_BOT": "abc123"}
        )
        assert _parameters(script)[0] == "    [hashtable[]]$Installs = @("
        assert "$WslConfigPath" not in script
        assert "$KeepaliveTask" not in script
        assert script.endswith(
            "\n".join(runner_windows_provision.render_windows_install_lines()) + "\n"
        )


class TestWslconfig:
    def test_no_memory_floor_writes_no_wslconfig_and_takes_no_path(self) -> None:
        spec = _host(keepalive=None, floor=None, installs=[])
        assert runner_windows_provision.render_wslconfig_lines(spec) == []
        assert runner_windows_provision.render_wslconfig_parameters(spec) == []
