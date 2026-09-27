"""The WSL keepalive's parameters and lines, as the Windows provision renders them.

The registration is executed by tests/pester/rendered-provision-windows.Tests.ps1
over the committed provision: a real task under a minted name whose action
is a stand-in wsl.exe, read back through Task Scheduler's COM service as S4U
at Highest, triggered at boot, restarting 999 times with no time limit, and
deleted afterwards.
"""

from __future__ import annotations

import pytest

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_keepalive
from tests._runner_fixtures import a_base


def _host(keepalive: str | None) -> HostRunnerSpec:
    """A host whose only Windows-side declaration is the keepalive.

    Args:
        keepalive: The keepalive task's name, or ``None``.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=keepalive,
        wslconfig_min_memory_gb=None,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        installs=[],
        assets=[],
        base=a_base(),
    )


class TestRendered:
    """What renders without running."""

    def test_no_declared_task_renders_nothing(self) -> None:
        assert runner_keepalive.render_keepalive_parameters(_host(None)) == []
        assert runner_keepalive.render_keepalive_lines(_host(None)) == []

    def test_the_task_the_distro_and_wsl_are_parameters(self) -> None:
        assert runner_keepalive.render_keepalive_parameters(_host("wsl-keepalive")) == [
            "    [string]$KeepaliveTask = 'wsl-keepalive',",
            "    [string]$Distro = 'Ubuntu',",
            '    [string]$Wsl = "$env:SystemRoot\\System32\\wsl.exe",',
        ]

    def test_the_principal_is_the_running_account_through_s4u(self) -> None:
        lines = runner_keepalive.render_keepalive_lines(_host("wsl-keepalive"))
        assert lines[2:4] == [
            "$KeepaliveUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name",
            "$KeepalivePrincipal = New-ScheduledTaskPrincipal -UserId $KeepaliveUser "
            "-LogonType S4U -RunLevel Highest",
        ]

    def test_a_task_name_the_script_cannot_carry_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            runner_keepalive.render_keepalive_parameters(_host("wsl'keepalive"))
        assert str(refused.value) == (
            'keepalive_task "wsl\'keepalive" contains "\'", which cannot be embedded in a '
            "rendered script verbatim; rename the item rather than escaping it"
        )
