"""The rebuild's base stages, as rendered text.

The Windows base and the import are executed from their committed renders
under rendered/ by tests/pester/rendered-base-windows.Tests.ps1 and
rendered-base-import.Tests.ps1, under MCPs' PowerShell harness, with
stand-ins passed through their parameters (MCPs board task d69786fa). Until
then they ran here with PowerShell functions shadowing the cmdlets they
call, which the harness's standard does not admit. The bash stages need a
distro and root, so they are asserted as text. What stays here is what the
text itself must carry: the defaults a host runs with, the header, and the
refusals the renderer raises.
"""

from __future__ import annotations

import pathlib

import pytest

from fleet.contracts.runner_base import PinnedDownload
from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_base_render
from tests._runner_fixtures import a_base, a_ci_slice

#: A digest both pinned downloads carry.
_PIN = "ab" * 32


def _host(scratch: pathlib.Path) -> HostRunnerSpec:
    """A host whose scratch directory is the case's own.

    Args:
        scratch: The directory downloads land in.

    Returns:
        The spec: no memory floor and no machine PATH entries.
    """
    base = a_base()
    base["machine_path_entries"] = []
    base["wsl_msi"] = PinnedDownload(
        version="2.7.14", url="https://example.invalid/wsl.2.7.14.0.x64.msi", sha256=_PIN
    )
    base["rootfs"] = PinnedDownload(
        version="24.04-20240423", url="https://example.invalid/ubuntu.rootfs.tar.gz", sha256=_PIN
    )
    base["distro_dir"] = f"{scratch.as_posix()}/distro"
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=None,
        wslconfig_min_memory_gb=None,
        scratch_dir=scratch.as_posix(),
        gpu_required=False,
        systemd_timers=[],
        job_timeout_minutes=360,
        installs=[
            RunnerInstall(
                repo="wagner-austin/API",
                runner_name="lavender-wsl",
                side="wsl",
                service="actions.runner.wagner-austin-API.lavender-wsl.service",
                workdir="/home/gharunner/actions-runner-api-1/_work",
                labels=["lavender-wsl"],
                python_toolcache=[],
            )
        ],
        assets=[],
        base=base,
        ci_slice=a_ci_slice(),
    )


class TestTheWindowsBaseText:
    def test_the_roster_values_are_the_defaults_a_host_runs_with(self) -> None:
        script = runner_base_render.render_windows_base_script(_host(pathlib.Path("C:/stage")))

        assert "    [string]$Scratch = 'C:/stage'," in script
        assert "    [string]$WslVersion = '2.7.14'," in script
        assert f"    [string]$MsiSha256 = '{_PIN}'," in script
        assert "    [string]$MsiFile = 'wsl.2.7.14.0.x64.msi'," in script
        assert f"    [string]$PolicyKey = '{runner_base_render.EXECUTION_POLICY_KEY}'," in script
        assert "    [string[]]$PathEntries = @()," in script
        assert '    [string]$Wsl = "$env:SystemRoot\\System32\\wsl.exe"' in script

    def test_it_runs_under_the_strict_header_with_nothing_suppressed(self) -> None:
        script = runner_base_render.render_windows_base_script(_host(pathlib.Path("C:/stage")))

        assert "Set-StrictMode -Version Latest\n$ErrorActionPreference = 'Stop'\n" in script
        assert "2>$null" not in script
        assert "WindowsOptionalFeature" not in script
        assert "Get-ExecutionPolicy" not in script
        assert "Set-ExecutionPolicy" not in script

    def test_features_and_the_msi_ask_for_a_restart_by_the_one_exit_code(self) -> None:
        script = runner_base_render.render_windows_base_script(_host(pathlib.Path("C:/stage")))

        restart = f"-eq {runner_base_render.REBOOT_REQUIRED_EXIT})"
        assert script.count(restart) == 2

    def test_a_machine_path_entry_is_a_default(self) -> None:
        spec = _host(pathlib.Path("C:/stage"))
        spec["base"]["machine_path_entries"] = ["C:\\Program Files (x86)\\GnuWin32\\bin"]

        script = runner_base_render.render_windows_base_script(spec)

        assert "    [string[]]$PathEntries = @('C:\\Program Files (x86)\\GnuWin32\\bin')," in script

    def test_a_memory_floor_is_written_before_the_distro_first_starts(self) -> None:
        spec = _host(pathlib.Path("C:/stage"))
        spec["wslconfig_min_memory_gb"] = 26

        script = runner_base_render.render_windows_base_script(spec)

        assert "memory=26GB" in script
        assert '    [string]$WslConfigPath = "$env:USERPROFILE\\.wslconfig",' in script

    def test_no_floor_declares_no_wslconfig_path(self) -> None:
        script = runner_base_render.render_windows_base_script(_host(pathlib.Path("C:/stage")))

        assert "WslConfigPath" not in script


class TestTheImportText:
    def test_registrations_are_read_from_the_lxss_key_not_from_wsl_list(self) -> None:
        script = runner_base_render.render_import_script(_host(pathlib.Path("C:/stage")))

        assert f"    [string]$LxssKey = '{runner_base_render.LXSS_KEY}'," in script
        assert "--list" not in script
        assert "    & $Wsl --import $Distro $DistroDir $image --version 2" in script

    def test_the_roster_values_are_the_defaults_a_host_runs_with(self) -> None:
        script = runner_base_render.render_import_script(_host(pathlib.Path("C:/stage")))

        assert "    [string]$Distro = 'Ubuntu'," in script
        assert "    [string]$DistroDir = 'C:/stage/distro'," in script
        assert "    [string]$RootfsFile = 'ubuntu.rootfs.tar.gz'," in script
        assert "Set-StrictMode -Version Latest\n$ErrorActionPreference = 'Stop'\n" in script


class TestTheBashStages:
    def test_the_wsl_conf_stage_rewrites_only_a_differing_file(self) -> None:
        script = runner_base_render.render_wslconf_script()
        assert runner_base_render.WSL_CONF.rstrip("\n") in script
        assert 'if [ "$(cat /etc/wsl.conf 2>/dev/null)" != "$want" ]; then' in script
        assert f"echo {runner_base_render.WSLCONF_CHANGED_MARKER}" in script

    def test_the_linux_base_refuses_without_systemd_and_installs_the_roster(self) -> None:
        script = runner_base_render.render_linux_base_script(_host(pathlib.Path("C:/stage")))
        assert 'test "$(ps -p 1 -o comm=)" = systemd' in script
        assert "apt-get install -y -qq build-essential docker.io" in script
        assert "usermod -aG docker gharunner" in script
        assert script.rstrip().endswith("docker reachable as gharunner'")

    def test_a_package_name_that_is_not_plain_is_refused(self) -> None:
        spec = _host(pathlib.Path("C:/stage"))
        spec["base"]["apt_packages"] = ["docker.io; rm -rf /"]
        with pytest.raises(ValueError, match="not a plain package name"):
            runner_base_render.render_linux_base_script(spec)


class TestRefusals:
    def test_an_unscriptable_roster_value_is_refused(self) -> None:
        spec = _host(pathlib.Path("C:/stage"))
        spec["wsl_distro"] = "Ub'untu"
        with pytest.raises(ValueError, match="cannot be embedded"):
            runner_base_render.render_import_script(spec)

    def test_an_unscriptable_array_value_is_refused(self) -> None:
        spec = _host(pathlib.Path("C:/stage"))
        spec["base"]["windows_features"] = ["Virtual'Machine"]
        with pytest.raises(ValueError, match="windows feature"):
            runner_base_render.render_windows_base_script(spec)
