"""The rebuild's base stages: rendered, and the PowerShell ones run for real.

The Windows base and the import are executed under Windows PowerShell 5.1,
the interpreter the host runs them in, with functions defined ahead of the
rendered script standing in for the cmdlets that would change this machine
(features, msiexec, the download, the policy write, wsl.exe). PowerShell
resolves a function before a cmdlet or an executable of the same name, so
every other line runs as rendered -- ``Get-FileHash`` included, which is the
digest check under test. The bash stages need a distro and root, so they are
asserted as text.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

import pytest

from fleet.contracts.runner_base import PinnedDownload
from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_base_render
from fleet.core.dialect_windows import POWERSHELL_INVOCATION
from tests._runner_fixtures import a_base

#: The bytes every fake download writes.
_PAYLOAD = "fleet rebuild test payload"

#: Their digest, which the pins in :func:`_host` carry unless a case says not.
_PAYLOAD_SHA = hashlib.sha256(_PAYLOAD.encode("utf-8")).hexdigest()


def _host(scratch: pathlib.Path, *, pin: str = _PAYLOAD_SHA) -> HostRunnerSpec:
    """A host whose scratch directory is the test's own.

    Args:
        scratch: The directory downloads land in.
        pin: The digest both pinned downloads carry.

    Returns:
        The spec: no memory floor and no machine PATH entries, so running
        its Windows base touches neither this machine's ``.wslconfig`` nor
        its PATH.
    """
    base = a_base()
    base["machine_path_entries"] = []
    base["wsl_msi"] = PinnedDownload(
        version="2.7.14", url="https://example.invalid/wsl.2.7.14.0.x64.msi", sha256=pin
    )
    base["rootfs"] = PinnedDownload(
        version="24.04-20240423", url="https://example.invalid/ubuntu.rootfs.tar.gz", sha256=pin
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
    )


#: Stand-ins for everything the Windows base would change on this machine.
_FAKES = f"""function Get-WindowsOptionalFeature {{
    param([switch]$Online, [string]$FeatureName)
    [pscustomobject]@{{ State = $script:Features[$FeatureName] }}
}}
function Enable-WindowsOptionalFeature {{
    param([switch]$Online, [string]$FeatureName, [switch]$All, [switch]$NoRestart)
    $script:Features[$FeatureName] = 'Enabled'
    [pscustomobject]@{{ RestartNeeded = $script:Restart }}
}}
function Invoke-WebRequest {{
    param([string]$Uri, [string]$OutFile, [switch]$UseBasicParsing)
    [IO.File]::WriteAllText($OutFile, '{_PAYLOAD}')
}}
function Start-Process {{
    param([string]$FilePath, [object[]]$ArgumentList, [switch]$Wait, [switch]$PassThru)
    [pscustomobject]@{{ ExitCode = $script:MsiExit }}
}}
function Get-ExecutionPolicy {{ param([string]$Scope) $script:Policy }}
function Set-ItemProperty {{
    param([string]$LiteralPath, [string]$Name, [string]$Value)
    $script:Policy = $Value
}}
function wsl {{
    if ($args[0] -eq '--version') {{ return $script:WslVersion }}
    if ($args[0] -eq '--list') {{ return $script:Registered }}
    if ($args[0] -eq '--import') {{
        Write-Output ('IMPORTED ' + ($args[1..3] -join ' '))
        $global:LASTEXITCODE = $script:ImportExit
        return
    }}
    throw "unexpected wsl call: $args"
}}
"""


def _run(tmp_path: pathlib.Path, prelude: str, script: str) -> subprocess.CompletedProcess[str]:
    """Run a rendered script under Windows PowerShell 5.1 with the fakes.

    Args:
        tmp_path: Where the combined script is written.
        prelude: The case's variable assignments.
        script: The rendered script.

    Returns:
        The completed process.
    """
    path = tmp_path / "stage.ps1"
    path.write_text(prelude + _FAKES + script, encoding="utf-8")
    return subprocess.run(
        [*POWERSHELL_INVOCATION, str(path)], capture_output=True, text=True, check=False
    )


def _thrown(ran: subprocess.CompletedProcess[str], tmp_path: pathlib.Path) -> str:
    """The message a script threw, from a run that must have failed.

    Windows PowerShell writes a thrown message first on stderr, then
    ``At <script>:<line>``, and wraps every line at the console width, so
    the message is rejoined and cut at its location line.

    Args:
        ran: The completed process.
        tmp_path: Where :func:`_run` wrote the script.

    Returns:
        The thrown message.
    """
    assert ran.returncode == 1, ran.stdout
    joined = ran.stderr.replace("\n", "")
    return joined.split(f"At {tmp_path / 'stage.ps1'}")[0]


def _lines(ran: subprocess.CompletedProcess[str]) -> list[str]:
    """The non-empty output lines of a run that must have exited 0.

    Args:
        ran: The completed process.

    Returns:
        Its output lines.
    """
    assert ran.returncode == 0, ran.stderr
    return [line for line in ran.stdout.splitlines() if line.strip()]


_FRESH = """$script:Features = @{ 'VirtualMachinePlatform' = 'Disabled'; \
'Microsoft-Windows-Subsystem-Linux' = 'Enabled' }
$script:Restart = $true
$script:WslVersion = @()
$script:MsiExit = 3010
$script:Policy = 'Restricted'
"""

_LAID = """$script:Features = @{ 'VirtualMachinePlatform' = 'Enabled'; \
'Microsoft-Windows-Subsystem-Linux' = 'Enabled' }
$script:Restart = $false
$script:WslVersion = @('WSL version: 2.7.14.0', 'Kernel version: 6.6.87.2-1')
$script:MsiExit = 0
$script:Policy = 'RemoteSigned'
"""


@pytest.mark.skipif(sys.platform != "win32", reason="the stages are PowerShell; run them here")
class TestTheWindowsBaseRunsForReal:
    """Features, the WSL release, the policy, and the restart marker."""

    def test_a_stock_install_is_laid_and_asks_for_a_restart(self, tmp_path: pathlib.Path) -> None:
        script = runner_base_render.render_windows_base_script(_host(tmp_path))
        assert _lines(_run(tmp_path, _FRESH, script)) == [
            "enabled Windows feature VirtualMachinePlatform",
            "installed WSL 2.7.14",
            "set the LocalMachine execution policy to RemoteSigned",
            runner_base_render.REBOOT_MARKER,
        ]
        assert not (tmp_path / "wsl.2.7.14.0.x64.msi").exists()

    def test_a_laid_host_changes_nothing_and_asks_for_nothing(self, tmp_path: pathlib.Path) -> None:
        script = runner_base_render.render_windows_base_script(_host(tmp_path))
        assert _lines(_run(tmp_path, _LAID, script)) == []

    def test_an_msi_that_needs_no_restart_asks_for_none(self, tmp_path: pathlib.Path) -> None:
        prelude = _LAID.replace("@('WSL version: 2.7.14.0', 'Kernel version: 6.6.87.2-1')", "@()")
        script = runner_base_render.render_windows_base_script(_host(tmp_path))
        assert _lines(_run(tmp_path, prelude, script)) == ["installed WSL 2.7.14"]

    def test_a_download_that_fails_its_digest_is_refused_and_removed(
        self, tmp_path: pathlib.Path
    ) -> None:
        script = runner_base_render.render_windows_base_script(_host(tmp_path, pin="0" * 64))
        ran = _run(tmp_path, _FRESH, script)
        assert _thrown(ran, tmp_path) == (
            f"WSL MSI sha256 {_PAYLOAD_SHA} does not match the pin {'0' * 64}"
        )
        assert not (tmp_path / "wsl.2.7.14.0.x64.msi").exists()

    def test_a_failed_msiexec_throws_with_its_exit_code(self, tmp_path: pathlib.Path) -> None:
        prelude = _FRESH.replace("$script:MsiExit = 3010", "$script:MsiExit = 1603")
        script = runner_base_render.render_windows_base_script(_host(tmp_path))
        ran = _run(tmp_path, prelude, script)
        assert _thrown(ran, tmp_path) == "msiexec for WSL 2.7.14 exited 1603"


@pytest.mark.skipif(sys.platform != "win32", reason="the stages are PowerShell; run them here")
class TestTheImportRunsForReal:
    """The pinned image imported once, and never over a registered distro."""

    def test_a_registered_distro_is_left_alone(self, tmp_path: pathlib.Path) -> None:
        prelude = "$script:Registered = @('Ubuntu', 'docker-desktop')\n$script:ImportExit = 0\n"
        script = runner_base_render.render_import_script(_host(tmp_path))
        assert _lines(_run(tmp_path, prelude, script)) == []

    def test_an_absent_distro_is_imported_from_the_verified_image(
        self, tmp_path: pathlib.Path
    ) -> None:
        prelude = "$script:Registered = @()\n$script:ImportExit = 0\n"
        script = runner_base_render.render_import_script(_host(tmp_path))
        image = f"{tmp_path.as_posix()}/ubuntu.rootfs.tar.gz"
        assert _lines(_run(tmp_path, prelude, script)) == [
            f"IMPORTED Ubuntu {tmp_path.as_posix()}/distro {image}",
            "imported Ubuntu 24.04-20240423",
        ]
        assert (tmp_path / "distro").is_dir()
        assert not (tmp_path / "ubuntu.rootfs.tar.gz").exists()

    def test_a_failed_import_throws_with_its_exit_code(self, tmp_path: pathlib.Path) -> None:
        prelude = "$script:Registered = @()\n$script:ImportExit = 1\n"
        ran = _run(tmp_path, prelude, runner_base_render.render_import_script(_host(tmp_path)))
        assert _thrown(ran, tmp_path) == "wsl --import Ubuntu exited 1"
        # The verified image stays, so the re-run the error asks for resumes
        # without fetching 357 MB again.
        assert (tmp_path / "ubuntu.rootfs.tar.gz").read_text(encoding="utf-8") == _PAYLOAD


class TestRenderedText:
    """What the text claims, for the stages that cannot run here."""

    def test_the_windows_base_writes_the_policy_registry_value_directly(self) -> None:
        script = runner_base_render.render_windows_base_script(_host(pathlib.Path("C:/stage")))
        assert "Set-ItemProperty -LiteralPath $PolicyKey -Name ExecutionPolicy" in script
        assert "Set-ExecutionPolicy" not in script

    def test_a_machine_path_entry_is_appended_once(self) -> None:
        spec = _host(pathlib.Path("C:/stage"))
        spec["base"]["machine_path_entries"] = ["C:\\Program Files (x86)\\GnuWin32\\bin"]
        script = runner_base_render.render_windows_base_script(spec)
        assert (
            "if (@($MachinePath -split ';') -notcontains 'C:\\Program Files (x86)\\GnuWin32\\bin')"
            in script
        )

    def test_a_memory_floor_is_written_before_the_distro_first_starts(self) -> None:
        spec = _host(pathlib.Path("C:/stage"))
        spec["wslconfig_min_memory_gb"] = 26
        assert "memory=26GB" in runner_base_render.render_windows_base_script(spec)

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

    def test_an_unscriptable_roster_value_is_refused(self) -> None:
        spec = _host(pathlib.Path("C:/stage"))
        spec["wsl_distro"] = "Ub'untu"
        with pytest.raises(ValueError, match="cannot be embedded"):
            runner_base_render.render_import_script(spec)
