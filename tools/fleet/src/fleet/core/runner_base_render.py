"""Rendering the stages that turn a stock Windows install into a runner host.

``fleet-runners --rebuild`` (:mod:`fleet.core.runner_rebuild`) lays these,
in this order, before the roster's own ``provision.ps1`` and
``provision.sh`` (:mod:`fleet.core.runner_render`):

  windows base   PowerShell on the host: the optional features WSL needs,
                 the pinned WSL release, the execution policy, the machine
                 PATH and ``.wslconfig``. Prints :data:`REBOOT_MARKER` when
                 Windows asks for a restart before WSL can start a VM.
  import         PowerShell on the host: the pinned distro image, imported
                 as the roster's ``wsl_distro`` when it is not registered.
  wsl.conf       bash inside the distro: systemd as PID 1 and root as the
                 default user. Prints :data:`WSLCONF_CHANGED_MARKER` when it
                 wrote the file, because the distro must then be restarted.
  linux base     bash inside the distro: the roster's packages, docker
                 enabled, and the runner account in the docker group.

Each stage is IDEMPOTENT, and that is the design rather than a courtesy: a
rebuild interrupted by a reboot, a dropped ssh session or a failed download
is finished by running the same command again, and a stage whose work is
already done does nothing. Every step was learned by hand on the reinstalled
lavender on 2026-09-26 (board task 1aa6a021); the notes on that task's
thread, and on 9e5c7a17's, are the measurements.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_render
from fleet.core.script_values import scriptable

#: The line the Windows base stage prints when Windows must restart first.
REBOOT_MARKER = "FLEET-REBOOT-REQUIRED"

#: The line the wsl.conf stage prints when it rewrote the file.
WSLCONF_CHANGED_MARKER = "FLEET-WSLCONF-CHANGED"

#: The distro's ``/etc/wsl.conf``. systemd as PID 1 is what runs docker and
#: every runner service; root as the default user is what the keepalive task
#: and the audit's ``wsl -d`` calls expect.
WSL_CONF = "[boot]\nsystemd=true\n\n[user]\ndefault=root\n"

#: msiexec's exit code for "installed; a restart completes it".
MSI_REBOOT_EXIT = 3010


def _file_name(url: str) -> str:
    """The last path segment of a download URL.

    Args:
        url: The URL.

    Returns:
        The file name the download is saved under.
    """
    return url.rsplit("/", 1)[1]


def _verified_download_lines(url: str, sha256: str, target: str, label: str) -> list[str]:
    """PowerShell lines that download a pinned file and refuse wrong bytes.

    Args:
        url: The pinned URL.
        sha256: The pinned digest, lowercase.
        target: The Windows path to save to, already validated.
        label: What the file is, for the refusal.

    Returns:
        The lines. A file already at ``target`` is re-verified rather than
        re-downloaded, so an interrupted rebuild resumes without fetching
        the image twice, and a corrupt leftover is refused by its digest.
    """
    return [
        f"if (-not (Test-Path '{target}')) {{",
        f"    Invoke-WebRequest -Uri '{url}' -OutFile '{target}' -UseBasicParsing",
        "}",
        f"$Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath '{target}').Hash.ToLower()",
        f"if ($Digest -ne '{sha256}') {{",
        f"    Remove-Item -LiteralPath '{target}'",
        f"    throw ('{label} sha256 ' + $Digest + ' does not match the pin {sha256}')",
        "}",
    ]


def render_windows_base_script(spec: HostRunnerSpec) -> str:
    """The Windows base stage for one host.

    Args:
        spec: The host's roster entry.

    Returns:
        The complete PowerShell text. It prints :data:`REBOOT_MARKER` when
        an enabled feature or the WSL install asks for a restart.

    Raises:
        ValueError: When a roster value cannot be embedded verbatim.
    """
    base = spec["base"]
    scratch = scriptable(spec["scratch_dir"], label="scratch_dir")
    lines: list[str] = [
        "$ErrorActionPreference = 'Stop'",
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
        "$ProgressPreference = 'SilentlyContinue'",
        "$env:WSL_UTF8 = '1'",
        "$Reboot = $false",
        f"New-Item -ItemType Directory -Force -Path '{scratch}' | Out-Null",
    ]
    for raw_feature in base["windows_features"]:
        feature = scriptable(raw_feature, label="windows feature")
        lines += [
            f"$Feature = Get-WindowsOptionalFeature -Online -FeatureName '{feature}'",
            "if ($Feature.State -ne 'Enabled') {",
            f"    $Enabled = Enable-WindowsOptionalFeature -Online -FeatureName '{feature}' "
            "-All -NoRestart",
            "    if ($Enabled.RestartNeeded) { $Reboot = $true }",
            f"    Write-Output 'enabled Windows feature {feature}'",
            "}",
        ]
    msi = base["wsl_msi"]
    version = scriptable(msi["version"], label="wsl_msi version")
    msi_path = f"{scratch}/{scriptable(_file_name(msi['url']), label='wsl_msi file')}"
    lines += [
        # wsl --version is absent or different on a stock install; the MSI's
        # own wsl.exe reports 'WSL version: <version>.0'.
        "$WslVersion = (@(wsl --version 2>$null) -join ' ')",
        f"if ($WslVersion -notmatch ('WSL[^:]*:\\s*' + [regex]::Escape('{version}'))) {{",
        *(
            "    " + line
            for line in _verified_download_lines(
                scriptable(msi["url"], label="wsl_msi url"), msi["sha256"], msi_path, "WSL MSI"
            )
        ),
        f"    $Install = Start-Process msiexec.exe -ArgumentList @('/i', '{msi_path}', "
        "'/qn', '/norestart') -Wait -PassThru",
        f"    if ($Install.ExitCode -eq {MSI_REBOOT_EXIT}) {{ $Reboot = $true }}",
        f"    elseif ($Install.ExitCode -ne 0) {{ throw ('msiexec for WSL {version} exited ' "
        "+ $Install.ExitCode) }",
        f"    Remove-Item -LiteralPath '{msi_path}'",
        f"    Write-Output 'installed WSL {version}'",
        "}",
    ]
    policy = scriptable(base["execution_policy"], label="execution_policy")
    lines += [
        # The registry value Set-ExecutionPolicy -Scope LocalMachine writes,
        # written directly: under this script's own Process-scope Bypass the
        # cmdlet reports the LocalMachine change as overridden and, with
        # ErrorActionPreference Stop, throws after succeeding.
        "$PolicyKey = 'HKLM:\\SOFTWARE\\Microsoft\\PowerShell\\1\\ShellIds\\Microsoft.PowerShell'",
        f"if ((@(Get-ExecutionPolicy -Scope LocalMachine) -join '') -ne '{policy}') {{",
        f"    Set-ItemProperty -LiteralPath $PolicyKey -Name ExecutionPolicy -Value '{policy}'",
        f"    Write-Output 'set the LocalMachine execution policy to {policy}'",
        "}",
    ]
    for raw_entry in base["machine_path_entries"]:
        entry = scriptable(raw_entry, label="machine PATH entry")
        lines += [
            "$MachinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')",
            f"if (@($MachinePath -split ';') -notcontains '{entry}') {{",
            "    [Environment]::SetEnvironmentVariable('Path', "
            f"($MachinePath.TrimEnd(';') + ';{entry}'), 'Machine')",
            f"    Write-Output 'added to the machine PATH: {entry}'",
            "}",
        ]
    lines += runner_render.render_wslconfig_lines(spec)
    lines += [f"if ($Reboot) {{ Write-Output '{REBOOT_MARKER}' }}", "exit 0"]
    return "\n".join(lines) + "\n"


def render_import_script(spec: HostRunnerSpec) -> str:
    """The distro import stage for one host.

    Args:
        spec: The host's roster entry.

    Returns:
        The complete PowerShell text: the pinned image downloaded, verified
        and imported as ``wsl_distro`` under ``distro_dir`` when the distro
        is not registered, and nothing when it is.

    Raises:
        ValueError: When a roster value cannot be embedded verbatim.
    """
    base = spec["base"]
    distro = scriptable(spec["wsl_distro"], label="wsl_distro")
    scratch = scriptable(spec["scratch_dir"], label="scratch_dir")
    distro_dir = scriptable(base["distro_dir"], label="distro_dir")
    rootfs = base["rootfs"]
    image = f"{scratch}/{scriptable(_file_name(rootfs['url']), label='rootfs file')}"
    version = scriptable(rootfs["version"], label="rootfs version")
    lines: list[str] = [
        "$ErrorActionPreference = 'Stop'",
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
        "$ProgressPreference = 'SilentlyContinue'",
        "$env:WSL_UTF8 = '1'",
        "$Registered = @(wsl --list --quiet 2>$null | ForEach-Object { $_.Trim() })",
        f"if ($Registered -notcontains '{distro}') {{",
        *(
            "    " + line
            for line in _verified_download_lines(
                scriptable(rootfs["url"], label="rootfs url"), rootfs["sha256"], image, "rootfs"
            )
        ),
        f"    New-Item -ItemType Directory -Force -Path '{distro_dir}' | Out-Null",
        f"    wsl --import '{distro}' '{distro_dir}' '{image}' --version 2",
        "    if ($LASTEXITCODE -ne 0) {",
        f"        throw ('wsl --import {distro} exited ' + $LASTEXITCODE)",
        "    }",
        f"    Remove-Item -LiteralPath '{image}'",
        f"    Write-Output 'imported {distro} {version}'",
        "}",
        "exit 0",
    ]
    return "\n".join(lines) + "\n"


def render_wslconf_script() -> str:
    """The wsl.conf stage, run inside the distro as root.

    Returns:
        The bash text. It rewrites ``/etc/wsl.conf`` only when it differs
        from :data:`WSL_CONF` and then prints :data:`WSLCONF_CHANGED_MARKER`.
    """
    return (
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "want=$(cat <<'WSL_CONF_EOF'",
                WSL_CONF.rstrip("\n"),
                "WSL_CONF_EOF",
                ")",
                'if [ "$(cat /etc/wsl.conf 2>/dev/null)" != "$want" ]; then',
                "    printf '%s\\n' \"$want\" > /etc/wsl.conf",
                f"    echo {WSLCONF_CHANGED_MARKER}",
                "fi",
            ]
        )
        + "\n"
    )


def render_linux_base_script(spec: HostRunnerSpec) -> str:
    """The Linux base stage for one host, run inside the distro as root.

    Args:
        spec: The host's roster entry.

    Returns:
        The bash text: refuses unless PID 1 is systemd, installs the
        roster's packages, enables docker, creates the runner account in
        the docker group, and proves the account can reach docker.

    Raises:
        ValueError: When a package name is not a plain apt name.
    """
    packages = spec["base"]["apt_packages"]
    for package in packages:
        if not all(c.isalnum() or c in ".+-" for c in package):
            raise ValueError(f"apt package {package!r} is not a plain package name")
    return (
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "export DEBIAN_FRONTEND=noninteractive",
                'test "$(ps -p 1 -o comm=)" = systemd || '
                '{ echo "PID 1 is not systemd; /etc/wsl.conf has not taken" >&2; exit 2; }',
                "apt-get update -qq",
                f"apt-get install -y -qq {' '.join(packages)}",
                "systemctl enable --now docker",
                "id gharunner >/dev/null 2>&1 || useradd -m -s /bin/bash gharunner",
                "usermod -aG docker gharunner",
                "sudo -u gharunner -H docker ps >/dev/null",
                "echo 'linux base ready: packages installed, docker reachable as gharunner'",
            ]
        )
        + "\n"
    )


__all__ = [
    "MSI_REBOOT_EXIT",
    "REBOOT_MARKER",
    "WSLCONF_CHANGED_MARKER",
    "WSL_CONF",
    "render_import_script",
    "render_linux_base_script",
    "render_windows_base_script",
    "render_wslconf_script",
]
