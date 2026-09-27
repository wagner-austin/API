"""Rendering the stages that turn a stock Windows install into a runner host.

``fleet-runners --rebuild`` (:mod:`fleet.core.runner_rebuild`) lays these,
in this order, before the roster's own ``provision.ps1`` and
``provision.sh`` (:mod:`fleet.core.runner_windows_provision`,
:mod:`fleet.core.runner_render`):

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
from fleet.core import runner_windows_provision
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter
from fleet.core.runner_machine_env import (
    render_machine_environment_lines,
    render_machine_environment_parameters,
)
from fleet.core.script_values import scriptable

#: The line the Windows base stage prints when Windows must restart first.
REBOOT_MARKER = "FLEET-REBOOT-REQUIRED"

#: The line the wsl.conf stage prints when it rewrote the file.
WSLCONF_CHANGED_MARKER = "FLEET-WSLCONF-CHANGED"

#: The distro's ``/etc/wsl.conf``. systemd as PID 1 is what runs docker and
#: every runner service; root as the default user is what the keepalive task
#: and the audit's ``wsl -d`` calls expect.
WSL_CONF = "[boot]\nsystemd=true\n\n[user]\ndefault=root\n"

#: ERROR_SUCCESS_REBOOT_REQUIRED: the exit code msiexec and dism both give
#: for "done; a restart completes it".
REBOOT_REQUIRED_EXIT = 3010

#: The registry key whose ``LongPathsEnabled`` lifts Win32's 260-character
#: path limit, read by the Windows base and by the audit.
LONG_PATHS_KEY = "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem"

#: The registry key whose ``ExecutionPolicy`` value is the LocalMachine
#: execution policy, the value ``Set-ExecutionPolicy -Scope LocalMachine``
#: writes.
EXECUTION_POLICY_KEY = "HKLM:\\SOFTWARE\\Microsoft\\PowerShell\\1\\ShellIds\\Microsoft.PowerShell"

#: The registry key WSL registers each distro under, one subkey carrying
#: its ``DistributionName``, for the account that registered it.
LXSS_KEY = "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss"


def _file_name(url: str) -> str:
    """The last path segment of a download URL.

    Args:
        url: The URL.

    Returns:
        The file name the download is saved under.
    """
    return url.rsplit("/", 1)[1]


def _parameter(name: str, value: str, label: str) -> str:
    """A param-block line whose default is a single-quoted roster value.

    Args:
        name: The parameter's name.
        value: The roster value.
        label: What the value is, for a refusal.

    Returns:
        ``[string]$<name> = '<value>',``.

    Raises:
        ValueError: When the value cannot be embedded verbatim.
    """
    return f"    [string]${name} = '{scriptable(value, label=label)}',"


def _array_parameter(name: str, values: list[str], label: str) -> str:
    """A param-block line whose default is an array of roster values.

    Args:
        name: The parameter's name.
        values: The roster values, in order.
        label: What each value is, for a refusal.

    Returns:
        ``[string[]]$<name> = @('<a>', '<b>'),``.

    Raises:
        ValueError: When a value cannot be embedded verbatim.
    """
    quoted = ", ".join(f"'{scriptable(value, label=label)}'" for value in values)
    return f"    [string[]]${name} = @({quoted}),"


def _verified_download_lines(url: str, sha256: str, target: str, label: str) -> list[str]:
    """PowerShell lines that download a pinned file and refuse wrong bytes.

    Args:
        url: The variable holding the pinned URL, such as ``$MsiUrl``.
        sha256: The variable holding the pinned digest, lowercase.
        target: The variable holding the Windows path to save to.
        label: What the file is, for the refusal.

    Returns:
        The lines. A file already at the target is re-verified rather than
        re-downloaded, so an interrupted rebuild resumes without fetching
        the image twice, and a corrupt leftover is refused by its digest.
    """
    return [
        f"if (-not (Test-Path -LiteralPath {target})) {{",
        f"    Invoke-WebRequest -Uri {url} -OutFile {target} -UseBasicParsing",
        "}",
        f"$Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath {target}).Hash.ToLower()",
        f"if ($Digest -ne {sha256}) {{",
        f"    Remove-Item -LiteralPath {target}",
        f"    throw ('{label} sha256 ' + $Digest + ' does not match the pin ' + {sha256})",
        "}",
    ]


def render_windows_base_script(spec: HostRunnerSpec) -> str:
    """The Windows base stage for one host.

    Args:
        spec: The host's roster entry.

    UNDER THE STRICT HEADER, EVERY EFFECT A PARAMETER (MCPs board task
    d69786fa). The roster's values are the defaults, so the rebuild runs the
    script with no arguments; a Pester suite over the committed render
    passes stand-ins for ``dism.exe``, ``msiexec.exe``, ``wsl.exe`` and
    ``git``, scratch HKCU keys for the three HKLM ones, a PATH variable of
    its own and a ``file://`` download whose digest it computed. The
    features are read and enabled through ``dism.exe /english`` rather than
    the optional-feature cmdlets, which need elevation and cannot be stood
    in for; ``dism`` exits 3010 when a restart completes the change. Every
    registry value is read with ``GetValue``, which answers ``$null`` for an
    absent value where a property read fails under strict mode.

    Args:
        spec: The host's roster entry.

    Returns:
        The complete PowerShell text. It prints :data:`REBOOT_MARKER` when
        an enabled feature or the WSL install asks for a restart, or when a
        machine variable changed (:mod:`fleet.core.runner_machine_env`).

    Raises:
        ValueError: When a roster value cannot be embedded verbatim.
    """
    base = spec["base"]
    msi = base["wsl_msi"]
    lines: list[str] = [
        "param(",
        _parameter("Scratch", spec["scratch_dir"], "scratch_dir"),
        _array_parameter("Features", base["windows_features"], "windows feature"),
        _parameter("WslVersion", msi["version"], "wsl_msi version"),
        _parameter("MsiUrl", msi["url"], "wsl_msi url"),
        _parameter("MsiSha256", msi["sha256"], "wsl_msi sha256"),
        _parameter("MsiFile", _file_name(msi["url"]), "wsl_msi file"),
        _parameter("Policy", base["execution_policy"], "execution_policy"),
        _parameter("PolicyKey", EXECUTION_POLICY_KEY, "policy key"),
        _parameter("FileSystemKey", LONG_PATHS_KEY, "file system key"),
        _array_parameter("PathEntries", base["machine_path_entries"], "machine PATH entry"),
        "    [string]$PathVariable = 'Path',",
        "    [string]$PathScope = 'Machine',",
        *render_machine_environment_parameters(spec),
        *runner_windows_provision.render_wslconfig_parameters(spec),
        "    [string]$Git = 'git',",
        "    " + system32_parameter("Dism", "dism.exe") + ",",
        "    " + system32_parameter("Msiexec", "msiexec.exe") + ",",
        "    " + system32_parameter("Wsl", "wsl.exe"),
        ")",
        *STRICT_HEADER,
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
        "$ProgressPreference = 'SilentlyContinue'",
        "$env:WSL_UTF8 = '1'",
        "$Reboot = $false",
        "[void][System.IO.Directory]::CreateDirectory($Scratch)",
        "foreach ($feature in $Features) {",
        "    $info = & $Dism /online /english /get-featureinfo /featurename:$feature",
        "    if ($LASTEXITCODE -ne 0) {",
        "        throw ('dism could not read the Windows feature ' + $feature + ', exit ' "
        "+ $LASTEXITCODE)",
        "    }",
        "    if (@($info | Where-Object { $_ -match '^State : Enabled' }).Count -eq 0) {",
        "        $null = & $Dism /online /english /enable-feature /featurename:$feature /all "
        "/norestart",
        f"        if ($LASTEXITCODE -eq {REBOOT_REQUIRED_EXIT}) {{",
        "            $Reboot = $true",
        "        } elseif ($LASTEXITCODE -ne 0) {",
        "            throw ('dism could not enable the Windows feature ' + $feature + ', exit ' "
        "+ $LASTEXITCODE)",
        "        }",
        "        Write-Output ('enabled Windows feature ' + $feature)",
        "    }",
        "}",
        # wsl --version is absent or different on a stock install, where it
        # exits non-zero; the MSI's own wsl.exe reports 'WSL version: <v>.0'.
        "$answer = & $Wsl --version",
        "$installed = ($LASTEXITCODE -eq 0) -and "
        "((@($answer) -join ' ') -match ('WSL[^:]*:\\s*' + [regex]::Escape($WslVersion)))",
        "if (-not $installed) {",
        "    $msiPath = Join-Path $Scratch $MsiFile",
        *(
            "    " + line
            for line in _verified_download_lines("$MsiUrl", "$MsiSha256", "$msiPath", "WSL MSI")
        ),
        "    $install = Start-Process -FilePath $Msiexec -ArgumentList @('/i', "
        "\"`\"$msiPath`\"\", '/qn', '/norestart') -Wait -PassThru -NoNewWindow",
        f"    if ($install.ExitCode -eq {REBOOT_REQUIRED_EXIT}) {{",
        "        $Reboot = $true",
        "    } elseif ($install.ExitCode -ne 0) {",
        "        throw ('msiexec for WSL ' + $WslVersion + ' exited ' + $install.ExitCode)",
        "    }",
        "    Remove-Item -LiteralPath $msiPath",
        "    Write-Output ('installed WSL ' + $WslVersion)",
        "}",
        # The registry value Set-ExecutionPolicy -Scope LocalMachine writes,
        # read and written directly: under this script's own Process-scope
        # Bypass the cmdlet reports the LocalMachine change as overridden
        # and, with ErrorActionPreference Stop, throws after succeeding.
        "if ([string](Get-Item -LiteralPath $PolicyKey).GetValue('ExecutionPolicy') -ne $Policy) {",
        "    Set-ItemProperty -LiteralPath $PolicyKey -Name ExecutionPolicy -Value $Policy",
        "    Write-Output ('set the LocalMachine execution policy to ' + $Policy)",
        "}",
        # Long paths, which a fresh Windows install leaves off. Measured on
        # the reinstalled lavender, 2026-09-26: MCPs CI's supervisor Pester
        # case extracts a tree under NETWORK SERVICE's temp directory, whose
        # paths pass 260 characters, and CreateDirectory failed with 'Could
        # not find a part of the path' until LongPathsEnabled was 1; git's
        # own checkout needs core.longpaths for the same trees.
        "if ((Get-Item -LiteralPath $FileSystemKey).GetValue('LongPathsEnabled') -ne 1) {",
        "    Set-ItemProperty -LiteralPath $FileSystemKey -Name LongPathsEnabled -Value 1 "
        "-Type DWord",
        "    Write-Output 'enabled Win32 long paths'",
        "}",
        # git exits 1 for a key that is not set, which is the stock answer.
        "$held = & $Git config --system --get core.longpaths",
        "if ($LASTEXITCODE -gt 1) {",
        "    throw ('git config --system --get core.longpaths exited ' + $LASTEXITCODE)",
        "}",
        "if ((@($held) -join '') -ne 'true') {",
        "    & $Git config --system core.longpaths true",
        "    if ($LASTEXITCODE -ne 0) {",
        "        throw ('git config --system core.longpaths exited ' + $LASTEXITCODE)",
        "    }",
        "    Write-Output 'set git core.longpaths for the system'",
        "}",
        "foreach ($entry in $PathEntries) {",
        "    $current = [string][Environment]::GetEnvironmentVariable($PathVariable, $PathScope)",
        "    $parts = @($current -split ';' | Where-Object { $_ -ne '' })",
        "    if ($parts -notcontains $entry) {",
        "        [Environment]::SetEnvironmentVariable($PathVariable, "
        "((@($parts) + $entry) -join ';'), $PathScope)",
        "        Write-Output ('added to the machine PATH: ' + $entry)",
        "    }",
        "}",
        *render_machine_environment_lines(),
        *runner_windows_provision.render_wslconfig_lines(spec),
        "if ($Reboot) {",
        f"    Write-Output '{REBOOT_MARKER}'",
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_import_script(spec: HostRunnerSpec) -> str:
    """The distro import stage for one host.

    Args:
        spec: The host's roster entry.

    WHICH DISTROS ARE REGISTERED IS READ FROM THE REGISTRY, the ``Lxss`` key
    each registration writes a ``DistributionName`` under, not from
    ``wsl --list --quiet``: that answers in UTF-16 unless told otherwise and
    exits non-zero on a host with none registered, which is exactly the host
    this stage imports on, so the script piped it through ``2>$null`` and
    ignored the exit. Under the strict header the key is read directly, and
    an absent key is a host that has never registered one. The key, the
    download and ``wsl.exe`` are parameters defaulting to the roster's, so
    the Pester suite over the committed render passes a scratch key, a
    ``file://`` image and a stand-in (MCPs board task d69786fa).

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
    rootfs = base["rootfs"]
    lines: list[str] = [
        "param(",
        _parameter("Distro", spec["wsl_distro"], "wsl_distro"),
        _parameter("Scratch", spec["scratch_dir"], "scratch_dir"),
        _parameter("DistroDir", base["distro_dir"], "distro_dir"),
        _parameter("RootfsVersion", rootfs["version"], "rootfs version"),
        _parameter("RootfsUrl", rootfs["url"], "rootfs url"),
        _parameter("RootfsSha256", rootfs["sha256"], "rootfs sha256"),
        _parameter("RootfsFile", _file_name(rootfs["url"]), "rootfs file"),
        _parameter("LxssKey", LXSS_KEY, "lxss key"),
        "    " + system32_parameter("Wsl", "wsl.exe"),
        ")",
        *STRICT_HEADER,
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
        "$ProgressPreference = 'SilentlyContinue'",
        "$registered = @()",
        "if (Test-Path -LiteralPath $LxssKey) {",
        "    $registered = @(Get-ChildItem -LiteralPath $LxssKey | "
        "ForEach-Object { [string]$_.GetValue('DistributionName') })",
        "}",
        "if ($registered -notcontains $Distro) {",
        "    [void][System.IO.Directory]::CreateDirectory($Scratch)",
        "    $image = Join-Path $Scratch $RootfsFile",
        *(
            "    " + line
            for line in _verified_download_lines("$RootfsUrl", "$RootfsSha256", "$image", "rootfs")
        ),
        "    [void][System.IO.Directory]::CreateDirectory($DistroDir)",
        "    & $Wsl --import $Distro $DistroDir $image --version 2",
        "    if ($LASTEXITCODE -ne 0) {",
        "        throw ('wsl --import ' + $Distro + ' exited ' + $LASTEXITCODE)",
        "    }",
        "    Remove-Item -LiteralPath $image",
        "    Write-Output ('imported ' + $Distro + ' ' + $RootfsVersion)",
        "}",
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
    "EXECUTION_POLICY_KEY",
    "LONG_PATHS_KEY",
    "LXSS_KEY",
    "REBOOT_MARKER",
    "REBOOT_REQUIRED_EXIT",
    "WSLCONF_CHANGED_MARKER",
    "WSL_CONF",
    "render_import_script",
    "render_linux_base_script",
    "render_windows_base_script",
    "render_wslconf_script",
]
