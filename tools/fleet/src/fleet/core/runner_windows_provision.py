"""The Windows side of a runner host's provision: ``provision.ps1``, and onboarding's.

Split out of :mod:`fleet.core.runner_render` when the Windows script became
a strict, parameterized render (MCPs board task d69786fa, A2). It writes
``.wslconfig`` when the roster declares a memory floor, registers and starts
the keepalive scheduled task when one is declared, and installs every
Windows-side runner.

ONE LOOP, THE INSTALLS AS DATA. Each Windows-side install is one hashtable
in the ``$Installs`` parameter, and a single loop body installs, configures,
binds to SYSTEM, sets restart-on-failure, starts, and seeds the Python tool
cache for each. The body is the same text for every host and every install,
so the Pester suite over the committed render executes all of it by passing
installs of its own under TestDrive, and the roster's installs are the
parameter's default.

EVERY ACT A PARAMETER. The runner release and the NuGet Python package are
URLs (a suite serves them as ``file://`` archives it built), python.exe's
file name, sc.exe, wsl.exe and ``.wslconfig``'s path are parameters, and
the three service acts (read a service's Win32_Service row, stop it, start
it) are script blocks whose defaults are the cmdlets, so nothing in a test
touches this machine's services.

TOKENS ARE A PARAMETER TOO. A registration token arrives in the per-repo
environment variable :func:`fleet.core.runner_install.token_variable` names.
An operator running the rendered file sets them by hand; ``--rebuild`` and
``--onboard`` pass the ones they minted as ``$Tokens``, whose pairs the
script sets first. The committed render carries none.
"""

from __future__ import annotations

from collections.abc import Mapping

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter
from fleet.core.runner_account import (
    WINDOWS_SERVICE_ACCOUNT,
    render_service_account_lines,
    render_service_running_lines,
)
from fleet.core.runner_install import RUNNER_VERSION, install_root_for, token_variable
from fleet.core.runner_keepalive import render_keepalive_lines, render_keepalive_parameters
from fleet.core.runner_recovery import render_windows_recovery_lines
from fleet.core.script_values import scriptable

#: Where ``.wslconfig`` lives: the running account's profile.
WSLCONFIG_PATH = '"$env:USERPROFILE\\.wslconfig"'

#: The pinned Windows runner release.
RUNNER_URL = (
    f"https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/"
    f"actions-runner-win-x64-{RUNNER_VERSION}.zip"
)

#: The NuGet CPython package, ``{0}`` the exact version.
PYTHON_PACKAGE_URL = "https://www.nuget.org/api/v2/package/python/{0}"


def render_wslconfig_lines(spec: HostRunnerSpec) -> list[str]:
    """PowerShell lines that write ``.wslconfig`` for a declared memory floor.

    Shared by ``provision.ps1`` and the rebuild's Windows base stage, which
    writes it BEFORE the distro is first started so the VM boots into the
    floor rather than needing a restart to take it. The file is
    ``$WslConfigPath``, which both scripts take as a parameter
    (:func:`render_wslconfig_parameters`) so their Pester suites write under
    TestDrive.

    Args:
        spec: The host's roster entry.

    Returns:
        The lines, none when the roster declares no floor.
    """
    floor = spec["wslconfig_min_memory_gb"]
    if floor is None:
        return []
    return [
        "$WslConfig = @(",
        "  '[wsl2]',",
        f"  'memory={floor}GB',",
        "  'swap=8GB'",
        ")",
        "Set-Content -LiteralPath $WslConfigPath -Value $WslConfig -Encoding ascii",
        "Write-Output 'wrote .wslconfig; the ceiling applies when the VM next starts'",
    ]


def render_wslconfig_parameters(spec: HostRunnerSpec) -> list[str]:
    """The param-block line naming ``.wslconfig``'s path.

    Args:
        spec: The host's roster entry.

    Returns:
        ``[string]$WslConfigPath = "$env:USERPROFILE\\.wslconfig",``, or
        nothing when the roster declares no floor and so no file is written.
    """
    if spec["wslconfig_min_memory_gb"] is None:
        return []
    return [f"    [string]$WslConfigPath = {WSLCONFIG_PATH},"]


def render_install_record(install: RunnerInstall) -> str:
    """One Windows-side install as the ``$Installs`` hashtable literal.

    Args:
        install: A windows-side install.

    Returns:
        ``@{ Repo = ...; Name = ...; Labels = ...; Service = ...;
        Directory = ...; Workdir = ...; TokenVariable = ...; Python = @(...) }``,
        the paths backslashed and the labels comma-joined as config.cmd
        takes them.

    Raises:
        ValueError: When a value cannot be embedded verbatim; see
            :func:`fleet.core.script_values.scriptable`.
    """
    fields = {
        "Repo": scriptable(install["repo"], label="repo"),
        "Name": scriptable(install["runner_name"], label="runner_name"),
        "Labels": scriptable(",".join(install["labels"]), label="labels"),
        "Service": scriptable(install["service"], label="service"),
        "Directory": scriptable(install_root_for(install), label="workdir"),
        "Workdir": scriptable(install["workdir"].replace("/", "\\"), label="workdir"),
        "TokenVariable": token_variable(install["repo"]),
    }
    versions = ", ".join(
        f"'{scriptable(version, label='python_toolcache')}'"
        for version in install["python_toolcache"]
    )
    pairs = "; ".join(f"{key} = '{value}'" for key, value in fields.items())
    return f"@{{ {pairs}; Python = @({versions}) }}"


def _install_parameters(installs: list[RunnerInstall], tokens: Mapping[str, str]) -> list[str]:
    """The param-block lines every Windows install script shares.

    Args:
        installs: The windows-side installs, in roster order.
        tokens: Registration tokens by variable name; empty for a render an
            operator runs with the variables set by hand.

    Returns:
        ``$Installs`` through ``$Sc``, the last without a trailing comma.

    Raises:
        ValueError: When a value cannot be embedded verbatim.
    """
    records = [f"        {render_install_record(install)}" for install in installs]
    token_pairs = "; ".join(
        f"'{scriptable(name, label='token variable')}' = '{scriptable(value, label='token')}'"
        for name, value in tokens.items()
    )
    return [
        "    [hashtable[]]$Installs = @(",
        ",\n".join(records),
        "    ),",
        f"    [hashtable]$Tokens = @{{{token_pairs}}},",
        f"    [string]$RunnerUrl = '{RUNNER_URL}',",
        f"    [string]$PythonPackageUrl = '{PYTHON_PACKAGE_URL}',",
        "    [string]$PythonName = 'python.exe',",
        "    [scriptblock]$GetService = { param([string]$Name) "
        "@(Get-CimInstance Win32_Service -Filter \"Name='$Name'\") },",
        "    [scriptblock]$StopService = { param([string]$Name) Stop-Service -Name $Name },",
        "    [scriptblock]$StartService = { param([string]$Name) Start-Service -Name $Name },",
        "    " + system32_parameter("Sc", "sc.exe"),
    ]


def render_windows_install_lines() -> list[str]:
    """The loop that sets the tokens and provisions every ``$Installs`` entry.

    Lifted from the tree-bot install of 2026-09-09 (the last hand-rolled
    runner on this fleet, by operator mandate) and from the layout its
    siblings share: an install directory beside ``C:\\actions-runner``
    named for the repo, the pinned runner release, and ``config.cmd
    --runasservice`` so the service survives reboots. ``--replace`` takes
    back a registration of the same name, which is what a rebuilt host must
    do: its old runners are still registered, offline, under exactly the
    names the roster gives the new ones (board task 1aa6a021). An install
    whose ``.runner`` file exists is already configured and is left alone,
    because config.cmd refuses a second configuration and ``--rebuild``
    re-runs every stage over a host that may be half built. The service runs
    as SYSTEM, a deliberate privilege decision whose reasons and cost
    :mod:`fleet.core.runner_account` states.

    Returns:
        The lines. A missing token throws ``FLEET_RUNNER_TOKEN_MISSING`` and
        a refused configuration ``FLEET_RUNNER_CONFIG_REFUSED``, each naming
        the install.
    """
    return [
        # Windows PowerShell 5.1 may offer GitHub only TLS 1.0 and 1.1.
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12",
        "foreach ($TokenName in @($Tokens.Keys)) {",
        "    Set-Item -LiteralPath ('env:' + $TokenName) -Value $Tokens[$TokenName]",
        "}",
        "foreach ($Install in $Installs) {",
        "    $Directory = [string]$Install.Directory",
        "    $Workdir = [string]$Install.Workdir",
        "    $ServiceName = [string]$Install.Service",
        "    $Token = [Environment]::GetEnvironmentVariable([string]$Install.TokenVariable)",
        "    if ([string]::IsNullOrEmpty($Token)) {",
        '        throw ("FLEET_RUNNER_TOKEN_MISSING: set " + $Install.TokenVariable + '
        "' to a fresh registration token for ' + $Install.Repo)",
        "    }",
        "    New-Item -ItemType Directory -Force -Path $Directory | Out-Null",
        "    if (-not (Test-Path -LiteralPath (Join-Path $Directory 'config.cmd'))) {",
        "        $Zip = Join-Path $Directory 'runner.zip'",
        "        Invoke-WebRequest -Uri $RunnerUrl -OutFile $Zip -UseBasicParsing",
        "        Expand-Archive -LiteralPath $Zip -DestinationPath $Directory -Force",
        "        Remove-Item -LiteralPath $Zip",
        "    }",
        "    if (-not (Test-Path -LiteralPath (Join-Path $Directory '.runner'))) {",
        "        & (Join-Path $Directory 'config.cmd') --unattended "
        "--url ('https://github.com/' + $Install.Repo) --token $Token --name $Install.Name "
        "--labels $Install.Labels --runasservice "
        f"--windowslogonaccount '{WINDOWS_SERVICE_ACCOUNT}' --replace",
        "        if ($LASTEXITCODE -ne 0) {",
        '            throw ("FLEET_RUNNER_CONFIG_REFUSED: config.cmd for " + $Install.Repo + '
        "' ' + $Install.Name + ' exited ' + $LASTEXITCODE)",
        "        }",
        "    }",
        *(f"    {line}" for line in render_service_account_lines()),
        *(f"    {line}" for line in render_windows_recovery_lines()),
        *(f"    {line}" for line in render_service_running_lines()),
        *(f"    {line}" for line in _python_toolcache_lines()),
        "}",
    ]


def _python_toolcache_lines() -> list[str]:
    """The loop that seeds the install's Python tool cache from ``$Install.Python``.

    On self-hosted Windows, ``actions/setup-python`` INSTALLS when the tool
    cache misses -- via the python.org all-users installer, whose registry
    writes the runner's service account rightly lacks (measured on tree-bot
    run 34394258775: SecurityException in Remove-Item Registry). Seeding the
    cache turns setup-python's install path into its find path. The source is
    the NuGet CPython package: full xcopy-deployable Python with pip, no
    installer, no registry. The find path wants only
    ``Python/<ver>/x64/python.exe`` plus an ``x64.complete`` marker. The
    versions are exact on purpose: NuGet carries only versions python.org
    built for Windows, and a floating spec would re-create the drift this
    package exists to prevent.

    The package carries pip as a module but no ``Scripts`` directory, and
    setup-python puts ``Scripts`` on PATH, so a workflow's bare ``pip``
    fails on a seed that stops at the copy (measured on MCPs run
    36254365786, session-audit: CommandNotFoundException for pip). pip is
    therefore reinstalled from the package's own bundled wheel, offline,
    which writes the entry points; ``Scripts\\pip.exe`` is both the
    idempotency test and the verification.

    Returns:
        The lines; an already-seeded version is left alone, and each way a
        seed can fail throws a named error.
    """
    return [
        "$ToolDir = Join-Path $Workdir '_tool'",
        "foreach ($Version in @($Install.Python)) {",
        "    $Target = Join-Path $ToolDir ('Python\\' + $Version + '\\x64')",
        "    if (-not (Test-Path -LiteralPath (Join-Path $Target 'Scripts\\pip.exe'))) {",
        "        $Work = Join-Path $env:TEMP ([guid]::NewGuid().ToString('N'))",
        "        New-Item -ItemType Directory -Path $Work | Out-Null",
        "        $Package = Join-Path $Work 'python.nupkg.zip'",
        "        Invoke-WebRequest -Uri ($PythonPackageUrl -f $Version) -OutFile $Package "
        "-UseBasicParsing",
        "        Expand-Archive -LiteralPath $Package -DestinationPath $Work -Force",
        "        New-Item -ItemType Directory -Force -Path $Target | Out-Null",
        "        Copy-Item -Path (Join-Path $Work 'tools\\*') -Destination $Target -Recurse -Force",
        "        $Wheel = @(Get-ChildItem -LiteralPath "
        "(Join-Path $Target 'Lib\\ensurepip\\_bundled') -Filter 'pip-*.whl')",
        "        if ($Wheel.Count -ne 1) {",
        '            throw ("FLEET_PYTHON_SEED_WHEEL: expected one bundled pip wheel in seeded " '
        "+ $Version + ', found ' + $Wheel.Count)",
        "        }",
        "        & (Join-Path $Target $PythonName) -m pip install --force-reinstall --no-deps "
        "--no-index --no-warn-script-location --disable-pip-version-check $Wheel[0].FullName",
        "        if ($LASTEXITCODE -ne 0) {",
        '            throw ("FLEET_PYTHON_SEED_PIP: the pip reinstall in seeded " + $Version + '
        "' exited ' + $LASTEXITCODE)",
        "        }",
        "        if (-not (Test-Path -LiteralPath (Join-Path $Target 'Scripts\\pip.exe'))) {",
        '            throw ("FLEET_PYTHON_SEED_PIP_MISSING: pip.exe is missing in seeded " '
        "+ $Version)",
        "        }",
        "        New-Item -ItemType File -Force -Path "
        "(Join-Path $ToolDir ('Python\\' + $Version + '\\x64.complete')) | Out-Null",
        "        Remove-Item -LiteralPath $Work -Recurse -Force",
        "    }",
        "}",
    ]


def render_windows_provision_script(spec: HostRunnerSpec, tokens: Mapping[str, str]) -> str:
    """The complete ``provision.ps1`` for one host.

    Args:
        spec: The host's roster entry.
        tokens: Registration tokens by variable name, set before any install;
            empty for the file an operator runs with the variables set.

    Returns:
        The script: the ``.wslconfig`` floor and keepalive task when
        declared, then every windows-side install. It says so rather than
        being empty when the roster declares none of those, so the run order
        the CLI prints holds for every host.

    Raises:
        ValueError: When a roster value or token cannot be embedded verbatim.
    """
    installs = [install for install in spec["installs"] if install["side"] == "windows"]
    lines = [
        "# provision.ps1 -- Windows side. Rendered by fleet-runners; run as the",
        "# machine's interactive user. Re-run after any change: every step is",
        "# idempotent, and an install already configured is left alone.",
        "param(",
        *render_wslconfig_parameters(spec),
        *render_keepalive_parameters(spec),
        *_install_parameters(installs, tokens),
        ")",
        *STRICT_HEADER,
        *render_wslconfig_lines(spec),
        *render_keepalive_lines(spec),
        *render_windows_install_lines(),
    ]
    if spec["wslconfig_min_memory_gb"] is None and spec["keepalive_task"] is None and not installs:
        lines.append("Write-Output 'nothing declared for the Windows side of this host'")
    return "\n".join(lines) + "\n"


def render_windows_onboard_script(install: RunnerInstall, tokens: Mapping[str, str]) -> str:
    """The script ``--onboard`` runs for one Windows-side install.

    The same parameters and loop as ``provision.ps1`` without the host-level
    ``.wslconfig`` and keepalive, which onboarding one runner does not touch.

    Args:
        install: The windows-side install.
        tokens: Its registration token, by variable name.

    Returns:
        The script.

    Raises:
        ValueError: When a value cannot be embedded verbatim.
    """
    lines = [
        "param(",
        *_install_parameters([install], tokens),
        ")",
        *STRICT_HEADER,
        *render_windows_install_lines(),
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "PYTHON_PACKAGE_URL",
    "RUNNER_URL",
    "WSLCONFIG_PATH",
    "render_install_record",
    "render_windows_install_lines",
    "render_windows_onboard_script",
    "render_windows_provision_script",
    "render_wslconfig_lines",
    "render_wslconfig_parameters",
]
