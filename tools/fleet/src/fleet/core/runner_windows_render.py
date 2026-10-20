"""The two PowerShell scripts a Windows runner host receives: its install and its audit.

MCPs board task 1e9ab738 (decision 5eab8a7b): loki's runners are installed
and audited by ``fleet-runners`` from the roster, like lavender's, and never
by hand. Both scripts are committed under ``rendered/`` for one example
install (:mod:`fleet.core.rendered_powershell`) and executed there by a
Pester suite to every command and branch; the copy a host receives differs
from the committed one only in its parameter defaults, its own installs and,
for the install, the registration tokens fleet-runners minted.

EVERY NATIVE ACT IS ``sc.exe``, BY ABSOLUTE PATH, AS A PARAMETER. The service
acts (set the recovery actions, read the state and configuration, start) go
through ``sc.exe`` rather than ``*-Service`` cmdlets, so a suite passes a
stand-in that answers as each case needs and nothing on the test machine
changes (:mod:`fleet.core.powershell_text` says why the default is a string).
The runner release is a URL parameter, which a suite serves as a ``file://``
archive it built, holding a ``config.cmd`` stand-in.

AN INSTALL IS IDEMPOTENT, step by step. The release is unpacked only where
``config.cmd`` is absent and registered only where ``.runner`` is absent, as
``config.cmd`` refuses a second run; the recovery actions are set every run,
and the service started only when ``sc.exe`` does not read it RUNNING.
"""

from __future__ import annotations

from collections.abc import Mapping

from fleet.contracts.runner_windows import WindowsRunnerHost, WindowsRunnerInstall
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter
from fleet.core.runner_audit_types import ExpectedCheck
from fleet.core.runner_install import RUNNER_VERSION

#: The committed install script's file stem under ``rendered/``.
INSTALL_STEM = "runner-windows-install"

#: The committed audit script's file stem under ``rendered/``.
AUDIT_STEM = "runner-windows-audit"

#: The pinned Windows runner release, the same version as the Linux one.
RUNNER_ZIP_URL = (
    f"https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/"
    f"actions-runner-win-x64-{RUNNER_VERSION}.zip"
)

#: The account every runner service runs as. corvis-stick's images arm
#: mounts a VHDX, which needs it.
SERVICE_ACCOUNT = "NT AUTHORITY\\SYSTEM"

#: How long the service manager waits before each restart, in milliseconds.
RESTART_DELAY_MS = 60000

#: After how long without a failure the restart count resets, in seconds.
RESTART_RESET_SECONDS = 86400

#: Why the state row matters.
SERVICE_REASON = "the runner's service is how its jobs run"

#: Why the configuration row matters.
ACCOUNT_REASON = (
    "a service that does not start with the machine leaves its jobs queued after every "
    "reboot, and corvis-stick's images arm mounts a VHDX, which needs SYSTEM"
)

#: Why the recovery row matters.
RECOVERY_REASON = (
    "a runner the service manager does not restart stays down until someone notices, and "
    "every job asking its label queues meanwhile"
)


def _example() -> list[WindowsRunnerInstall]:
    """The one install the committed renders carry.

    Returns:
        chat's runner on loki, the shape every roster install takes.
    """
    return [
        WindowsRunnerInstall(
            repo="wagner-austin/chat",
            runner_name="loki",
            service="actions.runner.wagner-austin-chat.loki",
            root="C:/actions-runner/chat",
            labels=["loki"],
        )
    ]


def _installs_parameter(entries: list[str]) -> list[str]:
    """The ``$Installs`` param-block lines for some hashtable literals.

    Args:
        entries: One ``@{ ... }`` literal per install.

    Returns:
        The lines, the last one with no trailing comma since ``$Installs``
        is the last parameter.
    """
    body = [f"        {entry}," for entry in entries[:-1]] + [f"        {entries[-1]}"]
    return ["    [hashtable[]]$Installs = @(", *body, "    )"]


def _install_entry(install: WindowsRunnerInstall) -> str:
    """One install as the install script's hashtable literal.

    Args:
        install: The install; the roster decoder admitted every value as
            embeddable in a single-quoted string.

    Returns:
        The literal.
    """
    return (
        f"@{{ Repo = '{install['repo']}'; Name = '{install['runner_name']}'; "
        f"Labels = '{','.join(install['labels'])}'; Root = '{install['root']}'; "
        f"Service = '{install['service']}' }}"
    )


def _tokens_literal(tokens: Mapping[str, str]) -> str:
    """The registration tokens as a hashtable literal.

    Args:
        tokens: Each repository's token; empty for the committed render.

    Returns:
        ``@{}``, or one ``'<repo>' = '<token>'`` pair per repository.

    Raises:
        ValueError: When a token is not plain alphanumeric, as every one
            GitHub mints is: anything else would be embedded verbatim.
    """
    for repo, token in tokens.items():
        if not token.isalnum():
            raise ValueError(f"the registration token for {repo} is not alphanumeric")
    pairs = "; ".join(f"'{repo}' = '{token}'" for repo, token in sorted(tokens.items()))
    return "@{" + (f" {pairs} " if pairs else "") + "}"


def render_install_script(installs: list[WindowsRunnerInstall], tokens: Mapping[str, str]) -> str:
    """The script that installs, registers and keeps running each runner service.

    Args:
        installs: The host's installs, at least one.
        tokens: A registration token per repository whose runner is not yet
            registered; empty for the committed render.

    Returns:
        The complete script.

    Raises:
        ValueError: As :func:`_tokens_literal` says.
    """
    lines = [
        "param(",
        f"    [string]$RunnerZipUrl = '{RUNNER_ZIP_URL}',",
        f"    {system32_parameter('Sc', 'sc.exe')},",
        f"    [hashtable]$Tokens = {_tokens_literal(tokens)},",
        *_installs_parameter([_install_entry(install) for install in installs]),
        ")",
        *STRICT_HEADER,
        "foreach ($install in $Installs) {",
        "    $root = $install.Root",
        '    if (-not (Test-Path -LiteralPath "$root/config.cmd")) {',
        "        [void][IO.Directory]::CreateDirectory($root)",
        '        Invoke-WebRequest -Uri $RunnerZipUrl -OutFile "$root/runner.zip" -UseBasicParsing',
        '        Expand-Archive -LiteralPath "$root/runner.zip" -DestinationPath $root -Force',
        '        Remove-Item -LiteralPath "$root/runner.zip"',
        '        "runner: unpacked $RunnerZipUrl into $root"',
        "    }",
        '    if (-not (Test-Path -LiteralPath "$root/.runner")) {',
        "        if (-not $Tokens.ContainsKey($install.Repo)) {",
        '            throw "RUNNER_TOKEN_MISSING: no registration token for $($install.Repo); '
        'fleet-runners --rebuild yes mints one"',
        "        }",
        '        & "$root/config.cmd" --unattended --url "https://github.com/$($install.Repo)" '
        "--token $Tokens[$install.Repo] --name $install.Name --labels $install.Labels --replace "
        f"--runasservice --windowslogonaccount '{SERVICE_ACCOUNT}'",
        "        if ($LASTEXITCODE -ne 0) {",
        '            throw "RUNNER_CONFIG_FAILED: config.cmd for $($install.Repo) exited '
        '$LASTEXITCODE"',
        "        }",
        '        "runner: registered $($install.Repo) runner $($install.Name)"',
        "    }",
        f"    $null = & $Sc failure $install.Service reset= {RESTART_RESET_SECONDS} actions= "
        f"restart/{RESTART_DELAY_MS}/restart/{RESTART_DELAY_MS}/restart/{RESTART_DELAY_MS}",
        "    if ($LASTEXITCODE -ne 0) {",
        '        throw "RUNNER_SERVICE_FAILED: sc.exe failure $($install.Service) exited '
        '$LASTEXITCODE"',
        "    }",
        "    $state = (& $Sc query $install.Service) -join ' '",
        "    if ($LASTEXITCODE -ne 0) {",
        '        throw "RUNNER_SERVICE_FAILED: sc.exe query $($install.Service) exited '
        '$LASTEXITCODE"',
        "    }",
        "    if ($state -notmatch 'STATE\\s+:\\s+4\\s+RUNNING') {",
        "        $null = & $Sc start $install.Service",
        "        if ($LASTEXITCODE -ne 0) {",
        '            throw "RUNNER_SERVICE_FAILED: sc.exe start $($install.Service) exited '
        '$LASTEXITCODE"',
        "        }",
        "    }",
        '    "runner: $($install.Repo) runner $($install.Name) runs as $($install.Service), '
        'restarted on failure"',
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_audit_script(installs: list[WindowsRunnerInstall]) -> str:
    """The script that prints one ``CHECK`` line per row :func:`expected_checks` names.

    Args:
        installs: The host's installs, at least one.

    Returns:
        The complete script. It reads and changes nothing but ``sc.exe``'s
        answers, and exits 0: drift is data.
    """
    entries = [f"@{{ Service = '{install['service']}' }}" for install in installs]
    lines = [
        "param(",
        f"    {system32_parameter('Sc', 'sc.exe')},",
        *_installs_parameter(entries),
        ")",
        *STRICT_HEADER,
        "function Get-Answer {",
        "    param([string]$Tool, [string[]]$Arguments)",
        "    $text = ((& $Tool @Arguments) -join ' ') -replace '\\s+', ' '",
        "    return [pscustomobject]@{ Code = $LASTEXITCODE; Text = $text.Trim() }",
        "}",
        "foreach ($install in $Installs) {",
        "    $service = $install.Service",
        "    $state = Get-Answer $Sc @('query', $service)",
        "    if ($state.Code -eq 0 -and $state.Text -match 'STATE : 4 RUNNING') {",
        '        "CHECK service:$service OK"',
        "    } else {",
        '        "CHECK service:$service DRIFT sc.exe query exited $($state.Code): $($state.Text)"',
        "    }",
        "    $config = Get-Answer $Sc @('qc', $service)",
        "    if ($config.Code -eq 0 -and $config.Text -match 'START_TYPE : 2 AUTO_START' "
        "-and $config.Text -match 'SERVICE_START_NAME : LocalSystem') {",
        '        "CHECK account:$service OK"',
        "    } else {",
        '        "CHECK account:$service DRIFT sc.exe qc exited $($config.Code): $($config.Text)"',
        "    }",
        "    $failure = Get-Answer $Sc @('qfailure', $service)",
        "    if ($failure.Code -eq 0 -and $failure.Text -match 'FAILURE_ACTIONS : RESTART') {",
        '        "CHECK recovery:$service OK"',
        "    } else {",
        '        "CHECK recovery:$service DRIFT sc.exe qfailure exited $($failure.Code): '
        '$($failure.Text)"',
        "    }",
        "}",
        # Drift is data: the last sc.exe read may have exited non-zero, and
        # that status is not the audit's.
        "exit 0",
    ]
    return "\n".join(lines) + "\n"


def expected_checks(host: WindowsRunnerHost) -> list[ExpectedCheck]:
    """Every check the audit script reports for this host, in order.

    The single source of the script's lines and the transcript scorer's
    expectations (:func:`fleet.core.runner_transcript.score_transcript`).

    Args:
        host: The host.

    Returns:
        Three rows per install: its state, its configuration and its
        recovery actions.
    """
    checks: list[ExpectedCheck] = []
    for install in host["installs"]:
        service = install["service"]
        checks += [
            ExpectedCheck(
                check_id=f"service:{service}",
                reason=f"{SERVICE_REASON}: {install['repo']}'s runner {install['runner_name']}",
            ),
            ExpectedCheck(check_id=f"account:{service}", reason=ACCOUNT_REASON),
            ExpectedCheck(check_id=f"recovery:{service}", reason=RECOVERY_REASON),
        ]
    return checks


def committed_install_script() -> str:
    """The install script committed under ``rendered/``, for the example install.

    Returns:
        The script, with no tokens.
    """
    return render_install_script(_example(), {})


def committed_audit_script() -> str:
    """The audit script committed under ``rendered/``, for the example install.

    Returns:
        The script.
    """
    return render_audit_script(_example())


__all__ = [
    "ACCOUNT_REASON",
    "AUDIT_STEM",
    "INSTALL_STEM",
    "RECOVERY_REASON",
    "RESTART_DELAY_MS",
    "RESTART_RESET_SECONDS",
    "RUNNER_ZIP_URL",
    "SERVICE_ACCOUNT",
    "SERVICE_REASON",
    "committed_audit_script",
    "committed_install_script",
    "expected_checks",
    "render_audit_script",
    "render_install_script",
]
