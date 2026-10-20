"""The two PowerShell scripts a Windows runner host receives: its install and its audit.

MCPs board task 1e9ab738 (decision 5eab8a7b): loki's runners are installed
and audited by ``fleet-runners`` from the roster, like lavender's, and never
by hand. Both scripts are committed under ``rendered/`` for one example host
(:mod:`fleet.core.rendered_powershell`) and executed there by a Pester suite
to every command and branch; the copy a host receives differs from the
committed one only in its parameter defaults, its own base and installs and,
for the install, the registration tokens fleet-runners minted.

EVERY NATIVE ACT IS ``sc.exe``, BY ABSOLUTE PATH, AS A PARAMETER. The service
acts (set the recovery actions, read the state and configuration, stop,
start) go through ``sc.exe`` rather than ``*-Service`` cmdlets, so a suite
passes a stand-in that answers as each case needs and nothing on the test
machine changes (:mod:`fleet.core.powershell_text` says why the default is a
string). The runner release is a URL parameter, which a suite serves as a
``file://`` archive it built, holding a ``config.cmd`` stand-in. The host's
base (:mod:`fleet.core.runner_windows_base_render`) runs first and its
effects are parameters too.

AN INSTALL IS IDEMPOTENT, step by step. The base writes only what differs;
the release is unpacked only where ``config.cmd`` is absent and registered
only where ``.runner`` is absent, as ``config.cmd`` refuses a second run; the
recovery actions are set every run; and the service is started when
``sc.exe`` does not read it RUNNING, or stopped and started again when the
base wrote a machine variable, which a running listener never reads.

EVERY ROSTER VALUE IS REFUSED HERE when it could not be embedded verbatim
(:func:`fleet.core.script_values.scriptable`), as every renderer does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from platform_core.json_utils import dump_json_str

from fleet.contracts.runner_windows import WindowsRunnerHost, WindowsRunnerInstall
from fleet.contracts.runner_windows_base import MachineVariable, WindowsHostBase
from fleet.core import runner_windows_base_render as base_render
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter
from fleet.core.runner_audit_types import ExpectedCheck
from fleet.core.runner_install import RUNNER_VERSION
from fleet.core.script_values import scriptable

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

#: The audit's ``-match`` pattern for that account in ``sc.exe qc``'s output.
#: config.cmd hands the name to CreateService as given, and the service
#: manager keeps it so: measured on loki 2026-10-20, a service created with
#: obj= "NT AUTHORITY\SYSTEM" reads SERVICE_START_NAME : NT AUTHORITY\SYSTEM,
#: not LocalSystem. The backslash is doubled, as a regex needs.
_START_NAME_PATTERN = "SERVICE_START_NAME : " + SERVICE_ACCOUNT.replace("\\", "\\\\")

#: How long the service manager waits before each restart, in milliseconds.
RESTART_DELAY_MS = 60000

#: After how long without a failure the restart count resets, in seconds.
RESTART_RESET_SECONDS = 86400

#: How long the install waits for a runner service it stopped to read
#: STOPPED, in seconds. A listener ends its job's worker first, and a minute
#: covers a worker's cleanup without hiding a service that will not stop.
STOP_WAIT_SECONDS = 60

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


def _example() -> WindowsRunnerHost:
    """The one host the committed renders carry.

    Returns:
        loki with chat's runner and one machine variable, the shape every
        roster host takes.
    """
    return WindowsRunnerHost(
        name="loki",
        host="loki",
        payload_dir="C:/fleet/runner-payloads",
        base=WindowsHostBase(
            execution_policy="RemoteSigned",
            machine_environment=[
                MachineVariable(
                    name="POETRY_CACHE_DIR",
                    value="C:\\fleet\\poetry",
                    reason="the runners run as SYSTEM",
                )
            ],
        ),
        installs=[
            WindowsRunnerInstall(
                repo="wagner-austin/chat",
                runner_name="loki",
                service="actions.runner.wagner-austin-chat.loki",
                root="C:/actions-runner/chat",
                labels=["loki"],
            )
        ],
    )


def _installs_parameter(rows: Sequence[Mapping[str, str]]) -> str:
    """The ``$Installs`` parameter: the host's installs as one JSON string.

    A string, not hashtable literals, because a constant default is no
    command: a literal is, and only a run that leaves the default in place
    reaches it, which for the install means a run on the machine's own
    runner directories. The body decodes it with ``ConvertFrom-Json``, which
    every run executes. Every field passed :func:`scriptable`, so the JSON's
    only double quotes are its own and a single-quoted string carries it
    verbatim.

    Args:
        rows: One object per install, its fields named as the script reads
            them.

    Returns:
        The line, with no trailing comma since ``$Installs`` is the last
        parameter.
    """
    return f"    [string]$Installs = '{dump_json_str(list(rows))}'"


def _install_row(install: WindowsRunnerInstall) -> dict[str, str]:
    """One install as the install script reads it.

    Args:
        install: The install.

    Returns:
        Its repository, runner name, comma-joined labels (``config.cmd``'s
        ``--labels`` form), directory and service.

    Raises:
        ValueError: When a value could not be embedded verbatim.
    """
    return {
        "Repo": scriptable(install["repo"], label="repo"),
        "Name": scriptable(install["runner_name"], label="runner_name"),
        "Labels": scriptable(",".join(install["labels"]), label="labels"),
        "Root": scriptable(install["root"], label="root"),
        "Service": scriptable(install["service"], label="service"),
    }


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


#: The install's per-runner service lines, after its registration: the
#: recovery actions, then the state, then a stop when the base wrote a
#: machine variable, then a start of a service that is not running.
_SERVICE_LINES: tuple[str, ...] = (
    f"    $null = & $Sc failure $install.Service reset= {RESTART_RESET_SECONDS} actions= "
    f"restart/{RESTART_DELAY_MS}/restart/{RESTART_DELAY_MS}/restart/{RESTART_DELAY_MS}",
    "    if ($LASTEXITCODE -ne 0) {",
    '        throw "RUNNER_SERVICE_FAILED: sc.exe failure $($install.Service) exited '
    '$LASTEXITCODE"',
    "    }",
    "    $state = (& $Sc query $install.Service) -join ' '",
    "    if ($LASTEXITCODE -ne 0) {",
    '        throw "RUNNER_SERVICE_FAILED: sc.exe query $($install.Service) exited $LASTEXITCODE"',
    "    }",
    "    $running = $state -match 'STATE\\s+:\\s+4\\s+RUNNING'",
    "    if ($running -and $restart) {",
    "        $null = & $Sc stop $install.Service",
    "        if ($LASTEXITCODE -ne 0) {",
    '            throw "RUNNER_SERVICE_FAILED: sc.exe stop $($install.Service) exited '
    '$LASTEXITCODE"',
    "        }",
    "        $stopped = $false",
    "        foreach ($second in 1..$StopWaitSeconds) {",
    "            $state = (& $Sc query $install.Service) -join ' '",
    "            if ($LASTEXITCODE -ne 0) {",
    '                throw "RUNNER_SERVICE_FAILED: sc.exe query $($install.Service) exited '
    '$LASTEXITCODE"',
    "            }",
    "            if ($state -match 'STATE\\s+:\\s+1\\s+STOPPED') {",
    "                $stopped = $true",
    "                break",
    "            }",
    "            Start-Sleep -Seconds 1",
    "        }",
    "        if (-not $stopped) {",
    '            throw "RUNNER_SERVICE_FAILED: $($install.Service) did not stop within '
    '$StopWaitSeconds s"',
    "        }",
    '        "runner: stopped $($install.Service) so it starts with the machine variables"',
    "        $running = $false",
    "    }",
    "    if (-not $running) {",
    "        $null = & $Sc start $install.Service",
    "        if ($LASTEXITCODE -ne 0) {",
    '            throw "RUNNER_SERVICE_FAILED: sc.exe start $($install.Service) exited '
    '$LASTEXITCODE"',
    "        }",
    "    }",
    '    "runner: $($install.Repo) runner $($install.Name) runs as $($install.Service), '
    'restarted on failure"',
)


def render_install_script(host: WindowsRunnerHost, tokens: Mapping[str, str]) -> str:
    """The script that lays the host's base and installs and keeps running each runner.

    Args:
        host: The host, with at least one install.
        tokens: A registration token per repository whose runner is not yet
            registered; empty for the committed render.

    Returns:
        The complete script.

    Raises:
        ValueError: As :func:`_tokens_literal` and :func:`_install_row` say,
            or a machine variable that could not be embedded.
    """
    lines = [
        "param(",
        f"    [string]$RunnerZipUrl = '{RUNNER_ZIP_URL}',",
        f"    {system32_parameter('Sc', 'sc.exe')},",
        f"    [hashtable]$Tokens = {_tokens_literal(tokens)},",
        *base_render.parameter_lines(host["base"]),
        f"    [int]$StopWaitSeconds = {STOP_WAIT_SECONDS},",
        _installs_parameter([_install_row(install) for install in host["installs"]]),
        ")",
        *STRICT_HEADER,
        *base_render.INSTALL_LINES,
        "foreach ($install in ($Installs | ConvertFrom-Json)) {",
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
        *_SERVICE_LINES,
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_audit_script(host: WindowsRunnerHost) -> str:
    """The script that prints one ``CHECK`` line per row :func:`expected_checks` names.

    Args:
        host: The host, with at least one install.

    Returns:
        The complete script. It reads and changes nothing but ``sc.exe``'s,
        git's and the registry's answers, and exits 0: drift is data.

    Raises:
        ValueError: When a service name or a machine variable could not be
            embedded verbatim.
    """
    services = [
        {"Service": scriptable(install["service"], label="service")} for install in host["installs"]
    ]
    lines = [
        "param(",
        f"    {system32_parameter('Sc', 'sc.exe')},",
        *base_render.parameter_lines(host["base"]),
        _installs_parameter(services),
        ")",
        *STRICT_HEADER,
        "function Get-Answer {",
        "    param([string]$Tool, [string[]]$Arguments)",
        "    $text = ((& $Tool @Arguments) -join ' ') -replace '\\s+', ' '",
        "    return [pscustomobject]@{ Code = $LASTEXITCODE; Text = $text.Trim() }",
        "}",
        *base_render.AUDIT_LINES,
        "foreach ($install in ($Installs | ConvertFrom-Json)) {",
        "    $service = $install.Service",
        "    $state = Get-Answer $Sc @('query', $service)",
        "    if ($state.Code -eq 0 -and $state.Text -match 'STATE : 4 RUNNING') {",
        '        "CHECK service:$service OK"',
        "    } else {",
        '        "CHECK service:$service DRIFT sc.exe query exited $($state.Code): $($state.Text)"',
        "    }",
        "    $config = Get-Answer $Sc @('qc', $service)",
        "    if ($config.Code -eq 0 -and $config.Text -match 'START_TYPE : 2 AUTO_START' "
        f"-and $config.Text -match '{_START_NAME_PATTERN}') {{",
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
        # Drift is data: the last native read may have exited non-zero, and
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
        The base's rows (:func:`fleet.core.runner_windows_base_render.expected_checks`),
        then three per install: its state, its configuration and its
        recovery actions.
    """
    checks = base_render.expected_checks(host["base"])
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
    """The install script committed under ``rendered/``, for the example host.

    Returns:
        The script, with no tokens.
    """
    return render_install_script(_example(), {})


def committed_audit_script() -> str:
    """The audit script committed under ``rendered/``, for the example host.

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
    "STOP_WAIT_SECONDS",
    "committed_audit_script",
    "committed_install_script",
    "expected_checks",
    "render_audit_script",
    "render_install_script",
]
