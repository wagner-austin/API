"""Scoring a CI host against the runner roster.

One rendered PowerShell script per host, sent and run by path under the same
rule as every other remote act in this package (:mod:`fleet.core.remote`: the
bytes that run are the bytes that were sent). The script emits exactly one
``CHECK <id> OK`` or ``CHECK <id> DRIFT <detail>`` line per declared item, in
declaration order, and exits 0 -- drift is data, not a command failure, so a
host with three problems reports three lines rather than dying on the first.

THE TRANSCRIPT IS VALIDATED AGAINST THE ROSTER, NOT JUST PARSED. The script
promises one line per expected check in a known order; a transcript with a
missing, duplicated, reordered or malformed line means the script died midway
or the transport mangled it, and scoring what remains would let a crashed
audit read as a clean host. That is ``RUNNER_AUDIT_UNPARSABLE``, raised, not
reported -- an audit that could not complete has established nothing.

WSL OUTPUT IS FORCED TO UTF-8. ``wsl.exe`` writes UTF-16LE when its output is
redirected, which arrives here as NUL-riddled text that matches nothing;
``WSL_UTF8=1`` at the top of the driver is what every check inside the distro
depends on to be comparable at all.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.node import NodePlatform
from fleet.contracts.runners import HostRunnerSpec
from fleet.core import remote
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter
from fleet.core.runner_account import (
    SERVICE_ACCOUNT_REASON,
    render_service_account_check_lines,
    service_account_check_id,
)
from fleet.core.runner_base_render import EXECUTION_POLICY_KEY, LONG_PATHS_KEY
from fleet.core.runner_machine_env import (
    MACHINE_ENVIRONMENT_KEY,
    machine_variable_check_id,
    render_machine_environment_check_lines,
)
from fleet.core.script_values import scriptable

#: File name the rendered audit script lands under in the host's scratch_dir.
AUDIT_SCRIPT_NAME = "fleet-runner-audit.ps1"

#: The row holding a host to long paths, both Win32's and git's.
LONG_PATHS_CHECK_ID = "long-paths:win32-and-git"


class ExpectedCheck(TypedDict):
    """One check the audit script promises to report.

    Attributes:
        check_id: Stable identifier, derived from the roster entry so two
            audits of one host agree line for line.
        reason: Why the item matters -- printed beside a drift line so the
            reader learns what breaks without opening the roster.
    """

    check_id: str
    reason: str


class AuditFinding(TypedDict):
    """One check's verdict.

    Attributes:
        check_id: The check that reported.
        ok: Whether the host satisfies it.
        detail: What the host actually said, for the drift line.
        reason: The roster's reason for the check.
    """

    check_id: str
    ok: bool
    detail: str
    reason: str


class AuditOutcome(TypedDict):
    """What auditing a host produced, or why it produced nothing.

    Same shape as :class:`fleet.core.probe.ProbeOutcome` and for the same
    caller: an audit over several hosts must report an unreachable one as a
    line beside the hosts that answered, not as a crash that hides them.

    Attributes:
        findings: Every check's verdict, or ``None`` when the host could not
            be audited at all.
        reason: Why there are no findings. Empty when there are.
    """

    findings: list[AuditFinding] | None
    reason: str


def expected_checks(spec: HostRunnerSpec) -> list[ExpectedCheck]:
    """Every check the audit script for this host will report, in order.

    The single source of both the script's Write-Check lines and the transcript
    validator, so the two cannot disagree about what a complete audit is.

    Args:
        spec: The host's roster entry.

    Returns:
        One entry per check, in the order the script emits them.
    """
    checks: list[ExpectedCheck] = []
    keepalive = spec["keepalive_task"]
    if keepalive is not None:
        checks.append(
            ExpectedCheck(
                check_id=f"keepalive:{keepalive}",
                reason="the scheduled task is the only thing holding the WSL VM open",
            )
        )
    floor = spec["wslconfig_min_memory_gb"]
    if floor is not None:
        checks.append(
            ExpectedCheck(
                check_id=f"memory-floor:{floor}gb",
                reason="two CI jobs must fit in the VM at once",
            )
        )
    checks.append(
        ExpectedCheck(
            check_id=disk_check_id(spec),
            reason="a runner host holds nothing that needs keeping, so its distro stays near "
            "the idle baseline; growth past the ceiling is a leak, as on 2026-09-25",
        )
    )
    checks.append(
        ExpectedCheck(
            check_id=f"execution-policy:LocalMachine:{spec['base']['execution_policy']}",
            reason="a Windows runner's PowerShell steps run scripts from _temp, which the "
            "Restricted default refuses",
        )
    )
    checks.append(
        ExpectedCheck(
            check_id=LONG_PATHS_CHECK_ID,
            reason="CI trees under the runner service's temp directory pass 260 characters, "
            "which a fresh install refuses",
        )
    )
    checks.extend(
        ExpectedCheck(check_id=machine_variable_check_id(variable), reason=variable["reason"])
        for variable in spec["base"]["machine_environment"]
    )
    if spec["gpu_required"]:
        # One per WSL runner, on that runner's own PATH: the jobs that
        # digest the card are the runner's, and a host-level check from an
        # interactive shell passed on the rebuilt lavender while its runners
        # could not find nvidia-smi (board task 1aa6a021).
        checks.extend(
            ExpectedCheck(
                check_id=f"gpu:{install['repo']}:wsl:{install['runner_name']}",
                reason="runner jobs on this host digest a real GPU",
            )
            for install in spec["installs"]
            if install["side"] == "wsl"
        )
    for timer in spec["systemd_timers"]:
        checks.append(
            ExpectedCheck(
                check_id=f"timer:{timer}",
                reason="host hygiene is part of the provision",
            )
        )
    for install in spec["installs"]:
        checks.append(
            ExpectedCheck(
                check_id=f"service:{install['side']}:{install['service']}",
                reason=f"runs the {install['repo']} runner {install['runner_name']}",
            )
        )
        checks.append(
            ExpectedCheck(
                check_id=f"workdir:{install['repo']}:{install['side']}:{install['runner_name']}",
                reason="the install's _work tree, which its venvs are path-bound to",
            )
        )
        if install["side"] == "windows":
            checks.append(
                ExpectedCheck(
                    check_id=service_account_check_id(install), reason=SERVICE_ACCOUNT_REASON
                )
            )
    for asset in spec["assets"]:
        checks.append(ExpectedCheck(check_id=f"asset:{asset['path']}", reason=asset["reason"]))
        if asset["sha256"] is not None:
            checks.append(ExpectedCheck(check_id=f"sha256:{asset['path']}", reason=asset["reason"]))
        if asset["writable"]:
            checks.append(
                ExpectedCheck(check_id=f"writable:{asset['path']}", reason=asset["reason"])
            )
    return checks


def disk_check_id(spec: HostRunnerSpec) -> str:
    """The disk row's id, which carries the ceiling and the idle baseline.

    Both numbers are IN the id, so every audit line for the row, OK or
    DRIFT, states what is allowed and what an idle rebuilt host measured
    (board task 1aa6a021, A3).

    Args:
        spec: The host's roster entry.

    Returns:
        E.g. ``disk:/:ceiling-150gb:baseline-46gb@2026-09-26``.
    """
    disk = spec["base"]["disk"]
    return (
        f"disk:/:ceiling-{disk['ceiling_gb']}gb:"
        f"baseline-{disk['baseline_gb']}gb@{disk['baseline_measured']}"
    )


def _disk_check_lines(spec: HostRunnerSpec) -> list[str]:
    """Script lines that hold the distro's root to the roster's ceiling.

    Args:
        spec: The host.

    Returns:
        The lines: ``df -BG`` of ``/`` inside the distro, parsed to whole GB,
        and the Write-Check. An unreadable answer parses to -1 and drifts with df's
        own words and exit, never passes.
    """
    disk = spec["base"]["disk"]
    ceiling = disk["ceiling_gb"]
    return [
        "$Probe = Invoke-InDistro $Cmd $Wsl $Distro 'df -BG --output=used /'",
        "$DiskLine = (@($Probe.Lines | Select-Object -Skip 1 -First 1) -join '')",
        "$UsedGb = -1",
        "if ($DiskLine -match '(\\d+)G') {",
        "    $UsedGb = [int]$Matches[1]",
        "}",
        f"Write-Check '{disk_check_id(spec)}' ($UsedGb -ge 0 -and $UsedGb -le {ceiling}) "
        f"('the distro root uses ' + $UsedGb + ' GB against a ceiling of {ceiling} GB; an "
        f"idle rebuilt host used {disk['baseline_gb']} GB on {disk['baseline_measured']}; "
        "df said: ' + $Probe.Text)",
    ]


def _emit_wsl_state_check(check_id: str, argv: str, expected: str) -> list[str]:
    """Script lines for a check that compares a WSL command's first line.

    Args:
        check_id: The check to report.
        argv: The command after ``wsl -d <distro> --``, already validated.
        expected: The exact first output line that means OK.

    Returns:
        The PowerShell lines.
    """
    return [
        f'$Probe = Invoke-InDistro $Cmd $Wsl $Distro "{argv}"',
        "$State = (@($Probe.Lines | Select-Object -First 1) -join '')",
        f"Write-Check '{check_id}' ($State -eq '{expected}') ('it said: ' + $Probe.Text)",
    ]


def _emit_wsl_test_check(check_id: str, test_flag: str, path: str) -> list[str]:
    """Script lines for a check driven by ``test`` inside the distro.

    Args:
        check_id: The check to report.
        test_flag: The ``test`` flag, e.g. ``-e`` or ``-w``.
        path: The path to test, already validated.

    Returns:
        The PowerShell lines.
    """
    return [
        f"$Probe = Invoke-InDistro $Cmd $Wsl $Distro \"test {test_flag} '{path}'\"",
        f"Write-Check '{check_id}' ($Probe.Exit -eq 0) ('test {test_flag} exited ' + $Probe.Exit)",
    ]


def _gpu_check_lines(spec: HostRunnerSpec) -> list[str]:
    """Script lines that ask each WSL runner's own PATH for the GPU.

    The row asks for exit 0 as well as a name: the probe's words include
    wsl's own stderr, so a distro that is not there answers "There is no
    distribution with the supplied name.", which is not empty and passed the
    row until the Pester suite's sick host showed it (MCPs board task
    d69786fa).

    Args:
        spec: The host, with ``gpu_required`` set.

    Returns:
        Two lines per WSL install: nvidia-smi run with PATH read from the
        runner's ``.path`` (the PATH runsvc.sh gives its jobs), and the
        Write-Check.

    Raises:
        ValueError: When a runner directory cannot be embedded verbatim.
    """
    lines: list[str] = []
    for install in (i for i in spec["installs"] if i["side"] == "wsl"):
        runner_dir = scriptable(install["workdir"].rsplit("/", 1)[0], label="workdir")
        check_id = f"gpu:{install['repo']}:wsl:{install['runner_name']}"
        lines += [
            f"$Probe = Invoke-InDistro $Cmd $Wsl $Distro \"sh -c 'PATH=`$(cat {runner_dir}/.path) "
            "nvidia-smi --query-gpu=name --format=csv,noheader'\"",
            "$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')",
            f"Write-Check '{check_id}' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) "
            "('nvidia-smi on the runner PATH said: ' + $Probe.Text)",
        ]
    return lines


def render_audit_script(spec: HostRunnerSpec) -> str:
    """The PowerShell audit driver for one host.

    UNDER THE STRICT HEADER, EVERY PROBE A PARAMETER (MCPs board task
    d69786fa). It ran under ``Continue`` with each ``wsl`` read piped
    through ``2>$null``, so a probe that failed and a line that threw alike
    left the transcript short. Every native now runs through one
    ``Invoke-Probe``, cmd.exe carrying the ``2>&1``, and answers its exit
    and its words, which a DRIFT row carries; nothing is suppressed and no
    Write-Check can be skipped. ``cmd.exe``, ``wsl.exe``, ``schtasks.exe``, ``git``
    and the three registry keys are parameters with plain-string defaults,
    and the Windows-side reads (a service's state and account, a workdir's
    presence) are script-block parameters whose defaults only read, so the
    Pester suite over the committed render runs the defaults and stand-ins
    alike.

    Args:
        spec: The host's roster entry.

    Returns:
        The complete script text. Emits one CHECK line per entry of
        :func:`expected_checks`, in that order, and exits 0 -- drift is data.

    Raises:
        ValueError: When a roster value cannot be embedded verbatim; see
            :func:`fleet.core.script_values.scriptable`.
    """
    distro = scriptable(spec["wsl_distro"], label="wsl_distro")
    lines: list[str] = [
        "param(",
        f"    [string]$Distro = '{distro}',",
        f"    [string]$PolicyKey = '{EXECUTION_POLICY_KEY}',",
        f"    [string]$FileSystemKey = '{LONG_PATHS_KEY}',",
        f"    [string]$EnvironmentKey = '{MACHINE_ENVIRONMENT_KEY}',",
        "    [scriptblock]$GetService = { param([string]$Name) "
        "@(Get-CimInstance Win32_Service -Filter \"Name='$Name'\") },",
        "    [scriptblock]$TestWorkdir = { param([string]$Path) Test-Path -LiteralPath $Path },",
        "    [string]$Git = 'git',",
        "    " + system32_parameter("Schtasks", "schtasks.exe") + ",",
        "    " + system32_parameter("Wsl", "wsl.exe") + ",",
        "    " + system32_parameter("Cmd", "cmd.exe"),
        ")",
        *STRICT_HEADER,
        # wsl.exe writes UTF-16LE when redirected; forcing UTF-8 is what
        # makes every comparison below possible. See the module docstring.
        "$env:WSL_UTF8 = '1'",
        "function Write-Check {",
        "    param([string]$CheckId, [bool]$Ok, [string]$Detail)",
        "    if ($Ok) {",
        "        Write-Output ('CHECK ' + $CheckId + ' OK')",
        "    } else {",
        "        $flat = $Detail -replace \"[`r`n]+\", ' '",
        "        Write-Output ('CHECK ' + $CheckId + ' DRIFT ' + $flat)",
        "    }",
        "}",
        "function Invoke-Probe {",
        "    param([string]$Shell, [string]$Line)",
        '    $said = & $Shell /d /s /c "$Line 2>&1"',
        "    $exit = $LASTEXITCODE",
        "    $all = [string[]]@($said | ForEach-Object { [string]$_ })",
        "    return [pscustomobject]@{ Exit = $exit; Lines = $all; "
        "Text = (($all -join ' ') + ' (exit ' + $exit + ')') }",
        "}",
        "function Invoke-InDistro {",
        "    param([string]$Shell, [string]$WslPath, [string]$Name, [string]$Command)",
        '    return Invoke-Probe $Shell "`"$WslPath`" -d $Name -- $Command"',
        "}",
    ]
    keepalive = spec["keepalive_task"]
    if keepalive is not None:
        task = scriptable(keepalive, label="keepalive_task")
        lines += [
            f'$Probe = Invoke-Probe $Cmd "`"$Schtasks`" /query /tn {task} /fo csv"',
            "$KeepaliveRow = (@($Probe.Lines | Select-Object -Skip 1 -First 1) -join '')",
            f"Write-Check 'keepalive:{task}' ($KeepaliveRow -match '\"Running\"') "
            "('schtasks row: ' + $Probe.Text)",
        ]
    floor = spec["wslconfig_min_memory_gb"]
    if floor is not None:
        # Measured VM memory rather than the .wslconfig text: the ceiling
        # that matters is the one the VM actually got, and a config file the
        # VM has not been restarted into satisfies the text check while the
        # jobs still swap.
        floor_mb = floor * 1000
        lines += [
            "$Probe = Invoke-InDistro $Cmd $Wsl $Distro 'free -m'",
            "$MemLine = (@($Probe.Lines | Where-Object { $_ -match '^Mem:' }) -join '')",
            "$TotalMb = 0",
            "if ($MemLine -match 'Mem:\\s+(\\d+)') {",
            "    $TotalMb = [int]$Matches[1]",
            "}",
            f"Write-Check 'memory-floor:{floor}gb' ($TotalMb -ge {floor_mb}) "
            "('the VM reports ' + $TotalMb + ' MB; free said: ' + $Probe.Text)",
        ]
    lines += _disk_check_lines(spec)
    policy = scriptable(spec["base"]["execution_policy"], label="execution_policy")
    lines += [
        "$Policy = [string](Get-Item -LiteralPath $PolicyKey).GetValue('ExecutionPolicy')",
        f"Write-Check 'execution-policy:LocalMachine:{policy}' ($Policy -eq '{policy}') "
        "('the LocalMachine ExecutionPolicy value is: ' + $Policy)",
        "$LongPaths = [string](Get-Item -LiteralPath $FileSystemKey).GetValue('LongPathsEnabled')",
        '$Probe = Invoke-Probe $Cmd "`"$Git`" config --system --get core.longpaths"',
        "$GitLongPaths = (@($Probe.Lines) -join '')",
        f"Write-Check '{LONG_PATHS_CHECK_ID}' ($LongPaths -eq '1' -and $GitLongPaths -eq 'true') "
        "('LongPathsEnabled=' + $LongPaths + ' git core.longpaths=' + $Probe.Text)",
        *render_machine_environment_check_lines(spec),
    ]
    if spec["gpu_required"]:
        lines += _gpu_check_lines(spec)
    for timer in spec["systemd_timers"]:
        name = scriptable(timer, label="systemd timer")
        lines += _emit_wsl_state_check(f"timer:{name}", f"systemctl is-enabled '{name}'", "enabled")
    for install in spec["installs"]:
        service = scriptable(install["service"], label="service")
        workdir = scriptable(install["workdir"], label="workdir")
        repo = scriptable(install["repo"], label="repo")
        runner_name = scriptable(install["runner_name"], label="runner_name")
        if install["side"] == "wsl":
            lines += _emit_wsl_state_check(
                f"service:wsl:{service}", f"systemctl is-active '{service}'", "active"
            )
            lines += _emit_wsl_test_check(f"workdir:{repo}:wsl:{runner_name}", "-d", workdir)
        else:
            # Windows-side installs are checked NATIVELY: the driver already
            # runs in the host's PowerShell, so the service and the workdir
            # are one read away rather than one wsl hop away. The state is
            # JOINED from the rows, so an absent service is '' and drifts.
            lines += [
                f"$Service = @(& $GetService '{service}')",
                "$ServiceState = (@($Service | ForEach-Object { [string]$_.State }) -join '')",
                f"Write-Check 'service:windows:{service}' ($ServiceState -eq 'Running') "
                "('Win32_Service State: ' + $ServiceState)",
                f"Write-Check 'workdir:{repo}:windows:{runner_name}' "
                f"([bool](& $TestWorkdir '{workdir}')) "
                f"('Test-Path {workdir}')",
                *render_service_account_check_lines(install),
            ]
    for asset in spec["assets"]:
        path = scriptable(asset["path"], label="asset path")
        lines += _emit_wsl_test_check(f"asset:{path}", "-e", path)
        pin = asset["sha256"]
        if pin is not None:
            # The output is JOINED, never cast. [string] over a pipeline that
            # emitted nothing is $null in Windows PowerShell 5.1, not '', so
            # for a MISSING pinned asset $Sum.StartsWith threw and the
            # transcript lacked this one line: the audit then refused to
            # score the host at all, as a script that died midway, instead
            # of reporting one drifted check (the 2026-09-26 lavender
            # rebuild, board task 1aa6a021).
            lines += [
                f"$Probe = Invoke-InDistro $Cmd $Wsl $Distro \"sha256sum '{path}'\"",
                "$Sum = (@($Probe.Lines | Select-Object -First 1) -join '')",
                f"Write-Check 'sha256:{path}' ($Sum.StartsWith('{pin}')) "
                "('sha256sum said: ' + $Probe.Text)",
            ]
        if asset["writable"]:
            lines += _emit_wsl_test_check(f"writable:{path}", "-w", path)
    lines.append("exit 0")
    return "\n".join(lines) + "\n"


def parse_audit_transcript(spec: HostRunnerSpec, output: str) -> list[AuditFinding]:
    """Validate a transcript against the roster and score it.

    Args:
        spec: The host's roster entry.
        output: The audit script's standard output.

    Returns:
        One finding per expected check, in declaration order.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` when any line is not a CHECK
            line, a DRIFT line carries no detail, or the sequence of check
            ids is not exactly the expected one. A transcript that stops
            early is a script that died midway, and the checks it never
            reached must not read as passed.
    """
    expected = expected_checks(spec)
    findings: list[AuditFinding] = []
    rows: list[tuple[str, bool, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split(" ", 3)
        if len(parts) < 3 or parts[0] != "CHECK" or parts[2] not in ("OK", "DRIFT"):
            raise _unparsable(spec, f"line is not a CHECK line: {line!r}")
        check_id, verdict = parts[1], parts[2]
        detail = parts[3] if len(parts) == 4 else ""
        if verdict == "DRIFT" and not detail.strip():
            raise _unparsable(spec, f"DRIFT line for {check_id} carries no detail")
        rows.append((check_id, verdict == "OK", detail))
    reported_ids = [row[0] for row in rows]
    expected_ids = [check["check_id"] for check in expected]
    if reported_ids != expected_ids:
        raise _unparsable(
            spec,
            f"expected checks {expected_ids} but the transcript reported {reported_ids}; "
            "a transcript that stops early is a script that died midway",
        )
    for (check_id, ok, detail), check in zip(rows, expected, strict=True):
        findings.append(
            AuditFinding(check_id=check_id, ok=ok, detail=detail, reason=check["reason"])
        )
    return findings


def _unparsable(spec: HostRunnerSpec, detail: str) -> AppError[FleetErrorCode]:
    """The error for a transcript that cannot be scored.

    Args:
        spec: The host whose transcript failed.
        detail: What was wrong with it.

    Returns:
        The error to raise.
    """
    return AppError(
        FleetErrorCode.RUNNER_AUDIT_UNPARSABLE,
        f"the audit transcript from {spec['name']} cannot be scored: {detail}",
    )


def attempt_audit_host(spec: HostRunnerSpec) -> AuditOutcome:
    """Audit one host, reporting unreachability as a value.

    Args:
        spec: The host's roster entry.

    Returns:
        Every check's verdict, or the reason the host could not be audited.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` as
            :func:`parse_audit_transcript` describes. Deliberately NOT folded
            into the outcome: an unreachable host is a fleet condition to
            report beside the others, while an unscorable transcript from a
            host that answered is a fault in this tooling or its transport,
            and must stop the audit rather than print as one more line.
    """
    outcome = remote.attempt_script(
        spec["host"],
        f"{spec['scratch_dir']}/{AUDIT_SCRIPT_NAME}",
        render_audit_script(spec),
        platform=NodePlatform.WINDOWS,
    )
    failure = outcome["failure"]
    if failure is not None:
        return AuditOutcome(findings=None, reason=failure["message"])
    return AuditOutcome(findings=parse_audit_transcript(spec, outcome["output"]), reason="")


__all__ = [
    "AUDIT_SCRIPT_NAME",
    "LONG_PATHS_CHECK_ID",
    "AuditFinding",
    "AuditOutcome",
    "ExpectedCheck",
    "attempt_audit_host",
    "disk_check_id",
    "expected_checks",
    "parse_audit_transcript",
    "render_audit_script",
]
