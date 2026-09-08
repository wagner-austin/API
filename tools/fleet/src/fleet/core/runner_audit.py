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

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import remote

#: File name the rendered audit script lands under in the host's scratch_dir.
AUDIT_SCRIPT_NAME = "fleet-runner-audit.ps1"

#: Characters a value must not carry to be embedded in the rendered script.
#:
#: Values land inside single-quoted PowerShell strings, where a quote ends the
#: string and CR/LF end the statement. Escaping is refused in favour of
#: rejection: every legitimate service name, path and task name in the roster
#: is plain, so a value carrying one of these is a roster error to surface,
#: not a case to accommodate.
_UNSCRIPTABLE = ("'", '"', "\r", "\n", "`", "$")


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


def _scriptable(value: str, *, label: str) -> str:
    """Refuse a value the rendered script could not carry verbatim.

    Args:
        value: The roster value about to be embedded.
        label: What the value is, for the error.

    Returns:
        The value, unchanged.

    Raises:
        JSONTypeError-free ValueError: deliberately a plain ValueError --
            this is a roster-content precondition of the renderer, not a
            JSON-shape fault, and the message names the field to fix.
    """
    for forbidden in _UNSCRIPTABLE:
        if forbidden in value:
            raise ValueError(
                f"{label} {value!r} contains {forbidden!r}, which cannot be embedded in "
                "the audit script verbatim; rename the item rather than escaping it"
            )
    return value


def expected_checks(spec: HostRunnerSpec) -> list[ExpectedCheck]:
    """Every check the audit script for this host will report, in order.

    The single source of both the script's Emit lines and the transcript
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
    if spec["gpu_required"]:
        checks.append(
            ExpectedCheck(
                check_id="gpu:nvidia-smi",
                reason="runner jobs on this host digest a real GPU",
            )
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
                check_id=f"service:{install['service']}",
                reason=f"runs the {install['repo']} runner {install['runner_name']}",
            )
        )
        checks.append(
            ExpectedCheck(
                check_id=f"workdir:{install['repo']}:{install['runner_name']}",
                reason="the install's _work tree, which its venvs are path-bound to",
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


def _emit_wsl_state_check(distro: str, check_id: str, argv: str, expected: str) -> list[str]:
    """Script lines for a check that compares a WSL command's first line.

    Args:
        distro: The WSL distribution to run inside.
        check_id: The check to report.
        argv: The command after ``wsl -d <distro> --``, already validated.
        expected: The exact first output line that means OK.

    Returns:
        The PowerShell lines.
    """
    return [
        f"$State = (wsl -d '{distro}' -- {argv} 2>$null | Select-Object -First 1)",
        f"Emit '{check_id}' ([string]$State -eq '{expected}') ('it said: ' + [string]$State)",
    ]


def _emit_wsl_test_check(distro: str, check_id: str, test_flag: str, path: str) -> list[str]:
    """Script lines for a check driven by ``test`` inside the distro.

    Args:
        distro: The WSL distribution to run inside.
        check_id: The check to report.
        test_flag: The ``test`` flag, e.g. ``-e`` or ``-w``.
        path: The path to test, already validated.

    Returns:
        The PowerShell lines.
    """
    return [
        f"wsl -d '{distro}' -- test {test_flag} '{path}' 2>$null | Out-Null",
        f"Emit '{check_id}' ($LASTEXITCODE -eq 0) ('test {test_flag} exited ' + $LASTEXITCODE)",
    ]


def render_audit_script(spec: HostRunnerSpec) -> str:
    """The PowerShell audit driver for one host.

    Args:
        spec: The host's roster entry.

    Returns:
        The complete script text. Emits one CHECK line per entry of
        :func:`expected_checks`, in that order, and exits 0 -- drift is data.

    Raises:
        ValueError: When a roster value cannot be embedded verbatim; see
            :func:`_scriptable`.
    """
    distro = _scriptable(spec["wsl_distro"], label="wsl_distro")
    lines: list[str] = [
        "$ErrorActionPreference = 'Continue'",
        # wsl.exe writes UTF-16LE when redirected; forcing UTF-8 is what
        # makes every comparison below possible. See the module docstring.
        "$env:WSL_UTF8 = '1'",
        "function Emit([string]$CheckId, [bool]$Ok, [string]$Detail) {",
        "  if ($Ok) { Write-Output ('CHECK ' + $CheckId + ' OK') }",
        "  else { Write-Output ('CHECK ' + $CheckId + ' DRIFT ' "
        "+ ($Detail -replace \"[`r`n]+\", ' ')) }",
        "}",
    ]
    keepalive = spec["keepalive_task"]
    if keepalive is not None:
        task = _scriptable(keepalive, label="keepalive_task")
        lines += [
            f"$KeepaliveRow = [string](schtasks /query /tn '{task}' /fo csv 2>$null "
            "| Select-Object -Skip 1 -First 1)",
            f"Emit 'keepalive:{task}' ($KeepaliveRow -match '\"Running\"') "
            "('schtasks row: ' + $KeepaliveRow)",
        ]
    floor = spec["wslconfig_min_memory_gb"]
    if floor is not None:
        # Measured VM memory rather than the .wslconfig text: the ceiling
        # that matters is the one the VM actually got, and a config file the
        # VM has not been restarted into satisfies the text check while the
        # jobs still swap.
        floor_mb = floor * 1000
        lines += [
            f"$MemLine = [string](wsl -d '{distro}' -- free -m 2>$null | Select-String '^Mem:')",
            "$TotalMb = 0",
            "if ($MemLine -match 'Mem:\\s+(\\d+)') { $TotalMb = [int]$Matches[1] }",
            f"Emit 'memory-floor:{floor}gb' ($TotalMb -ge {floor_mb}) "
            "('the VM reports ' + $TotalMb + ' MB')",
        ]
    if spec["gpu_required"]:
        lines += [
            f"$GpuName = [string](wsl -d '{distro}' -- nvidia-smi --query-gpu=name "
            "--format=csv,noheader 2>$null | Select-Object -First 1)",
            "Emit 'gpu:nvidia-smi' ($GpuName.Trim().Length -gt 0) ('nvidia-smi said: ' + $GpuName)",
        ]
    for timer in spec["systemd_timers"]:
        name = _scriptable(timer, label="systemd timer")
        lines += _emit_wsl_state_check(
            distro, f"timer:{name}", f"systemctl is-enabled '{name}'", "enabled"
        )
    for install in spec["installs"]:
        service = _scriptable(install["service"], label="service")
        workdir = _scriptable(install["workdir"], label="workdir")
        repo = _scriptable(install["repo"], label="repo")
        runner_name = _scriptable(install["runner_name"], label="runner_name")
        lines += _emit_wsl_state_check(
            distro, f"service:{service}", f"systemctl is-active '{service}'", "active"
        )
        lines += _emit_wsl_test_check(distro, f"workdir:{repo}:{runner_name}", "-d", workdir)
    for asset in spec["assets"]:
        path = _scriptable(asset["path"], label="asset path")
        lines += _emit_wsl_test_check(distro, f"asset:{path}", "-e", path)
        pin = asset["sha256"]
        if pin is not None:
            lines += [
                f"$Sum = [string](wsl -d '{distro}' -- sha256sum '{path}' 2>$null "
                "| Select-Object -First 1)",
                f"Emit 'sha256:{path}' ($Sum.StartsWith('{pin}')) ('sha256sum said: ' + $Sum)",
            ]
        if asset["writable"]:
            lines += _emit_wsl_test_check(distro, f"writable:{path}", "-w", path)
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
    )
    failure = outcome["failure"]
    if failure is not None:
        return AuditOutcome(findings=None, reason=failure["message"])
    return AuditOutcome(findings=parse_audit_transcript(spec, outcome["output"]), reason="")


__all__ = [
    "AUDIT_SCRIPT_NAME",
    "AuditFinding",
    "AuditOutcome",
    "ExpectedCheck",
    "attempt_audit_host",
    "expected_checks",
    "parse_audit_transcript",
    "render_audit_script",
]
