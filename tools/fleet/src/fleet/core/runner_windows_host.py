"""A Windows runner host's two acts over ssh: its audit, and its rebuild.

MCPs board task 1e9ab738 (decision 5eab8a7b). The scripts are
:mod:`fleet.core.runner_windows_render`'s; each is sent to the host's
``payload_dir`` and run in its ssh session, which holds an elevated token
(:func:`fleet.core.remote.root_invocation`), and the audit's lines are
scored by the reader a Linux host's are
(:func:`fleet.core.runner_transcript.score_transcript`).

A rebuild mints one registration token per repository, as a Linux host's
does (:mod:`fleet.core.runner_rebuild`), runs the install, and audits the
host as proof. Each step of the install is idempotent, so a rebuild cut
short is finished by running it again.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.node import NodePlatform
from fleet.contracts.runner_windows import WindowsRunnerHost
from fleet.core import remote
from fleet.core.runner_audit_types import AuditFinding, AuditOutcome
from fleet.core.runner_github import mint_registration_token
from fleet.core.runner_transcript import score_transcript
from fleet.core.runner_windows_render import (
    AUDIT_STEM,
    INSTALL_STEM,
    expected_checks,
    render_audit_script,
    render_install_script,
)

#: The audit's deadline: twelve ``sc.exe`` reads for four runners.
AUDIT_TIMEOUT_SECONDS = 120

#: The install's deadline: on a first install, four runner releases of about
#: 100 MB each are fetched and four ``config.cmd`` registrations run.
INSTALL_TIMEOUT_SECONDS = 1800


class WindowsRebuildReport(TypedDict):
    """What rebuilding a Windows runner host did and found.

    Attributes:
        steps: One line per act, the install's own lines among them.
        findings: The audit's verdicts afterwards.
    """

    steps: list[str]
    findings: list[AuditFinding]


def payload_path(host: WindowsRunnerHost, stem: str) -> str:
    """Where one of the host's scripts lands.

    Args:
        host: The host.
        stem: The script's stem.

    Returns:
        ``<payload_dir>/<stem>.ps1``.
    """
    return f"{host['payload_dir']}/{stem}.ps1"


def attempt_audit_windows_host(host: WindowsRunnerHost) -> AuditOutcome:
    """Audit one Windows runner host, reporting unreachability as a value.

    Args:
        host: The host.

    Returns:
        Every check's verdict, or the reason the host could not be audited.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` from the scorer, for the
            reason :func:`fleet.core.runner_audit.attempt_audit_host` gives.
    """
    outcome = remote.attempt_root_script(
        host["host"],
        payload_path(host, AUDIT_STEM),
        render_audit_script(host),
        platform=NodePlatform.WINDOWS,
        timeout_seconds=AUDIT_TIMEOUT_SECONDS,
    )
    failure = outcome["failure"]
    if failure is not None:
        return AuditOutcome(findings=None, reason=failure["message"])
    findings = score_transcript(host["name"], expected_checks(host), outcome["output"])
    return AuditOutcome(findings=findings, reason="")


def rebuild_windows_host(host: WindowsRunnerHost) -> WindowsRebuildReport:
    """Install, register and start every runner the roster gives a Windows host, then audit it.

    Args:
        host: The host.

    Returns:
        The steps and the audit's findings.

    Raises:
        AppError: ``RUNNER_TOKEN_UNAVAILABLE`` from the mint; the remote
            layer's codes from the install; ``NODE_UNREACHABLE`` when the
            host stops answering between the install and its audit; and
            ``RUNNER_AUDIT_UNPARSABLE`` from the scorer.
    """
    repos = sorted({install["repo"] for install in host["installs"]})
    tokens = {repo: mint_registration_token(repo) for repo in repos}
    output = remote.run_root_script(
        host["host"],
        payload_path(host, INSTALL_STEM),
        render_install_script(host, tokens),
        platform=NodePlatform.WINDOWS,
        timeout_seconds=INSTALL_TIMEOUT_SECONDS,
    )
    steps = [f"tokens: minted for {len(tokens)} repositories"]
    steps += [line.strip() for line in output.splitlines() if line.strip()]
    audit = attempt_audit_windows_host(host)
    findings = audit["findings"]
    if findings is None:
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE,
            f"{host['name']} stopped answering between its install and its audit: "
            f"{audit['reason']}",
        )
    return WindowsRebuildReport(steps=steps, findings=findings)


__all__ = [
    "AUDIT_TIMEOUT_SECONDS",
    "INSTALL_TIMEOUT_SECONDS",
    "WindowsRebuildReport",
    "attempt_audit_windows_host",
    "payload_path",
    "rebuild_windows_host",
]
