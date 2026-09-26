"""The account a Windows-side runner service runs as.

THIS IS A PRIVILEGE DECISION AND IT WAS MADE DELIBERATELY, in MCPs'
``scripts/ops/provision-actions-runner.ps1``, whose ``-ServiceAccount``
defaults to SYSTEM with the reasoning this module carries forward.
``config.cmd --runasservice`` alone installs the service as NETWORK SERVICE,
and that account cannot do what these repositories' jobs need:

- ``actions/setup-python`` writes registry keys when its cache misses;
- node-gyp needs a toolchain NETWORK SERVICE cannot install;
- MCPs' supervisor Pester case registers real scheduled tasks, which
  NETWORK SERVICE is refused ("CimException: Access is denied" on MCPs run
  36254365786, the first CI run after lavender's reinstall).

SYSTEM can do all three. THE COST, SAID PLAINLY: workflow code then runs with
full machine privilege on the host. That is acceptable only because every
repository with a Windows-side install is private, so no fork's pull request
can run on these runners. On a public repository it would hand the machine
to strangers.

The rebuild's first recipe dropped the ``--windowslogonaccount`` argument,
and the reinstalled lavender came up as NETWORK SERVICE. That is why the
account is both rendered into ``config.cmd`` and CONVERGED: a service
already installed under another account is rebound in place.
"""

from __future__ import annotations

from fleet.contracts.runners import RunnerInstall
from fleet.core.script_values import scriptable

#: The account as ``config.cmd --windowslogonaccount`` names it.
WINDOWS_SERVICE_ACCOUNT = "NT AUTHORITY\\SYSTEM"

#: The same account as ``Win32_Service.StartName`` and ``sc.exe obj=`` name it.
WINDOWS_SERVICE_START_NAME = "LocalSystem"


def render_service_account_lines(install: RunnerInstall) -> list[str]:
    """PowerShell lines that bind an installed runner service to SYSTEM.

    Run after the install, whose ``config.cmd`` already names the account;
    these lines rebind a service that an earlier install left under another
    one. CHANGING THE ACCOUNT ORPHANS THE WORK TREE: ``_work`` belongs to
    whichever account created it, and git under the new one refuses the
    checkout as "dubious ownership" (the tracked-imports failure the MCPs
    script records). The tree is removed rather than waved through with
    ``safe.directory``, because the mismatch is real. That also removes the
    tool cache inside it, so these lines run before the cache is seeded.

    Args:
        install: A windows-side install.

    Returns:
        The lines. A service already running as SYSTEM is left alone; a
        missing service or a failed rebind throws, and the stopped service
        stays stopped for the audit to report.

    Raises:
        ValueError: When the service name or workdir cannot be embedded
            verbatim; see :func:`fleet.core.script_values.scriptable`.
    """
    service = scriptable(install["service"], label="service")
    workdir = scriptable(install["workdir"], label="workdir").replace("/", "\\")
    return [
        f"$Service = Get-CimInstance Win32_Service -Filter \"Name='{service}'\"",
        f"if ($null -eq $Service) {{ throw 'service {service} is not installed' }}",
        f"if ($Service.StartName -ne '{WINDOWS_SERVICE_START_NAME}') {{",
        f"    Stop-Service -Name '{service}'",
        f"    & sc.exe config '{service}' obj= {WINDOWS_SERVICE_START_NAME}",
        "    if ($LASTEXITCODE -ne 0) { "
        f"throw 'sc.exe config {service} exited ' + $LASTEXITCODE }}",
        f"    if (Test-Path -LiteralPath '{workdir}') {{ "
        f"Remove-Item -LiteralPath '{workdir}' -Recurse -Force }}",
        f"    Start-Service -Name '{service}'",
        f"    Write-Output ('rebound {service} from ' + $Service.StartName + "
        f"' to {WINDOWS_SERVICE_START_NAME} and removed its work tree')",
        "}",
    ]


__all__ = [
    "WINDOWS_SERVICE_ACCOUNT",
    "WINDOWS_SERVICE_START_NAME",
    "render_service_account_lines",
]
