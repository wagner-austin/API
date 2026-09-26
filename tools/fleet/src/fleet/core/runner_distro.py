"""Running a bash payload inside a runner host's WSL distro, as root.

Lifted out of :mod:`fleet.core.runner_onboard` when ``--rebuild`` became the
second caller (board task 1aa6a021): onboarding provisions one runner this
way, and a rebuild provisions the distro's base, its ``wsl.conf`` and every
runner the same way. One implementation, so the route that survived the
ssh->cmd->powershell->wsl->bash gauntlet is the only route.

THE PAYLOAD IS A FILE, NEVER A COMMAND LINE. The bash text is sent to the
host's ``scratch_dir`` and run by path through a one-line PowerShell driver,
per :mod:`fleet.core.remote`'s rule. That also matters for one measured
reason beyond quoting: ``bash -c`` through System32's WSL launcher
re-expands its argument, which turned a ``sed`` expression's ``$`` into
``0`` in the 2026-09-26 review re-run of this package.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import remote


def windows_to_wsl_path(path: str) -> str:
    """A Windows drive path as the distro sees it.

    Args:
        path: A forward-slashed drive-letter path, e.g. ``C:/fleet/stage``.

    Returns:
        The ``/mnt/<drive>/...`` form.

    Raises:
        ValueError: On a path with no drive letter -- translating a
            relative or POSIX path would silently point the distro at the
            wrong file.
    """
    if len(path) < 3 or not path[0].isalpha() or path[1] != ":" or path[2] != "/":
        raise ValueError(f"not a forward-slashed drive path: {path!r}")
    return f"/mnt/{path[0].lower()}{path[2:]}"


def run_distro_script(spec: HostRunnerSpec, stem: str, body: str, *, timeout_seconds: int) -> str:
    """Send a bash payload to the host and run it inside the distro as root.

    Args:
        spec: The host.
        stem: The file-name stem; the payload lands as ``<stem>.sh`` and its
            driver as ``<stem>-driver.ps1`` in the host's ``scratch_dir``.
        body: The complete bash text, shebang included.
        timeout_seconds: The deadline for running it; the caller names it
            because only the caller knows whether it installs packages.

    Returns:
        The payload's standard output.

    Raises:
        AppError: ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            remote layer; a payload exiting non-zero surfaces as the latter
            with its own stderr.
    """
    windows_payload_path = f"{spec['scratch_dir']}/{stem}.sh"
    remote.send_script(spec["host"], windows_payload_path, body, platform="windows")
    driver = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "$env:WSL_UTF8 = '1'",
            f"wsl -d '{spec['wsl_distro']}' -u root -- bash "
            f"'{windows_to_wsl_path(windows_payload_path)}'",
            "exit $LASTEXITCODE",
        ]
    )
    return remote.run_script_within(
        spec["host"],
        f"{spec['scratch_dir']}/{stem}-driver.ps1",
        driver,
        platform="windows",
        timeout_seconds=timeout_seconds,
    )


__all__ = ["run_distro_script", "windows_to_wsl_path"]
