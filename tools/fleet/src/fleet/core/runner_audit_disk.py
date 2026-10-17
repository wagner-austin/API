"""Holding a runner host's distro root to its ceiling.

One row per host, the audit's (:mod:`fleet.core.runner_audit`): ``df -BG``
of ``/`` inside the distro against the roster's ceiling, the idle baseline
of a rebuilt host carried in the row's id. The rows that say where a grown
root went are the cache rows (:mod:`fleet.core.runner_audit_cache`).
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec


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


def render_disk_check_lines(spec: HostRunnerSpec) -> list[str]:
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


__all__ = ["disk_check_id", "render_disk_check_lines"]
