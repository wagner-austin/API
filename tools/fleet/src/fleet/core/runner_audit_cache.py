"""Holding a runner host's caches to their ceilings.

The disk row (:func:`fleet.core.runner_audit.disk_check_id`) holds the
distro's root to one number, which says a host has grown but not where. On
lavender the root went from 10 GB to 97 GB in the day after its 2026-09-26
rebuild (MCPs board task bfca20e6): 50 GB was the runner account's cache,
which pip, npm, poetry and playwright fill across jobs and nothing empties,
and about 23 GB was the eight installs' ``_work`` trees. Each of those is a
row of its own here, so the audit that reports the growth names the
directory it is in.

MEASURED WITH ``du -s -BG`` INSIDE THE DISTRO, PARSED TO WHOLE GB. An
unreadable answer (a missing directory, a du that failed) parses to -1 and
drifts with du's own words and exit, never passes. Only WSL installs have a
row: a Windows install's ``_work`` is on the host's own disk, which the
distro ceiling does not govern.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from fleet.contracts.runners import HostRunnerSpec
from fleet.core.script_values import scriptable

#: Why the cache row exists, printed beside its drift line.
CACHE_REASON = (
    "pip, npm, poetry and playwright fill the runner account's cache across jobs and "
    "nothing empties it; it was 50 GB of lavender's 97 GB on 2026-09-27"
)

#: Why each _work row exists, printed beside its drift line.
WORK_REASON = (
    "an install's _work tree keeps every checkout and venv its jobs made; a tree past "
    "its ceiling is a job leaving state behind"
)


class CacheRow(TypedDict):
    """One directory the audit holds to a ceiling.

    Attributes:
        check_id: ``cache:<path>:ceiling-<N>gb``; the ceiling is in the id
            so every line for the row states what is allowed.
        path: The directory inside the distro, absolute.
        ceiling_gb: The most it may hold, in GB.
        reason: Why the row exists.
    """

    check_id: str
    path: str
    ceiling_gb: int
    reason: str


def _row(path: str, ceiling_gb: int, reason: str) -> CacheRow:
    """One row for a directory.

    Args:
        path: The directory inside the distro.
        ceiling_gb: Its ceiling in GB.
        reason: Why the row exists.

    Returns:
        The row, its id carrying the path and the ceiling.
    """
    return CacheRow(
        check_id=f"cache:{path}:ceiling-{ceiling_gb}gb",
        path=path,
        ceiling_gb=ceiling_gb,
        reason=reason,
    )


def cache_rows(spec: HostRunnerSpec) -> list[CacheRow]:
    """Every cache row for a host, in the order the script reports them.

    Args:
        spec: The host's roster entry.

    Returns:
        The runner account's cache first, then one row per WSL install's
        ``_work`` tree in roster order.
    """
    disk = spec["base"]["disk"]
    rows = [_row(disk["cache_path"], disk["cache_ceiling_gb"], CACHE_REASON)]
    rows.extend(
        _row(install["workdir"], disk["work_ceiling_gb"], WORK_REASON)
        for install in spec["installs"]
        if install["side"] == "wsl"
    )
    return rows


def render_cache_check_lines(spec: HostRunnerSpec) -> list[str]:
    """Script lines that measure each cache row inside the distro.

    Args:
        spec: The host's roster entry.

    Returns:
        Six lines per row: ``du -s -BG`` of the directory, its first line
        parsed to whole GB (-1 when unreadable), and the Write-Check.

    Raises:
        ValueError: When a path cannot be embedded verbatim; see
            :func:`fleet.core.script_values.scriptable`.
    """
    lines: list[str] = []
    for row in cache_rows(spec):
        path = scriptable(row["path"], label="cache path")
        ceiling = row["ceiling_gb"]
        lines += [
            f"$Probe = Invoke-InDistro $Cmd $Wsl $Distro \"du -s -BG '{path}'\"",
            "$CacheGb = -1",
            "if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\\d+)G') {",
            "    $CacheGb = [int]$Matches[1]",
            "}",
            f"Write-Check '{row['check_id']}' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and "
            f"$CacheGb -le {ceiling}) ('{path} holds ' + $CacheGb + ' GB against a ceiling of "
            f"{ceiling} GB; du said: ' + $Probe.Text)",
        ]
    return lines


__all__ = [
    "CACHE_REASON",
    "WORK_REASON",
    "CacheRow",
    "cache_rows",
    "render_cache_check_lines",
]
