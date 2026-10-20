"""Holding a runner host's root to its ceiling.

One row per host, the audit's (:mod:`fleet.core.runner_audit`): ``df -BG``
of ``/`` against the roster's ceiling, the idle baseline of a rebuilt host
carried in the row's id. The rows that say where a grown root went are the
cache rows (:mod:`fleet.core.runner_audit_cache`).
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
    """Audit lines that hold the host's root to the roster's ceiling.

    Args:
        spec: The host.

    Returns:
        The bash lines: ``df -BG`` of ``/``, its second line read as whole
        GB by the audit's ``gb_at``, and the ``check``. An answer that is
        not a size leaves the figure empty, which drifts with df's own
        words and exit, never passes.
    """
    disk = spec["base"]["disk"]
    ceiling = disk["ceiling_gb"]
    return [
        "probe df -BG --output=used /",
        "used=$(gb_at 2)",
        f'check \'{disk_check_id(spec)}\' "$(verdict_of within "$used" {ceiling})" '
        f'"the root uses ${{used:-an unreadable figure}} GB against a ceiling of {ceiling} GB; '
        f"an idle rebuilt host used {disk['baseline_gb']} GB on {disk['baseline_measured']}; "
        'df said: $PROBE_TEXT"',
    ]


__all__ = ["disk_check_id", "render_disk_check_lines"]
