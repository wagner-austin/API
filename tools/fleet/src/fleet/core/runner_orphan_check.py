"""The audit row that fails a runner holding what a finished job left behind.

MCPs board task 53528106, A2 and A4. A cancelled GitHub Actions job kills
its worker, and what the job daemonized lives on: on 2026-10-02 lavender's
runners held 158 such processes and 17 GB with no job running, and nothing
in the audit could see it. This row asks each runner how many of its
processes lie outside its service's own tree, are older than the host's
``job_timeout_minutes``, and live while no ``Runner.Worker`` runs: any such
process outlived every job that could own it. A Worker older than that
timeout runs no job GitHub still holds, so it counts itself rather than
shielding the rest (on 2026-10-02 at 10:09Z one such Worker kept 12.7 GB in
an API runner for 3 h 22 min).

The row asks through the reaper's own ``--audit`` mode
(:mod:`fleet.core.runner_reaper_render`), so the audit measures exactly what
the reaper acts on, and a host without the reaper drifts on that row as well
as on its timer's.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core.runner_reaper_render import REAPER_PATH
from fleet.core.script_values import scriptable

#: Why the row matters, printed beside a drift line.
ORPHAN_REASON = (
    "a finished job's processes hold the runner's memory until something kills them; on "
    "2026-10-02 they held 17 GB of lavender's CI budget and GitHub read every runner offline"
)


def orphan_check_id(install: RunnerInstall) -> str:
    """The row's id for one install, unique across repositories.

    Args:
        install: The install.

    Returns:
        E.g. ``orphans:wagner-austin/MCPs:lavender``.
    """
    return f"orphans:{install['repo']}:{install['runner_name']}"


def render_orphan_check_lines(spec: HostRunnerSpec, install: RunnerInstall) -> list[str]:
    """Audit lines for one install's row.

    Args:
        spec: The host, for its job timeout.
        install: The install.

    Returns:
        The bash lines. The row passes only when the reaper exits 0 and
        counts nothing.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    seconds = spec["job_timeout_minutes"] * 60
    minutes = spec["job_timeout_minutes"]
    service = scriptable(install["service"], label="service")
    return [
        f"probe {REAPER_PATH} --audit {seconds} '{service}'",
        f"check '{orphan_check_id(install)}' \"$(verdict_of answered 0)\" "
        f'"processes older than {minutes} minutes that a finished job left in {service}, '
        'or under a Worker past that job timeout; the reaper counted: $PROBE_TEXT"',
    ]


__all__ = ["ORPHAN_REASON", "orphan_check_id", "render_orphan_check_lines"]
