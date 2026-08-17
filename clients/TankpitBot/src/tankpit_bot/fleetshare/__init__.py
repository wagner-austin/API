"""Fleet knowledge sharing: same-team bots exchanging beliefs on disk.

See [[fleet-coordination]] and the module docstrings of
:mod:`tankpit_bot.fleetshare.report` (the write half) and
:mod:`tankpit_bot.fleetshare.merge` (the read half).
"""

from tankpit_bot.fleetshare.merge import (
    FLEET_REPORT_TTL_MS,
    FleetMergeSummaryDict,
    merge_fleet_reports,
    read_team_reports,
)
from tankpit_bot.fleetshare.report import (
    ENEMY_SIGHTING_TTL_MS,
    FLEET_REPORT_FILENAME,
    build_fleet_report,
    write_fleet_report,
)
from tankpit_bot.fleetshare.role import resolve_fleet_role
from tankpit_bot.fleetshare.types import (
    FLEET_ROLES,
    FleetContainerSightingDict,
    FleetEnemySightingDict,
    FleetReportDict,
    FleetRole,
)

__all__ = [
    "ENEMY_SIGHTING_TTL_MS",
    "FLEET_REPORT_FILENAME",
    "FLEET_REPORT_TTL_MS",
    "FLEET_ROLES",
    "FleetContainerSightingDict",
    "FleetEnemySightingDict",
    "FleetMergeSummaryDict",
    "FleetReportDict",
    "FleetRole",
    "build_fleet_report",
    "merge_fleet_reports",
    "read_team_reports",
    "resolve_fleet_role",
    "write_fleet_report",
]
