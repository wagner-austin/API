"""Reading teammates' fleet reports and merging them into local belief.

The read half of the knowledge exchange ([[fleet-coordination]]). Each
tick the bot lists its siblings' ``knowledge.json`` files, keeps the
fresh SAME-TEAM ones, and merges their content through the existing
observation pathways — tank sightings via ``apply_tank_observation``
with the ``fleet_report`` fact source (never advancing the viewport
gate, so a merged sighting can never provoke a phantom shot), and
container sightings via ``merge_container_sighting`` (add/refresh
only; local wire is the higher trust tier). Teammates' held combat
locks land in ``ws.fleet_engaged_target_ids`` — the focus-fire signal
threat ranking prefers.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONTypeError, load_json_str, require_int
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.fleetshare.codecs import decode_fleet_report
from tankpit_bot.fleetshare.report import FLEET_REPORT_FILENAME
from tankpit_bot.fleetshare.types import FleetReportDict
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import merge_container_sighting, remove_container
from tankpit_bot.state.scan_coverage import merge_scanned_coverage
from tankpit_bot.state.tank_mutations import apply_tank_observation
from tankpit_bot.state.types import make_tank_observation

FLEET_REPORT_TTL_MS = 10_000
"""Maximum age of a sibling report worth merging.

Sized to a handful of server windows: a live bot rewrites its report
every tick (~2 s), so anything older than this belongs to a stopped or
wedged process and must not steer the living."""


class FleetMergeSummaryDict(TypedDict):
    """What one merge pass actually did, for the tick log.

    Attributes:
        reports: Fresh same-team reports merged.
        enemies: Enemy sightings applied to the registry.
        containers: Container sightings merged into the atlas.
        removed: Local beliefs dropped by teammates' removals.
        scanned: Coverage tiles advanced in the shared scan map.
    """

    reports: int
    enemies: int
    containers: int
    removed: int
    scanned: int


_REPORT_READ_ATTEMPTS = 3
"""Bounded re-open budget for a sibling report hit mid-replace.

Windows refuses a concurrent open with ``PermissionError`` for the
microseconds ``os.replace`` swaps the report file — the first
two-fighter session at full tick rate hit it (arterial tick 264,
2026-08-26 03:01:06, ``[Errno 13]`` on artax's knowledge.json; POSIX
rename never refuses the open, which is why 2,500+ single-bot and
short pair sessions never saw it). The writer's swap completes within
the failed open itself, so an immediate retry lands. A persistent
denial across the whole budget is no longer the race — it is a real
permission fault, and it still raises."""


def _read_report_text(path: Path) -> str:
    """Read a sibling report, absorbing the Windows replace window.

    Args:
        path: The report file.

    Returns:
        The file text.

    Raises:
        PermissionError: When every attempt in the budget is denied —
            a real permission fault, never the transient swap race.
    """
    attempt = 0
    while True:
        try:
            return _test_hooks.read_text(path)
        except PermissionError:
            attempt += 1
            if attempt >= _REPORT_READ_ATTEMPTS:
                raise


def read_team_reports(
    own_instance: str,
    own_team: int,
    now_ms: int,
) -> list[FleetReportDict]:
    """Read the fresh same-team reports of every sibling bot.

    Args:
        own_instance: This bot's instance name ("" for the sole-bot
            namespace) — its own report file is skipped.
        own_team: This bot's team; other teams' reports are not merged
            (knowledge sharing is an alliance).
        now_ms: Current wall-clock ms for the freshness bound.

    Returns:
        Fresh same-team sibling reports, in path order.

    Raises:
        JSONTypeError: If a sibling report fails validation — writes
            are atomic, so a malformed file is a genuine bug, not a
            torn read.
    """
    root = bot_run_dir("")
    own_path = bot_run_dir(own_instance) / FLEET_REPORT_FILENAME
    paths = list(_test_hooks.glob_paths(root, f"*/{FLEET_REPORT_FILENAME}"))
    sole_path = root / FLEET_REPORT_FILENAME
    if _test_hooks.path_exists(sole_path):
        paths.append(sole_path)
    reports: list[FleetReportDict] = []
    for path in paths:
        if path == own_path:
            continue
        parsed = load_json_str(_read_report_text(path))
        if not isinstance(parsed, dict):
            raise JSONTypeError(f"fleet report must be an object, got {type(parsed).__name__}")
        # Freshness gates BEFORE full decode, two-sided: every fleet
        # bot shares one machine clock, so a report stamped in the
        # FUTURE is clock-domain garbage (a leftover file from a
        # different clock regime -- the sim seam's stepped clock vs
        # live wall-clock artifacts), and a stale file may predate
        # the current report schema entirely (a dead build's last
        # write). Neither is ever consumed, so neither is validated
        # beyond the one key the gate reads -- while a FRESH report
        # that fails validation still raises, because a live
        # mixed-schema fleet is a genuine bug.
        age_ms = now_ms - require_int(parsed, "written_ms")
        if not 0 <= age_ms <= FLEET_REPORT_TTL_MS:
            continue
        report = decode_fleet_report(parsed)
        if report["team"] != own_team:
            continue
        reports.append(report)
    return reports


def _merge_enemy_sightings(
    ws: WorldService,
    report: FleetReportDict,
    own_tank_id: int,
    own_team: int,
) -> int:
    """Apply one report's enemy sightings to the local registry.

    A sighting applies only when FRESHER than the local entry (a
    teammate's old news never regresses local truth), and never for
    this bot's own tank or any same-team tank.

    Args:
        ws: The session's world service.
        report: One fresh same-team report.
        own_tank_id: This bot's tank id.
        own_team: This bot's team.

    Returns:
        How many sightings landed.
    """
    merged = 0
    for sighting in report["enemies"]:
        if sighting["tank_id"] == own_tank_id or sighting["team"] == own_team:
            continue
        existing = ws.world_state["tanks"].get(str(sighting["tank_id"]))
        if existing is not None and existing["timestamp_ms"] >= sighting["observed_ms"]:
            continue
        ws.world_state = apply_tank_observation(
            ws.world_state,
            make_tank_observation(
                tank_id=sighting["tank_id"],
                timestamp_ms=sighting["observed_ms"],
                is_wire_sourced=False,
                storage_source="world_state",
                fact_source="fleet_report",
                position_is_authoritative=True,
                position=(sighting["x"], sighting["y"]),
                team=sighting["team"],
                rank=sighting["rank"],
                damage_state=sighting["damage_state"],
                name=sighting["name"],
            ),
        )
        merged += 1
    return merged


def _merge_container_knowledge(ws: WorldService, report: FleetReportDict) -> tuple[int, int]:
    """Apply one report's container sightings and removals.

    Sightings honour the tombstone law (a locally disproven tile only
    re-admits a sighting OBSERVED AFTER the disproof) and the
    freshness law inside :func:`~tankpit_bot.state.merge_container_sighting`.
    Removals are the negative half (fleet ruling 2026-08-14: "does it
    update the equipment for everyone when one of them takes the
    discovered equipment?"): a teammate's removal drops any local
    belief OBSERVED BEFORE it, and the tombstone is inherited so
    third parties' stale sightings stay out here too. A local belief
    FRESHER than the removal survives -- the tile may have respawned
    since the teammate's disproof.

    Args:
        ws: The session's world service.
        report: One fresh same-team report.

    Returns:
        ``(sightings_merged, beliefs_removed)``.
    """
    sightings_merged = 0
    beliefs_removed = 0
    for container in report["containers"]:
        disproof_key = f"{container['x']},{container['y']}"
        if ws.container_disproofs.get(disproof_key, 0) >= container["observed_ms"]:
            continue
        merged = merge_container_sighting(
            ws.world_state,
            container["x"],
            container["y"],
            container["is_fuel"],
            container["volume"],
            container["observed_ms"],
        )
        if merged is not ws.world_state:
            sightings_merged += 1
            ws.world_state = merged
    for removal in report["removed"]:
        removal_key = f"{removal['x']},{removal['y']}"
        recorded = ws.container_disproofs.get(removal_key, 0)
        if removal["removed_ms"] > recorded:
            ws.container_disproofs[removal_key] = removal["removed_ms"]
        existing_belief = ws.world_state["containers"].get(removal_key)
        if existing_belief is not None and existing_belief["timestamp_ms"] <= removal["removed_ms"]:
            ws.world_state = remove_container(
                ws.world_state,
                removal["x"],
                removal["y"],
                ws.world_state["timestamp_ms"],
            )
            beliefs_removed += 1
    return (sightings_merged, beliefs_removed)


def merge_fleet_reports(
    ws: WorldService,
    reports: list[FleetReportDict],
    *,
    own_tank_id: int,
    own_team: int,
) -> FleetMergeSummaryDict:
    """Merge teammates' reports into this session's beliefs.

    Enemy sightings, container sightings, container removals, and
    scan coverage each apply through their concern's own applier;
    ``ws.fleet_engaged_target_ids`` is REPLACED with the engaged
    locks of exactly these reports, so a teammate that disengages or
    goes silent stops steering acquisition within one exchange.

    Args:
        ws: The session's world service.
        reports: Fresh same-team reports from :func:`read_team_reports`.
        own_tank_id: This bot's tank id (never merged as an enemy).
        own_team: This bot's team (same-team sightings are skipped).

    Returns:
        Summary of what the pass merged.
    """
    enemies_merged = 0
    containers_merged = 0
    removed_merged = 0
    scanned_merged = 0
    engaged: dict[int, int] = {}
    consented: set[int] = set()
    for report in reports:
        consented.update(report["combat_consent_ids"])
        target_id = report["engaged_target_id"]
        if target_id not in (-1, own_tank_id):
            recorded = engaged.get(target_id, 0)
            if report["written_ms"] > recorded:
                engaged[target_id] = report["written_ms"]
        enemies_merged += _merge_enemy_sightings(ws, report, own_tank_id, own_team)
        sightings, removals = _merge_container_knowledge(ws, report)
        containers_merged += sightings
        removed_merged += removals
        ws.world_state, advanced = merge_scanned_coverage(
            ws.world_state,
            [(tile["x"], tile["y"], tile["observed_ms"]) for tile in report["scanned"]],
        )
        scanned_merged += advanced
    ws.fleet_engaged_target_ids = engaged
    # Wholesale replacement, the engaged-ids pattern: a departed
    # sibling's shared consent ages out with its report TTL; by then
    # the human has long since granted the survivor organic consent
    # (or the fight is over).
    ws.fleet_consented_tank_ids = consented
    return FleetMergeSummaryDict(
        reports=len(reports),
        enemies=enemies_merged,
        containers=containers_merged,
        removed=removed_merged,
        scanned=scanned_merged,
    )


__all__ = [
    "FLEET_REPORT_TTL_MS",
    "FleetMergeSummaryDict",
    "merge_fleet_reports",
    "read_team_reports",
]
