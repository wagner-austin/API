"""Building and writing this bot's fleet knowledge report.

The write half of the knowledge exchange ([[fleet-coordination]]):
once per tick the bot serializes its fresh beliefs — enemy sightings,
container atlas, held combat lock — and atomically replaces
``knowledge.json`` in its own run directory. Reading teammates'
reports lives in :mod:`tankpit_bot.fleetshare.merge`.
"""

from __future__ import annotations

from platform_core.json_utils import dump_json_str

from tankpit_bot import _test_hooks
from tankpit_bot.fleetshare.codecs import encode_fleet_report
from tankpit_bot.fleetshare.types import (
    FleetContainerRemovalDict,
    FleetContainerSightingDict,
    FleetEnemySightingDict,
    FleetReportDict,
    FleetRole,
    FleetScannedTileDict,
)
from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.scan_coverage import FORAGE_COVERAGE_TTL_MS
from tankpit_bot.state.types import has_known_position

FLEET_REPORT_FILENAME = "knowledge.json"
"""The report's file name inside each bot's run directory — a sibling
of ``hud.json``, riding the same channel the fleet page reads."""

CONTAINER_SIGHTING_TTL_MS = 60_000
"""Maximum age of a container belief worth publishing.

Stricter than the local larder memory (600 s) by design: the reporter
can re-verify its own stale belief with a cheap local radar, but a
receiver must TRAVEL to act on it -- and in a co-farmed room (two
fleet bots plus ~27 practice bots eating containers) a sighting much
older than a minute is more likely a phantom than a prize. The
2026-08-14 240 s run measured the cost of the unbounded share: 18
code-4 empty-container disproofs on arterial, most against aged
imports."""

ENEMY_SIGHTING_TTL_MS = 30_000
"""Maximum age of an enemy positional belief worth publishing.

Bounded by the pursuit-relevant horizon: a position older than this is
map-stale everywhere (the 0x4C atlas refreshes far faster), so
publishing it would only feed teammates noise their own map already
beats."""


def _enemy_rows(ws: WorldService, own_team: int, now_ms: int) -> list[FleetEnemySightingDict]:
    """Collect the fresh enemy sightings worth publishing.

    Args:
        ws: The session's world service.
        own_team: The reporter's team (allies are not sightings).
        now_ms: Freshness bound reference.

    Returns:
        Sightings within :data:`ENEMY_SIGHTING_TTL_MS`.
    """
    rows: list[FleetEnemySightingDict] = []
    for tank in ws.world_state["tanks"].values():
        if tank["team"] == own_team or tank["liveness"] != "alive":
            continue
        if not has_known_position(tank):
            continue
        if now_ms - tank["last_position_update_ms"] > ENEMY_SIGHTING_TTL_MS:
            continue
        rows.append(
            FleetEnemySightingDict(
                tank_id=tank["tank_id"],
                name=tank["name"],
                team=tank["team"],
                rank=tank["rank"],
                x=tank["x"],
                y=tank["y"],
                damage_state=tank["damage_state"],
                observed_ms=tank["last_position_update_ms"],
            )
        )
    return rows


def _container_rows(ws: WorldService, now_ms: int) -> list[FleetContainerSightingDict]:
    """Collect the container beliefs worth publishing.

    Args:
        ws: The session's world service.
        now_ms: Freshness bound reference.

    Returns:
        Unfailed beliefs within :data:`CONTAINER_SIGHTING_TTL_MS`.
    """
    rows: list[FleetContainerSightingDict] = []
    for container in ws.world_state["containers"].values():
        if container["failed_pickups"] > 0:
            # A locally disproven pickup is not knowledge worth
            # exporting — the mark is this bot's own negative verdict.
            continue
        if now_ms - container["timestamp_ms"] > CONTAINER_SIGHTING_TTL_MS:
            # Aged beliefs stay home: the reporter can re-verify them
            # with a cheap local radar; a receiver must travel.
            continue
        rows.append(
            FleetContainerSightingDict(
                x=container["x"],
                y=container["y"],
                is_fuel=container["is_fuel"],
                volume=container["volume"],
                observed_ms=container["timestamp_ms"],
            )
        )
    return rows


def _removal_rows(ws: WorldService, now_ms: int) -> list[FleetContainerRemovalDict]:
    """Collect the recent container removals worth publishing.

    The reporter's tombstone map IS its removal ledger — every local
    disproof (code-4, emptied pickup, unreachable, radar-stated
    empty) stamped a tile there. Only removals inside the container
    share horizon matter: a sighting older than
    :data:`CONTAINER_SIGHTING_TTL_MS` is never shared, so a removal
    older than that has nothing left to refute.

    Args:
        ws: The session's world service.
        now_ms: Freshness bound reference.

    Returns:
        Removals within :data:`CONTAINER_SIGHTING_TTL_MS`.
    """
    rows: list[FleetContainerRemovalDict] = []
    for key, removed_ms in ws.container_disproofs.items():
        if now_ms - removed_ms > CONTAINER_SIGHTING_TTL_MS:
            continue
        tile_x, tile_y = key.split(",")
        rows.append(FleetContainerRemovalDict(x=int(tile_x), y=int(tile_y), removed_ms=removed_ms))
    return rows


def _scanned_rows(ws: WorldService, now_ms: int) -> list[FleetScannedTileDict]:
    """Collect the live scan-coverage tiles worth publishing.

    Args:
        ws: The session's world service.
        now_ms: Freshness bound reference.

    Returns:
        Tiles within :data:`FORAGE_COVERAGE_TTL_MS`.
    """
    rows: list[FleetScannedTileDict] = []
    for key, scanned_ms in ws.world_state["scanned_tiles"].items():
        if now_ms - scanned_ms > FORAGE_COVERAGE_TTL_MS:
            # Expired coverage answers no sibling's "worth a radar?"
            # question -- the shared scan map carries live tiles only.
            continue
        tile_x, tile_y = key.split(",")
        rows.append(FleetScannedTileDict(x=int(tile_x), y=int(tile_y), observed_ms=scanned_ms))
    return rows


def _combat_consent_rows(ws: WorldService) -> list[int]:
    """Tank ids whose combat-consent evidence this session holds.

    The same two proofs :func:`~tankpit_bot.bot.ai.threat_primitives.
    human_combat_consented` reads locally — they chatted this session,
    or they struck us (a ``taken`` row in the damage book). Published
    so a sibling inherits the consent (operator ruling 2026-08-26:
    a human who engages one tank of our color has consented to
    fighting the color).

    Args:
        ws: The session's world service.

    Returns:
        Sorted consent-evidence tank ids.
    """
    ids = set(ws.chat_seen_tank_ids)
    ids.update(int(key) for key in ws.damage_book["taken"])
    return sorted(ids)


def build_fleet_report(
    ws: WorldService,
    *,
    instance: str,
    role: FleetRole,
    engaged_target_id: int,
    forage_goal_x: int,
    forage_goal_y: int,
    collect_claim_x: int,
    collect_claim_y: int,
    now_ms: int,
) -> FleetReportDict | None:
    """Assemble this bot's knowledge offer from its current beliefs.

    Args:
        ws: The session's world service.
        instance: This bot's instance name ("" for the sole-bot
            namespace).
        role: This bot's fleet role.
        engaged_target_id: The held combat lock (-1 for none).
        forage_goal_x: The latched forage-frontier goal X (-1 none).
        forage_goal_y: The latched forage-frontier goal Y (-1 none).
        collect_claim_x: The held collect-plan container X (-1 none).
        collect_claim_y: The held collect-plan container Y (-1 none).
        now_ms: Current wall-clock ms (stamped as ``written_ms`` and
            used for the sighting freshness bound).

    Returns:
        The report, or ``None`` before the session has an established
        self (no tank id yet means there is nothing attributable to
        offer).
    """
    self_state = ws.world_state["self_state"]
    if self_state is None or self_state["tank_id"] == 0:
        return None
    if ws.selected_room is None:
        # No room, no shareable coordinates -- pre-join ticks offer
        # nothing (same contract as the no-self return above).
        return None
    own_team = self_state["team"]
    return FleetReportDict(
        instance=instance,
        team=own_team,
        room=ws.selected_room,
        tank_id=self_state["tank_id"],
        role=role,
        x=self_state["x"],
        y=self_state["y"],
        engaged_target_id=engaged_target_id,
        forage_goal_x=forage_goal_x,
        forage_goal_y=forage_goal_y,
        collect_claim_x=collect_claim_x,
        collect_claim_y=collect_claim_y,
        combat_consent_ids=_combat_consent_rows(ws),
        written_ms=now_ms,
        enemies=_enemy_rows(ws, own_team, now_ms),
        containers=_container_rows(ws, now_ms),
        removed=_removal_rows(ws, now_ms),
        scanned=_scanned_rows(ws, now_ms),
    )


def write_fleet_report(report: FleetReportDict) -> None:
    """Atomically replace this bot's on-disk knowledge report.

    Args:
        report: The report to publish (its ``instance`` field names
            the run directory to write into).
    """
    path = bot_run_dir(report["instance"]) / FLEET_REPORT_FILENAME
    _test_hooks.replace_text(path, dump_json_str(encode_fleet_report(report)))


__all__ = [
    "CONTAINER_SIGHTING_TTL_MS",
    "ENEMY_SIGHTING_TTL_MS",
    "FLEET_REPORT_FILENAME",
    "build_fleet_report",
    "write_fleet_report",
]
