"""Radar scan handling and resource reconciliation for world state.

Handles radar scan results, differential cache promotion, viewport scan
confirmation, and stale resource reconciliation within the radar envelope.

Container and mine state both live in their own registries
(``world.containers`` / ``world.mines``), populated by every wire signal
that carries them -- 0x5A viewport patches, 0x43 cache updates, 0x40
overlay updates, 0x4B mine placements, and these radar handlers. There
is no parallel store in ``world.terrain`` to reconcile against.
"""

from __future__ import annotations

from typing import TypeVar

from tankpit_bot import _test_hooks
from tankpit_bot.protocol import RadarContainerDict, RadarMineClearDict, RadarMineDict
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    ContainerStateDict,
    MineStateDict,
    WorldStateDict,
    add_mine_from_radar,
    coord_key,
    record_scanned_tiles,
    remove_mine,
    update_container_from_radar,
)
from tankpit_bot.state.scan_coverage import (
    free_radar_revealed_tiles,
    viewport_tiles,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


def _radar_revealed_tiles(ws: WorldService) -> list[tuple[int, int]]:
    """Return the exact tile set the current server radar revealed.

    Extra radar reveals every tile in the viewport. Free radar reveals
    a ``(2r+1)x(2r+1)`` block around the tank, where
    ``r = free_radar_radius(rank)``, intersected with the viewport
    (see :func:`tankpit_bot.physics.capacity.free_radar_radius`).
    When the tank position is unknown (self_state not yet observed),
    the server falls back to the extra-radar geometry -- mirror that.

    Args:
        ws: World service instance.

    Returns:
        Tiles the current radar revealed.
    """
    left, top, right, bottom = viewport_visible_bounds(ws.world_state["viewport"])
    self_state = ws.world_state["self_state"]
    if ws.current_radar_uses_extra() or self_state is None:
        return viewport_tiles(left, top, right, bottom)
    return free_radar_revealed_tiles(
        self_state["x"],
        self_state["y"],
        left,
        top,
        right,
        bottom,
        self_state["rank"],
    )


def update_world_state_from_radar(
    ws: WorldService,
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict],
    mine_clears: list[RadarMineClearDict],
) -> None:
    """Update world state with radar scan results.

    The 0x4F body is a delta sync of the scanned area (JS handler
    ``ch`` applies every entry as a per-tile write): container entries
    with ``volume == 0`` are authoritative removals, and mine-clear
    entries (overlay >= 8) drop any tracked mine at the tile.

    Args:
        ws: World service instance.
        containers: List of containers from radar.
        mines: List of mines from radar.
        mine_clears: Tiles the server declared mine-free.
    """
    ts = _test_hooks.get_current_time_ms()
    ws.pending_radar_empty_delta_ms = 0
    # mark_radar_scan_complete also answers any pending
    # container-desync latch -- every radar response shape counts.
    ws.mark_radar_scan_complete()
    ws.clear_failed_move_targets()
    reconcile_radar_viewport_resources(ws, containers, mines)
    viewport = ws.world_state["viewport"]
    ws.world_state = record_scanned_tiles(ws.world_state, _radar_revealed_tiles(ws), ts)
    if ws.current_radar_uses_extra():
        ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])

    for c in containers:
        if c["volume"] == 0:
            # Radar states empty fuel authoritatively -- the mutation
            # below removes any tracked belief, and the tombstone
            # keeps the fleet merge from re-importing a teammate's
            # older sighting of the drained tile
            # ([[fleet-coordination]] negative knowledge).
            ws.container_disproofs[f"{c['x']},{c['y']}"] = ts
        ws.world_state = update_container_from_radar(
            ws.world_state,
            c["x"],
            c["y"],
            c["volume"],
            ts,
            refresh_kind="radar_response",
        )
    for m in mines:
        ws.world_state = add_mine_from_radar(
            ws.world_state,
            m["x"],
            m["y"],
            m["team"],
            ts,
        )
    for mc in mine_clears:
        ws.world_state = remove_mine(ws.world_state, mc["x"], mc["y"], ts)


def _radar_envelope_containers(ws: WorldService) -> list[RadarContainerDict]:
    """Return the containers currently tracked inside the radar envelope.

    Used by the differential-refresh path: when the server signals
    "radar confirmed the cache is current", we replay the existing
    ``world.containers`` entries within radar bounds back through
    :func:`update_container_from_radar` to bump their refresh metadata.

    Args:
        ws: World service instance.

    Returns:
        Radar-shaped container records for every tracked container in
        the current radar envelope.
    """
    left, top, right, bottom = ws.radar_bounds()
    result: list[RadarContainerDict] = []
    for container in ws.world_state["containers"].values():
        x = container["x"]
        y = container["y"]
        if not (left <= x <= right and top <= y <= bottom):
            continue
        volume = container["volume"] if container["is_fuel"] else -1
        result.append(RadarContainerDict(x=x, y=y, volume=volume))
    return result


def update_world_state_from_radar_cache(ws: WorldService) -> None:
    """Promote a differential radar cache refresh to authoritative containers.

    The wire signaled "radar saw containers but isn't sending the full
    list -- the cache is current." With the single-store container
    architecture, the per-tile mutators (0x5A / 0x43 / pickup) keep
    ``world.containers`` continuously accurate, so this handler just
    bumps refresh metadata on the in-envelope entries.

    Args:
        ws: World service instance.
    """
    ts = _test_hooks.get_current_time_ms()
    envelope = _radar_envelope_containers(ws)
    ws.mark_radar_scan_complete()
    ws.clear_failed_move_targets()
    viewport = ws.world_state["viewport"]
    ws.world_state = record_scanned_tiles(ws.world_state, _radar_revealed_tiles(ws), ts)
    if ws.current_radar_uses_extra():
        ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])
    for container in envelope:
        ws.world_state = update_container_from_radar(
            ws.world_state,
            container["x"],
            container["y"],
            container["volume"],
            ts,
            refresh_kind="radar_cache_refresh",
        )
    emit_world(
        "Radar cache refresh: refreshed %d containers in current envelope",
        len(envelope),
    )


def update_world_state_from_radar_known_resources(ws: WorldService) -> None:
    """Confirm a zero-delta differential radar without discarding known resources.

    Preserves existing authoritative containers inside the radar bounds.

    Args:
        ws: World service instance.
    """
    ts = _test_hooks.get_current_time_ms()
    envelope = _radar_envelope_containers(ws)
    ws.mark_radar_scan_complete()
    ws.clear_failed_move_targets()
    viewport = ws.world_state["viewport"]
    ws.world_state = record_scanned_tiles(ws.world_state, _radar_revealed_tiles(ws), ts)
    if ws.current_radar_uses_extra():
        ws.clear_failed_scan_viewport(viewport["left"], viewport["top"])
    for rc in envelope:
        ws.world_state = update_container_from_radar(
            ws.world_state,
            rc["x"],
            rc["y"],
            rc["volume"],
            ts,
            refresh_kind="radar_known_resources",
        )
    emit_world(
        "Radar differential refresh: preserved %d known containers in viewport",
        len(envelope),
    )


_RadarEntityT = TypeVar("_RadarEntityT", ContainerStateDict, MineStateDict)


def _without_stale_radar_entries(
    entries: dict[str, _RadarEntityT],
    bounds: tuple[int, int, int, int],
    radar_keys: set[str],
) -> dict[str, _RadarEntityT] | None:
    """Return ``entries`` without stale radar-sourced ones, or ``None`` if unchanged.

    An entry is stale when its latest confirmation came from a radar,
    it sits inside the scan bounds, and the new radar response does not
    list it. Entries confirmed by any other source (0x5A viewport
    patches, 0x43 cache updates, mine placements) are never removed --
    the radar response says nothing about visible entities.

    Args:
        entries: Current registry keyed by ``"x,y"``.
        bounds: ``(left, top, right, bottom)`` scan bounds, inclusive.
        radar_keys: ``"x,y"`` keys the radar response listed.

    Returns:
        Pruned copy of the registry, or ``None`` when nothing is stale.
    """
    left, top, right, bottom = bounds
    pruned: dict[str, _RadarEntityT] | None = None
    for key, entry in entries.items():
        if entry["source"] != "radar":
            continue
        x = entry["x"]
        y = entry["y"]
        if left <= x <= right and top <= y <= bottom and key not in radar_keys:
            if pruned is None:
                pruned = dict(entries)
            del pruned[key]
    return pruned


def reconcile_radar_viewport_resources(
    ws: WorldService,
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict] | None,
) -> None:
    """Reconcile radar-sourced viewport resources against a fresh radar scan.

    A radar response lists only the HIDDEN entities the scan revealed —
    already-visible containers and mines are on screen and are NOT
    re-sent (live run 2026-07-01 20:20:10: the landing 0x5A registered
    7 visible containers, the scan-on-landing radar listed just 2
    hidden ones, and the old whole-envelope reconcile deleted all 7
    visible entries, including a 1000+ volume fuel container). Only
    entries whose latest confirmation came from a PREVIOUS radar are
    stale when a new radar omits them; the visible layer is owned by
    the 0x5A viewport patches and 0x43 cache updates and must be left
    alone.

    Args:
        ws: World service instance.
        containers: Containers returned by radar.
        mines: Mines returned by radar. ``None`` skips mine reconciliation.
    """
    bounds = ws.radar_bounds()
    radar_container_keys = {coord_key(item["x"], item["y"]) for item in containers}

    new_containers = _without_stale_radar_entries(
        ws.world_state["containers"],
        bounds,
        radar_container_keys,
    )
    new_mines: dict[str, MineStateDict] | None = None
    if mines is not None:
        radar_mine_keys = {coord_key(item["x"], item["y"]) for item in mines}
        new_mines = _without_stale_radar_entries(
            ws.world_state["mines"],
            bounds,
            radar_mine_keys,
        )

    if new_containers is None and new_mines is None:
        return

    ws.world_state = WorldStateDict(
        self_state=ws.world_state["self_state"],
        tanks=ws.world_state["tanks"],
        containers=ws.world_state["containers"] if new_containers is None else new_containers,
        mines=ws.world_state["mines"] if new_mines is None else new_mines,
        terrain=ws.world_state["terrain"],
        viewport=ws.world_state["viewport"],
        scanned_tiles=ws.world_state["scanned_tiles"],
        timestamp_ms=ws.world_state["timestamp_ms"],
    )


def handle_radar_ack(ws: WorldService, found: bool) -> None:
    """Process a RadarAck by promoting any pending differential refresh.

    Args:
        ws: World service instance.
        found: Whether the RadarAck reports resources exist.
    """
    if ws.consume_pending_radar_empty_delta():
        if found:
            if _radar_envelope_containers(ws):
                update_world_state_from_radar_cache(ws)
            else:
                update_world_state_from_radar_known_resources(ws)
        else:
            update_world_state_from_radar(ws, [], [], [])
    else:
        ws.mark_radar_scan_complete()


__all__ = [
    "handle_radar_ack",
    "reconcile_radar_viewport_resources",
    "update_world_state_from_radar",
    "update_world_state_from_radar_cache",
    "update_world_state_from_radar_known_resources",
]
