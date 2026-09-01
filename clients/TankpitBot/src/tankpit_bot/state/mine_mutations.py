"""Mine world-state mutations.

Split from :mod:`tankpit_bot.state.container_mutations` (2026-08-14,
file-size ceiling): mines and containers share the coordinate-keyed
world-state shape but nothing else -- placement provenance, radar
refresh, and detonation removal live here.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.facts.provenance import make_provenance
from tankpit_bot.state.types import (
    WorldStateDict,
    coord_key,
    make_mine_state,
)
from tankpit_bot.types.constants import EntitySource

log = get_logger(__name__)


def add_mine(
    state: WorldStateDict,
    x: int,
    y: int,
    mine_type: int,
    tank_id: int,
    team: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Add mine from MinePlacement message.

    Args:
        state: Current world state.
        x: Mine X coordinate.
        y: Mine Y coordinate.
        mine_type: Type of mine.
        tank_id: ID of placing tank.
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with mine added.
    """
    new_mine = make_mine_state(
        x=x,
        y=y,
        mine_type=mine_type,
        tank_id=tank_id,
        team=team,
        source="viewport",
        timestamp_ms=timestamp_ms,
        provenance=make_provenance("wire_0x4B_mine_placement", []),
    )

    key = coord_key(x, y)
    new_mines = dict(state["mines"])
    new_mines[key] = new_mine

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=new_mines,
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def add_mine_from_radar(
    state: WorldStateDict,
    x: int,
    y: int,
    team: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Add or refresh a mine discovered via radar scan (0x4F-tunneled).

    Radar mine entries are 3 bytes wide -- ``x, y, team`` -- and carry
    NEITHER ``mine_type`` NOR the placer's ``tank_id``. Those fields are
    only knowable via wire MinePlacement (V.K / 0x4B, per tpclient.js
    handler ``Dg.h``). When radar refreshes a tile where a wire-placed
    mine already lives, this mutator must preserve the wire-known
    ``mine_type`` and ``tank_id`` -- they came from a richer source and
    radar cannot reproduce them.

    Merge rules:
      * New tile (no existing mine): seed with ``mine_type=0``,
        ``tank_id=-1``, ``source="radar"``.
      * Existing wire-sourced mine: preserve ``mine_type`` and
        ``tank_id``, keep ``source="viewport"`` (still wire-richer),
        advance ``timestamp_ms``, update ``team`` to the radar value
        (the wire team is authoritative on placement and the radar
        team is authoritative on refresh -- a placement followed by a
        radar sighting at the same tile is the same mine, and team
        cannot legally change for an undetonated mine, so this
        difference indicates the wire team field went stale and should
        be re-synced).
      * Existing radar-sourced mine: refresh as before with
        ``source="radar"``.

    Args:
        state: Current world state.
        x: Mine X coordinate.
        y: Mine Y coordinate.
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        timestamp_ms: Message timestamp.

    Returns:
        New ``WorldStateDict`` with the mine added or refreshed.
    """
    key = coord_key(x, y)
    existing = state["mines"].get(key)
    if existing is None:
        merged_mine_type = 0
        merged_tank_id = -1
        merged_source: EntitySource = "radar"
    elif existing["source"] == "viewport":
        merged_mine_type = existing["mine_type"]
        merged_tank_id = existing["tank_id"]
        merged_source = "viewport"
    else:
        merged_mine_type = existing["mine_type"]
        merged_tank_id = existing["tank_id"]
        merged_source = "radar"

    new_mine = make_mine_state(
        x=x,
        y=y,
        mine_type=merged_mine_type,
        tank_id=merged_tank_id,
        team=team,
        source=merged_source,
        timestamp_ms=timestamp_ms,
    )

    new_mines = dict(state["mines"])
    new_mines[key] = new_mine

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=new_mines,
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def merge_mine_sighting(
    state: WorldStateDict,
    x: int,
    y: int,
    mine_type: int,
    tank_id: int,
    team: int,
    observed_ms: int,
) -> WorldStateDict:
    """Merge one teammate-reported hostile mine into local belief.

    The fleet knowledge law ([[fleet-coordination]]), applied to the
    mine layer (operator order 2026-09-01: "have a mine aware layer
    between the bots"): remote knowledge only ADDS or REFRESHES — it
    never removes a local belief and never outranks one of equal or
    fresher age (own wire is the higher trust tier). The merged
    belief carries ``source="world_state"`` (the non-own-sensing tier,
    the same stamp fleet-merged containers use so own-viewport miners
    exclude imports) with ``fleet_report`` provenance, and contact
    disproofs (0x45 in view, the exact-landing receipt) treat it
    exactly like any other remembered mine.

    Args:
        state: Current world state.
        x: Mine X.
        y: Mine Y.
        mine_type: Wire mine-type byte from the reporter.
        tank_id: Layer's tank id as the reporter recorded it.
        team: The mine's team.
        observed_ms: The reporter's belief timestamp for the mine.

    Returns:
        New ``WorldStateDict`` with the sighting merged, or the input
        state unchanged when local belief is at least as fresh.
    """
    key = coord_key(x, y)
    existing = state["mines"].get(key)
    if existing is not None and existing["timestamp_ms"] >= observed_ms:
        return state
    new_mine = make_mine_state(
        x=x,
        y=y,
        mine_type=mine_type,
        tank_id=tank_id,
        team=team,
        source="world_state",
        timestamp_ms=observed_ms,
        provenance=make_provenance("fleet_report", []),
    )
    new_mines = dict(state["mines"])
    new_mines[key] = new_mine
    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=new_mines,
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=state["timestamp_ms"],
    )


def remove_mine(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Remove mine after detonation.

    Args:
        state: Current world state.
        x: Mine X coordinate.
        y: Mine Y coordinate.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with mine removed.
    """
    key = coord_key(x, y)
    if key not in state["mines"]:
        return state

    new_mines = dict(state["mines"])
    del new_mines[key]

    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=new_mines,
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def apply_tile_overlay_update(
    state: WorldStateDict,
    x: int,
    y: int,
    overlay_value: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Reconcile ``world.mines`` from one tile's wire-decoded overlay byte.

    The 0x5A ``ViewportUpdate`` and 0x40 ``OverlayUpdate`` messages both
    carry the same per-tile mine-layer byte:

    * ``overlay_value`` in ``0..7`` -> mine present; team encoded in the
      low 2 bits (``team = overlay_value & 3``).
    * ``overlay_value >= 8`` (the decoder maps 8..15 to ``255``) -> tile
      has no mine: drop any tracked mine.

    The 0x5A / 0x40 path does NOT carry the placer's ``tank_id`` or the
    mine ``mine_type`` -- only 0x4B ``MinePlacement`` provides those. If
    a wire-rich mine already lives at the tile, preserve those fields
    while refreshing ``team`` and ``timestamp_ms``; otherwise seed with
    ``mine_type=0``, ``tank_id=-1``. Mirrors the merge policy in
    :func:`add_mine_from_radar`.

    Args:
        state: Current world state.
        x: Tile X coordinate.
        y: Tile Y coordinate.
        overlay_value: Decoded overlay byte (``0..7`` = mine, else clear).
        timestamp_ms: Message timestamp.

    Returns:
        New ``WorldStateDict`` with ``world.mines`` reconciled for this tile.
    """
    if not 0 <= overlay_value <= 7:
        return remove_mine(state, x, y, timestamp_ms)

    team = overlay_value & 3
    key = coord_key(x, y)
    existing = state["mines"].get(key)
    if existing is None:
        merged_mine_type = 0
        merged_tank_id = -1
    else:
        merged_mine_type = existing["mine_type"]
        merged_tank_id = existing["tank_id"]

    new_mine = make_mine_state(
        x=x,
        y=y,
        mine_type=merged_mine_type,
        tank_id=merged_tank_id,
        team=team,
        source="viewport",
        timestamp_ms=timestamp_ms,
    )
    new_mines = dict(state["mines"])
    new_mines[key] = new_mine
    return WorldStateDict(
        self_state=state["self_state"],
        tanks=state["tanks"],
        containers=state["containers"],
        mines=new_mines,
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


__all__ = [
    "add_mine",
    "add_mine_from_radar",
    "apply_tile_overlay_update",
    "merge_mine_sighting",
    "remove_mine",
]
