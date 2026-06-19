"""Ingest per-tick tank truth from the page-client registry.

The client tank registry (``activeGame.P.j``) carries verified per-tank
truth: ``u`` = damage tier (matched wire ``damage_state`` 5/5 on
enemies and 19/19 on self), ``h`` = team, and ``j``/``i`` = the drawn
viewport column/row. Coordinates map to world tiles as
``(viewport_left + col - 1, viewport_top + row - 1)`` -- verified
against 199/214 joinable self-position samples and 58/60 wire shot
targets across runs 20260611-004505/013801/035438.

``j``/``i`` are where the sprite is DRAWN; entries with the -8 draw
sentinel (roster duplicates, tanks never seen) are skipped. ``P``/``U``
mirror j/i on some samples and read -8 on others -- INCLUDING samples
of live tanks mid-firefight (purple-3 in run 20260611-004505 carried
P=-8 through 41 samples while our confirmed hits landed), so P is a
render-frame artifact, NOT a presence flag; a P-based filter shipped
briefly on 2026-06-11 and skipped live targets (run 110445: 59 skips,
0 kills). Stale leftover entries (tanks that died or drove away keep
their last drawn state for minutes) are NOT distinguishable by any
single captured field; the shot-response backstop in combat_strategy
(miss on a stationary target at shot range -> block) is the working
defense until wire-traffic presence is cracked from raw captures.

The wire goes silent on enemy positions between movement messages; the
registry re-anchors every present enemy once per tick, which is what
lets HUNT engage visible enemies from current positions instead of
stale map intel.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_combat import (
    clear_death_anchor,
    get_death_anchor,
)
from tankpit_bot.sniffer.world_state_tanks import (
    update_world_state_from_client_registry,
    update_world_state_from_tank_damage,
)
from tankpit_bot.state.types import ViewportStateDict, WorldStateDict

_REGISTRY_COLLECTION_KEY = "P.j"
_NOT_PRESENT_SENTINEL = -8
# world tile = viewport origin + live coord - 1 (the grid is 1-indexed
# against the viewport frame; the 0/17 ring is the radar's +1-tile
# reveal beyond the 16x16 viewport).
_RENDER_ORIGIN_OFFSET = -1


class RegistryTankDict(TypedDict):
    """One tank read from the client registry.

    Attributes:
        tank_id: Tank ID (shared with the wire ID space).
        name: Tank name.
        team: Team ID (registry ``h``; red=0, purple=1, blue=2, orange=3).
        damage_tier: Damage tier (registry ``u``; 0=full/unsynced,
            3=light, 2=medium, 1=critical).
        drawn_col: Drawn viewport column (registry ``j``; -8 = not drawn).
        drawn_row: Drawn viewport row (registry ``i``; -8 = not drawn).
    """

    tank_id: int
    name: str
    team: int
    damage_tier: int
    drawn_col: int
    drawn_row: int


def decode_registry_tank(
    item: dict[str, int | float | bool | str | None],
) -> RegistryTankDict | None:
    """Decode one registry collection item into a typed tank reading.

    Args:
        item: Raw captured registry entry (flat key/value map).

    Returns:
        Typed reading, or ``None`` when any required field is absent --
        the page capture caps fields per item, so a partial entry is an
        expected state of the channel, not corrupt input.

    Raises:
        ValueError: When a present field has the wrong type; the page
            client changed shape and silent skipping would hide it.
    """
    ints: dict[str, int] = {}
    for key in ("id", "u", "h", "j", "i"):
        raw = item.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(
                f"registry tank field {key!r} must be an int, got {type(raw).__name__}"
            )
        ints[key] = raw
    name = item.get("name")
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError(f"registry tank field 'name' must be a str, got {type(name).__name__}")
    return RegistryTankDict(
        tank_id=ints["id"],
        name=name,
        team=ints["h"],
        damage_tier=ints["u"],
        drawn_col=ints["j"],
        drawn_row=ints["i"],
    )


def _ingest_one_registry_entry(
    item: dict[str, int | float | bool | str | None],
    self_tank_id: int,
    viewport: ViewportStateDict,
) -> str:
    """Try to ingest one registry entry into world state.

    Args:
        item: Raw registry collection entry.
        self_tank_id: The bot's own tank ID (excluded).
        viewport: Current viewport for coordinate mapping.

    Returns:
        ``"corpse"`` if the entry is a death-tile sprite,
        ``"ingested"`` if successfully refined, or ``"skip"``
        if the entry was filtered.
    """
    tank = decode_registry_tank(item)
    if tank is None:
        return "skip"
    if tank["tank_id"] == self_tank_id:
        return "skip"
    if _NOT_PRESENT_SENTINEL in (tank["drawn_col"], tank["drawn_row"]):
        return "skip"
    world_x = viewport["left"] + tank["drawn_col"] + _RENDER_ORIGIN_OFFSET
    world_y = viewport["top"] + tank["drawn_row"] + _RENDER_ORIGIN_OFFSET
    ws = get_world_service()
    death_tile = get_death_anchor(ws, tank["tank_id"])
    if death_tile is not None:
        if (world_x, world_y) == death_tile:
            return "corpse"
        clear_death_anchor(ws, tank["tank_id"])
    refined = update_world_state_from_client_registry(
        ws,
        tank["tank_id"],
        tank["name"],
        tank["team"],
        world_x,
        world_y,
    )
    if not refined:
        return "skip"
    if tank["damage_tier"] != 0:
        update_world_state_from_tank_damage(
            ws,
            tank["tank_id"],
            tank["damage_tier"],
            refresh_wire_timestamp=False,
        )
    return "ingested"


def register_tank_truth_from_page_snapshot(
    snapshot: PageClientSnapshotDict,
    world: WorldStateDict,
) -> int:
    """Re-anchor every rendered wire-known tank from the registry.

    The wire vouches for presence; the registry only refines. A drawn
    entry the wire never announced is a stale afterimage -- the class
    that absorbed 52 wasted shots in run 20260611-103309 (the tank had
    died and respawned elsewhere ~10s later; only its sprite state
    lingered). Only tanks already present in world state (established
    by a wire message) are refined here.

    Position, team, and name are ingested for every DRAWN, wire-known,
    non-self tank (``j != -8``). The damage tier is ingested only when
    nonzero: 0 means both "full" and "unsynced", and stale registry
    entries park at 0 (run 20260611-003415: red-6 held u=0 through a
    fight the wire tracked), so a zero can never overwrite a wire-known
    tier.

    Corpse detection: a tank rendered at its recorded death tile (set
    by :func:`mark_tank_killed`) is a corpse sprite, not a live tank.
    The client registry keeps rendering corpses for minutes (run
    20260611-092159 re-ingested the dead purple-1 and spent three
    minutes shooting it). A tank observed at a NEW tile clears the
    death anchor permanently -- that is respawn evidence.

    Args:
        snapshot: Page-client snapshot captured this tick.
        world: Current world state providing self identity and viewport
            origin for the coordinate mapping.

    Returns:
        Number of tanks ingested this tick.
    """
    if snapshot["map_visible"] is True:
        return 0
    self_state = world["self_state"]
    if self_state is None:
        return 0
    items = snapshot["world_collections"].get(_REGISTRY_COLLECTION_KEY)
    if items is None:
        return 0
    viewport = world["viewport"]
    ingested = 0
    corpse_count = 0
    for item in items:
        result = _ingest_one_registry_entry(
            item,
            self_state["tank_id"],
            viewport,
        )
        if result == "corpse":
            corpse_count += 1
        elif result == "ingested":
            ingested += 1
    if ingested > 0 or corpse_count > 0:
        emit_diagnostic(
            diagnostic_kind="registry_truth_ingested",
            tank_count=ingested,
            corpse_count=corpse_count,
        )
    return ingested


def reset_registry_truth() -> None:
    """Reset module state for test isolation.

    No module-level state is held here -- death anchors live in
    :mod:`tankpit_bot.sniffer.world_state` and are cleared by
    ``reset_world_state``. This function exists so the diagnostics
    conftest can call it uniformly without coupling to the reset
    protocol.
    """


__all__ = [
    "RegistryTankDict",
    "decode_registry_tank",
    "register_tank_truth_from_page_snapshot",
    "reset_registry_truth",
]
