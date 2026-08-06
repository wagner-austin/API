"""Tank state TypedDict + factory + encode/decode.

Phase 1c of the self-observing architecture: the tank carries the full
fact metadata flat -- ``source`` plus the four freshness timestamps
(pre-existing) and the ``confidence`` / ``provenance`` fields. The
Fact[T] projection lives in :mod:`tankpit_bot.facts.tank_facts`.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.facts.provenance import (
    ProvenanceChainDict,
    decode_provenance,
    encode_provenance,
    make_provenance,
)
from tankpit_bot.facts.source import FactSource
from tankpit_bot.state.types.constants import (
    EntitySource,
    TankLiveness,
    require_entity_source,
    require_tank_liveness,
)

# The viewport-presence horizon for ``last_viewport_observation_ms``:
# how long a viewport-sourced observation still proves the tank is in
# the bot's local sensing window. Lives here, beside the field it
# gates, because two independent consumers ask the same question and
# must never drift apart: HUNT acquisition
# (:func:`tankpit_bot.bot.ai.threats.analyze_threats` -- may I engage
# this tank) and tile occupancy
# (:func:`tankpit_bot.state.occupancy.occupied_tank_keys` -- does this
# tank's body block a walk route).
#
# Live-run 2026-06-21 tracking probe captured the cost of skipping the
# gate: open_map's 0x4C MapData refreshed every-tank timestamps, then
# global TankStatusSync kept them fresh, so analyze_threats returned 27
# tanks while the JS client's viewport registry held only 1. Set to
# 5 s -- viewport-bound updates (0x47 Movement, 0x3D MovementResponse,
# 0x28 TankEntry) arrive every 1-3 s for tanks the bot can actually
# see, so 5 s tolerates short cadence gaps but rejects the 5-6 s global
# broadcast cycle that wire-presence alone cannot.
VIEWPORT_PRESENCE_TTL_MS = 5000


def has_known_position(tank: TankStateDict) -> bool:
    """Return whether this tank's ``(x, y)`` was ever actually observed.

    The registry upserts tanks from every message kind, and the login
    choreography makes the position-less kinds FIRST: the server opens
    every session with a full-roster 0x21 TankInfo dump (name + team,
    no coordinates), and positions only arrive with the first
    position-bearing sync — measured 2026-08-04 across three captures
    (113 tanks, every one 0x21-first, 9-46 s to first position). Until
    then the entry sits at the construction default ``(0, 0)`` — a
    coordinate that is also a legal tile. Any consumer that reads
    ``(x, y)`` without asking this question aims at, walks around, or
    walls off the map corner.

    Two conditions, either sufficient:

    * The coordinates differ from the ``(0, 0)`` construction default.
      This is how radar EnemyDetect and DOM-registry refinements
      qualify: they write real (tile-coarse) coordinates but
      deliberately do not advance ``last_position_update_ms`` (that
      field is the kill-shot gate and must stay wire-authoritative).
    * ``last_position_update_ms`` is nonzero — an authoritative
      position message has stated the coordinates, which also covers
      the pathological tank standing exactly on (0, 0).

    This predicate is the ONLY place the ``(0, 0)`` default may be
    compared against; the guard (``scripts/state_sentinel_rules.py``)
    bans the inline idiom everywhere else.

    Args:
        tank: Tank registry entry to test.

    Returns:
        True when ``(tank["x"], tank["y"])`` reflects an observation
        rather than the constructor default.
    """
    if tank["x"] != 0 or tank["y"] != 0:
        return True
    return tank["last_position_update_ms"] > 0


def has_real_coordinates(tank: TankStateDict) -> bool:
    """Return whether ``(x, y)`` holds actual coordinates, not the default.

    The strict-coordinates sibling of :func:`has_known_position`: no
    freshness escape hatch. The login roster's 0x28/0x21 entries carry
    ``(0, 0)`` with an advancing freshness stamp, so
    ``has_known_position`` (correctly, for occupancy) counts them
    known — but the MAP-POSITION DEFER
    (``state/mutations.py::apply_tank_observation``) must not protect
    that construction default from the 0x4C snapshot's real fix. The
    pathological tank standing exactly on the map corner loses this
    coin toss, exactly as it does for radar refinements.

    This module remains the ONLY home of the ``(0, 0)`` comparison
    (guard: ``scripts/state_sentinel_rules.py``).

    Args:
        tank: Tank registry entry to test.

    Returns:
        True when the stored coordinates differ from the ``(0, 0)``
        construction default.
    """
    return tank["x"] != 0 or tank["y"] != 0


_DEFAULT_FACT_SOURCE_BY_ENTITY_SOURCE: dict[EntitySource, FactSource] = {
    "viewport": "wire_0x28_tank_entry",
    "radar": "wire_0x48_enemy_detect",
    "world_state": "wire_0x4C_map_data",
}


def tank_default_fact_source(source: EntitySource) -> FactSource:
    """Return a synthetic default fact source for a coarse entity source.

    Direct ``make_tank_state`` construction (tests, fixtures) has no
    wire message behind it; this maps the coarse label to that label's
    canonical channel (viewport entry, radar enemy-detect, map data).
    The observation pipeline always overrides with the true message
    kind (``TankObservation.fact_source``).

    Args:
        source: Coarse observed source.

    Returns:
        Canonical fact source for that coarse label.
    """
    return _DEFAULT_FACT_SOURCE_BY_ENTITY_SOURCE[source]


class TankStateDict(TypedDict):
    """State of a single tank in the game world.

    Four independent freshness timestamps lock the freshness model:

    * ``timestamp_ms`` advances on ANY observation source (wire OR map).
      Used to keep a tank in the registry as a HUNT acquisition
      candidate even when only the map snapshot has confirmed it.

    * ``last_wire_seen_ms`` advances only on WIRE-SOURCED observations
      (viewport, radar, movement response, enemy detection,
      TankStatusSync). Map snapshot updates deliberately do NOT advance
      it — a tank truly present talks on the wire; a departed
      afterimage goes silent on the wire while the map keeps re-listing
      it.

    * ``last_position_update_ms`` advances ONLY when an observation
      carries a fresh ``(x, y)`` value. Damage-only wire messages
      (TankStatusSync, TankStatusShort) refresh ``last_wire_seen_ms``
      but NOT this field. This is the kill-shot gate — only fire at a
      tank whose position is structurally proven recent, never at a
      stale registry entry being kept alive by status-only broadcasts.

    * ``last_viewport_observation_ms`` advances ONLY when an observation
      carries ``storage_source == "viewport"`` -- proving the tank was
      in the bot's local sensing window when the wire arrived.
      MapData snapshots and global TankStatusSync broadcasts do NOT
      advance it (they fire for tanks anywhere on the map). This is
      the HUNT acquisition gate: ``analyze_threats`` filters on this
      timestamp so only tanks the bot can actually see are eligible
      to engage. Live-run 2026-06-21 tracking probe: 26 of 27 tanks
      passed every other gate (timestamp_ms, last_wire_seen_ms,
      last_position_update_ms) while the JS client's tank registry
      had none of them in view -- 0x4C MapData refreshes everyone's
      position-and-wire timestamps; 0x2E TankStatusSync broadcasts
      every ~5 s for every alive tank globally. Without this gate
      the bot's threat list is the global roster, not the visible
      one.

    The three-timestamp model exists because the broadcast cadences
    differ by message kind. 0x2E TankStatusSync broadcasts globally
    every ~2 s for every active tank regardless of viewport, so a
    single "any wire activity" timestamp would never expire and the
    bot would keep firing at stale registry positions. Position-bearing
    messages (0x3D MovementResponse, 0x47 Movement, 0x28 TankEntry,
    container TankUpdate*) refresh on a slower viewport-bound cadence.

    Attributes:
        tank_id: Unique identifier for this tank.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0=red, 1=purple, 2=blue, 3=orange).
        rank: Military rank (0 recruit .. 8 general).
        damage_state: Fuel-quartile health tier (0=near death ..
            3=full; corpus-fitted 2026-07-23, [[deactivation-format]]).
        direction: Sprite direction byte. Low nibble (0-15) = facing
            heading, high nibble carries state flags. Bit 5 (value 32)
            is the DEAD flag -- the game client sets direction to 32 or
            33 on deactivation (tpclient.js ``Pg.prototype.h``). Check
            ``direction >= 32`` to detect dead/corpse tanks. Verified
            across 42 corpse transitions in capture data (2026-06-18).
        name: Player name.
        is_bot: Whether this is a bot player.
        is_self: Whether this is the player's own tank.
        source: Which observed source most recently confirmed this tank.
        timestamp_ms: Wall-clock ms of the most recent observation by
            ANY source (wire OR map). Acquisition gate.
        last_wire_seen_ms: Wall-clock ms of the most recent
            wire-sourced observation. Wire-presence gate.
        last_position_update_ms: Wall-clock ms of the most recent
            wire-sourced observation that carried a fresh ``(x, y)``.
            Kill-shot gate.
        last_viewport_observation_ms: Wall-clock ms of the most recent
            observation whose ``storage_source`` was ``"viewport"`` --
            i.e., proof the tank was in the bot's local sensing window
            when the wire arrived. HUNT acquisition gate. Zero means
            the tank has never been viewport-confirmed; threat
            analysis must filter it out regardless of the other
            freshness timestamps.
        liveness: Three-state lifecycle gate. ``alive`` is the default.
            ``deactivated`` is set on 0x41 Deactivation -- the tank is a
            corpse on the tile for ~22 s until the server cleans it up
            with 0x58 TankRemove. ``removed`` is set on 0x58 -- the
            tile is empty and MapData entries for this id must be
            skipped (tombstone). Any per-tank wire (TankInfo,
            TankEntry, MovementResponse, TankStatusSync, Movement) flips
            a non-alive tank back to ``alive`` -- the respawn flow.
            ``analyze_threats`` filters to ``liveness == "alive"``;
            ``_combat_shoot`` thus cannot fire at a corpse or empty
            tile. Empirical capture 2026-06-20: bot used to shoot the
            corpse 3 times during the 22 s window because no 0x41
            handler updated the tank state.
    """

    tank_id: int
    x: int
    y: int
    team: int
    rank: int
    damage_state: int
    direction: int
    name: str
    is_bot: bool
    is_self: bool
    source: EntitySource
    timestamp_ms: int
    last_wire_seen_ms: int
    last_position_update_ms: int
    last_viewport_observation_ms: int
    liveness: TankLiveness
    # Last 0x53 ShootEvent attributed to this tank. ``last_aim_x``,
    # ``last_aim_y`` are the wire-reported barrel-aim coords at the
    # moment of fire; for straight shots they coincide with the impact
    # tile, for homing fire they can diverge. ``last_aim_weapon``
    # records which weapon fired (0=single, 1=dual, 2=missile,
    # 3=homing). ``last_aim_ms`` is the wall-clock so consumers can
    # treat the aim as stale once it ages past combat-tempo. All four
    # default to -1 / 0 when no shot has yet been observed.
    last_aim_x: int
    last_aim_y: int
    last_aim_weapon: int
    last_aim_ms: int
    # Phase 1c fact metadata. ``confidence`` is the trust in this tank
    # belief ([0.0, 1.0]; fresh observations record 1.0 -- decay by age
    # is a consumer policy, Phase 3). ``provenance`` records the wire
    # channel of the most recent observation that refreshed this tank
    # (``TankObservation.fact_source``).
    confidence: float
    provenance: ProvenanceChainDict


def make_tank_state(
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
    damage_state: int,
    name: str,
    is_bot: bool,
    is_self: bool,
    source: EntitySource = "viewport",
    timestamp_ms: int = 0,
    last_wire_seen_ms: int = 0,
    last_position_update_ms: int = 0,
    last_viewport_observation_ms: int = 0,
    direction: int = 0,
    liveness: TankLiveness = "alive",
    last_aim_x: int = -1,
    last_aim_y: int = -1,
    last_aim_weapon: int = -1,
    last_aim_ms: int = 0,
    confidence: float = 1.0,
    provenance: ProvenanceChainDict | None = None,
) -> TankStateDict:
    """Create a tank state.

    Args:
        tank_id: Unique tank identifier.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0 recruit .. 8 general).
        damage_state: Fuel-quartile health tier (0=near death .. 3=full).
        name: Player name.
        is_bot: Whether this is a bot.
        is_self: Whether this is the player's tank.
        source: Which observed source confirmed this tank.
        timestamp_ms: Wall-clock ms of the most recent observation by
            ANY source. Zero means never observed.
        last_wire_seen_ms: Wall-clock ms of the most recent
            wire-sourced observation. Zero means never wire-confirmed.
        last_position_update_ms: Wall-clock ms of the most recent
            wire-sourced observation that carried fresh ``(x, y)``.
            Zero means the position has never been wire-confirmed.
        last_viewport_observation_ms: Wall-clock ms of the most recent
            observation whose ``storage_source`` was ``"viewport"``.
            Zero means the tank has never been viewport-confirmed.
        direction: Sprite direction byte. 0-31 = alive facing,
            32-33 = dead corpse.
        liveness: Three-state lifecycle gate. Defaults to ``alive``.
            See :class:`TankStateDict` for the full semantics.
        confidence: Trust in this belief. Fresh observations use 1.0.
        provenance: Origin plus derivation references. When omitted,
            derived from ``source`` via :func:`tank_default_fact_source`
            (synthetic default for direct construction; the observation
            pipeline always supplies the true message channel).

    Returns:
        TankStateDict with the provided values.
    """
    resolved_provenance = (
        make_provenance(tank_default_fact_source(source), []) if provenance is None else provenance
    )
    return TankStateDict(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        damage_state=damage_state,
        direction=direction,
        name=name,
        is_bot=is_bot,
        is_self=is_self,
        source=source,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
        last_viewport_observation_ms=last_viewport_observation_ms,
        liveness=liveness,
        last_aim_x=last_aim_x,
        last_aim_y=last_aim_y,
        last_aim_weapon=last_aim_weapon,
        last_aim_ms=last_aim_ms,
        confidence=confidence,
        provenance=resolved_provenance,
    )


def encode_tank_state(state: TankStateDict) -> JSONObject:
    """Encode TankStateDict to JSON-serializable dict.

    Args:
        state: TankStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "tank_id": state["tank_id"],
        "x": state["x"],
        "y": state["y"],
        "team": state["team"],
        "rank": state["rank"],
        "damage_state": state["damage_state"],
        "direction": state["direction"],
        "name": state["name"],
        "is_bot": state["is_bot"],
        "is_self": state["is_self"],
        "source": state["source"],
        "timestamp_ms": state["timestamp_ms"],
        "last_wire_seen_ms": state["last_wire_seen_ms"],
        "last_position_update_ms": state["last_position_update_ms"],
        "last_viewport_observation_ms": state["last_viewport_observation_ms"],
        "liveness": state["liveness"],
        "last_aim_x": state["last_aim_x"],
        "last_aim_y": state["last_aim_y"],
        "last_aim_weapon": state["last_aim_weapon"],
        "last_aim_ms": state["last_aim_ms"],
        "confidence": state["confidence"],
        "provenance": encode_provenance(state["provenance"]),
    }


def decode_tank_state(data: JSONObject) -> TankStateDict:
    """Decode TankStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated TankStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    source = require_entity_source(data, "source")
    confidence = require_float(data, "confidence") if "confidence" in data else 1.0
    provenance = (
        decode_provenance(require_dict(data, "provenance"))
        if "provenance" in data
        else make_provenance(tank_default_fact_source(source), [])
    )
    return TankStateDict(
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        team=require_int(data, "team"),
        rank=require_int(data, "rank"),
        damage_state=require_int(data, "damage_state"),
        direction=require_int(data, "direction"),
        name=require_str(data, "name"),
        is_bot=require_bool(data, "is_bot"),
        is_self=require_bool(data, "is_self"),
        source=source,
        timestamp_ms=require_int(data, "timestamp_ms"),
        last_wire_seen_ms=require_int(data, "last_wire_seen_ms"),
        last_position_update_ms=require_int(data, "last_position_update_ms"),
        last_viewport_observation_ms=_optional_int(data, "last_viewport_observation_ms", 0),
        liveness=require_tank_liveness(data, "liveness"),
        last_aim_x=_optional_int(data, "last_aim_x", -1),
        last_aim_y=_optional_int(data, "last_aim_y", -1),
        last_aim_weapon=_optional_int(data, "last_aim_weapon", -1),
        last_aim_ms=_optional_int(data, "last_aim_ms", 0),
        confidence=confidence,
        provenance=provenance,
    )


def _optional_int(data: JSONObject, key: str, default: int) -> int:
    """Read an optional int field from JSON, falling back to ``default``.

    Used for tank-state fields added after the on-disk format
    stabilised; older snapshots / fixtures lack the new keys and must
    decode cleanly without them.

    Args:
        data: JSON object being decoded.
        key: Field name to look up.
        default: Value to return when the key is absent.

    Returns:
        The int value at ``data[key]`` if present and an int; otherwise
        ``default``.

    Raises:
        JSONTypeError: When the key is present but the value is not an
            int (a hard type mismatch we want to surface, not silently
            paper over).
    """
    if key not in data:
        return default
    return require_int(data, key)


__all__ = [
    "VIEWPORT_PRESENCE_TTL_MS",
    "TankStateDict",
    "decode_tank_state",
    "encode_tank_state",
    "has_known_position",
    "make_tank_state",
    "tank_default_fact_source",
]
