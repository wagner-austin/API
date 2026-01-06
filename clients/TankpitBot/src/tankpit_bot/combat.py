"""Combat tracking for Tankpit game log analysis.

Provides utilities to parse combat log lines and track per-target
and entity-to-entity combat statistics for correlation with WebSocket messages.
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# TypedDicts
# =============================================================================


CombatEventType = Literal[
    "hit_by_player",
    "hit_by_enemy",
    "hit_by_unknown",
    "deactivated",
    "destroyed",
    "entity_hit",
    "entity_deactivated",
    "entity_destroyed",
]


class CombatEvent(TypedDict):
    """A single combat event.

    Attributes:
        event_type: Type of combat event.
        attacker: Name of attacker (e.g., "blue-7") or "player" or "unknown".
        target: Name of target (e.g., "red-9") or "player".
    """

    event_type: CombatEventType
    attacker: str
    target: str


class CombatStats(TypedDict):
    """Combat statistics for a single target (player's perspective).

    Attributes:
        name: Target name (e.g., "blue-7").
        hits_given: Number of times player hit this target.
        hits_received: Number of times this target hit player.
        deactivated: Whether player deactivated this target.
        destroyed: Whether player destroyed this target.
    """

    name: str
    hits_given: int
    hits_received: int
    deactivated: bool
    destroyed: bool


class EntityPairStats(TypedDict):
    """Combat statistics between two entities.

    Tracks hits, deactivations, and destructions from one entity to another.
    Use make_entity_pair_key to create the key for lookup.

    Attributes:
        attacker: Name of the attacking entity.
        target: Name of the target entity.
        hits: Number of times attacker hit target.
        deactivated: Whether attacker deactivated target.
        destroyed: Whether attacker destroyed target.
    """

    attacker: str
    target: str
    hits: int
    deactivated: bool
    destroyed: bool


def make_entity_pair_key(attacker: str, target: str) -> str:
    """Create a unique key for an attacker->target pair.

    Args:
        attacker: Name of the attacking entity.
        target: Name of the target entity.

    Returns:
        String key in format "attacker->target".
    """
    return f"{attacker}->{target}"


# =============================================================================
# Combat Parsing Functions
# =============================================================================


# Regex patterns for combat log parsing - player-centric
_HIT_BY_PLAYER_PATTERN = re.compile(r"^You hit (.+)$")
_HIT_BY_ENEMY_PATTERN = re.compile(r"^(.+) hit you$")
_DEACTIVATED_BY_PLAYER_PATTERN = re.compile(r"^(.+) has been deactivated by you$")
_DESTROYED_BY_PLAYER_PATTERN = re.compile(r"^(.+) has been destroyed by you$")

# Regex patterns for entity-to-entity combat (non-player)
_ENTITY_HIT_PATTERN = re.compile(r"^(.+) hit (.+)$")
_ENTITY_DEACTIVATED_PATTERN = re.compile(r"^(.+) has been deactivated by (.+)$")
_ENTITY_DESTROYED_PATTERN = re.compile(r"^(.+) has been destroyed by (.+)$")


def _parse_player_centric(stripped: str) -> CombatEvent | None:
    """Parse player-centric combat patterns.

    Handles: You hit X, X hit you, You are hit, X deactivated/destroyed by you.

    Args:
        stripped: Stripped log line.

    Returns:
        CombatEvent if matched, None otherwise.
    """
    # "You hit {target}"
    match = _HIT_BY_PLAYER_PATTERN.match(stripped)
    if match:
        return CombatEvent(event_type="hit_by_player", attacker="player", target=match.group(1))

    # "{attacker} hit you"
    match = _HIT_BY_ENEMY_PATTERN.match(stripped)
    if match:
        return CombatEvent(event_type="hit_by_enemy", attacker=match.group(1), target="player")

    # "You are hit" (off-screen attacker)
    if stripped == "You are hit":
        return CombatEvent(event_type="hit_by_unknown", attacker="unknown", target="player")

    # "{target} has been deactivated by you"
    match = _DEACTIVATED_BY_PLAYER_PATTERN.match(stripped)
    if match:
        return CombatEvent(event_type="deactivated", attacker="player", target=match.group(1))

    # "{target} has been destroyed by you"
    match = _DESTROYED_BY_PLAYER_PATTERN.match(stripped)
    if match:
        return CombatEvent(event_type="destroyed", attacker="player", target=match.group(1))

    return None


def _parse_entity_to_entity(stripped: str) -> CombatEvent | None:
    """Parse entity-to-entity combat patterns.

    Handles: X hit Y, X deactivated/destroyed by Y (where neither is player).

    Args:
        stripped: Stripped log line.

    Returns:
        CombatEvent if matched, None otherwise.
    """
    # "{target} has been deactivated by {attacker}" (not "by you")
    match = _ENTITY_DEACTIVATED_PATTERN.match(stripped)
    if match:
        target_name: str = match.group(1)
        attacker_name: str = match.group(2)
        if attacker_name.lower() != "you":
            return CombatEvent(
                event_type="entity_deactivated",
                attacker=attacker_name,
                target=target_name,
            )

    # "{target} has been destroyed by {attacker}" (not "by you")
    match = _ENTITY_DESTROYED_PATTERN.match(stripped)
    if match:
        destroyed_target: str = match.group(1)
        destroyed_attacker: str = match.group(2)
        if destroyed_attacker.lower() != "you":
            return CombatEvent(
                event_type="entity_destroyed",
                attacker=destroyed_attacker,
                target=destroyed_target,
            )

    # "{attacker} hit {target}" (not "hit you", not "You hit")
    match = _ENTITY_HIT_PATTERN.match(stripped)
    if match:
        hit_attacker: str = match.group(1)
        hit_target: str = match.group(2)
        if hit_attacker.lower() != "you" and hit_target.lower() != "you":
            return CombatEvent(event_type="entity_hit", attacker=hit_attacker, target=hit_target)

    return None


def parse_combat_line(line: str) -> CombatEvent | None:
    """Parse a combat log line into a structured event.

    Parses player-centric events (You hit X, X hit you) as well as
    entity-to-entity events (blue-7 hit red-9).

    Args:
        line: A single log line.

    Returns:
        CombatEvent if the line is a combat event, None otherwise.
    """
    stripped = line.strip()

    # Try player-centric patterns first (more specific)
    result = _parse_player_centric(stripped)
    if result is not None:
        return result

    # Try entity-to-entity patterns
    return _parse_entity_to_entity(stripped)


def _make_empty_combat_stats(name: str) -> CombatStats:
    """Create empty combat stats for a target.

    Args:
        name: Target name.

    Returns:
        CombatStats with all counts at 0.
    """
    return CombatStats(
        name=name,
        hits_given=0,
        hits_received=0,
        deactivated=False,
        destroyed=False,
    )


def _make_empty_entity_pair_stats(attacker: str, target: str) -> EntityPairStats:
    """Create empty entity pair stats.

    Args:
        attacker: Attacker entity name.
        target: Target entity name.

    Returns:
        EntityPairStats with all counts at 0.
    """
    return EntityPairStats(
        attacker=attacker,
        target=target,
        hits=0,
        deactivated=False,
        destroyed=False,
    )


# =============================================================================
# CombatTracker Class
# =============================================================================


class CombatTracker:
    """Tracks combat events over time.

    Parses combat log lines into structured events and maintains:
    - Per-target statistics (player's perspective)
    - Entity-to-entity statistics (any attacker to any target)
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self._stats: dict[str, CombatStats] = {}
        self._entity_pair_stats: dict[str, EntityPairStats] = {}
        self._events: list[CombatEvent] = []
        self._unknown_hits_received: int = 0

    def _get_or_create_stats(self, name: str) -> CombatStats:
        """Get or create player-centric stats for a target.

        Args:
            name: Target name.

        Returns:
            CombatStats for the target.
        """
        if name not in self._stats:
            self._stats[name] = _make_empty_combat_stats(name)
        return self._stats[name]

    def _get_or_create_entity_pair_stats(self, attacker: str, target: str) -> EntityPairStats:
        """Get or create stats for an entity pair.

        Args:
            attacker: Attacker entity name.
            target: Target entity name.

        Returns:
            EntityPairStats for the pair.
        """
        key = make_entity_pair_key(attacker, target)
        if key not in self._entity_pair_stats:
            self._entity_pair_stats[key] = _make_empty_entity_pair_stats(attacker, target)
        return self._entity_pair_stats[key]

    def _record_player_event(self, event: CombatEvent) -> None:
        """Record a player-centric combat event.

        Args:
            event: The combat event (player is attacker or target).
        """
        event_type = event["event_type"]

        if event_type == "hit_by_player":
            stats = self._get_or_create_stats(event["target"])
            self._stats[event["target"]] = CombatStats(
                name=stats["name"],
                hits_given=stats["hits_given"] + 1,
                hits_received=stats["hits_received"],
                deactivated=stats["deactivated"],
                destroyed=stats["destroyed"],
            )
        elif event_type == "hit_by_enemy":
            stats = self._get_or_create_stats(event["attacker"])
            self._stats[event["attacker"]] = CombatStats(
                name=stats["name"],
                hits_given=stats["hits_given"],
                hits_received=stats["hits_received"] + 1,
                deactivated=stats["deactivated"],
                destroyed=stats["destroyed"],
            )
        elif event_type == "hit_by_unknown":
            self._unknown_hits_received += 1
        elif event_type == "deactivated":
            stats = self._get_or_create_stats(event["target"])
            self._stats[event["target"]] = CombatStats(
                name=stats["name"],
                hits_given=stats["hits_given"],
                hits_received=stats["hits_received"],
                deactivated=True,
                destroyed=stats["destroyed"],
            )
        else:  # event_type == "destroyed"
            stats = self._get_or_create_stats(event["target"])
            self._stats[event["target"]] = CombatStats(
                name=stats["name"],
                hits_given=stats["hits_given"],
                hits_received=stats["hits_received"],
                deactivated=stats["deactivated"],
                destroyed=True,
            )

    def _record_entity_pair_event(self, event: CombatEvent) -> None:
        """Record an entity-to-entity combat event.

        Args:
            event: The combat event between two entities.
        """
        attacker = event["attacker"]
        target = event["target"]
        key = make_entity_pair_key(attacker, target)
        event_type = event["event_type"]

        stats = self._get_or_create_entity_pair_stats(attacker, target)

        if event_type == "entity_hit":
            self._entity_pair_stats[key] = EntityPairStats(
                attacker=attacker,
                target=target,
                hits=stats["hits"] + 1,
                deactivated=stats["deactivated"],
                destroyed=stats["destroyed"],
            )
        elif event_type == "entity_deactivated":
            self._entity_pair_stats[key] = EntityPairStats(
                attacker=attacker,
                target=target,
                hits=stats["hits"],
                deactivated=True,
                destroyed=stats["destroyed"],
            )
        else:  # event_type == "entity_destroyed"
            self._entity_pair_stats[key] = EntityPairStats(
                attacker=attacker,
                target=target,
                hits=stats["hits"],
                deactivated=stats["deactivated"],
                destroyed=True,
            )

    def record_event(self, event: CombatEvent) -> None:
        """Record a combat event and update stats.

        Args:
            event: The combat event to record.
        """
        self._events.append(event)
        event_type = event["event_type"]

        # Route to appropriate handler based on event type
        if event_type in ("entity_hit", "entity_deactivated", "entity_destroyed"):
            self._record_entity_pair_event(event)
        else:
            self._record_player_event(event)

    def process_log_line(self, line: str) -> CombatEvent | None:
        """Process a log line and record any combat event.

        Args:
            line: Log line to process.

        Returns:
            CombatEvent if the line was a combat event, None otherwise.
        """
        event = parse_combat_line(line)
        if event is not None:
            self.record_event(event)
        return event

    def get_stats(self, name: str) -> CombatStats | None:
        """Get player-centric stats for a specific target.

        Args:
            name: Target name.

        Returns:
            CombatStats or None if not encountered.
        """
        return self._stats.get(name)

    def get_all_stats(self) -> list[CombatStats]:
        """Get player-centric stats for all encountered targets.

        Returns:
            List of CombatStats for all targets.
        """
        return list(self._stats.values())

    def get_entity_pair_stats(self, attacker: str, target: str) -> EntityPairStats | None:
        """Get stats for a specific entity pair.

        Args:
            attacker: Attacker entity name.
            target: Target entity name.

        Returns:
            EntityPairStats or None if pair not encountered.
        """
        key = make_entity_pair_key(attacker, target)
        return self._entity_pair_stats.get(key)

    def get_all_entity_pair_stats(self) -> list[EntityPairStats]:
        """Get stats for all entity pairs.

        Returns:
            List of EntityPairStats for all tracked pairs.
        """
        return list(self._entity_pair_stats.values())

    def get_events(self) -> list[CombatEvent]:
        """Get all recorded combat events.

        Returns:
            List of all CombatEvents.
        """
        return list(self._events)

    def get_unknown_hits_received(self) -> int:
        """Get count of hits from off-screen attackers.

        Returns:
            Number of "You are hit" events.
        """
        return self._unknown_hits_received

    def log_event(self, event: CombatEvent) -> None:
        """Log a combat event.

        Args:
            event: Event to log.
        """
        event_type = event["event_type"]
        if event_type == "hit_by_player":
            stats = self._stats.get(event["target"])
            count = stats["hits_given"] if stats else 0
            log.info("[COMBAT:HIT] You -> %s (total: %d)", event["target"], count)
        elif event_type == "hit_by_enemy":
            stats = self._stats.get(event["attacker"])
            count = stats["hits_received"] if stats else 0
            log.info("[COMBAT:HIT] %s -> You (total: %d)", event["attacker"], count)
        elif event_type == "hit_by_unknown":
            log.info(
                "[COMBAT:HIT] ??? -> You (off-screen, total: %d)",
                self._unknown_hits_received,
            )
        elif event_type == "deactivated":
            log.info("[COMBAT:DEACTIVATED] You deactivated %s", event["target"])
        elif event_type == "destroyed":
            log.info("[COMBAT:DESTROYED] You destroyed %s", event["target"])
        elif event_type == "entity_hit":
            pair_stats = self.get_entity_pair_stats(event["attacker"], event["target"])
            count = pair_stats["hits"] if pair_stats else 0
            log.info(
                "[COMBAT:HIT] %s -> %s (total: %d)",
                event["attacker"],
                event["target"],
                count,
            )
        elif event_type == "entity_deactivated":
            log.info(
                "[COMBAT:DEACTIVATED] %s deactivated %s",
                event["attacker"],
                event["target"],
            )
        else:  # event_type == "entity_destroyed"
            log.info(
                "[COMBAT:DESTROYED] %s destroyed %s",
                event["attacker"],
                event["target"],
            )


# =============================================================================
# Encode/Decode Functions
# =============================================================================


VALID_COMBAT_EVENT_TYPES: frozenset[str] = frozenset(
    [
        "hit_by_player",
        "hit_by_enemy",
        "hit_by_unknown",
        "deactivated",
        "destroyed",
        "entity_hit",
        "entity_deactivated",
        "entity_destroyed",
    ]
)


def validate_combat_event_type(value: str) -> CombatEventType:
    """Validate and narrow a string to a CombatEventType literal.

    Args:
        value: String value to validate.

    Returns:
        The validated event type as a Literal type.

    Raises:
        ValueError: If value is not a valid event type.
    """
    if value == "hit_by_player":
        return "hit_by_player"
    if value == "hit_by_enemy":
        return "hit_by_enemy"
    if value == "hit_by_unknown":
        return "hit_by_unknown"
    if value == "deactivated":
        return "deactivated"
    if value == "destroyed":
        return "destroyed"
    if value == "entity_hit":
        return "entity_hit"
    if value == "entity_deactivated":
        return "entity_deactivated"
    if value == "entity_destroyed":
        return "entity_destroyed"
    msg = f"Invalid combat event type '{value}', must be one of {VALID_COMBAT_EVENT_TYPES}"
    raise ValueError(msg)


def encode_combat_event(event: CombatEvent) -> JSONObject:
    """Encode CombatEvent to JSON-serializable dict.

    Args:
        event: Event to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "event_type": event["event_type"],
        "attacker": event["attacker"],
        "target": event["target"],
    }


def decode_combat_event(obj: JSONObject) -> CombatEvent:
    """Decode JSON object to CombatEvent.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated CombatEvent.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If event_type is invalid.
    """
    event_type_str = require_str(obj, "event_type")
    event_type = validate_combat_event_type(event_type_str)
    attacker = require_str(obj, "attacker")
    target = require_str(obj, "target")
    return CombatEvent(event_type=event_type, attacker=attacker, target=target)


def encode_combat_stats(stats: CombatStats) -> JSONObject:
    """Encode CombatStats to JSON-serializable dict.

    Args:
        stats: Stats to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "name": stats["name"],
        "hits_given": stats["hits_given"],
        "hits_received": stats["hits_received"],
        "deactivated": stats["deactivated"],
        "destroyed": stats["destroyed"],
    }


def decode_combat_stats(obj: JSONObject) -> CombatStats:
    """Decode JSON object to CombatStats.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated CombatStats.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    name = require_str(obj, "name")
    hits_given = require_int(obj, "hits_given")
    hits_received = require_int(obj, "hits_received")
    deactivated = require_bool(obj, "deactivated")
    destroyed = require_bool(obj, "destroyed")
    return CombatStats(
        name=name,
        hits_given=hits_given,
        hits_received=hits_received,
        deactivated=deactivated,
        destroyed=destroyed,
    )


def encode_entity_pair_stats(stats: EntityPairStats) -> JSONObject:
    """Encode EntityPairStats to JSON-serializable dict.

    Args:
        stats: Stats to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "attacker": stats["attacker"],
        "target": stats["target"],
        "hits": stats["hits"],
        "deactivated": stats["deactivated"],
        "destroyed": stats["destroyed"],
    }


def decode_entity_pair_stats(obj: JSONObject) -> EntityPairStats:
    """Decode JSON object to EntityPairStats.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated EntityPairStats.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    attacker = require_str(obj, "attacker")
    target = require_str(obj, "target")
    hits = require_int(obj, "hits")
    deactivated = require_bool(obj, "deactivated")
    destroyed = require_bool(obj, "destroyed")
    return EntityPairStats(
        attacker=attacker,
        target=target,
        hits=hits,
        deactivated=deactivated,
        destroyed=destroyed,
    )


__all__ = [
    "VALID_COMBAT_EVENT_TYPES",
    "CombatEvent",
    "CombatEventType",
    "CombatStats",
    "CombatTracker",
    "EntityPairStats",
    "decode_combat_event",
    "decode_combat_stats",
    "decode_entity_pair_stats",
    "encode_combat_event",
    "encode_combat_stats",
    "encode_entity_pair_stats",
    "make_entity_pair_key",
    "parse_combat_line",
    "validate_combat_event_type",
]
