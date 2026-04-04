"""Unified game state management for Tankpit.

Provides a central manager that aggregates:
- Combat tracker state (player stats and entity pair stats)
- Inventory state (item counts and enabled states)
- Player location (from game log parsing)
- Nearby entities (from radar detections)
- Room and session information

All state is immutable - updates create new TypedDict instances.
"""

from __future__ import annotations

import re
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
    require_list,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot.browser import GameLogEntry, GameLogScraper
from tankpit_bot.combat import (
    CombatStats,
    CombatTracker,
    EntityPairStats,
    decode_combat_stats,
    decode_entity_pair_stats,
    encode_combat_stats,
    encode_entity_pair_stats,
)
from tankpit_bot.inventory import (
    InventoryState,
    decode_inventory_state,
    encode_inventory_state,
)
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state

log = get_logger(__name__)


# =============================================================================
# TypedDicts
# =============================================================================


class LocationState(TypedDict):
    """Player's current location.

    Attributes:
        x: X coordinate on the map.
        y: Y coordinate on the map.
        raw: Original location string from game log.
    """

    x: int
    y: int
    raw: str


class NearbyEntity(TypedDict):
    """An entity detected nearby (via radar).

    Attributes:
        name: Entity name (e.g., "blue-7").
        direction: Cardinal direction (N, S, E, W, NE, etc.).
        coordinates: Target coordinates as "x,y" string.
        is_private: Whether the entity is marked as private.
    """

    name: str
    direction: str
    coordinates: str
    is_private: bool


class SessionInfo(TypedDict):
    """Current session information.

    Attributes:
        session_id: Unique session identifier.
        start_timestamp_ms: Session start time in milliseconds.
        magic_key: XOR magic key for protocol encoding (if captured).
        tank_name: Player's tank name (if known).
    """

    session_id: str
    start_timestamp_ms: int
    magic_key: str
    tank_name: str


class GameStateSnapshot(TypedDict):
    """Complete snapshot of current game state.

    Attributes:
        session: Session information.
        location: Player's current location.
        inventory: Current inventory state.
        combat_stats: Player-centric combat statistics.
        entity_pair_stats: Entity-to-entity combat statistics.
        nearby_entities: Currently detected nearby entities.
        unknown_hits_received: Count of hits from off-screen attackers.
    """

    session: SessionInfo
    location: LocationState
    inventory: InventoryState
    combat_stats: list[CombatStats]
    entity_pair_stats: list[EntityPairStats]
    nearby_entities: list[NearbyEntity]
    unknown_hits_received: int


# =============================================================================
# Parsing Functions
# =============================================================================


# Pattern for radar detection: "blue-7 (private) detected to W [57,135]"
_RADAR_PATTERN = re.compile(r"^(.+?)\s*(\(private\))?\s*detected to\s+([NSEW]+)\s+\[(\d+),(\d+)\]$")


def parse_location(location_str: str) -> LocationState:
    """Parse a location string into structured state.

    Args:
        location_str: Location string like "123,456" or empty.

    Returns:
        LocationState with parsed coordinates.
    """
    if not location_str or "," not in location_str:
        return LocationState(x=0, y=0, raw=location_str)

    parts = location_str.split(",")
    if len(parts) != 2:
        return LocationState(x=0, y=0, raw=location_str)

    x_str, y_str = parts[0].strip(), parts[1].strip()
    if not x_str.isdigit() or not y_str.isdigit():
        return LocationState(x=0, y=0, raw=location_str)

    return LocationState(x=int(x_str), y=int(y_str), raw=location_str)


def parse_radar_detection(text: str) -> NearbyEntity | None:
    """Parse a radar detection log line into NearbyEntity.

    Expected format: "blue-7 (private) detected to W [57,135]"

    Args:
        text: Log line text.

    Returns:
        NearbyEntity if parsed successfully, None otherwise.
    """
    match = _RADAR_PATTERN.match(text.strip())
    if not match:
        return None

    name: str = match.group(1).strip()
    is_private = match.group(2) is not None
    direction: str = match.group(3)
    x_coord: str = match.group(4)
    y_coord: str = match.group(5)

    return NearbyEntity(
        name=name,
        direction=direction,
        coordinates=f"{x_coord},{y_coord}",
        is_private=is_private,
    )


def _make_empty_location() -> LocationState:
    """Create an empty location state.

    Returns:
        LocationState with zero coordinates.
    """
    return LocationState(x=0, y=0, raw="")


def _make_empty_session() -> SessionInfo:
    """Create an empty session info.

    Returns:
        SessionInfo with empty values.
    """
    return SessionInfo(
        session_id="",
        start_timestamp_ms=0,
        magic_key="",
        tank_name="",
    )


# =============================================================================
# GameStateManager Class
# =============================================================================


class GameStateManager:
    """Manages unified game state from multiple sources.

    Aggregates state from:
    - CombatTracker: Combat events and statistics
    - InventoryScraper: Inventory items and changes
    - GameLogScraper: Location and nearby entity detection

    Provides immutable state snapshots for analysis and correlation.
    """

    def __init__(self) -> None:
        """Initialize the state manager."""
        self._combat_tracker: CombatTracker | None = None
        self._game_log_scraper: GameLogScraper | None = None
        self._session_info = _make_empty_session()
        self._location = _make_empty_location()
        self._nearby_entities: list[NearbyEntity] = []

    def set_combat_tracker(self, tracker: CombatTracker) -> None:
        """Set the combat tracker.

        Args:
            tracker: CombatTracker instance to use.
        """
        self._combat_tracker = tracker

    def set_game_log_scraper(self, scraper: GameLogScraper) -> None:
        """Set the game log scraper.

        Args:
            scraper: GameLogScraper instance to use.
        """
        self._game_log_scraper = scraper

    def update_session(
        self,
        *,
        session_id: str | None = None,
        start_timestamp_ms: int | None = None,
        magic_key: str | None = None,
        tank_name: str | None = None,
    ) -> None:
        """Update session information.

        Only provided fields are updated; None values are ignored.

        Args:
            session_id: Session identifier.
            start_timestamp_ms: Session start time.
            magic_key: XOR magic key.
            tank_name: Player's tank name.
        """
        self._session_info = SessionInfo(
            session_id=(session_id if session_id is not None else self._session_info["session_id"]),
            start_timestamp_ms=(
                start_timestamp_ms
                if start_timestamp_ms is not None
                else self._session_info["start_timestamp_ms"]
            ),
            magic_key=(magic_key if magic_key is not None else self._session_info["magic_key"]),
            tank_name=(tank_name if tank_name is not None else self._session_info["tank_name"]),
        )

    def process_game_log_entry(self, entry: GameLogEntry) -> None:
        """Process a game log entry to update state.

        Extracts location updates and radar detections.

        Args:
            entry: Game log entry to process.
        """
        text = entry["text"]
        category = entry["category"]

        # Update location from LOCATION entries
        if category == "location" and text.startswith("LOCATION:"):
            loc_str = text.replace("LOCATION:", "").strip()
            self._location = parse_location(loc_str)

        # Update nearby entities from action (radar) entries
        if category == "action" and "detected to" in text:
            entity = parse_radar_detection(text)
            if entity is not None:
                self._update_nearby_entity(entity)

    def _update_nearby_entity(self, entity: NearbyEntity) -> None:
        """Update or add a nearby entity.

        Replaces existing entry with same name or appends new entry.

        Args:
            entity: NearbyEntity to update or add.
        """
        # Remove existing entry with same name
        self._nearby_entities = [e for e in self._nearby_entities if e["name"] != entity["name"]]
        self._nearby_entities.append(entity)

    def clear_nearby_entities(self) -> None:
        """Clear all tracked nearby entities.

        Call this when entities may have moved out of range or on room change.
        """
        self._nearby_entities = []

    def get_location(self) -> LocationState:
        """Get current player location.

        Returns:
            Current LocationState.
        """
        return self._location

    def get_nearby_entities(self) -> list[NearbyEntity]:
        """Get list of nearby entities.

        Returns:
            List of NearbyEntity currently tracked.
        """
        return list(self._nearby_entities)

    def get_session_info(self) -> SessionInfo:
        """Get current session information.

        Returns:
            Current SessionInfo.
        """
        return self._session_info

    def get_snapshot(self) -> GameStateSnapshot:
        """Get a complete snapshot of current game state.

        Returns:
            GameStateSnapshot with all current state.
        """
        # Get inventory from binary protocol tracking
        inventory: InventoryState = get_inventory_state()

        # Get combat stats from tracker or use empty
        combat_stats: list[CombatStats]
        entity_pair_stats: list[EntityPairStats]
        unknown_hits: int

        if self._combat_tracker is not None:
            combat_stats = self._combat_tracker.get_all_stats()
            entity_pair_stats = self._combat_tracker.get_all_entity_pair_stats()
            unknown_hits = self._combat_tracker.get_unknown_hits_received()
        else:
            combat_stats = []
            entity_pair_stats = []
            unknown_hits = 0

        return GameStateSnapshot(
            session=self._session_info,
            location=self._location,
            inventory=inventory,
            combat_stats=combat_stats,
            entity_pair_stats=entity_pair_stats,
            nearby_entities=list(self._nearby_entities),
            unknown_hits_received=unknown_hits,
        )


# =============================================================================
# Encode/Decode Functions
# =============================================================================


def encode_location_state(state: LocationState) -> JSONObject:
    """Encode LocationState to JSON-serializable dict.

    Args:
        state: Location state to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "x": state["x"],
        "y": state["y"],
        "raw": state["raw"],
    }


def decode_location_state(obj: JSONObject) -> LocationState:
    """Decode JSON object to LocationState.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated LocationState.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    x = require_int(obj, "x")
    y = require_int(obj, "y")
    raw = require_str(obj, "raw")
    return LocationState(x=x, y=y, raw=raw)


def encode_nearby_entity(entity: NearbyEntity) -> JSONObject:
    """Encode NearbyEntity to JSON-serializable dict.

    Args:
        entity: Entity to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "name": entity["name"],
        "direction": entity["direction"],
        "coordinates": entity["coordinates"],
        "is_private": entity["is_private"],
    }


def decode_nearby_entity(obj: JSONObject) -> NearbyEntity:
    """Decode JSON object to NearbyEntity.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated NearbyEntity.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    name = require_str(obj, "name")
    direction = require_str(obj, "direction")
    coordinates = require_str(obj, "coordinates")
    is_private = require_bool(obj, "is_private")
    return NearbyEntity(
        name=name,
        direction=direction,
        coordinates=coordinates,
        is_private=is_private,
    )


def encode_session_info(info: SessionInfo) -> JSONObject:
    """Encode SessionInfo to JSON-serializable dict.

    Args:
        info: Session info to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "session_id": info["session_id"],
        "start_timestamp_ms": info["start_timestamp_ms"],
        "magic_key": info["magic_key"],
        "tank_name": info["tank_name"],
    }


def decode_session_info(obj: JSONObject) -> SessionInfo:
    """Decode JSON object to SessionInfo.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated SessionInfo.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    session_id = require_str(obj, "session_id")
    start_timestamp_ms = require_int(obj, "start_timestamp_ms")
    magic_key = require_str(obj, "magic_key")
    tank_name = require_str(obj, "tank_name")
    return SessionInfo(
        session_id=session_id,
        start_timestamp_ms=start_timestamp_ms,
        magic_key=magic_key,
        tank_name=tank_name,
    )


def encode_game_state_snapshot(snapshot: GameStateSnapshot) -> JSONObject:
    """Encode GameStateSnapshot to JSON-serializable dict.

    Args:
        snapshot: Snapshot to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "session": encode_session_info(snapshot["session"]),
        "location": encode_location_state(snapshot["location"]),
        "inventory": encode_inventory_state(snapshot["inventory"]),
        "combat_stats": [encode_combat_stats(s) for s in snapshot["combat_stats"]],
        "entity_pair_stats": [encode_entity_pair_stats(s) for s in snapshot["entity_pair_stats"]],
        "nearby_entities": [encode_nearby_entity(e) for e in snapshot["nearby_entities"]],
        "unknown_hits_received": snapshot["unknown_hits_received"],
    }


def decode_game_state_snapshot(obj: JSONObject) -> GameStateSnapshot:
    """Decode JSON object to GameStateSnapshot.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated GameStateSnapshot.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If nested objects are invalid.
    """
    # Decode session
    session_obj = obj.get("session")
    if not isinstance(session_obj, dict):
        raise ValueError("session must be a dict")
    session = decode_session_info(session_obj)

    # Decode location
    location_obj = obj.get("location")
    if not isinstance(location_obj, dict):
        raise ValueError("location must be a dict")
    location = decode_location_state(location_obj)

    # Decode inventory
    inventory_obj = obj.get("inventory")
    if not isinstance(inventory_obj, dict):
        raise ValueError("inventory must be a dict")
    inventory = decode_inventory_state(inventory_obj)

    # Decode combat stats
    combat_stats_raw = require_list(obj, "combat_stats")
    combat_stats: list[CombatStats] = []
    for i, item in enumerate(combat_stats_raw):
        if not isinstance(item, dict):
            raise ValueError(f"combat_stats[{i}] must be a dict")
        combat_stats.append(decode_combat_stats(item))

    # Decode entity pair stats
    entity_pair_stats_raw = require_list(obj, "entity_pair_stats")
    entity_pair_stats: list[EntityPairStats] = []
    for i, item in enumerate(entity_pair_stats_raw):
        if not isinstance(item, dict):
            raise ValueError(f"entity_pair_stats[{i}] must be a dict")
        entity_pair_stats.append(decode_entity_pair_stats(item))

    # Decode nearby entities
    nearby_entities_raw = require_list(obj, "nearby_entities")
    nearby_entities: list[NearbyEntity] = []
    for i, item in enumerate(nearby_entities_raw):
        if not isinstance(item, dict):
            raise ValueError(f"nearby_entities[{i}] must be a dict")
        nearby_entities.append(decode_nearby_entity(item))

    unknown_hits_received = require_int(obj, "unknown_hits_received")

    return GameStateSnapshot(
        session=session,
        location=location,
        inventory=inventory,
        combat_stats=combat_stats,
        entity_pair_stats=entity_pair_stats,
        nearby_entities=nearby_entities,
        unknown_hits_received=unknown_hits_received,
    )


__all__ = [
    "GameStateManager",
    "GameStateSnapshot",
    "LocationState",
    "NearbyEntity",
    "SessionInfo",
    "decode_game_state_snapshot",
    "decode_location_state",
    "decode_nearby_entity",
    "decode_session_info",
    "encode_game_state_snapshot",
    "encode_location_state",
    "encode_nearby_entity",
    "encode_session_info",
    "parse_location",
    "parse_radar_detection",
]
