"""Combat event tracking over time."""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.combat import (
    CombatEvent,
    CombatStats,
    EntityPairStats,
    _make_empty_combat_stats,
    _make_empty_entity_pair_stats,
    make_entity_pair_key,
    parse_combat_line,
)

log = get_logger(__name__)


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


__all__ = [
    "CombatTracker",
]
