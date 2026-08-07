"""Tests for the tank-observation invariants.

The three rules the mutator must never violate: timestamps advance,
wire-seen requires wire, and position freshness requires an
authoritative position.
"""

from __future__ import annotations

from tankpit_bot.state.tank_mutations import apply_tank_observation
from tankpit_bot.state.types import (
    make_empty_world_state,
    make_tank_observation,
)
from tests.world_state._observation_fixtures import make_world_with_seed


class TestInvariantTimestampAlwaysAdvances:
    """``timestamp_ms`` advances on every observation."""

    def test_any_observation_advances_timestamp(self) -> None:
        """Even a fully-null observation advances ``timestamp_ms``."""
        state, key = make_world_with_seed(tank_id=42, timestamp_ms=1000)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["timestamp_ms"] == 5000

    def test_map_only_observation_advances_timestamp(self) -> None:
        """A map-only observation still advances ``timestamp_ms``."""
        state, key = make_world_with_seed(tank_id=42, timestamp_ms=1000)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["timestamp_ms"] == 5000


class TestInvariantWireSeenRequiresWire:
    """``last_wire_seen_ms`` advances only on wire-sourced observations."""

    def test_wire_observation_advances_wire_seen(self) -> None:
        """Any wire observation refreshes ``last_wire_seen_ms``."""
        state, key = make_world_with_seed(tank_id=42, last_wire_seen_ms=900)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_wire_seen_ms"] == 5000

    def test_map_observation_does_not_advance_wire_seen(self) -> None:
        """Map-sourced observations leave ``last_wire_seen_ms`` untouched."""
        state, key = make_world_with_seed(tank_id=42, last_wire_seen_ms=900)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_wire_seen_ms"] == 900

    def test_map_observation_starts_wire_seen_at_zero_for_new_tank(self) -> None:
        """A first-sight map observation must start wire-seen at zero."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=99,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["99"]["last_wire_seen_ms"] == 0


class TestInvariantPositionFreshnessRequiresAuthoritativePosition:
    """``last_position_update_ms`` advances iff ``position_is_authoritative`` AND ``position``.

    ``position_is_authoritative`` decouples the kill-shot gate from the
    wire-presence gate: MAP_DATA snapshots are not wire-sourced (a
    departed tank can linger in the snapshot for minutes) but their
    listed coordinates ARE the server's authoritative statement of
    where each tank IS at snapshot time, so they advance the position
    freshness gate without claiming wire presence. Radar EnemyDetect
    and DOM-scraped client-registry refinements do NOT (tile-coarse /
    out-of-band estimates).
    """

    def test_wire_with_position_advances_position_freshness(self) -> None:
        """0x3D-like observation refreshes ``last_position_update_ms``."""
        state, key = make_world_with_seed(tank_id=42, last_position_update_ms=800)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_position_update_ms"] == 5000

    def test_wire_without_position_preserves_position_freshness(self) -> None:
        """TankStatusSync-like observation MUST NOT lie about position freshness.

        This is the locked invariant whose violation produced the
        stale-registry combat-miss loop in 2026-06-19 runs.
        """
        state, key = make_world_with_seed(tank_id=42, last_position_update_ms=800)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            damage_state=2,
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_position_update_ms"] == 800

    def test_authoritative_map_position_advances_position_freshness(self) -> None:
        """MAP_DATA's listed coordinates ARE the server's authoritative position.

        A wire-quiet stationary target stays kill-shot-fresh after the
        bot opens the map, even though the wire-presence stamp
        deliberately does NOT advance (a departed tank can linger in
        the snapshot for minutes). Live run 20260620-191622 fix: the
        bot was blocking targets it was actively engaging because the
        gate could only advance on wire-position-bearing messages.
        """
        state, key = make_world_with_seed(tank_id=42, last_position_update_ms=800)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=True,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_position_update_ms"] == 5000

    def test_map_position_defers_to_a_fresh_position(self) -> None:
        """A 0x4C fix never overwrites a position updated inside its aging window.

        The 2026-08-06 delta mining (2,851 same-tank map/wire pairs
        within 2 s): the snapshot payload AGES before arrival — 53%
        disagree with a fresh wire fix by walk steps or teleport hops,
        zero decode artifacts. Presence stays exact (liveness rule 3
        untouched); the position and its freshness both hold, so an
        aged snapshot cannot smear a kill-shot-fresh wire fix.
        """
        state, key = make_world_with_seed(tank_id=42, x=10, y=20, last_position_update_ms=4500)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=True,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert (result["tanks"][key]["x"], result["tanks"][key]["y"]) == (10, 20)
        assert result["tanks"][key]["last_position_update_ms"] == 4500

    def test_wire_position_is_never_deferred(self) -> None:
        """The defer is map-only: a wire fix always lands, however fresh the seed."""
        state, key = make_world_with_seed(tank_id=42, x=10, y=20, last_position_update_ms=4500)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            position_is_authoritative=True,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert (result["tanks"][key]["x"], result["tanks"][key]["y"]) == (50, 60)
        assert result["tanks"][key]["last_position_update_ms"] == 5000

    def test_non_authoritative_position_preserves_position_freshness(self) -> None:
        """Radar / DOM-refinement positions are not authoritative; freshness stays.

        Radar EnemyDetect (0x48) returns a tile-coarse estimate that
        may not match the target's actual wire position by the next
        tick. Client-registry refinements come from DOM scrape, an
        out-of-band channel with no server proof. Neither must gate a
        kill shot.
        """
        state, key = make_world_with_seed(tank_id=42, last_position_update_ms=800)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=False,
            storage_source="radar",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_position_update_ms"] == 800

    def test_authoritative_map_position_starts_freshness_for_new_tank(
        self,
    ) -> None:
        """A first-sight authoritative map observation seeds position freshness."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=99,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=True,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["99"]["last_position_update_ms"] == 5000

    def test_non_authoritative_position_starts_freshness_at_zero_for_new_tank(
        self,
    ) -> None:
        """A first-sight radar / DOM-refinement observation cannot seed freshness."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=99,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=False,
            storage_source="radar",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["99"]["last_position_update_ms"] == 0

    def test_wire_status_only_starts_position_freshness_at_zero_for_new_tank(
        self,
    ) -> None:
        """A first-sight damage-only wire observation cannot bootstrap position freshness."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=99,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            damage_state=2,
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["99"]["last_position_update_ms"] == 0
