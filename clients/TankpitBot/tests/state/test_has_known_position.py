"""Tests for the canonical tank position-provenance predicate."""

from __future__ import annotations

from tankpit_bot.state.types import TankStateDict, has_known_position, make_tank_state


def _tank(*, x: int, y: int, last_position_update_ms: int = 0) -> TankStateDict:
    """Build a tank carrying exactly the fields the predicate reads.

    Args:
        x: Tile X coordinate.
        y: Tile Y coordinate.
        last_position_update_ms: Authoritative-position timestamp.

    Returns:
        Tank state for predicate tests.
    """
    return make_tank_state(
        tank_id=7,
        x=x,
        y=y,
        team=1,
        rank=1,
        damage_state=3,
        name="nope",
        is_bot=False,
        is_self=False,
        last_position_update_ms=last_position_update_ms,
    )


class TestHasKnownPosition:
    """The (0, 0) construction default versus observed coordinates."""

    def test_login_roster_phantom_has_no_known_position(self) -> None:
        """A 0x21-created tank: default (0, 0), never position-synced."""
        assert has_known_position(_tank(x=0, y=0)) is False

    def test_nonzero_coordinates_are_a_known_position(self) -> None:
        """Any observed coordinate differs from the default. This is
        also how radar EnemyDetect qualifies: it writes real coords
        without advancing the authoritative-position timestamp."""
        assert has_known_position(_tank(x=201, y=143)) is True

    def test_single_nonzero_axis_is_a_known_position(self) -> None:
        """Map-edge tiles like (0, 143) are real positions."""
        assert has_known_position(_tank(x=0, y=143)) is True
        assert has_known_position(_tank(x=143, y=0)) is True

    def test_authoritative_zero_zero_is_a_known_position(self) -> None:
        """A tank an authoritative message placed exactly on (0, 0)
        is at (0, 0), not a phantom."""
        assert has_known_position(_tank(x=0, y=0, last_position_update_ms=1_000)) is True
