"""Tests for target acquisition and relay travel."""

from __future__ import annotations

from tankpit_bot.bot.ai.threats import (
    find_locked_target_pursuit,
)
from tests.bot.ai._threat_fixtures import (
    _self_at,
    _tank,
    _world,
)


class TestFindLockedTargetPursuit:
    """Tests for ``find_locked_target_pursuit`` (locked-target chase)."""

    def test_returns_none_when_no_lock(self) -> None:
        """``locked_target_id == -1`` means no lock to pursue."""
        world = _world({})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=-1,
            killed={},
        )

        assert result is None

    def test_returns_none_when_locked_target_killed(self) -> None:
        """A locked target on the kill cooldown is not pursued.

        Once the bot has observed a kill, the cooldown applies even
        if the registry still lists the tank. Otherwise the pursuit
        path would re-engage corpses for a few seconds.
        """
        tank = _tank("50", x=105, y=100, team=1)
        tank["timestamp_ms"] = 100000
        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={"50": 99500},
        )

        assert result is None

    def test_returns_none_when_target_not_in_registry(self) -> None:
        """A locked id that is no longer in ``world["tanks"]`` cannot be pursued."""
        world = _world({})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        assert result is None

    def test_returns_none_when_target_deactivated(self) -> None:
        """A deactivated (corpse-window) tank is not a pursuit target."""
        tank = _tank("50", x=105, y=100, team=1, liveness="deactivated")
        tank["timestamp_ms"] = 100000
        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        assert result is None

    def test_returns_none_when_target_at_origin(self) -> None:
        """A tank with no position-bearing wire (still at (0,0)) is unfireable."""
        tank = _tank("50", x=0, y=0, team=1)
        tank["timestamp_ms"] = 100000
        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        assert result is None

    def test_returns_pursuit_threat_even_when_timestamp_is_stale(self) -> None:
        """Pursuit fires at the cached coords regardless of timestamp staleness.

        The earlier 5 s freshness gate was removed 2026-06-22 -- it
        was tripping on tanks the server stopped broadcasting 0x2E
        for (typically because they teleported far away), ending
        pursuit prematurely. Ammo only decrements on confirmed hit,
        so over-pursuing burns no resources -- the loop is bounded
        by 0x41 Deactivation or the kill cooldown, both authoritative
        death signals.
        """
        tank = _tank("50", x=105, y=100, team=1, name="prey")
        tank["timestamp_ms"] = 80000  # 20 s stale at now_ms=100000 -- old gate would have tripped

        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        if result is None:
            raise AssertionError("expected pursuit threat even with stale timestamp")
        assert result["tank_id"] == 50
        assert result["x"] == 105
        assert result["y"] == 100

    def test_returns_pursuit_threat_for_alive_locked_target(self) -> None:
        """An alive locked target returns a pursuit threat at cached coords."""
        tank = _tank("50", x=105, y=100, team=1, name="prey")
        tank["timestamp_ms"] = 100000
        world = _world({"50": tank})

        result = find_locked_target_pursuit(
            world,
            _self_at(),
            locked_target_id=50,
            killed={},
        )

        if result is None:
            raise AssertionError("expected a pursuit threat for alive target")
        assert result["tank_id"] == 50
        assert result["x"] == 105
        assert result["y"] == 100


def test_pursuit_trace_dead_when_target_missing_from_registry() -> None:
    """An id absent from the registry has no live trace (audit gap pin)."""
    from tankpit_bot.bot.ai.threat_primitives import pursuit_trace_is_live
    from tests.bot.ai._support import make_world

    world, _self_state = make_world(self_x=100, self_y=100, fuel=800)

    assert pursuit_trace_is_live(world, 999, 100000) is False


def test_pursuit_homing_budget_spent_for_a_vanished_target() -> None:
    """A stamped target missing from the registry stays spent, never re-fired."""
    from tankpit_bot.bot.ai.threat_primitives import pursuit_homing_budget_spent
    from tests.bot.ai._support import make_world

    world, _self_state = make_world(self_x=100, self_y=100, fuel=800)

    assert pursuit_homing_budget_spent(world, 999, 999, 95000) is True


def test_pursuit_homing_budget_fresh_for_a_different_target() -> None:
    """The stamp binds to one target id; another lock has a fresh budget."""
    from tankpit_bot.bot.ai.threat_primitives import pursuit_homing_budget_spent
    from tests.bot.ai._support import make_world

    world, _self_state = make_world(self_x=100, self_y=100, fuel=800)

    assert pursuit_homing_budget_spent(world, 999, 42, 95000) is False
