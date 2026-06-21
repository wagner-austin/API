"""Integration test: low-fuel state triggers RECOVER_FUEL mode selection.

When ``self_state["fuel"]`` is at or below ``fuel_low_threshold``,
``try_collect_fuel`` must return a non-None decision and ``decide()``
must pick the RECOVER_FUEL mode. When the fuel stays above the
threshold the function returns None so the bot continues whatever it
was doing.

This is the highest-leverage refuel test: the threshold gate is the
sole reason the bot tears itself away from combat to feed. Locking it
in prevents drift on the most safety-critical decision boundary.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.recover_fuel_mode import try_collect_fuel
from tankpit_bot.state.types import (
    SelfStateDict,
    WorldStateDict,
    make_container_state,
)
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def _world_with_visible_fuel(
    *,
    self_x: int,
    self_y: int,
    fuel: int,
    fuel_container_x: int,
    fuel_container_y: int,
    fuel_volume: int,
) -> tuple[WorldStateDict, SelfStateDict]:
    """Build a world state with a self position, low fuel, and a nearby fuel pickup.

    Args:
        self_x: Self X position.
        self_y: Self Y position.
        fuel: Self fuel amount.
        fuel_container_x: Fuel container X position.
        fuel_container_y: Fuel container Y position.
        fuel_volume: Fuel volume reported by radar.

    Returns:
        Tuple of (world_state, self_state).
    """
    container = make_container_state(
        x=fuel_container_x,
        y=fuel_container_y,
        volume=fuel_volume,
        is_fuel=True,
        timestamp_ms=100_000,
    )
    return make_world(
        self_x=self_x,
        self_y=self_y,
        fuel=fuel,
        containers={f"{fuel_container_x},{fuel_container_y}": container},
    )


class TestRefuelTriggersBelowThreshold:
    """Integration tests for the fuel-low collection gate."""

    def test_collect_fuel_returns_decision_below_threshold(self) -> None:
        """At fuel == low_threshold the recovery mode produces a decision."""
        world, self_state = _world_with_visible_fuel(
            self_x=131,
            self_y=122,
            fuel=300,  # default fuel_low_threshold
            fuel_container_x=132,
            fuel_container_y=122,
            fuel_volume=600,
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100_000,
            None,
            "",
        )
        assert ctx.config["fuel_low_threshold"] == 300

        decision = try_collect_fuel(ctx)

        if decision is None:
            raise AssertionError("try_collect_fuel must fire at fuel == threshold")
        # Recovery dispatches a move/pickup toward the visible fuel
        # container at (132, 122) (one tile east of self).
        cmd = decision["command"]
        assert cmd["cmd_type"] in ("pickup_fuel", "move", "teleport")

    def test_collect_fuel_returns_none_above_threshold(self) -> None:
        """Above ``fuel_low_threshold`` the recovery mode stays silent."""
        world, self_state = _world_with_visible_fuel(
            self_x=131,
            self_y=122,
            fuel=301,  # one above default low_threshold (300)
            fuel_container_x=132,
            fuel_container_y=122,
            fuel_volume=600,
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100_000,
            None,
            "",
        )

        decision = try_collect_fuel(ctx)

        assert decision is None, "try_collect_fuel must NOT fire above threshold"

    def test_collect_critical_fuel_at_critical_threshold(self) -> None:
        """At critical fuel the gate fires regardless of mode ownership.

        ``try_collect_critical_fuel`` interrupts any in-progress mode,
        including combat. This is the safety-of-last-resort path; the
        regression it prevents is the 20260612 fuel-stranding incident.
        """
        from tankpit_bot.bot.ai.recover_fuel_mode import try_collect_critical_fuel

        world, self_state = _world_with_visible_fuel(
            self_x=131,
            self_y=122,
            fuel=300,  # default fuel_critical_threshold (per ``make_default_ai_config``)
            fuel_container_x=132,
            fuel_container_y=122,
            fuel_volume=600,
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100_000,
            None,
            "",
        )
        assert ctx.config["fuel_critical_threshold"] == 300

        decision = try_collect_critical_fuel(ctx)

        if decision is None:
            raise AssertionError(
                "try_collect_critical_fuel must fire at fuel == critical_threshold"
            )
        cmd = decision["command"]
        assert cmd["cmd_type"] in ("pickup_fuel", "move", "teleport")
