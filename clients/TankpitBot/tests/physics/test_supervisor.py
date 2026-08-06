"""The 0x52 refusal laws — every byte-mined branch pinned.

One law, two consumers: the sim emits refusals with these predicates
and the bot predicts them before dispatch. The branch table comes
from the 2026-08-01 fuel-choreography mining (~1,600 windows) and the
2026-08-02 20-kill soak's 48 live at-cap refusal receipts.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import fuel_capacity, inventory_capacity
from tankpit_bot.physics.supervisor import (
    TELEPORT_RING1_COST_SLACK,
    equipment_pickup_refusal,
    fuel_pickup_close_code,
    fuel_pickup_refusal,
    teleport_refusal,
)
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_EMPTY_CONTAINER,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
    SUPERVISOR_ERROR_INVENTORY_FULL,
    SUPERVISOR_ERROR_TANK_FULL,
)


class TestFuelPickupCloseCode:
    """Close-by-stockedness: code 5 while stocked, code 4 once drained."""

    def test_stocked_container_closes_with_the_clamp_receipt(self) -> None:
        """Any remaining fuel reads code 5 — the clamp SUCCESS close."""
        assert fuel_pickup_close_code(231) == SUPERVISOR_ERROR_TANK_FULL
        assert fuel_pickup_close_code(1) == SUPERVISOR_ERROR_TANK_FULL

    def test_drained_container_closes_empty(self) -> None:
        """Remaining zero reads code 4."""
        assert fuel_pickup_close_code(0) == SUPERVISOR_ERROR_EMPTY_CONTAINER


class TestFuelPickupRefusal:
    """The two locally-provable no-transfer outcomes."""

    def test_known_drained_container_predicts_empty(self) -> None:
        """A believed-drained container transfers nothing: code 4."""
        assert fuel_pickup_refusal(500, 1, 0) == SUPERVISOR_ERROR_EMPTY_CONTAINER

    def test_tank_at_rank_capacity_predicts_tank_full(self) -> None:
        """Exactly-full fuel against a stocked container: code 5 — the
        shape the 20-kill soak sent 48 times, 2 s after each kill."""
        assert fuel_pickup_refusal(fuel_capacity(1), 1, 508) == SUPERVISOR_ERROR_TANK_FULL

    def test_drained_wins_over_full_when_both_hold(self) -> None:
        """A full tank at a drained container closes code 4, not 5 —
        the close-by-stockedness law reads the CONTAINER."""
        assert fuel_pickup_refusal(fuel_capacity(1), 1, 0) == SUPERVISOR_ERROR_EMPTY_CONTAINER

    def test_headroom_against_stock_transfers(self) -> None:
        """One fuel below cap is a transfer — no refusal."""
        assert fuel_pickup_refusal(fuel_capacity(1) - 1, 1, 508) is None

    def test_capacity_is_rank_derived(self) -> None:
        """1100 fuel refuses at private but transfers at corporal —
        the mid-session-promotion case the stale-rank bug would hit."""
        assert fuel_pickup_refusal(1100, 1, 508) == SUPERVISOR_ERROR_TANK_FULL
        assert fuel_pickup_refusal(1100, 2, 508) is None


class TestEquipmentPickupRefusal:
    """Code 7 exactly when all five slots sit at the rank cap."""

    def test_all_slots_at_cap_refuses(self) -> None:
        """Five full slots: code 7, the container stays."""
        cap = inventory_capacity(1)
        assert equipment_pickup_refusal([cap] * 5, 1) == SUPERVISOR_ERROR_INVENTORY_FULL

    def test_one_deficient_slot_grants(self) -> None:
        """A single deficient slot makes the pickup grantable."""
        cap = inventory_capacity(1)
        assert equipment_pickup_refusal([cap, cap, cap, cap - 1, cap], 1) is None

    def test_cap_is_rank_derived(self) -> None:
        """All-20 refuses at recruit but grants at private — the
        bot-20260725-211120 promotion-crossing law."""
        assert equipment_pickup_refusal([20] * 5, 0) == SUPERVISOR_ERROR_INVENTORY_FULL
        assert equipment_pickup_refusal([20] * 5, 1) is None


class TestTeleportRefusal:
    """The affordability law: cost above fuel refuses, equal lands dry."""

    def test_cost_above_fuel_refuses(self) -> None:
        """One fuel short: code 8."""
        assert teleport_refusal(847, 848) == SUPERVISOR_ERROR_INSUFFICIENT_FUEL

    def test_cost_equal_to_fuel_lands(self) -> None:
        """Spending the tank exactly dry is legal."""
        assert teleport_refusal(848, 848) is None

    def test_ring1_slack_bounds_the_displacement_discount(self) -> None:
        """floor(6 * sqrt(2)) = 8 < 9: the slack covers the cheapest
        ring-1 landing, so a target-based prediction never refuses a
        hop the server could have landed."""
        assert TELEPORT_RING1_COST_SLACK == 9
        assert teleport_refusal(839, 848 - TELEPORT_RING1_COST_SLACK) is None
        assert (
            teleport_refusal(838, 848 - TELEPORT_RING1_COST_SLACK)
            == SUPERVISOR_ERROR_INSUFFICIENT_FUEL
        )
