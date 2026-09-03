"""Tests for :mod:`tankpit_bot.physics.capacity`.

Reference values (see ``wiki/pages/game-economy.md``,
``wiki/pages/radar-mechanics.md``): each of the 9 rank rows is
enumerated explicitly, no parametrize magic, so a future formula
change fails on the specific rank row it broke.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import (
    DEPOSIT_FLOOR,
    RESERVE_REFERENCE_RANK,
    free_radar_radius,
    fuel_capacity,
    inventory_capacity,
    rank_scaled_reserve,
)


class TestFuelCapacity:
    """Capacity formula ``1000 + 100 * rank``, verified at ranks 1/3/6/7."""

    def test_recruit_capacity_is_1000(self) -> None:
        """Rank 0 (recruit) holds 1000 fuel.

        Formula plus the login-full-tank observation (viewport probe
        20260725-190352: fuel_before exactly 1000 on the freshly
        deactivated recruit account). The 2026-07-25 over-1000
        readings at wire rank 0 were a MID-SESSION PROMOTION to
        private, not a recruit cap ([[game-economy]]).
        """
        assert fuel_capacity(0) == 1000

    def test_private_capacity_is_1100(self) -> None:
        """Rank 1 (private) holds 1100 fuel.

        Verified 2026-07-06 by wire 0x52 code-5 tank-full at exactly 1100
        and by max deposit of 1000 (= 1100 - 100 floor).
        """
        assert fuel_capacity(1) == 1100

    def test_corporal_capacity_is_1200(self) -> None:
        """Rank 2 (corporal) holds 1200 fuel (formula, no direct measurement)."""
        assert fuel_capacity(2) == 1200

    def test_sergeant_capacity_is_1300(self) -> None:
        """Rank 3 (sergeant) holds 1300 fuel.

        Verified 2026-07-06 by max deposit of 1200 (= 1300 - 100 floor).
        """
        assert fuel_capacity(3) == 1300

    def test_lieutenant_capacity_is_1400(self) -> None:
        """Rank 4 (lieutenant) holds 1400 fuel (formula, no direct measurement)."""
        assert fuel_capacity(4) == 1400

    def test_captain_capacity_is_1500(self) -> None:
        """Rank 5 (captain) holds 1500 fuel (formula, no direct measurement)."""
        assert fuel_capacity(5) == 1500

    def test_major_capacity_is_1600(self) -> None:
        """Rank 6 (major) holds 1600 fuel.

        Verified 2026-07-06 by max deposit of 1500 (= 1600 - 100 floor).
        """
        assert fuel_capacity(6) == 1600

    def test_colonel_capacity_is_1700(self) -> None:
        """Rank 7 (colonel) holds 1700 fuel.

        Verified 2026-07-06 by max deposit of 1598
        (= 1700 - 100 floor - ~2 fuel walked to the deposit tile).
        """
        assert fuel_capacity(7) == 1700

    def test_general_capacity_is_1800(self) -> None:
        """Rank 8 (general) holds 1800 fuel (formula, no direct measurement)."""
        assert fuel_capacity(8) == 1800


class TestDepositFloor:
    """Server-enforced deposit floor, verified at four ranks 2026-07-06."""

    def test_deposit_floor_is_100(self) -> None:
        """A max deposit always leaves exactly 100 fuel in the tank."""
        assert DEPOSIT_FLOOR == 100


class TestFreeRadarRadius:
    """Radius formula ``2 + rank // 3``, verified at ranks 1/3/4/6/7."""

    def test_recruit_radius_is_2(self) -> None:
        """Rank 0 (recruit) built-in radar covers a 5x5 (radius 2)."""
        assert free_radar_radius(0) == 2

    def test_private_radius_is_2(self) -> None:
        """Rank 1 (private) built-in radar covers a 5x5 (radius 2).

        Verified 2026-06-12 corpus: ~120 built-in scans, zero reveals
        beyond chebyshev 2.
        """
        assert free_radar_radius(1) == 2

    def test_corporal_radius_is_2(self) -> None:
        """Rank 2 (corporal) built-in radar covers a 5x5 (radius 2)."""
        assert free_radar_radius(2) == 2

    def test_sergeant_radius_is_3(self) -> None:
        """Rank 3 (sergeant) built-in radar covers a 7x7 (radius 3).

        Verified 2026-07-06: (128,120) -> (128,123). Sergeant was
        chosen specifically to discriminate the two candidate
        step-boundary formulas.
        """
        assert free_radar_radius(3) == 3

    def test_lieutenant_radius_is_3(self) -> None:
        """Rank 4 (lieutenant) built-in radar covers a 7x7 (radius 3).

        Verified 2026-07-06: (111,129) -> (111,126).
        """
        assert free_radar_radius(4) == 3

    def test_captain_radius_is_3(self) -> None:
        """Rank 5 (captain) built-in radar covers a 7x7 (radius 3)."""
        assert free_radar_radius(5) == 3

    def test_major_radius_is_4(self) -> None:
        """Rank 6 (major) built-in radar covers a 9x9 (radius 4).

        Verified 2026-07-06: (234,5) -> (238,5). Major was chosen
        specifically to discriminate the two candidate step-boundary
        formulas.
        """
        assert free_radar_radius(6) == 4

    def test_colonel_radius_is_4(self) -> None:
        """Rank 7 (colonel) built-in radar covers a 9x9 (radius 4).

        Verified 2026-07-06: (165,125) -> (165,129).
        """
        assert free_radar_radius(7) == 4

    def test_general_radius_is_4(self) -> None:
        """Rank 8 (general) built-in radar covers a 9x9 (radius 4)."""
        assert free_radar_radius(8) == 4


class TestInventoryCapacity:
    """Per-slot cap ``20 + 5 * rank`` from the official rules table."""

    def test_recruit_cap_is_20(self) -> None:
        """Rank 0 (recruit) caps each slot at 20 (rules table).

        The 2026-07-25 over-20 counts at wire rank 0 were a
        mid-session promotion to private, not a recruit cap.
        """
        assert inventory_capacity(0) == 20

    def test_private_cap_is_25(self) -> None:
        """Rank 1 (private) caps each slot at 25.

        Matches the 0x52 code-7 refusals observed at 25 in live runs.
        """
        assert inventory_capacity(1) == 25

    def test_corporal_cap_is_30(self) -> None:
        """Rank 2 (corporal) caps each slot at 30."""
        assert inventory_capacity(2) == 30

    def test_sergeant_cap_is_35(self) -> None:
        """Rank 3 (sergeant) caps each slot at 35."""
        assert inventory_capacity(3) == 35

    def test_lieutenant_cap_is_40(self) -> None:
        """Rank 4 (lieutenant) caps each slot at 40."""
        assert inventory_capacity(4) == 40

    def test_captain_cap_is_45(self) -> None:
        """Rank 5 (captain) caps each slot at 45."""
        assert inventory_capacity(5) == 45

    def test_major_cap_is_50(self) -> None:
        """Rank 6 (major) caps each slot at 50."""
        assert inventory_capacity(6) == 50

    def test_colonel_cap_is_55(self) -> None:
        """Rank 7 (colonel) caps each slot at 55."""
        assert inventory_capacity(7) == 55

    def test_general_cap_is_60(self) -> None:
        """Rank 8 (general) caps each slot at 60."""
        assert inventory_capacity(8) == 60


class TestRankScaledReserve:
    """The reserve scaler behind [[flag-triage-20260902]] row 6."""

    def test_exact_at_the_reference_rank(self) -> None:
        """The lieutenant tuning survives the scaling untouched."""
        assert rank_scaled_reserve(200, RESERVE_REFERENCE_RANK) == 200
        assert rank_scaled_reserve(100, RESERVE_REFERENCE_RANK) == 100
        assert rank_scaled_reserve(450, RESERVE_REFERENCE_RANK) == 450

    def test_scales_down_with_capacity_below_the_reference(self) -> None:
        """A private's reserves are 11/14 of a lieutenant's."""
        assert rank_scaled_reserve(200, 1) == 157
        assert rank_scaled_reserve(100, 1) == 78
        assert rank_scaled_reserve(450, 1) == 353

    def test_scales_up_with_capacity_above_the_reference(self) -> None:
        """A general's reserves grow with the tank they protect."""
        assert rank_scaled_reserve(200, 8) == 257
        assert rank_scaled_reserve(450, 8) == 578

    def test_recruit_floor_is_proportional_to_the_smallest_tank(self) -> None:
        """Rank 0: capacity 1000 scales the reference by 5/7."""
        assert rank_scaled_reserve(200, 0) == 142
