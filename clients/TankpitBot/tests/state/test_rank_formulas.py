"""Tests for :mod:`tankpit_bot.state.rank_formulas`.

Reference values (see ``wiki/pages/game-economy.md``,
``wiki/pages/radar-mechanics.md``): each of the 9 rank rows is
enumerated explicitly, no parametrize magic, so a future formula
change fails on the specific rank row it broke.
"""

from __future__ import annotations

from tankpit_bot.state.rank_formulas import free_radar_radius, fuel_capacity


class TestFuelCapacity:
    """Capacity formula ``1000 + 100 * rank``, verified at ranks 1/3/6/7."""

    def test_recruit_capacity_is_1000(self) -> None:
        """Rank 0 (recruit) holds 1000 fuel (formula, no direct measurement)."""
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
