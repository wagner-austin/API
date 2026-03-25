"""Tests for protocol enums.

Tests for Rank, Team, Equipment, and TerrainType enumerations.
"""

from __future__ import annotations

from tankpit_bot.protocol import (
    RANK_FUEL,
    Equipment,
    Rank,
    Team,
    TerrainType,
)


class TestRankEnum:
    """Tests for Rank enumeration."""

    def test_rank_values(self) -> None:
        """Rank values are 0-8."""
        assert int(Rank.RECRUIT) == 0
        assert int(Rank.PRIVATE) == 1
        assert int(Rank.CORPORAL) == 2
        assert int(Rank.SERGEANT) == 3
        assert int(Rank.LIEUTENANT) == 4
        assert int(Rank.CAPTAIN) == 5
        assert int(Rank.MAJOR) == 6
        assert int(Rank.COLONEL) == 7
        assert int(Rank.GENERAL) == 8

    def test_rank_fuel_mapping(self) -> None:
        """RANK_FUEL maps ranks to starting fuel."""
        assert RANK_FUEL[Rank.RECRUIT] == 1000
        assert RANK_FUEL[Rank.PRIVATE] == 1100
        assert RANK_FUEL[Rank.CORPORAL] == 1200
        assert RANK_FUEL[Rank.SERGEANT] == 1300
        assert RANK_FUEL[Rank.LIEUTENANT] == 1400
        assert RANK_FUEL[Rank.CAPTAIN] == 1500
        assert RANK_FUEL[Rank.MAJOR] == 1600
        assert RANK_FUEL[Rank.COLONEL] == 1700
        assert RANK_FUEL[Rank.GENERAL] == 1800


class TestTeamEnum:
    """Tests for Team enumeration."""

    def test_team_values(self) -> None:
        """Team values are 0-3."""
        assert int(Team.RED) == 0
        assert int(Team.PURPLE) == 1
        assert int(Team.BLUE) == 2
        assert int(Team.ORANGE) == 3


class TestEquipmentEnum:
    """Tests for Equipment enumeration."""

    def test_equipment_values(self) -> None:
        """Equipment values are 0-4."""
        assert int(Equipment.ARMOR_SHIELD) == 0
        assert int(Equipment.DUAL_SHOT) == 1
        assert int(Equipment.MISSILE_SHOT) == 2
        assert int(Equipment.HOMING_SHOT) == 3
        assert int(Equipment.EXTRA_RADAR) == 4


class TestTerrainTypeEnum:
    """Tests for TerrainType enumeration."""

    def test_terrain_values(self) -> None:
        """TerrainType values match protocol."""
        assert int(TerrainType.GROUND) == 0
        assert int(TerrainType.ROCK_A) == 1
        assert int(TerrainType.ROCK_B) == 2
        assert int(TerrainType.ROCK_AB) == 3
        assert int(TerrainType.FERRY) == 5
        assert int(TerrainType.FERRY_ROCK) == 7
