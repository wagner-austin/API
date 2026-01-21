"""Tests for state module constants."""

from tankpit_bot.state import (
    ASCII_ALLY,
    ASCII_ENEMY,
    ASCII_EQUIPMENT,
    ASCII_FERRY,
    ASCII_FUEL,
    ASCII_GROUND,
    ASCII_MINE,
    ASCII_ROCK,
    ASCII_SELF,
    ASCII_UNKNOWN,
    ASCII_WATER,
    DAMAGE_CRITICAL,
    DAMAGE_FULL,
    DAMAGE_LIGHT,
    DAMAGE_MEDIUM,
    TEAM_BLUE,
    TEAM_ORANGE,
    TEAM_PURPLE,
    TEAM_RED,
    TERRAIN_FERRY,
    TERRAIN_FERRY_ROCK,
    TERRAIN_GROUND,
    TERRAIN_ROCK_A,
    TERRAIN_ROCK_AB,
    TERRAIN_ROCK_B,
)


class TestConstants:
    """Tests for module constants."""

    def test_terrain_constants(self) -> None:
        """Verify terrain type constants."""
        assert TERRAIN_GROUND == 0
        assert TERRAIN_ROCK_A == 1
        assert TERRAIN_ROCK_B == 2
        assert TERRAIN_ROCK_AB == 3
        assert TERRAIN_FERRY == 5
        assert TERRAIN_FERRY_ROCK == 7

    def test_team_constants(self) -> None:
        """Verify team ID constants."""
        assert TEAM_RED == 0
        assert TEAM_PURPLE == 1
        assert TEAM_BLUE == 2
        assert TEAM_ORANGE == 3

    def test_damage_constants(self) -> None:
        """Verify damage state constants."""
        assert DAMAGE_FULL == 0
        assert DAMAGE_LIGHT == 1
        assert DAMAGE_MEDIUM == 2
        assert DAMAGE_CRITICAL == 3

    def test_ascii_constants(self) -> None:
        """Verify ASCII character constants."""
        assert ASCII_GROUND == "."
        assert ASCII_ROCK == "#"
        assert ASCII_FERRY == "~"
        assert ASCII_WATER == "W"
        assert ASCII_FUEL == "F"
        assert ASCII_EQUIPMENT == "E"
        assert ASCII_MINE == "*"
        assert ASCII_SELF == "@"
        assert ASCII_ENEMY == "T"
        assert ASCII_ALLY == "A"
        assert ASCII_UNKNOWN == "?"
