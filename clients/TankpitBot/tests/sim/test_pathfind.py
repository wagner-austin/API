"""The deterministic quadrant-keyed router."""

from __future__ import annotations

from tankpit_bot.sim.pathfind import route


def _open_ground(x: int, y: int) -> bool:
    """Everything is passable."""
    return True


def test_same_tile_routes_empty() -> None:
    """Already at the destination: empty path."""
    assert route(_open_ground, 5, 5, 5, 5) == ""


def test_ne_quadrant_goes_horizontal_first() -> None:
    """Northeast diagonals lead with east (the measured exception)."""
    assert route(_open_ground, 10, 10, 13, 8) == "eeenn"


def test_other_quadrants_go_vertical_first() -> None:
    """SE/SW/NW diagonals lead with the vertical leg."""
    assert route(_open_ground, 10, 10, 13, 12) == "sseee"
    assert route(_open_ground, 10, 10, 7, 12) == "sswww"
    assert route(_open_ground, 10, 10, 7, 8) == "nnwww"


def test_straight_lines() -> None:
    """Pure-axis routes have no turn."""
    assert route(_open_ground, 10, 10, 10, 13) == "sss"
    assert route(_open_ground, 10, 10, 12, 10) == "ee"


def test_blocked_primary_leg_falls_to_secondary_l() -> None:
    """A wall across the primary L flips to the other L."""

    def passable(x: int, y: int) -> bool:
        return (x, y) != (10, 11)

    assert route(passable, 10, 10, 12, 12) == "eess"


def test_block_on_second_leg_also_flips() -> None:
    """The primary L is rejected even when the block is on leg two."""

    def passable(x: int, y: int) -> bool:
        return (x, y) != (11, 12)

    assert route(passable, 10, 10, 12, 12) == "eess"


def test_both_ls_blocked_takes_deterministic_staircase() -> None:
    """With both Ls blocked, BFS finds a shortest detour, stably."""
    blocked = {(10, 11), (11, 10)}

    def passable(x: int, y: int) -> bool:
        return (x, y) not in blocked

    first = route(passable, 10, 10, 12, 12)
    second = route(passable, 10, 10, 12, 12)
    assert first == "wsseee"
    assert second == "wsseee"


def test_unreachable_destination_returns_none() -> None:
    """A destination sealed behind walls has no route."""
    walls = {(1, 0), (1, 1), (0, 1)}

    def passable(x: int, y: int) -> bool:
        return (x, y) not in walls and 0 <= x <= 3 and 0 <= y <= 3

    assert route(passable, 0, 0, 3, 3) is None
