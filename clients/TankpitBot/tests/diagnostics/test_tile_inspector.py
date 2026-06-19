"""Integration tests for the terrain tile inspector.

Tests exercise the REAL :class:`tankpit_bot.terrain.TerrainMap` against
the real :file:`field01_r.gif` checked into the repository, the real
:func:`tankpit_bot.bot.ai.equipment.find_teleport_landing_tile` /
:func:`tankpit_bot.bot.ai.equipment.is_reachable` helpers, and the
real CLI entrypoint. Nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.tile_inspector import (
    _parse_coord_pair,
    _parse_optional_flag,
    _require_positional,
    inspect_tile,
    main,
    render_tile_inspection,
)

_FIELD01_GIF = Path(__file__).resolve().parents[2] / "field01_r.gif"


def test_inspect_tile_marks_passable_ground_at_known_open_spawn() -> None:
    """Spawn around (131, 110) on Practice (field01) is passable ground."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=131,
        target_y=110,
        from_x=-1,
        from_y=-1,
    )

    assert report["field_image"] == "field01_r.gif"
    assert report["target_in_bounds"] is True
    assert report["target_passable"] is True
    assert report["target_terrain"] == "."
    assert report["landing_resolution"] == "target_is_passable"
    assert report["landing_tile_x"] == 131
    assert report["landing_tile_y"] == 110
    assert report["reachable"] is False  # no origin supplied


def test_inspect_tile_marks_water_at_known_water_spawn() -> None:
    """The far-west water column on field01 reports as impassable water."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=125,
        target_y=126,
        from_x=-1,
        from_y=-1,
    )

    assert report["target_passable"] is False
    assert report["target_terrain"] == "W"


def test_inspect_tile_reports_out_of_bounds_target() -> None:
    """A coordinate outside 0..255 is flagged ``in_bounds=False``."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=999,
        target_y=999,
        from_x=-1,
        from_y=-1,
    )

    assert report["target_in_bounds"] is False
    assert report["target_passable"] is False
    assert report["target_terrain"] == " "


def test_inspect_tile_with_origin_runs_reachability() -> None:
    """Supplying a from-origin enables the production A* reachability check."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=131,
        target_y=110,
        from_x=131,
        from_y=126,
    )

    assert report["from_x"] == 131
    assert report["from_y"] == 126
    assert report["reachable"] is True


def test_inspect_tile_neighbors_cover_eight_compass_directions() -> None:
    """The neighbor list always carries exactly the 8 compass directions."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=131,
        target_y=110,
        from_x=-1,
        from_y=-1,
    )

    directions = [neighbor["direction"] for neighbor in report["neighbors"]]
    assert directions == ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def test_inspect_tile_neighbor_in_bounds_flag_handles_map_edge() -> None:
    """Neighbors of (0, 0) include four out-of-bounds tiles."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=0,
        target_y=0,
        from_x=-1,
        from_y=-1,
    )

    by_direction = {n["direction"]: n for n in report["neighbors"]}
    assert by_direction["N"]["in_bounds"] is False
    assert by_direction["NE"]["in_bounds"] is False
    assert by_direction["NW"]["in_bounds"] is False
    assert by_direction["W"]["in_bounds"] is False
    assert by_direction["E"]["in_bounds"] is True
    assert by_direction["S"]["in_bounds"] is True


def test_inspect_tile_raises_on_missing_gif(tmp_path: Path) -> None:
    """A missing field GIF raises ``FileNotFoundError`` immediately."""
    with pytest.raises(FileNotFoundError, match="field GIF not found"):
        inspect_tile(
            tmp_path / "does_not_exist.gif",
            target_x=0,
            target_y=0,
            from_x=-1,
            from_y=-1,
        )


def test_render_tile_inspection_includes_target_and_neighbor_summary() -> None:
    """The rendered report names the target tile and lists every neighbor."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=131,
        target_y=110,
        from_x=-1,
        from_y=-1,
    )

    rendered = render_tile_inspection(report)

    assert "TANKPIT TILE INSPECTION" in rendered
    assert "Target: (131, 110)" in rendered
    assert "passable=True" in rendered
    assert "no origin provided" in rendered
    for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
        assert f" {direction:2s}" in rendered or f" {direction} " in rendered


def test_render_tile_inspection_renders_reachability_section_when_origin_present() -> None:
    """The reachability section only appears once a from-origin is set."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=131,
        target_y=110,
        from_x=131,
        from_y=126,
    )

    rendered = render_tile_inspection(report)

    assert "from=(131,126)" in rendered
    assert "reachable=True" in rendered


def test_inspect_tile_lands_on_target_when_impassable() -> None:
    """When the target tile is water, the landing tile is the target itself.

    The server handles displacement to the nearest passable tile. The
    landing resolution still describes the adjacent passable neighbor
    for diagnostic purposes.
    """
    # On field01 row 126 the western water column ends right before the
    # spawn ground column at x=131. (130,126) is water; its eastern
    # neighbor (131,126) is the passable spawn tile.
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=130,
        target_y=126,
        from_x=-1,
        from_y=-1,
    )

    assert report["target_passable"] is False
    assert report["target_terrain"] == "W"
    assert report["landing_resolution"] == "adjacent:E"
    assert report["landing_tile_x"] == 131
    assert report["landing_tile_y"] == 126


def test_render_tile_inspection_renders_no_landing_branch() -> None:
    """A target with no passable cardinal neighbor renders the NONE landing line."""
    report = inspect_tile(
        _FIELD01_GIF,
        target_x=999,
        target_y=999,
        from_x=-1,
        from_y=-1,
    )

    rendered = render_tile_inspection(report)

    assert "Landing tile: NONE" in rendered


def test_main_resolves_positional_args_and_optional_origin(tmp_path: Path) -> None:
    """``main()`` parses the positional path/X/Y + the optional ``--from`` flag."""
    argv_value = [
        "tankpit-tile-info",
        str(_FIELD01_GIF),
        "131",
        "110",
        "--from",
        "131,126",
    ]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: argv_value
    try:
        exit_code = main()
    finally:
        _test_hooks.get_argv = original_get_argv

    assert exit_code == 0


def test_main_runs_without_origin() -> None:
    """``main()`` works with only the positional args."""
    argv_value = ["tankpit-tile-info", str(_FIELD01_GIF), "131", "110"]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: argv_value
    try:
        exit_code = main()
    finally:
        _test_hooks.get_argv = original_get_argv

    assert exit_code == 0


def test_main_handles_empty_argv_via_test_hook() -> None:
    """An empty argv from the test hook raises a clear ``ValueError``."""
    original_get_argv = _test_hooks.get_argv
    empty: list[str] = []
    _test_hooks.get_argv = lambda: empty
    try:
        with pytest.raises(ValueError, match="missing required positional"):
            main()
    finally:
        _test_hooks.get_argv = original_get_argv


def test_parse_coord_pair_returns_tuple_of_ints() -> None:
    """``--from 131,126`` parses into the integer pair ``(131, 126)``."""
    assert _parse_coord_pair("131,126", flag="--from") == (131, 126)


def test_parse_coord_pair_rejects_single_value() -> None:
    """A single value without a comma is rejected."""
    with pytest.raises(ValueError, match="comma-separated"):
        _parse_coord_pair("131", flag="--from")


def test_parse_optional_flag_returns_none_when_absent() -> None:
    """An absent flag returns ``None`` rather than raising."""
    assert _parse_optional_flag(["tile", "131", "110"], "--from") is None


def test_parse_optional_flag_raises_when_value_missing() -> None:
    """Specifying a flag without a value raises ``ValueError``."""
    with pytest.raises(ValueError, match="requires a value"):
        _parse_optional_flag(["tile", "131", "110", "--from"], "--from")


def test_require_positional_raises_when_index_out_of_range() -> None:
    """Missing positional argument raises with the parameter name in the message."""
    with pytest.raises(ValueError, match="field-gif"):
        _require_positional([], 0, "field-gif")
