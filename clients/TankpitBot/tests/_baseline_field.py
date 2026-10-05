"""A field every baseline scenario can seed on, for tests that play them all.

``scripts.build_sim_baseline`` and ``scripts.sim_control`` both play every
entry of :data:`scripts.build_sim_baseline.SCENARIOS` through the real sim,
so both need the same stand-in for the field GIF and its terrain. It lives
here once rather than as a fixture copied into each test module.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.resources import data_directory
from tankpit_bot.sim.scenarios import SIM_FIELD
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

#: The ferry scenario's own water tile (``make_ferry_sim_world``).
FERRY_TILE = (118, 112)


def _load_baseline_terrain(gif_path: Path) -> TerrainMapProtocol:
    """Return an in-memory terrain the whole sweep can seed on.

    Open ground everywhere except the ferry scenario's own tile, which
    has to be WATER: the scenario floats a ferry there and the seed
    validator refuses to start a session whose furniture is on the wrong
    surface. On a real run the field GIF supplies that water; here this
    does.

    Args:
        gif_path: Ignored.

    Returns:
        The terrain map every baseline scenario can seed on.
    """
    del gif_path
    return InMemoryTerrainMap(terrain_data={FERRY_TILE: "W"})


@contextmanager
def baseline_field(fake_fs: FakeFileSystem) -> Generator[None, None, None]:
    """Give the sim a field GIF and the baseline terrain for the block.

    Args:
        fake_fs: The installed fake file system the GIF is written into.

    Yields:
        Nothing; the hooks are restored when the block ends.
    """
    fake_fs.write_text(data_directory() / SIM_FIELD, "fake-gif-bytes")
    real_terrain = _test_hooks.load_terrain_map
    _test_hooks.load_terrain_map = _load_baseline_terrain
    try:
        yield
    finally:
        _test_hooks.load_terrain_map = real_terrain


__all__ = ["FERRY_TILE", "baseline_field"]
