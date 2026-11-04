"""The hub's workspace as its ticks read it: the rolled registry beside the live records.

MCPs board task f19d66a5. The hub's ticks never read the registry in its
API checkout's working tree: :func:`fleet.core.rolled.extract_rolled_tree`
extracts the commit ``refs/fleet/rolled`` names and the agent reads that
extraction's registry. The hub cases read the working tree instead, so they
judged a registry the hub does not run. Fleet job 46ccbe7f failed both of
them on lacrymal at API 104bfeaee because the checkout was still at
41523d503, which pinned tools/fleet-execution-hub to diphtheria, while the
rolled ref, and every tick, already read 104bfeaee. The records stay the
checkout's, since the runners write them there as they run.
"""

from __future__ import annotations

import pathlib

import pytest

from fleet.cli import _config
from fleet.core import rolled


def hub_workspace(fleet_hub: pathlib.Path) -> _config.LoadedWorkspace:
    """Load the registry of the hub's rolled commit with the checkout's records.

    Args:
        fleet_hub: The hub's API checkout, as ``--fleet-hub`` names it.

    Returns:
        The workspace the hub's ticks run with.
    """
    tree = rolled.extract_rolled_tree(fleet_hub)
    if isinstance(tree, str):
        pytest.fail(f"the hub's rolled tree could not be read: {tree}")
    return _config.load_workspace(
        {
            _config.CONFIG_FLAG: tree["config"],
            _config.RECORDS_FLAG: str(fleet_hub / rolled.RECORDS_DIR),
        }
    )


__all__ = ["hub_workspace"]
