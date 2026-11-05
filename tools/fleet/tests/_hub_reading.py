"""The hub's live records, read with the registry of the commit under test.

MCPs board task 6b6caf74. The hub cases import the fleet package of the
export being checked, so they read the registry beside it: code and registry
from one commit, as every tick runs them (:mod:`fleet.core.rolled` extracts
both from the commit ``refs/fleet/rolled`` names). Reading the rolled
commit's registry instead mixed two commits whenever the checked sha was not
the rolled one, which is every hub run between a push and its roll: fleet job
e62ad33d failed both cases at API 37f52927b because its decoder required
``worker_cores_measurement`` and the registry still rolled at 0234136ea had
none. The checkout's working tree is not read either, since it lags the ref
(MCPs board task f19d66a5: fleet job 46ccbe7f judged a checkout at 41523d503
while the ticks ran 104bfeaee); the export is the checked sha itself. The
records stay the checkout's, since the runners write them there as they run.
"""

from __future__ import annotations

import pathlib

from fleet.cli import _config
from fleet.core import rolled
from tests._registry import SHIPPED_REGISTRY


def hub_workspace(fleet_hub: pathlib.Path) -> _config.LoadedWorkspace:
    """Load the registry of the commit under test with the checkout's records.

    Args:
        fleet_hub: The hub's API checkout, as ``--fleet-hub`` names it.

    Returns:
        The workspace this commit's ticks would run with on the hub.
    """
    return _config.load_workspace(
        {
            _config.CONFIG_FLAG: str(SHIPPED_REGISTRY),
            _config.RECORDS_FLAG: str(fleet_hub / rolled.RECORDS_DIR),
        }
    )


__all__ = ["hub_workspace"]
