"""Every enabled node's stage root, read from the hub against its records.

MCPs board task 3d9eec0d. A run's directory is on its node from the moment
its lease is taken until a retire removes it (:mod:`fleet.core.retire_owed`).
Measured 2026-10-18T18:11Z, the seven nodes held 106 run ids no retire would
ever reach: refused while
staging, orphaned when the hub moved, or left by a retire that failed. This
case reads, from the hub, every enabled node's stage root over the hub's own
ssh and the runners' records in the hub's API checkout, and fails on any
run's directory, staging directory or root script whose run is neither live
(a held lease, a running ledger row) nor still owed a retire. A node that does
not answer fails it too, since its stage root cannot be read.

``tools/fleet-execution-hub`` runs it, pinned to the hub, with
``--fleet-hub`` naming the checkout the hub's ticks run with.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from typing import Final

import pytest
from platform_core.errors import AppError

from fleet.cli import _config
from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.core import names, remote, retire_owed, rolled
from tests._host import READING_PROPERTY

#: The root scripts a run leaves under its stage root, by the prefix of each
#: stem :func:`fleet.core.names.root_script_stems` names.
_SCRIPT_PREFIXES: Final[tuple[str, ...]] = (
    names.make_directory_stem(""),
    names.stop_stem(""),
    names.retire_stem(""),
)

#: One stage-root entry that belongs to a run: an optional root-script prefix,
#: the run id ending in its start second, then the staging suffix and a
#: script's extension, each optional.
_RUN_ENTRY: Final = re.compile(
    "^(?:"
    + "|".join(re.escape(prefix) for prefix in _SCRIPT_PREFIXES)
    + r")?(?P<run>.+-\d{10})(?:"
    + re.escape(names.STAGE_SUFFIX)
    + r")?(?:\.ps1|\.sh)?$"
)


def _listing(node: NodeConfig) -> tuple[str, ...]:
    """Every entry of a node's stage root, by name, read over the hub's ssh.

    Args:
        node: The node.

    Returns:
        The names.

    Raises:
        AppError: When the node did not answer or the listing failed there.
    """
    root = node["stage_root"]
    argv = (
        ("powershell", "-NoProfile", "-Command", f"Get-ChildItem -Name '{root}'")
        if node["platform"] is NodePlatform.WINDOWS
        else ("ls", "-1", root)
    )
    return tuple(remote.run_ssh(node["host"], argv).split())


def run_entries(entries: tuple[str, ...]) -> dict[str, list[str]]:
    """The stage-root entries that belong to a run, by run id.

    Args:
        entries: A stage root's entry names.

    Returns:
        Each run id with its entries; a companion and its scripts carry no
        start second and are left out.
    """
    runs: dict[str, list[str]] = {}
    for entry in entries:
        matched = _RUN_ENTRY.match(entry)
        if matched is not None:
            runs.setdefault(matched.group("run"), []).append(entry)
    return runs


def test_a_run_s_entries_are_told_from_a_companion_s() -> None:
    run = "MCPs-mcp-shared-loki-1792134587"
    owned = [
        run,
        f"{run}.stage",
        f"mkdir-{run}.ps1",
        f"mkdir-{run}.stage.ps1",
        f"stop-{run}.sh",
        f"retire-{run}.sh",
    ]
    companions = [
        "MCPs",
        "MCPs.stage",
        "mkdir-MCPs.stage.sh",
        "reset-MCPs.sh",
        "logs",
        "fleet-capacity-loki.ps1",
    ]

    assert run_entries((*owned, *companions)) == {run: owned}


@pytest.mark.host_fleet_hub
def test_every_run_directory_on_every_node_is_live_or_owed_its_retire(
    fleet_hub: pathlib.Path, record_property: Callable[[str, str], None]
) -> None:
    loaded = _config.load_workspace(
        {
            _config.CONFIG_FLAG: str(fleet_hub / rolled.REGISTRY_DIRECTORY),
            _config.RECORDS_FLAG: str(fleet_hub / rolled.RECORDS_DIR),
        }
    )
    live = retire_owed.live_runs(loaded.leases, loaded.ledger)
    left: list[str] = []
    unanswered: list[str] = []
    readings: list[str] = []
    for alias, node in sorted(loaded.workspace["nodes"].items()):
        if not node["enabled"]:
            continue
        try:
            runs = run_entries(_listing(node))
        except AppError as refusal:
            unanswered.append(f"{alias}: {refusal.code}: {refusal.message}")
            continue
        owed = set(retire_owed.owed_on(loaded.retires, alias=alias))
        held = sorted(run for run in runs if run in live or run in owed)
        left.extend(
            f"{alias}: {run} ({', '.join(sorted(runs[run]))})"
            for run in sorted(runs)
            if run not in live and run not in owed
        )
        readings.append(f"{alias} {len(runs)} run(s), {len(held)} live or owed")

    record_property(READING_PROPERTY, "; ".join(readings))
    assert unanswered == [], f"stage roots not read: {unanswered}"
    assert left == [], f"run directories no live run or owed retire holds: {left}"
