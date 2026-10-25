"""fleet-nodes says, per node, whether its runners can claim and why not (MCPs 7e467416).

On 2026-09-29 thirty checks sat queued with one running, and why each node
took nothing was only in its runner's daily log on the hub. These drive
``fleet-nodes`` through the fake ssh runner and assert the lines it prints:
the runner's own pre-claim verdict on the same probe, the reason a retired
node gives, and what a node that does not answer says.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONObject

from fleet.cli import _config, nodes
from fleet.contracts.ledger import decode_ledger_entry
from fleet.core import _test_hooks, records
from tests._registry import write_registry
from tests._roster_fixtures import dispatch_of
from tests.conftest import (
    CALM,
    DEMO_PROJECT,
    PROBE_OK,
    FakeRun,
    failed,
    ok,
    workspace_document,
)

#: sedona's reading at 10:51Z scaled to the fixture: under the reservation.
STARVED = f"free_ram_gb=2.9\nfree_disk_gb=860.0\nreserved_cores=0\n{CALM}"


def _config_with(tmp_path: pathlib.Path, document: JSONObject) -> pathlib.Path:
    """Write a workspace document where fleet-nodes reads it.

    Args:
        tmp_path: pytest's per-test temporary directory.
        document: The document.

    Returns:
        The written registry directory.
    """
    return write_registry(tmp_path / "registry", document)


def _lines(config: pathlib.Path) -> list[str]:
    """What fleet-nodes prints, one line per node.

    Args:
        config: The workspace document.

    Returns:
        The lines.
    """
    return nodes.describe_fleet(_config.load_workspace({_config.CONFIG_FLAG: str(config)}))[0]


def test_a_node_with_room_says_its_lane_can_claim(tmp_path: pathlib.Path) -> None:
    _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

    assert _lines(_config_with(tmp_path, workspace_document())) == [
        "lavender: lavender: cpu-only, 27.0/32.0 GB RAM free, 860 GB disk free, 0 live run(s) "
        "holding 0 worker(s); node lane can claim"
    ]


def test_a_node_under_its_reservation_names_the_reservation(tmp_path: pathlib.Path) -> None:
    """sedona's cause that morning."""
    _test_hooks.run = FakeRun([ok(""), ok(STARVED)])

    assert _lines(_config_with(tmp_path, workspace_document())) == [
        "lavender: lavender: cpu-only, 2.9/32.0 GB RAM free, 860 GB disk free, 0 live run(s) "
        "holding 0 worker(s); node lane claims nothing: NODE_OWNER_RESERVED: lavender has 2.9 "
        "GB free against a reservation of 4.0 GB for whoever is using it, and 16 cores against "
        "2 reserved. Nothing is left for a dispatch; somebody is on this machine."
    ]


def test_a_node_whose_live_runs_fill_it_names_the_held_runs(tmp_path: pathlib.Path) -> None:
    """Two runs of 7 workers hold lavender's 14 spare cores (MCPs 939ec5c7)."""
    config = _config_with(tmp_path, workspace_document())
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config)})
    for run in ("run-1", "run-2"):
        records.append_ledger(
            loaded.ledger,
            decode_ledger_entry(
                {
                    "run_id": run,
                    "node": "lavender",
                    "host": "lavender",
                    "project": DEMO_PROJECT,
                    "agent": "opus-fleet-0904",
                    "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
                    "started_unix": 100,
                    "ended_unix": 100,
                    "outcome": "running",
                    "exit_code": -1,
                    "workers": 7,
                    "detail": "",
                }
            ),
        )
    _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

    (line,) = _lines(config)

    assert line.endswith(
        "2 live run(s) holding 14 worker(s); node lane claims nothing: NODE_OWNER_RESERVED: "
        "lavender has 27.0 GB free against a reservation of 4.0 GB for whoever is using it, and "
        "16 cores against 2 reserved, and its 2 live fleet run(s) hold 14 worker(s) and 15.4 GB. "
        "Nothing is left for a dispatch; its own fleet runs hold the rest, so it takes the next "
        "job when one of them ends."
    )


def test_a_retired_node_points_at_its_reason(tmp_path: pathlib.Path) -> None:
    document = workspace_document()
    dispatch_of(document, "lavender")["held"] = "reinstalling as NixOS"
    _test_hooks.run = FakeRun([])

    assert _lines(_config_with(tmp_path, document)) == [
        "lavender: DISABLED -- declared off in this workspace, not probed; "
        "its machine file gives the reason it is held"
    ]


def test_an_unreachable_node_says_so_with_the_transport_s_words(tmp_path: pathlib.Path) -> None:
    _test_hooks.run = FakeRun([failed(255, "Connection timed out during banner exchange")])

    (line,) = _lines(_config_with(tmp_path, workspace_document()))

    assert line.startswith("lavender: UNREACHABLE -- ")
    assert line.endswith("Connection timed out during banner exchange")
