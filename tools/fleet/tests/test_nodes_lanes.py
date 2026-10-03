"""fleet-nodes says, per node, whether its runners can claim and why not (MCPs 7e467416).

On 2026-09-29 thirty checks sat queued with one running, and why each node
took nothing was only in its runner's daily log on the hub. These drive
``fleet-nodes`` through the fake ssh runner and assert the lines it prints:
the runner's own pre-claim verdict for each lane on the same probe, the
reason a retired node gives, and what a WSL node's host reports when the
node does not answer.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import _config, nodes
from fleet.contracts.ledger import decode_ledger_entry
from fleet.core import _test_hooks, records
from tests.conftest import DEMO_PROJECT, PROBE_OK, FakeRun, failed, ok, workspace_document

#: sedona's reading at 10:51Z scaled to the fixture: under the reservation.
STARVED = "free_ram_gb=2.9\nfree_disk_gb=860.0\n"

#: lavender's listing, UTF-16LE as wsl.exe prints it (test_host_report.py).
LAVENDER_LIST = "  NAME      STATE           VERSION\r\n* Ubuntu    Running         2\r\n".encode(
    "utf-16-le"
).decode("utf-8", errors="replace")


def _config_with(tmp_path: pathlib.Path, document: JSONObject) -> pathlib.Path:
    """Write a workspace document where fleet-nodes reads it.

    Args:
        tmp_path: pytest's per-test temporary directory.
        document: The document.

    Returns:
        Its path.
    """
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    return path


def _nodes(document: JSONObject) -> JSONObject:
    """The fixture workspace's node map, to change in place.

    Args:
        document: A workspace document from the shared fixtures.

    Returns:
        Its ``nodes`` object.
    """
    declared = document["nodes"]
    assert isinstance(declared, dict)
    return declared


def _lavender(document: JSONObject) -> JSONObject:
    """The fixture workspace's one node, to change in place.

    Args:
        document: A workspace document from the shared fixtures.

    Returns:
        lavender's declaration inside it.
    """
    lavender = _nodes(document)["lavender"]
    assert isinstance(lavender, dict)
    return lavender


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
    """sedona's cause that morning, and with an elevated runner both lanes
    say it, since they share the node."""
    document = workspace_document()
    _lavender(document)["elevated"] = True
    _test_hooks.run = FakeRun([ok(""), ok(STARVED)])

    reserved = (
        "NODE_OWNER_RESERVED: lavender has 2.9 GB free against a reservation of 4.0 GB for "
        "whoever is using it, and 16 cores against 2 reserved. Nothing is left for a dispatch; "
        "somebody is on this machine."
    )
    assert _lines(_config_with(tmp_path, document)) == [
        "lavender: lavender: cpu-only, 2.9/32.0 GB RAM free, 860 GB disk free, 0 live run(s) "
        f"holding 0 worker(s); node lane claims nothing: {reserved}; "
        f"elevated lane claims nothing: {reserved}"
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
    _lavender(document)["enabled"] = False
    document["not_dispatchable"] = {"lavender": "its WSL VM holds the memory this lane measures"}
    _test_hooks.run = FakeRun([])

    assert _lines(_config_with(tmp_path, document)) == [
        "lavender: DISABLED -- declared off in this workspace, not probed; "
        "not_dispatchable gives the reason"
    ]


def test_an_unreachable_wsl_node_adds_what_its_host_reports(tmp_path: pathlib.Path) -> None:
    document = workspace_document()
    guest: JSONObject = {**_lavender(document), "host": "lavender-wsl", "platform": "linux"}
    guest["wsl_host"] = "lavender"
    _nodes(document)["lavender-wsl"] = guest
    _test_hooks.run = FakeRun(
        [
            ok(""),
            ok(PROBE_OK),
            failed(255, "Connection timed out during banner exchange"),
            ok(""),
            ok("free_ram_gb=0.3\nfree_disk_gb=612.4\n"),
            ok(LAVENDER_LIST),
        ]
    )

    lines = _lines(_config_with(tmp_path, document))

    assert lines[1].startswith("lavender-wsl: UNREACHABLE -- ")
    assert lines[1].endswith(
        "; its host lavender answers with 0.3 GB of 32.0 GB RAM and 612.4 GB disk free; "
        "wsl: Ubuntu Running"
    )
