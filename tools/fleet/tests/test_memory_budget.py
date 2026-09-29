"""A node's CI cap, reservation and smallest job fit its memory (MCPs board task 5d6e57e7).

On 2026-09-29 lavender's runners.slice was allowed 18 GB of the VM that
lavender-wsl runs in while fleet.json reserved 12 GB of it, on a node
declaring 25.4 GB, so the testdb lane could never dispatch while CI held its
cap. The first case reads the two real declarations, so a later edit to
either file that breaks the sum fails here; the others pin the rule on that
measured shape and on the nodes it does not reach.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str

from fleet.cli.runners import load_runner_spec
from fleet.contracts.runners import RunnerSpec
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace, require_node
from fleet.core.memory_budget import budget_gaps, slice_cap_gb, smallest_job_gb

PACKAGE = Path(__file__).resolve().parents[1]


def _workspace() -> FleetWorkspace:
    """The real fleet.json.

    Returns:
        The decoded workspace.
    """
    return decode_fleet_workspace(load_json_str((PACKAGE / "fleet.json").read_text("utf-8")))


def _runners() -> RunnerSpec:
    """The real runners.json.

    Returns:
        The decoded roster.
    """
    return load_runner_spec(str(PACKAGE / "runners.json"))


def _with_lane_reservation(workspace: FleetWorkspace, gb: float) -> FleetWorkspace:
    """The workspace with lavender-wsl's reservation set to ``gb``.

    Args:
        workspace: The real workspace.
        gb: The reservation to declare.

    Returns:
        A copy differing only there.
    """
    node = require_node(workspace, "lavender-wsl").copy()
    budget = node["budget"].copy()
    budget["reserved_ram_gb"] = gb
    node["budget"] = budget
    nodes = dict(workspace["nodes"])
    nodes["lavender-wsl"] = node
    changed = workspace.copy()
    changed["nodes"] = nodes
    return changed


def test_the_real_declarations_add_up_for_every_node() -> None:
    workspace = _workspace()
    runners = _runners()
    lane = require_node(workspace, "lavender-wsl")

    assert budget_gaps(workspace, runners) == ()
    assert (slice_cap_gb(lane, runners), lane["budget"]["reserved_ram_gb"], lane["ram_gb"]) == (
        18,
        6.0,
        25.4,
    )


def test_the_measured_split_of_the_afternoon_is_refused_naming_all_four_numbers() -> None:
    """runners.slice at 18 GB beside the old 12 GB reservation."""
    workspace = _with_lane_reservation(_workspace(), 12.0)

    assert budget_gaps(workspace, _runners()) == (
        "lavender-wsl: CI slice 18 GB + reservation 12.0 GB + smallest job 0.25 GB is 30.25 "
        "GB, past its 25.4 GB, so while CI holds its cap the node can never dispatch; lower "
        "the reservation in fleet.json or the slice in runners.json",
    )


def test_a_node_with_no_wsl_host_or_an_unrostered_one_has_no_slice() -> None:
    workspace = _workspace()
    runners = _runners()
    elsewhere = require_node(workspace, "lavender-wsl").copy()
    elsewhere["wsl_host"] = "sedona"

    assert slice_cap_gb(require_node(workspace, "diphtheria"), runners) == 0
    assert slice_cap_gb(elsewhere, runners) == 0


def test_a_node_that_can_serve_no_project_needs_no_room_for_one() -> None:
    lane = require_node(_workspace(), "lavender-wsl")

    assert smallest_job_gb(lane, ()) == 0.0
    assert smallest_job_gb(lane, tuple(_workspace()["projects"].values())) == 0.25
