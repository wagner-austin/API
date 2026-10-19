"""A node's CI cap, reservation and smallest job fit its memory (MCPs board task 5d6e57e7).

On 2026-09-29 lavender's runners.slice was allowed 18 GB of the VM that
lavender-wsl runs in while the registry reserved 12 GB of it, on a node
declaring 25.4 GB, so the testdb lane could never dispatch while CI held its
cap. The first case reads the two real declarations, so a later edit to
either file that breaks the sum fails here; the others pin the rule on that
measured shape and on the nodes it does not reach.

Since lavender was reimaged to Ubuntu on 2026-10-19 (MCPs board task
1e9ab738) its runners run on the machine itself, and a runners.json host
joins the node of its own name. lavender dispatches nothing until it is
provisioned, so the measured shape is a real Linux node's declaration
renamed lavender, with the guest's 25.4 GB.
"""

from __future__ import annotations

from pathlib import Path

from fleet.cli.runners import load_runner_spec
from fleet.contracts.node import NodeConfig
from fleet.contracts.runners import RunnerSpec
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace, require_node
from fleet.core.memory_budget import budget_gaps, slice_cap_gb, smallest_job_gb
from tests._registry import shipped_document

PACKAGE = Path(__file__).resolve().parents[1]


def _workspace() -> FleetWorkspace:
    """The real registry.

    Returns:
        The decoded workspace.
    """
    return decode_fleet_workspace(shipped_document())


def _runners() -> RunnerSpec:
    """The real runners.json.

    Returns:
        The decoded roster.
    """
    return load_runner_spec(str(PACKAGE / "runners.json"))


def _lavender(workspace: FleetWorkspace, reserved_gb: float) -> NodeConfig:
    """lacrymal's declaration as lavender, with the guest's memory and a reservation.

    Args:
        workspace: The real workspace.
        reserved_gb: The reservation to declare.

    Returns:
        The node.
    """
    node = require_node(workspace, "lacrymal").copy()
    budget = node["budget"].copy()
    budget["reserved_ram_gb"] = reserved_gb
    node["budget"] = budget
    node["ram_gb"] = 25.4
    return node


def _with_lavender(workspace: FleetWorkspace, reserved_gb: float) -> FleetWorkspace:
    """The workspace with :func:`_lavender` declared as the node lavender.

    Args:
        workspace: The real workspace.
        reserved_gb: lavender's reservation.

    Returns:
        A copy differing only there.
    """
    nodes = dict(workspace["nodes"])
    nodes["lavender"] = _lavender(workspace, reserved_gb)
    changed = workspace.copy()
    changed["nodes"] = nodes
    return changed


def test_the_real_declarations_add_up_for_every_node() -> None:
    workspace = _workspace()
    runners = _runners()

    assert budget_gaps(workspace, runners) == ()
    assert [host["name"] for host in runners["hosts"]] == ["lavender"]
    assert slice_cap_gb("lavender", runners) == 18


def test_the_measured_split_of_the_afternoon_is_refused_naming_all_four_numbers() -> None:
    """runners.slice at 18 GB beside the old 12 GB reservation."""
    workspace = _with_lavender(_workspace(), 12.0)

    assert budget_gaps(workspace, _runners()) == (
        "lavender: CI slice 18 GB + reservation 12.0 GB + smallest job 0.25 GB is 30.25 GB, "
        "past its 25.4 GB, so while CI holds its cap the node can never dispatch; lower the "
        "reservation in registry/machines/lavender.json or the slice in runners.json",
    )


def test_a_reservation_that_leaves_room_for_the_slice_and_a_job_passes() -> None:
    """The 6 GB the guest reserved after the afternoon fits beside the slice."""
    assert budget_gaps(_with_lavender(_workspace(), 6.0), _runners()) == ()


def test_a_machine_running_no_runners_here_has_no_slice() -> None:
    runners = _runners()

    assert slice_cap_gb("diphtheria", runners) == 0
    assert slice_cap_gb("lavender-wsl", runners) == 0


def test_a_node_that_can_serve_no_project_needs_no_room_for_one() -> None:
    workspace = _workspace()
    lane = _lavender(workspace, 6.0)

    assert smallest_job_gb(lane, ()) == 0.0
    assert smallest_job_gb(lane, tuple(workspace["projects"].values())) == 0.25
