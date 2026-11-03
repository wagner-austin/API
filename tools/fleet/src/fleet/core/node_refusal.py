"""A dispatch verdict, and the refusals of a node that is the wrong machine whatever its load.

Split out of :mod:`fleet.core.capacity` when the battery refusal (MCPs board
task 3268afdf) took that module past the 600-line ceiling. These are the
refusals :func:`fleet.core.capacity.assess` gives before it weighs any load:
a node that is not the one a project names, one that checks only its pinned
projects, one whose disk cannot hold a staged tree, and one running on its
battery. No load falling changes any of them, which is why they are one
concern and come first.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig


class DispatchVerdict(TypedDict):
    """What one node would do with one project, right now.

    Attributes:
        workers: Workers the node would grant, or zero when it would refuse.
        code: The error code a refusal carries, or None when it would accept.
            Carried rather than re-derived so the raising wrapper and the
            aggregating one cannot classify the same refusal differently.
        reason: Why it would refuse, or an empty string when it would accept.
    """

    workers: int
    code: FleetErrorCode | None
    reason: str


def wrong_node(node: NodeConfig, project: ProjectConfig) -> DispatchVerdict | None:
    """The refusal of a node that is the wrong one whatever its load.

    Args:
        node: The node's declaration.
        project: The work being dispatched.

    Returns:
        The refusal :func:`fleet.core.capacity.assess` gives first for a pin
        or a node that checks only its pinned projects, in that order; None
        when the node may be weighed.
    """
    pinned = project["node"]
    if pinned is not None and pinned != node["host"]:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_NOT_PINNED,
            reason=(
                f"{node['host']} is not {pinned}: the project runs on {pinned} alone, because "
                "it reads what only that host holds, so the answer is that node and never "
                "this one."
            ),
        )
    if pinned is None and node["budget"]["pinned_checks_only"]:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_NOT_PINNED,
            reason=(
                f"{node['host']} checks only the projects pinned to it: its own work leaves a "
                "check no margin under the 300 s budget, so the answer is any other node "
                "this project's tags fit."
            ),
        )
    return None


def machine_refusal(node: NodeConfig, state: NodeState) -> DispatchVerdict | None:
    """The refusal of a node whose machine cannot take a check, however light its load.

    Args:
        node: The node's declaration.
        state: What it reported when last probed.

    Returns:
        The refusal :func:`fleet.core.capacity.assess` gives after the tags
        for a disk short of the staged trees' reservation, then for a node
        running on its battery (MCPs board task 3268afdf); None when neither
        holds.
    """
    if state["free_disk_gb"] < node["budget"]["max_disk_gb"]:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_DISK_EXHAUSTED,
            reason=(
                f"{node['host']} has {state['free_disk_gb']:.0f} GB free and this workspace "
                f"reserves {node['budget']['max_disk_gb']:.0f} GB for staged trees. The first "
                "dispatch to a node is a cold stage of a whole monorepo."
            ),
        )
    if state["on_battery"]:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_ON_BATTERY,
            reason=(
                f"{node['host']} is running on its battery: its probe read a power supply "
                "discharging, and a check placed here drains it. It takes the next job once it "
                "is back on mains."
            ),
        )
    return None


__all__ = ["DispatchVerdict", "machine_refusal", "wrong_node"]
