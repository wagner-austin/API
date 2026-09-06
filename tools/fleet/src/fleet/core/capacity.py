"""Whether a node can take a dispatch right now, and how many workers it gets.

THE LOCAL ANALOGUE OF ``hpc3.core.gpu_supply``, and it exists for the same
measured reason. There, ``sbatch --test-only`` answered "would this be
ADMITTED" and a job queued behind an exhausted GPU model was admitted and then
sat for five hours. Here the equivalent mistake is cheaper to make and worse:
a node will happily accept any number of test workers and then thrash. On
2026-09-04 two overlapping suites on one box held 66 processes and 77.9 GB of
commit while doing no work at all, and nothing had refused either of them.

SO THE REFUSAL IS THE PRODUCT. Both public functions raise or return a worker
count. Neither hands back a "would_run: false" object, because a caller given
one treats it as advice and dispatches anyway -- which is exactly how the
estimated start that said "nine days" got read as noise.

WHY THERE IS A PURE ASSESSOR UNDERNEATH THEM. :func:`assess` returns a verdict
instead of raising, and it exists because :func:`first_fit` has to weigh
several nodes and report why each declined. Written the obvious way that would
mean catching the refusal of one node to carry on to the next -- softening a
failure to recover from it, which is the thing this codebase does not do. A
pure verdict makes the aggregation ordinary code and leaves both entry points
raising. It is not a "check then act" API for callers: nothing outside this
module reaches for it, and both wrappers raise on the same condition.

WHAT NONE OF IT DOES is pick a node for a reason other than capacity. Some
work wants a particular card, and this module is never the thing that knows
why.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.budget import admissible_workers
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


def assess(node: NodeConfig, state: NodeState, project: ProjectConfig) -> DispatchVerdict:
    """Weigh one node against one project without raising.

    The project's ``worker_ram_gb`` overrides the node's, because what a
    worker costs is a property of what the suite imports rather than of the
    machine. The node's figure describes its default tenant; the project's
    describes this one.

    Checks run cheapest-consequence first: concurrency, then disk, then
    memory. Memory is last because its message is the most specific and a
    reader should see it rather than a disk complaint that happens to also be
    true.

    Args:
        node: The node's declaration.
        state: What it reported when last probed.
        project: The work being dispatched.

    Returns:
        The verdict. ``workers`` is zero exactly when ``code`` is set.
    """
    if state["live_runs"] >= node["budget"]["max_concurrent_runs"]:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_OWNER_RESERVED,
            reason=(
                f"{node['host']} already holds {state['live_runs']} fleet run(s), its declared "
                f"limit of {node['budget']['max_concurrent_runs']}. Memory is checked against "
                "a snapshot, so this second bound exists because three dispatches that each "
                "fit alone do not fit together."
            ),
        )
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
    workers = admissible_workers(
        {**node["budget"], "worker_ram_gb": project["worker_ram_gb"]},
        logical_cores=node["logical_cores"],
        free_ram_gb=state["free_ram_gb"],
    )
    if workers <= 0:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_OWNER_RESERVED,
            reason=(
                f"{node['host']} has {state['free_ram_gb']:.1f} GB free against a reservation "
                f"of {node['budget']['reserved_ram_gb']:.1f} GB for whoever is using it, and "
                f"{node['logical_cores']} cores against {node['budget']['reserved_cores']} "
                "reserved. Nothing is left for a dispatch; somebody is on this machine."
            ),
        )
    if workers < project["minimum_workers"]:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_MEMORY_EXHAUSTED,
            reason=(
                f"{node['host']} affords {workers} worker(s) for a suite that declares a "
                f"minimum of {project['minimum_workers']}: {state['free_ram_gb']:.1f} GB "
                f"free, {project['worker_ram_gb']:.1f} GB per worker, "
                f"{node['budget']['reserved_ram_gb']:.1f} GB reserved for the node's owner. "
                "Dispatching anyway would run a suite at a fraction of its workers until its "
                "own lease expired underneath it."
            ),
        )
    return DispatchVerdict(workers=workers, code=None, reason="")


def plan_dispatch(node: NodeConfig, state: NodeState, project: ProjectConfig) -> int:
    """Decide how many workers one named node may give this project, or refuse.

    Args:
        node: The node's declaration.
        state: What it reported when last probed.
        project: The work being dispatched.

    Returns:
        Workers to grant, never fewer than the project's minimum.

    Raises:
        AppError: With ``NODE_OWNER_RESERVED`` when nothing is left after the
            owner's reservation or the node is at its concurrency limit,
            ``NODE_DISK_EXHAUSTED`` when the staged tree would not fit, or
            ``NODE_MEMORY_EXHAUSTED`` when the node affords fewer workers
            than the project can use. Distinct codes because the fixes
            differ: wait, clean up, or use a bigger node.
    """
    verdict = assess(node, state, project)
    if verdict["code"] is not None:
        raise AppError(verdict["code"], verdict["reason"])
    return verdict["workers"]


def first_fit(
    candidates: tuple[tuple[str, NodeConfig, NodeState], ...],
    project: ProjectConfig,
    *,
    unassessable: tuple[tuple[str, str], ...] = (),
) -> tuple[str, int]:
    """Choose the node that affords this project the most workers.

    Most workers rather than first that fits, because the fleet's nodes differ
    by more than a factor of two in free memory and a dispatch landing on the
    smallest one that technically qualifies wastes the rest.

    Ties keep the earlier candidate, so a workspace's node order is a
    tie-break a person can control rather than a detail of iteration.

    A NODE THAT COULD NOT BE ASSESSED IS A REFUSAL, NOT AN ABORT, and that is
    the whole reason ``unassessable`` exists. Two of this fleet's three nodes
    are laptops; one being asleep is the ordinary case, not a fault. Measured
    2026-09-05: the first real auto-select dispatch was refused outright
    because loki was off for a trip, while lavender had already answered and
    had room. Folding those nodes in here means a caller learns "loki is off
    AND sedona is full" in one answer, instead of learning about whichever
    node happened to be probed first.

    Args:
        candidates: ``(name, node, state)`` for every node that answered, in
            workspace order.
        project: The work being dispatched.
        unassessable: ``(name, reason)`` for every node that did not answer.
            Never chosen; carried so the refusal names them.

    Returns:
        The chosen node's name and its worker count.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` when nodes were tried and NONE of
            them answered, and with ``NODE_MEMORY_EXHAUSTED`` otherwise --
            including for a workspace that declares no nodes at all, which is
            a configuration fault rather than a fleet that is down. Both carry
            EVERY node's own refusal rather than the first.

            The two codes are the point, not a detail. "The fleet is off" and
            "the fleet is busy" send a reader to different places -- the
            tailnet, or the clock -- and a single code for both would send
            half of them to the wrong one. It is the same distinction
            ``refused`` draws against ``failed`` one layer up.
    """
    best_name = ""
    best_workers = 0
    refusals: list[str] = [f"{name}: {reason}" for name, reason in unassessable]
    for name, node, state in candidates:
        verdict = assess(node, state, project)
        if verdict["code"] is not None:
            refusals.append(f"{name}: {verdict['reason']}")
            continue
        if verdict["workers"] > best_workers:
            best_name, best_workers = name, verdict["workers"]
    if best_workers == 0:
        nothing_answered = not candidates and bool(unassessable)
        raise AppError(
            FleetErrorCode.NODE_UNREACHABLE
            if nothing_answered
            else FleetErrorCode.NODE_MEMORY_EXHAUSTED,
            "no node can take this dispatch right now. " + " | ".join(refusals),
        )
    return best_name, best_workers


__all__ = ["DispatchVerdict", "assess", "first_fit", "plan_dispatch"]
