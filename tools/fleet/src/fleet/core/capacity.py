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


class Unassessed(TypedDict):
    """A node that produced no verdict, and whether it was asked for one.

    THE ``asked`` FLAG IS THE WHOLE TYPE. Both kinds of node are equally
    unusable for this dispatch, so the obvious shape is one list of excuses --
    and that shape loses the only fact that tells the reader where to go. A
    node that was asked and stayed silent is a tailnet problem; a node nobody
    asked is a line in ``fleet.json``. Merging them sends half of every
    refusal's readers to the wrong file.

    Attributes:
        name: The node's workspace name.
        reason: Why it produced no verdict, in its own words where it had
            any.
        asked: Whether an ssh probe was actually made. False for a node the
            workspace declares disabled, which costs nothing precisely
            because nothing was sent.
    """

    name: str
    reason: str
    asked: bool


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


def _nothing_fits_code(
    candidates: tuple[tuple[str, NodeConfig, NodeState], ...],
    unassessed: tuple[Unassessed, ...],
) -> FleetErrorCode:
    """Classify a fleet-wide refusal by what actually happened.

    Three answers, because they send a reader to three different places: the
    tailnet, ``fleet.json``, or the clock. Order matters -- a node that was
    asked and stayed silent outranks one nobody asked, because it is the only
    one of the three with something to investigate.

    Args:
        candidates: Nodes that answered and were weighed.
        unassessed: Nodes that produced no verdict.

    Returns:
        The code the refusal carries.
    """
    if not candidates and any(entry["asked"] for entry in unassessed):
        return FleetErrorCode.NODE_UNREACHABLE
    if not candidates and unassessed:
        return FleetErrorCode.NODE_DISABLED
    return FleetErrorCode.NODE_MEMORY_EXHAUSTED


def first_fit(
    candidates: tuple[tuple[str, NodeConfig, NodeState], ...],
    project: ProjectConfig,
    *,
    unassessed: tuple[Unassessed, ...] = (),
) -> tuple[str, int]:
    """Choose the node that affords this project the most workers.

    Most workers rather than first that fits, because the fleet's nodes differ
    by more than a factor of two in free memory and a dispatch landing on the
    smallest one that technically qualifies wastes the rest.

    Ties keep the earlier candidate, so a workspace's node order is a
    tie-break a person can control rather than a detail of iteration.

    A NODE THAT COULD NOT BE ASSESSED IS A REFUSAL, NOT AN ABORT, and that is
    the whole reason ``unassessed`` exists. Two of this fleet's three nodes
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
        unassessed: Every node that produced no verdict, each carrying
            whether it was asked for one. Never chosen; carried so the
            refusal names them and can say which kind of nothing happened.

    Returns:
        The chosen node's name and its worker count.

    Raises:
        AppError: With one of three codes, chosen by
            :func:`_nothing_fits_code` and all carrying EVERY node's own
            refusal rather than the first:

            ``NODE_UNREACHABLE`` -- nodes were asked and none answered. The
            fleet is off; look at the tailnet.

            ``NODE_DISABLED`` -- nothing was asked, because every node this
            workspace declares is switched off in it. Nothing failed; look at
            ``fleet.json``.

            ``NODE_MEMORY_EXHAUSTED`` -- nodes answered and all refused, or
            the workspace declares no nodes at all, which is a configuration
            fault rather than a fleet that is down.

            The three are the point, not a detail. A single code would send
            two thirds of its readers to the wrong file. It is the same
            distinction ``refused`` draws against ``failed`` one layer up.
    """
    best_name = ""
    best_workers = 0
    refusals: list[str] = [f"{entry['name']}: {entry['reason']}" for entry in unassessed]
    for name, node, state in candidates:
        verdict = assess(node, state, project)
        if verdict["code"] is not None:
            refusals.append(f"{name}: {verdict['reason']}")
            continue
        if verdict["workers"] > best_workers:
            best_name, best_workers = name, verdict["workers"]
    if best_workers == 0:
        raise AppError(
            _nothing_fits_code(candidates, unassessed),
            "no node can take this dispatch right now. " + " | ".join(refusals),
        )
    return best_name, best_workers


__all__ = ["DispatchVerdict", "Unassessed", "assess", "first_fit", "plan_dispatch"]
