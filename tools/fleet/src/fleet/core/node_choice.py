"""Choosing among several nodes the one that affords a dispatch the most.

Split from :mod:`fleet.core.capacity`, which weighs ONE node against one
project (:func:`fleet.core.capacity.assess`); this module weighs every node
that answered, carries every refusal into one answer, and classifies a
fleet-wide refusal by what actually happened (MCPs board task 2ce5dc13, when
the CPU-pressure refusal took capacity past its line ceiling).

THE AGGREGATION IS ORDINARY CODE OVER PURE VERDICTS. Written over a raising
check, choosing among nodes would mean catching one node's refusal to carry
on to the next, softening a failure to recover from it; over
:func:`~fleet.core.capacity.assess`'s verdicts it is a loop.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.contracts.tags import node_tags
from fleet.core.capacity import assess


class Unassessed(TypedDict):
    """A node that produced no verdict, and whether it was asked for one.

    THE ``asked`` FLAG IS THE WHOLE TYPE. Both kinds of node are equally
    unusable for this dispatch, so the obvious shape is one list of excuses --
    and that shape loses the only fact that tells the reader where to go. A
    node that was asked and stayed silent is a tailnet problem; a node nobody
    asked is the ``enabled`` field of its ``registry/machines/<name>.json``.
    Merging them sends half of every
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


def _nothing_fits_code(
    candidates: tuple[tuple[str, NodeConfig, NodeState], ...],
    unassessed: tuple[Unassessed, ...],
    refused_with: tuple[FleetErrorCode, ...],
) -> FleetErrorCode:
    """Classify a fleet-wide refusal by what actually happened.

    Four answers, because they send a reader to four different places: the
    tailnet, a machine's ``registry/machines/<name>.json``, the project's
    ``required_tags``, or the clock.
    Order matters -- a node that was asked and stayed silent outranks one
    nobody asked, because it is the only one of the four with something to
    investigate.

    Args:
        candidates: Nodes that answered and were weighed.
        unassessed: Nodes that produced no verdict.
        refused_with: The code each weighed node refused with, in order.

    Returns:
        The code the refusal carries. ``NODE_LACKS_TAG`` only when EVERY
        weighed node refused for its tags: one node short of memory beside
        three of the wrong kind is still a capacity answer, because that one
        node will take the work later and the reader should wait for it.
    """
    if not candidates and any(entry["asked"] for entry in unassessed):
        return FleetErrorCode.NODE_UNREACHABLE
    if not candidates and unassessed:
        return FleetErrorCode.NODE_DISABLED
    if candidates and all(code is FleetErrorCode.NODE_NOT_PINNED for code in refused_with):
        return FleetErrorCode.NODE_NOT_PINNED
    wrong_node = {FleetErrorCode.NODE_LACKS_TAG, FleetErrorCode.NODE_NOT_PINNED}
    if candidates and all(code in wrong_node for code in refused_with):
        return FleetErrorCode.NODE_LACKS_TAG
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

    EACH CANDIDATE IS JUDGED ON ITS DECLARED TAGS (:func:`node_tags`). This
    is the direct dispatch a session runs, which takes no toolchain probe, so
    a project that requires a tool tag (``ffmpeg``) is refused
    ``NODE_LACKS_TAG`` here; the queue's node runners, which probe the
    toolchain every tick, are where such a project runs.

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
        AppError: With one of five codes, chosen by
            :func:`_nothing_fits_code` and all carrying EVERY node's own
            refusal rather than the first:

            ``NODE_UNREACHABLE`` -- nodes were asked and none answered. The
            fleet is off; look at the tailnet.

            ``NODE_DISABLED`` -- nothing was asked, because every node this
            workspace declares is switched off in it. Nothing failed; look at
            each machine's ``enabled`` in ``registry/machines/<name>.json``.

            ``NODE_LACKS_TAG`` -- nodes answered and every one of them is
            the wrong kind for this project's ``required_tags``, or the
            named node lacks them and every other one is not that node.
            Nothing is full; the fleet has no node of that kind, or the
            declaration asks for one it does not have.

            ``NODE_NOT_PINNED`` -- nodes answered and every one of them is
            not the node the project names, which did not answer. Look at
            that node.

            ``NODE_MEMORY_EXHAUSTED`` -- nodes answered and all refused, at
            least one of them on capacity, or the workspace declares no
            nodes at all, which is a configuration fault rather than a
            fleet that is down.

            The five are the point, not a detail. A single code would send
            most of its readers to the wrong file. It is the same
            distinction ``refused`` draws against ``failed`` one layer up.
    """
    best_name = ""
    best_workers = 0
    refusals: list[str] = [f"{entry['name']}: {entry['reason']}" for entry in unassessed]
    refused_with: list[FleetErrorCode] = []
    for name, node, state in candidates:
        verdict = assess(node, state, project, node_tags(node))
        if verdict["code"] is not None:
            refusals.append(f"{name}: {verdict['reason']}")
            refused_with.append(verdict["code"])
            continue
        if verdict["workers"] > best_workers:
            best_name, best_workers = name, verdict["workers"]
    if best_workers == 0:
        raise AppError(
            _nothing_fits_code(candidates, unassessed, tuple(refused_with)),
            "no node can take this dispatch right now. " + " | ".join(refusals),
        )
    return best_name, best_workers


__all__ = ["Unassessed", "first_fit"]
