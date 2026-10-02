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

from collections.abc import Sequence

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.budget import admissible_workers
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.contracts.tags import NodeTag, missing_tags, node_tags


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


#: How close to its ``memory.high`` a ``runners.slice`` must be to count as
#: at it: the kernel throttles the slice there, so its current hovers just
#: under the line rather than on it (measured 17179713536 of 17179869184).
SLICE_AT_HIGH = 0.98


def _who_holds_it(state: NodeState) -> str:
    """Name what holds a node's memory in a reservation refusal.

    Args:
        state: What the node reported.

    Returns:
        The node's CI runners, with the slice's numbers, when its probe read
        a ``runners.slice`` at its ``memory.high`` (MCPs board task
        5d6e57e7); otherwise the owner, as before.
    """
    ci_slice = state["ci_slice"]
    if ci_slice is None or ci_slice["current_gb"] < SLICE_AT_HIGH * ci_slice["high_gb"]:
        return "somebody is on this machine"
    return (
        f"its CI runners hold it: runners.slice is at {ci_slice['current_gb']:.1f} GB of its "
        f"{ci_slice['high_gb']:.1f} GB memory.high, so the lane waits for a CI job to end"
    )


def assess(
    node: NodeConfig, state: NodeState, project: ProjectConfig, carried: frozenset[NodeTag]
) -> DispatchVerdict:
    """Weigh one node against one project without raising.

    The project's ``worker_ram_gb`` overrides the node's, because what a
    worker costs is a property of what the suite imports rather than of the
    machine. The node's figure describes its default tenant; the project's
    describes this one.

    Checks run cheapest-consequence first: the project's required tags, then
    disk, then cores and memory. Tags come first because a node of the wrong
    kind is refused however idle it is, and a reader told "sedona is full"
    about a linux-only suite would wait for a node that can never take it.
    Memory is last because its message is the most specific and a reader
    should see it rather than a disk complaint that happens to also be true.

    HOW MANY RUNS A NODE HOLDS IS LIVE, NOT DECLARED (MCPs board task
    939ec5c7). Every node used to declare one concurrent run, and on
    2026-10-02 thirteen jobs queued behind four running while live probes
    showed memory a second run could have used. Now the node's live runs are
    charged first: the workers they were granted come off its cores and the
    memory those workers may hold comes off its free memory, whether or not
    they have spawned yet (:class:`fleet.contracts.node.LiveLoad`), and what
    is left is granted. One grant is capped at :func:`job_ceiling`, so the
    first job leaves room for a second instead of taking the whole node.

    Args:
        node: The node's declaration.
        state: What it reported when last probed.
        project: The work being dispatched.
        carried: The tags the runner judging it carries: the node's declared
            tags, plus on a node runner the tool tags this tick's toolchain
            probe found (:func:`fleet.contracts.tags.tool_tags`).

    Returns:
        The verdict. ``workers`` is zero exactly when ``code`` is set.
    """
    missing = missing_tags(carried, project["required_tags"])
    if missing:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_LACKS_TAG,
            reason=(
                f"{node['host']} lacks {', '.join(missing)}: the project requires "
                f"{', '.join(project['required_tags'])} and this node carries "
                f"{', '.join(sorted(carried))}. Tags come from the node's declaration and "
                "from the tools its toolchain probe found, so the answer is another node, or "
                "this one once the missing tool is installed."
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
    live = state["live"]
    workers = min(
        admissible_workers(
            {**node["budget"], "worker_ram_gb": project["worker_ram_gb"]},
            logical_cores=node["logical_cores"] - live["workers"],
            free_ram_gb=state["free_ram_gb"] - live["ram_gb"],
        ),
        job_ceiling(node, project),
    )
    if workers <= 0:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_OWNER_RESERVED,
            reason=(
                f"{node['host']} has {state['free_ram_gb']:.1f} GB free against a reservation "
                f"of {node['budget']['reserved_ram_gb']:.1f} GB for whoever is using it, and "
                f"{node['logical_cores']} cores against {node['budget']['reserved_cores']} "
                f"reserved{_live_clause(state)}. Nothing is left for a dispatch; "
                f"{_who_holds_it(state)}."
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
                f"{node['budget']['reserved_ram_gb']:.1f} GB reserved for the node's owner"
                f"{_live_clause(state)}. Dispatching anyway would run a suite at a fraction of "
                "its workers until its own lease expired underneath it."
            ),
        )
    return DispatchVerdict(workers=workers, code=None, reason="")


def job_ceiling(node: NodeConfig, project: ProjectConfig) -> int:
    """The most workers one dispatch may be granted on a node.

    Half the cores the node's owner leaves, rounded up, and never below the
    project's own minimum: a suite spends real time in serial install and
    build phases and xdist scales below linearly, so two half-width runs keep
    a node busier than one full-width one, and the cores pool in
    :func:`assess` still bounds the sum.

    Args:
        node: The node's declaration.
        project: The work being dispatched.

    Returns:
        The ceiling, at least ``project['minimum_workers']``.
    """
    spare = node["logical_cores"] - node["budget"]["reserved_cores"]
    return max(project["minimum_workers"], -(-spare // 2))


def _live_clause(state: NodeState) -> str:
    """Name what a node's live runs hold, for a capacity refusal.

    Args:
        state: What the node reported.

    Returns:
        An empty string when no fleet run is live on the node; otherwise
        ``, and its N live fleet run(s) hold W worker(s) and X GB``.
    """
    live = state["live"]
    if live["runs"] == 0:
        return ""
    return (
        f", and its {live['runs']} live fleet run(s) hold {live['workers']} worker(s) and "
        f"{live['ram_gb']:.1f} GB"
    )


def room_for_any(
    node: NodeConfig,
    state: NodeState,
    projects: Sequence[ProjectConfig],
    tags: frozenset[NodeTag],
) -> str | None:
    """Whether a node could take SOME dispatch right now, before one is chosen.

    The node runner's gate before it claims (board task fd5cabfa): a runner
    that claimed first and asked its node second took the oldest job off
    the queue and refused it, while a live node beside it found the lane
    empty. Measured 2026-09-21T10:00:02Z: loki, asleep, claimed
    ``libs/platform_core`` 48 ms before diphtheria claimed the next row,
    and closed it ``NODE_UNREACHABLE``. So the three checks that do not
    depend on the project run first, on the probe the runner has already
    paid for, and a node that fails one claims nothing this tick. The
    project-dependent checks (its tags, its minimum workers) still run after
    the claim, on the same probe.

    THE HYPOTHETICAL JOB IS THE SMALLEST ONE THIS RUNNER COULD TAKE (MCPs board
    task 865287f3). It used to be sized at the node's own worker_ram_gb, 1.1
    GB, so on 2026-09-29 diphtheria, with 16.2 GB free against a 16.0 GB
    reservation, claimed nothing for hours while the 0.25 GB roll gate that
    would have fitted waited, and every API roll waited behind it. A lower
    bound has to be the least any claimable job needs: one worker of the
    registered project with the smallest worker_ram_gb whose required tags
    this runner carries. The per-project assess after the claim is unchanged.

    Args:
        node: The node's declaration.
        state: What it reported when probed this tick.
        projects: Every registered project.
        tags: The tags this runner claims with
            (:func:`fleet.contracts.tags.runner_tags`).

    Returns:
        None when the node could take one worker of the smallest project it
        can serve (disk for a staged tree, then cores and memory after the
        owner's reservation and the node's live runs), else the ``CODE: reason`` line
        saying why it can take nothing, worded as :func:`assess` would word
        the same refusal, or ``NODE_LACKS_TAG`` when no registered project's
        tags fit this runner at all.
    """
    servable = [
        project["worker_ram_gb"] for project in projects if set(project["required_tags"]) <= tags
    ]
    if not servable:
        carried = ", ".join(sorted(tag.value for tag in tags)) or "no tags"
        return (
            f"{FleetErrorCode.NODE_LACKS_TAG.value}: this runner carries {carried}, and no "
            "registered project's required tags fit it"
        )
    default_tenant = ProjectConfig(
        worker_ram_gb=min(servable),
        minimum_workers=1,
        expected_minutes=1,
        exclusive_resources=(),
        external_paths=(),
        required_tags=(),
        source=None,
    )
    verdict = assess(node, state, default_tenant, tags)
    if verdict["code"] is None:
        return None
    return f"{verdict['code'].value}: {verdict['reason']}"


def plan_dispatch(
    node: NodeConfig, state: NodeState, project: ProjectConfig, carried: frozenset[NodeTag]
) -> int:
    """Decide how many workers one named node may give this project, or refuse.

    Args:
        node: The node's declaration.
        state: What it reported when last probed.
        project: The work being dispatched.
        carried: The tags the dispatching runner carries (:func:`assess`).

    Returns:
        Workers to grant, never fewer than the project's minimum.

    Raises:
        AppError: With ``NODE_LACKS_TAG`` when the node is missing a tag the
            project requires, ``NODE_OWNER_RESERVED`` when nothing is left
            after the owner's reservation and the node's live runs,
            ``NODE_DISK_EXHAUSTED`` when the staged tree would not
            fit, or ``NODE_MEMORY_EXHAUSTED`` when the node affords fewer
            workers than the project can use. Distinct codes because the
            fixes differ: another node, wait, clean up, or a bigger node.
    """
    verdict = assess(node, state, project, carried)
    if verdict["code"] is not None:
        raise AppError(verdict["code"], verdict["reason"])
    return verdict["workers"]


def _nothing_fits_code(
    candidates: tuple[tuple[str, NodeConfig, NodeState], ...],
    unassessed: tuple[Unassessed, ...],
    refused_with: tuple[FleetErrorCode, ...],
) -> FleetErrorCode:
    """Classify a fleet-wide refusal by what actually happened.

    Four answers, because they send a reader to four different places: the
    tailnet, ``fleet.json``, the project's ``required_tags``, or the clock.
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
    if candidates and all(code is FleetErrorCode.NODE_LACKS_TAG for code in refused_with):
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
        AppError: With one of four codes, chosen by
            :func:`_nothing_fits_code` and all carrying EVERY node's own
            refusal rather than the first:

            ``NODE_UNREACHABLE`` -- nodes were asked and none answered. The
            fleet is off; look at the tailnet.

            ``NODE_DISABLED`` -- nothing was asked, because every node this
            workspace declares is switched off in it. Nothing failed; look at
            ``fleet.json``.

            ``NODE_LACKS_TAG`` -- nodes answered and every one of them is
            the wrong kind for this project's ``required_tags``. Nothing is
            full; the fleet has no node of that kind, or the declaration
            asks for one it does not have.

            ``NODE_MEMORY_EXHAUSTED`` -- nodes answered and all refused, at
            least one of them on capacity, or the workspace declares no
            nodes at all, which is a configuration fault rather than a
            fleet that is down.

            The four are the point, not a detail. A single code would send
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


__all__ = [
    "DispatchVerdict",
    "Unassessed",
    "assess",
    "first_fit",
    "job_ceiling",
    "plan_dispatch",
    "room_for_any",
]
