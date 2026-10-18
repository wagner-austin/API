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
instead of raising, and it exists because
:func:`fleet.core.node_choice.first_fit` has to weigh several nodes and report
why each declined. Written the obvious way that would mean catching the
refusal of one node to carry on to the next -- softening a failure to recover
from it, which is the thing this codebase does not do. A pure verdict makes the
aggregation ordinary code and leaves both entry points raising. It is not a
"check then act" API for callers: what reaches for it aggregates verdicts or
raises on the same condition :func:`plan_dispatch` does.

WHAT NONE OF IT DOES is pick a node for a reason other than capacity. Some
work wants a particular card, and this module is never the thing that knows
why.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.budget import NodeBudget, admissible_workers
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.contracts.tags import NodeTag, missing_tags


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

#: The CPU pressure, in percent of the last minute, at which a node takes no
#: check (MCPs board task 2ce5dc13). diphtheria's day to 14:3xZ on 2026-10-16,
#: 873 one-minute readings of its load sampler: p50 6.9, p75 15.9, p90 30,
#: p95 39, max 73.7. MCPs/execution-deploy claimed there at 42 to 51 percent
#: ran 339 s of its 300 s budget, and claimed at 2 and about 10 percent ran
#: 224 s and 287 s. At 30 the node waits out 88 of those 873 minutes.
#: lavender-wsl, 16 cores, averaged 5.1 since boot (7561 s of pressure in
#: 149135 s of uptime) and read 1.6 to 16.6 over six minutes from 14:52Z
#: with its GitHub runners busy at load 8 to 17, where MCPs/execution-deploy
#: ran 274 s and 287 s, inside its budget, so the bound never holds it.
CPU_PRESSURE_BOUND_PCT = 30.0

#: The I/O pressure, in percent of the last minute, at which a node takes no
#: check (MCPs board task 814c4555). diphtheria's load sampler, 12,425
#: one-minute readings from 2026-10-10 to 2026-10-18 17:45Z: p50 10.1, p75
#: 18.6, p90 28.9, p95 37.8, max 91.3. Of its 468 checks in that span with a
#: budget reading, each run's time against its project's median, by the I/O
#: pressure it was claimed at: under 30, 380 runs at a median of 1.00 with
#: 11 over budget; 30 to 40, 18 at 0.99 with none over; 40 to 60, 27 at 1.06
#: with none over, so a disk this busy does not slow a check. None was
#: claimed at 60 or over until MCPs/transcriber-api at 71.5 percent, while a
#: 36 GB database dump and a VM install shared the disk, which ran about five
#: times its usual 91 to 266 s and was ended past its lease (fleet job
#: 347ae73b). 97 of the readings reach 60, 50 of them that afternoon.
IO_PRESSURE_BOUND_PCT = 60.0


def _project_budget(node: NodeConfig, project: ProjectConfig) -> NodeBudget:
    """A node's budget with the project's cost per worker in place of the node's.

    Args:
        node: The node's declaration.
        project: The work being weighed.

    Returns:
        The budget the project is weighed against.
    """
    return {**node["budget"], "worker_ram_gb": project["worker_ram_gb"]}


def _who_holds_it(node: NodeConfig, state: NodeState, budget: NodeBudget) -> str:
    """Name what holds a node's room in a reservation refusal.

    Args:
        node: The node's declaration.
        state: What the node reported.
        budget: The budget the project was weighed against.

    Returns:
        The node's CI runners, with the slice's numbers, when its probe read
        a ``runners.slice`` at its ``memory.high`` (MCPs board task
        5d6e57e7); else the node's own fleet runs, when without their
        charge it would have room (MCPs board task 939ec5c7: on 2026-10-03
        lavender-wsl, with 18.8 GB free and two runs holding seven workers,
        was said to have "somebody" on it); otherwise the owner.
    """
    ci_slice = state["ci_slice"]
    if ci_slice is not None and ci_slice["current_gb"] >= SLICE_AT_HIGH * ci_slice["high_gb"]:
        return (
            f"its CI runners hold it: runners.slice is at {ci_slice['current_gb']:.1f} GB of "
            f"its {ci_slice['high_gb']:.1f} GB memory.high, so the lane waits for a CI job to end"
        )
    unloaded = admissible_workers(
        budget, logical_cores=node["logical_cores"], free_ram_gb=state["free_ram_gb"]
    )
    if state["live"]["runs"] > 0 and unloaded > 0:
        return "its own fleet runs hold the rest, so it takes the next job when one of them ends"
    if state["reserved_cores"] > 0 and unloaded > 0:
        return (
            "its reservations hold the rest (work the fleet does not run, declared in "
            "~/.fleet/reservations), so it takes the next job when they end"
        )
    return "somebody is on this machine"


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

    WHERE A SECOND CHECK DOES NOT FIT ITS CPUS, THE NODE SAYS SO (MCPs board
    task a85ef09e). A node whose budget declares ``checks_at_once`` takes no
    run past that many, counting every run live or launching on it from any
    of its runners, before the pool is asked: on serendipity two checks
    inside the pool's 4 cores pushed MCPs/sms-gateway from 162 s to 306 s.

    WHERE THE CPUS ARE ALREADY SATURATED, NO CHECK IS PLACED (MCPs board task
    2ce5dc13). Cores are counted from the declaration less the fleet's own
    runs and reservations, so the stack's own work on diphtheria held none
    of them, and a check claimed there at 42 to 51 percent CPU pressure ran
    past its budget with every test passing. A node whose probe reads
    pressure at :data:`CPU_PRESSURE_BOUND_PCT` or over waits, after the
    ``checks_at_once`` refusal, whose cause is the more specific one.

    WHERE ITS DISK IS SATURATED, NO CHECK IS PLACED EITHER (MCPs board task
    814c4555). The CPU reading does not see a task stalled on the disk: a
    check claimed on diphtheria at 29.9 percent CPU and 71.5 percent I/O
    pressure ran five times its usual time. A node reading I/O pressure at
    :data:`IO_PRESSURE_BOUND_PCT` or over waits, after the CPU refusal.

    Args:
        node: The node's declaration.
        state: What it reported when last probed.
        project: The work being dispatched.
        carried: The tags the runner judging it carries: the node's declared
            tags, plus on a node runner the tool tags this tick's toolchain
            probe found (:func:`fleet.contracts.tags.tool_tags`).

    A PROJECT THAT NAMES ITS NODE IS REFUSED BY EVERY OTHER ONE, first,
    because no tag, disk or memory on another node changes that it reads
    what only the named host holds (MCPs board task bf7a9460). The workspace
    refuses a pin naming a node it does not declare, and every node's host
    is its registry name, so the node is matched by its host.

    A NODE THAT CHECKS ONLY ITS PINNED PROJECTS REFUSES EVERY OTHER ONE, next,
    for the same reason in reverse (MCPs board task d577036f): its own work
    leaves a check no margin under the budget, and nothing about the node
    changes that, so a project any other node can run is that node's.

    Returns:
        The verdict. ``workers`` is zero exactly when ``code`` is set.
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
    carried_at_once = node["budget"]["checks_at_once"]
    if carried_at_once is not None and live["runs"] >= carried_at_once:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_OWNER_RESERVED,
            reason=(
                f"{node['host']}'s CPUs carry {carried_at_once} budgeted check(s) at once "
                f"across its runners, and {live['runs']} are live or launching on it now"
                f"{_live_clause(state)}. A check runs wider than its test workers, so a "
                "second would push both past their budget; it takes the next job when one "
                "of them ends."
            ),
        )
    pressure = state["cpu_pressure_pct"]
    if pressure is not None and pressure >= CPU_PRESSURE_BOUND_PCT:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_OWNER_RESERVED,
            reason=(
                f"{node['host']}'s CPUs are saturated: a runnable task waited for one through "
                f"{pressure:.1f}% of the last minute, at or over the {CPU_PRESSURE_BOUND_PCT:.0f}% "
                f"a check is placed under{_live_clause(state)}. Free cores counted from its "
                "declaration do not show work the fleet does not run, and a check placed here "
                "would run past its budget; it takes the next job when the pressure falls."
            ),
        )
    io_pressure = state["io_pressure_pct"]
    if io_pressure is not None and io_pressure >= IO_PRESSURE_BOUND_PCT:
        return DispatchVerdict(
            workers=0,
            code=FleetErrorCode.NODE_OWNER_RESERVED,
            reason=(
                f"{node['host']}'s disk is saturated: a task stalled on I/O through "
                f"{io_pressure:.1f}% of the last minute, at or over the "
                f"{IO_PRESSURE_BOUND_PCT:.0f}% a check is placed under{_live_clause(state)}. "
                "Its CPUs do not show it, and a "
                "check placed here would run past its budget; it takes the next job when the "
                "pressure falls."
            ),
        )
    budget = _project_budget(node, project)
    # A reservation's cores come off beside the live runs' workers (MCPs board
    # task 84a2b690): a clean-VM smoke on diphtheria held 4 of its 8 cores
    # while two checks placed beside it ran 334 s and 411 s of their 300.
    workers = min(
        admissible_workers(
            budget,
            logical_cores=node["logical_cores"] - live["workers"] - state["reserved_cores"],
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
                f"{_who_holds_it(node, state, budget)}."
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
    """Name what a node's live runs and reservations hold, for a capacity refusal.

    Args:
        state: What the node reported.

    Returns:
        An empty string when no fleet run is live on the node and no
        reservation holds a core; otherwise ``, and its N live fleet run(s)
        hold W worker(s) and X GB`` and ``, and its reservations hold C
        core(s)``, each only when it holds something.
    """
    live = state["live"]
    runs = (
        ""
        if live["runs"] == 0
        else (
            f", and its {live['runs']} live fleet run(s) hold {live['workers']} worker(s) and "
            f"{live['ram_gb']:.1f} GB"
        )
    )
    held = (
        ""
        if state["reserved_cores"] == 0
        else f", and its reservations hold {state['reserved_cores']} core(s)"
    )
    return runs + held


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
    On a node that checks only its pinned projects, only those are servable,
    and the hypothetical job is pinned to the node judged, so the pin rules
    in :func:`assess` weigh it as the node's own.

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
        the same refusal, ``NODE_LACKS_TAG`` when no registered project's
        tags fit this runner at all, or ``NODE_NOT_PINNED`` when some fit and
        the node checks only its pinned projects and none is pinned to it.
    """
    fitting = [project for project in projects if set(project["required_tags"]) <= tags]
    if not fitting:
        carried = ", ".join(sorted(tag.value for tag in tags)) or "no tags"
        return (
            f"{FleetErrorCode.NODE_LACKS_TAG.value}: this runner carries {carried}, and no "
            "registered project's required tags fit it"
        )
    servable = [
        project["worker_ram_gb"]
        for project in fitting
        if project["node"] == node["host"] or not node["budget"]["pinned_checks_only"]
    ]
    if not servable:
        return (
            f"{FleetErrorCode.NODE_NOT_PINNED.value}: {node['host']} checks only the projects "
            "pinned to it, and no project this runner's tags fit is pinned to it"
        )
    default_tenant = ProjectConfig(
        worker_ram_gb=min(servable),
        minimum_workers=1,
        expected_minutes=1,
        exclusive_resources=(),
        external_paths=(),
        required_tags=(),
        source=None,
        node=node["host"],
    )
    verdict = assess(node, state, default_tenant, tags)
    if verdict["code"] is None:
        return None
    return f"{verdict['code'].value}: {verdict['reason']}"


def fitting_projects(
    node: NodeConfig,
    state: NodeState,
    projects: Mapping[str, ProjectConfig],
    carried: frozenset[NodeTag],
) -> tuple[str, ...]:
    """The registered projects a runner could launch on its node right now.

    What a node runner hands the queue with its claim (MCPs board task
    939ec5c7), so it is offered only a job it can run. Without it a node with
    room for the smallest project claimed the oldest job its tags matched and
    refused it on capacity, which closes a job for good: diphtheria did that
    to MCPs/execution-deploy twice on 2026-10-02 (8ccd89db, 86208e6b) while
    lavender-wsl, also tagged for it, might have run it later.

    Args:
        node: The node's declaration.
        state: What it reported when probed this tick.
        projects: Every registered project, by name.
        carried: The tags the runner claims with.

    Returns:
        The name of every project :func:`assess` accepts on this state, in
        name order; empty when none fits.
    """
    return tuple(
        name
        for name, project in sorted(projects.items())
        if assess(node, state, project, carried)["code"] is None
    )


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


__all__ = [
    "DispatchVerdict",
    "assess",
    "fitting_projects",
    "job_ceiling",
    "plan_dispatch",
    "room_for_any",
]
