"""Whether a node runner may claim this tick, and the tags it claims with.

Lifted out of :mod:`fleet.cli.node_agent`, which runs the tick, because the
gate grew a second product (MCPs board task 939ec5c7): besides the node's
measured state it returns the tags the runner claims with, read from the
toolchain probe this gate already pays for
(:func:`fleet.contracts.detection.detected_tags`), plus ``elevated`` for the
node's elevated runner (:func:`fleet.contracts.tags.runner_tags`). So
installing ffmpeg, a compiler or the test database on a node makes it
eligible for the jobs that need it on its next tick with no file edited, a
node without one takes every other job instead of claiming nothing, and how
the declaration differs from the answer is logged every tick
(:func:`fleet.contracts.detection.tag_drift`).

THE TOOLCHAIN IS ASKED BEFORE THE ROOM. Which projects a runner could take
depends on its tags, and the room check sizes the smallest of them, so the
tags are read first; the cost is one probe on a tick that finds no room.

EVERY ANSWER CARRIES THE TICK (A4). Whichever gate closes, the runner
records what it found, its tags, its load, the projects it fits and the
verdict, as its row on the queue (:mod:`fleet.core.tick_report`), so the
line this module logs is also the line ``fleet_status`` shows beside the
node.
"""

from __future__ import annotations

from platform_core.logging import get_logger
from typing_extensions import TypedDict

from fleet.cli import _config
from fleet.contracts.detection import detected_tags, tag_drift
from fleet.contracts.elevation import elevation_gap
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.runner_tick import RunnerTick, TickLoad
from fleet.contracts.tags import NodeTag, runner_tags
from fleet.contracts.toolchain import ToolReport
from fleet.core import capacity, host_report, names, probe, records, run_lease, toolchain

_log = get_logger(__name__)

#: The one tag a runner takes from its lane rather than from its probe.
ELEVATED_ONLY = frozenset({NodeTag.ELEVATED})


class Ready(TypedDict):
    """A node that may claim this tick.

    Attributes:
        state: What it reported when probed this tick.
        tags: The tags its runner claims with: every one its toolchain
            probe found this tick, plus ``elevated`` for the elevated runner.
        fits: The registered projects it could launch now
            (:func:`fleet.core.capacity.fitting_projects`), which its claim
            names so the queue offers it nothing it would refuse. Never
            empty: a node that fits none claims nothing.
    """

    state: NodeState
    tags: frozenset[NodeTag]
    fits: tuple[str, ...]


class Gate(TypedDict):
    """What the gate decided, and the tick to record for it.

    Attributes:
        ready: The node's state, tags and fits when it may claim, else None.
        tick: What this tick found, with the verdict of the gate that closed,
            or for a ready node ``claiming`` set and a verdict the claim
            replaces once it has asked the queue.
    """

    ready: Ready | None
    tick: RunnerTick


def ready_state(
    loaded: _config.LoadedWorkspace, *, alias: str, node: NodeConfig, elevated: bool
) -> Gate:
    """Ask the node, in order, whether it may claim at all this tick.

    Args:
        loaded: The workspace and its resolved record paths.
        alias: This node's workspace name.
        node: Its declaration.
        elevated: Whether this is the node's elevated runner, which also
            needs the probe to read an administrator's token.

    Returns:
        The gate. ``ready`` carries the node's measured state and claim tags
        when it is enabled (a stale task for a retired node asks nothing),
        answered, has every required tool, has room and (elevated) holds an
        administrator's token; otherwise it is None, with the gate that
        closed and its reason logged and carried as the tick's verdict, and
        for a node with a ``wsl_host`` what that host reports logged too
        (:mod:`fleet.core.host_report`). A tagged tool the node lacks, and a
        declaration its probe contradicts, close no gate: each is logged and
        the runner claims with what the probe found.
    """
    if not node["enabled"]:
        return closed(alias, elevated, None, "is disabled in fleet.json; claiming nothing")
    live = records.live_load(loaded.ledger, node=alias, projects=loaded.workspace["projects"])
    # The probes' scripts are named for this runner, never for the node: a
    # node's ordinary and elevated runners tick on the same minute, and one
    # shared path collided on serendipity every tick (MCPs board task
    # 939ec5c7, :func:`fleet.core.names.capacity_probe_stem`).
    writer = names.runner_name(alias, elevated=elevated)
    probed = probe.attempt_probe(node, live=live, writer=writer)
    state = probed["state"]
    if state is None:
        gate = closed(
            alias, elevated, None, f"did not answer; claiming nothing: {probed['reason']}"
        )
        if node["wsl_host"] is not None:
            seen = host_report.describe_wsl_host(
                loaded.workspace, loaded.ledger, node["wsl_host"], writer=writer
            )
            _log.info("%s did not answer, and %s", alias, seen)
        return gate
    load = tick_load(state)
    answered = toolchain.attempt_toolchain(node, writer=writer)
    if not isinstance(answered, tuple):
        return closed(
            alias,
            elevated,
            load,
            "did not answer the toolchain probe; claiming nothing: "
            f"{answered['code']}: {answered['message']}",
        )
    gap = toolchain.readiness_gap(alias, node, answered)
    if gap is not None:
        return closed(
            alias, elevated, load, f"cannot build; claiming nothing: {gap.code}: {gap.message}"
        )
    return _claimable(
        loaded, alias=alias, node=node, state=state, answered=answered, elevated=elevated
    )


def tick_load(state: NodeState) -> TickLoad:
    """Read the tick's load off a node's measured state.

    Args:
        state: What the node reported this tick.

    Returns:
        Its live runs and their workers, and its free memory.
    """
    return TickLoad(
        runs=state["live"]["runs"],
        workers=state["live"]["workers"],
        free_ram_gb=state["free_ram_gb"],
    )


def closed(
    alias: str,
    elevated: bool,
    load: TickLoad | None,
    verdict: str,
    *,
    tags: frozenset[NodeTag] = frozenset(),
    fits: tuple[str, ...] = (),
) -> Gate:
    """Log why a node claims nothing this tick, and carry it as the tick.

    Args:
        alias: This node's workspace name.
        elevated: Whether this is the node's elevated runner.
        load: What the node holds, or None when it was not probed or did
            not answer.
        verdict: Why it claims nothing, without the node's name, which the
            log line prefixes and the tick carries as its own field.
        tags: The tags its probe detected, when the tick got that far.
        fits: The projects it fits, when the tick got that far.

    Returns:
        A gate with no ready node and the tick to record.
    """
    _log.info("%s %s", alias, verdict)
    return Gate(
        ready=None,
        tick=RunnerTick(
            node=alias,
            elevated=elevated,
            tags=tuple(sorted(tags)),
            fits=fits,
            load=load,
            claiming=False,
            verdict=verdict,
        ),
    )


def _claimable(
    loaded: _config.LoadedWorkspace,
    *,
    alias: str,
    node: NodeConfig,
    state: NodeState,
    answered: tuple[ToolReport, ...],
    elevated: bool,
) -> Gate:
    """Read a ready node's claim tags and the projects it fits, or say why none.

    Args:
        loaded: The workspace and its resolved record paths.
        alias: This node's workspace name.
        node: Its declaration.
        state: What it reported when probed this tick.
        answered: What its toolchain probe answered, already judged ready.
        elevated: Whether this is the node's elevated runner.

    Returns:
        The gate: the node's state, claim tags and fitting projects, less
        every one a lease on the node holds now
        (:func:`fleet.core.run_lease.held_on_node`), so a claim never takes a
        job its lease would refuse; or no ready node when it has room for
        nothing, fits no project, fits only projects a lease holds, or
        (elevated) does not hold an administrator's token, with the reason
        logged and carried. The ready summary, a tagged tool it lacks and
        each difference from its declaration are logged first.
    """
    _log.info("%s toolchain ready: %s", alias, toolchain.ready_summary(answered))
    lacking = toolchain.tagged_gap(alias, answered)
    if lacking is not None:
        _log.info("%s", lacking)
    for drift in tag_drift(node, answered):
        _log.info("%s (%s) %s", alias, node["host"], drift)
    tags = detected_tags(node, answered) | (runner_tags(node, elevated=elevated) & ELEVATED_ONLY)
    load = tick_load(state)
    registered = loaded.workspace["projects"]
    full = capacity.room_for_any(node, state, tuple(registered.values()), tags)
    if full is not None:
        return closed(
            alias, elevated, load, f"has room for nothing; claiming nothing: {full}", tags=tags
        )
    roomy = capacity.fitting_projects(node, state, registered, tags)
    held = run_lease.held_on_node(
        loaded.leases,
        node=alias,
        names=roomy,
        projects=registered,
        node_local=loaded.workspace["node_local_resources"],
    )
    if held:
        _log.info(
            "%s leaves out what a lease on it holds now, which the queue keeps for another "
            "node or a later tick: %s",
            alias,
            ", ".join(held),
        )
    fits = tuple(name for name in roomy if name not in held)
    if roomy and not fits:
        return closed(
            alias,
            elevated,
            load,
            "has room, but every project it fits is held by a lease on it; claiming nothing",
            tags=tags,
        )
    if not fits:
        return closed(
            alias,
            elevated,
            load,
            "has room for one worker but not for any project's minimum; claiming nothing",
            tags=tags,
        )
    unelevated = elevation_gap(alias, node, answered) if elevated else None
    if unelevated is not None:
        return closed(
            alias,
            elevated,
            load,
            f"cannot launch elevated; claiming nothing: {unelevated.code}: {unelevated.message}",
            tags=tags,
            fits=fits,
        )
    return Gate(
        ready=Ready(state=state, tags=tags, fits=fits),
        tick=RunnerTick(
            node=alias,
            elevated=elevated,
            tags=tuple(sorted(tags)),
            fits=fits,
            load=load,
            claiming=True,
            verdict=f"fits {len(fits)} project(s); asking the queue",
        ),
    )


__all__ = ["ELEVATED_ONLY", "Gate", "Ready", "closed", "ready_state", "tick_load"]
