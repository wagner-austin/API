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
"""

from __future__ import annotations

from platform_core.logging import get_logger
from typing_extensions import TypedDict

from fleet.cli import _config
from fleet.contracts.detection import detected_tags, tag_drift
from fleet.contracts.elevation import elevation_gap
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.tags import NodeTag, runner_tags
from fleet.contracts.toolchain import ToolReport
from fleet.core import capacity, host_report, probe, records, toolchain

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


def ready_state(
    loaded: _config.LoadedWorkspace, *, alias: str, node: NodeConfig, elevated: bool
) -> Ready | None:
    """Ask the node, in order, whether it may claim at all this tick.

    Args:
        loaded: The workspace and its resolved record paths.
        alias: This node's workspace name.
        node: Its declaration.
        elevated: Whether this is the node's elevated runner, which also
            needs the probe to read an administrator's token.

    Returns:
        The node's measured state and claim tags when it is enabled (a stale
        task for a retired node asks nothing), answered, has every required
        tool, has room and (elevated) holds an administrator's token;
        otherwise None, with the gate that closed and its reason logged, and
        for a node with a ``wsl_host`` what that host reports
        (:mod:`fleet.core.host_report`). A tagged tool the node lacks, and a
        declaration its probe contradicts, close no gate: each is logged and
        the runner claims with what the probe found.
    """
    if not node["enabled"]:
        _log.info("%s is disabled in fleet.json; claiming nothing", alias)
        return None
    live = records.live_load(loaded.ledger, node=alias, projects=loaded.workspace["projects"])
    probed = probe.attempt_probe(node, live=live)
    state = probed["state"]
    if state is None:
        _log.info("%s did not answer; claiming nothing: %s", alias, probed["reason"])
        if node["wsl_host"] is not None:
            seen = host_report.describe_wsl_host(loaded.workspace, loaded.ledger, node["wsl_host"])
            _log.info("%s did not answer, and %s", alias, seen)
        return None
    answered = toolchain.attempt_toolchain(node)
    if not isinstance(answered, tuple):
        _log.info(
            "%s did not answer the toolchain probe; claiming nothing: %s: %s",
            alias,
            answered["code"],
            answered["message"],
        )
        return None
    gap = toolchain.readiness_gap(alias, node, answered)
    if gap is not None:
        _log.info("%s cannot build; claiming nothing: %s: %s", alias, gap.code, gap.message)
        return None
    return _claimable(
        loaded, alias=alias, node=node, state=state, answered=answered, elevated=elevated
    )


def _claimable(
    loaded: _config.LoadedWorkspace,
    *,
    alias: str,
    node: NodeConfig,
    state: NodeState,
    answered: tuple[ToolReport, ...],
    elevated: bool,
) -> Ready | None:
    """Read a ready node's claim tags and the projects it fits, or say why none.

    Args:
        loaded: The workspace and its resolved record paths.
        alias: This node's workspace name.
        node: Its declaration.
        state: What it reported when probed this tick.
        answered: What its toolchain probe answered, already judged ready.
        elevated: Whether this is the node's elevated runner.

    Returns:
        The node's state, claim tags and fitting projects; or None when it has
        room for nothing, fits no project, or (elevated) does not hold an
        administrator's token, with the reason logged. The ready summary, a
        tagged tool it lacks and each difference from its declaration are
        logged first.
    """
    _log.info("%s toolchain ready: %s", alias, toolchain.ready_summary(answered))
    lacking = toolchain.tagged_gap(alias, answered)
    if lacking is not None:
        _log.info("%s", lacking)
    for drift in tag_drift(node, answered):
        _log.info("%s (%s) %s", alias, node["host"], drift)
    tags = detected_tags(node, answered) | (runner_tags(node, elevated=elevated) & ELEVATED_ONLY)
    registered = loaded.workspace["projects"]
    full = capacity.room_for_any(node, state, tuple(registered.values()), tags)
    if full is not None:
        _log.info("%s has room for nothing; claiming nothing: %s", alias, full)
        return None
    fits = capacity.fitting_projects(node, state, registered, tags)
    if not fits:
        _log.info(
            "%s has room for one worker but not for any project's minimum; claiming nothing",
            alias,
        )
        return None
    unelevated = elevation_gap(alias, node, answered) if elevated else None
    if unelevated is not None:
        _log.info(
            "%s cannot launch elevated; claiming nothing: %s: %s",
            alias,
            unelevated.code,
            unelevated.message,
        )
        return None
    return Ready(state=state, tags=tags, fits=fits)


__all__ = ["Ready", "ready_state"]
