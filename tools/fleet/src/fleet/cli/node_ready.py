"""Whether a node runner may claim this tick, and the tags it claims with.

Lifted out of :mod:`fleet.cli.node_agent`, which runs the tick, because the
gate grew a second product (MCPs board task 939ec5c7): besides the node's
measured state it now returns the tags the runner claims with, and the tool
tags among them are read from the toolchain probe this gate already pays
for. A runner claims with its node's declared tags
(:func:`fleet.contracts.tags.runner_tags`) plus the tag of every tool the
probe found (:func:`fleet.contracts.tags.tool_tags`), so installing ffmpeg
on a node makes it eligible for grandma-api's check on its next tick with
no file edited, and a node without ffmpeg takes every other job instead of
claiming nothing.
"""

from __future__ import annotations

from platform_core.logging import get_logger
from typing_extensions import TypedDict

from fleet.cli import _config
from fleet.contracts.elevation import elevation_gap
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.tags import NodeTag, runner_tags, tool_tags
from fleet.core import capacity, host_report, probe, records, toolchain

_log = get_logger(__name__)


class Ready(TypedDict):
    """A node that may claim this tick.

    Attributes:
        state: What it reported when probed this tick.
        tags: The tags its runner claims with: the declared ones for this
            runner's lane, plus the tag of every tool its toolchain probe
            found this tick.
    """

    state: NodeState
    tags: frozenset[NodeTag]


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
        task for a retired node asks nothing), answered, has room, has every
        required tool and (elevated) holds an administrator's token;
        otherwise None, with the gate that closed and its reason logged, and
        for a node with a ``wsl_host`` what that host reports
        (:mod:`fleet.core.host_report`). A tagged tool the node lacks closes
        no gate: it is logged with its install command and the runner claims
        without its tag.
    """
    if not node["enabled"]:
        _log.info("%s is disabled in fleet.json; claiming nothing", alias)
        return None
    probed = probe.attempt_probe(node, live_runs=records.live_runs(loaded.ledger, node=alias))
    state = probed["state"]
    if state is None:
        _log.info("%s did not answer; claiming nothing: %s", alias, probed["reason"])
        if node["wsl_host"] is not None:
            seen = host_report.describe_wsl_host(loaded.workspace, loaded.ledger, node["wsl_host"])
            _log.info("%s did not answer, and %s", alias, seen)
        return None
    declared = runner_tags(node, elevated=elevated)
    projects = tuple(loaded.workspace["projects"].values())
    full = capacity.room_for_any(node, state, projects, declared)
    if full is not None:
        _log.info("%s has room for nothing; claiming nothing: %s", alias, full)
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
    _log.info("%s toolchain ready: %s", alias, toolchain.ready_summary(answered))
    lacking = toolchain.tagged_gap(alias, answered)
    if lacking is not None:
        _log.info("%s", lacking)
    unelevated = elevation_gap(alias, node, answered) if elevated else None
    if unelevated is not None:
        _log.info(
            "%s cannot launch elevated; claiming nothing: %s: %s",
            alias,
            unelevated.code,
            unelevated.message,
        )
        return None
    return Ready(state=state, tags=declared | tool_tags(answered))


__all__ = ["Ready", "ready_state"]
