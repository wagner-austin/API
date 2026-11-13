"""The passes a hub tick runs after its claim: publish, main, observe and reap.

Each is the control duty's (MCPs board task c39ee056), so only
``fleet-agent --lane hub`` runs them (:mod:`fleet.cli.agent`); a sessions
tick claims and stops. The module docstring of :mod:`fleet.cli.agent` says
why each comes in the order it does.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

from board_watch import config as board_config
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.contracts.machine import Machine
from fleet.core import declared, main_history, observe, runner_reap_dropped

_log = get_logger(__name__)


def publish_pass(
    credentials: McpCredentials, identity: JSONObject, read: declared.RegistryRead | str
) -> None:
    """Publish the rolled registry to the queue, which refuses undeclared work.

    The publication names the roster's node runners too, and the queue
    retires every other runner's tick; the log line names each one retired.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments; the publication is its.
        read: This tick's read of ``refs/fleet/rolled``
            (:func:`fleet.core.declared.read_declared`), or why it read
            nothing.

    Raises:
        AppError: From the queue call. A read that could not list or read
            the registry sends nothing and is logged with its code, every
            tick it happens, since publishing a registry not read would
            refuse declared work.
    """
    if isinstance(read, str):
        _log.warning("published nothing: %s", read)
        return
    publication = declared.publish_registry(credentials, read, identity=identity)
    _log.info(
        "published %d project(s) from the registry at %s (%d dropped since the last "
        "publication) and refused %d queued job(s) naming a project it does not declare; "
        "%d node runner(s) declared, retired: %s",
        publication["published"],
        read["commit"],
        publication["removed"],
        publication["refused"],
        len(read["runners"]),
        ", ".join(publication["retired"]) or "none",
    )


def main_pass(
    credentials: McpCredentials,
    identity: JSONObject,
    read: declared.RegistryRead | str,
    *,
    mirrors_root: pathlib.Path,
) -> None:
    """Publish the default-branch history of every repository the registry sources from.

    MCPs board task 8b3dcb31: the queue keeps it (``dispatch_main_publish``),
    so task_update's cancel gate reads only checks at main commits newer than
    a failure's. Each repository is fetched into the hub's mirror of its
    remote; one that cannot be read, or whose projects name two remotes, is
    logged with its code and left out, and the rest are sent at the
    registry read's instant, which was taken before any of them.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments; the publication is its.
        read: This tick's read of the rolled registry, or why it read
            nothing, in which case no main is read either.
        mirrors_root: The hub's mirrors directory.

    Raises:
        AppError: From the queue call, its refusal of a stale or rewritten
            history among them.
    """
    if isinstance(read, str):
        _log.warning("read no main history: the rolled registry was not read")
        return
    remotes, disagreeing = main_history.source_remotes(read["projects"])
    for line in disagreeing:
        _log.warning("%s", line)
    reads, unread = main_history.read_mains(mirrors_root, remotes)
    for line in unread:
        _log.warning("read no main history for %s", line)
    if not reads:
        _log.warning("published no main history: none of %d repositories was read", len(remotes))
        return
    heads = main_history.publish_mains(
        credentials, reads, read_at=read["checked_at"], identity=identity
    )
    _log.info(
        "published the main history of %d repositories: %s",
        len(heads),
        ", ".join(
            f"{head['repository']} at {head['head'][:12]} "
            f"(depth {head['depth']}, {head['added']} new)"
            for head in heads
        ),
    )


def observe_pass(
    machines: Mapping[str, Machine], identity: JSONObject, evidence_seconds: int
) -> None:
    """Record the Claude Code sessions of every enabled machine the arm observes.

    The board's credentials are read HERE, not at the top of the tick, and
    only when the roster names a machine to observe: they are the
    taskboard's (``TASKBOARD_MCP_API_KEY``), a different secret from the
    queue's, which the claim and publish passes need without them.

    Args:
        machines: The roster's machines, from the loaded workspace.
        identity: This runner's identity arguments.
        evidence_seconds: The workspace's ``native_evidence_seconds``.

    Raises:
        AppError: The board-watch credential codes when the taskboard's
            variables are unset, and the contract faults
            :func:`fleet.core.observe.observe_node` names. A node that does
            not answer is logged, not raised.
    """
    observed = observe.observable(machines)
    if not observed:
        _log.info(
            "observed no session: none of the roster's %d machine(s) is enabled and declares "
            "ledger_observer fleet-arm",
            len(machines),
        )
        return
    board = board_config.load_credentials()
    for outcome in observe.observe_pass(board, observed, identity, evidence_seconds):
        _log.info("%s", observe.render_outcome(outcome))


def reap_pass(repo_root: pathlib.Path) -> None:
    """End every runner Worker whose job GitHub dropped (MCPs board task 715b9ec5).

    Args:
        repo_root: The API checkout whose object store holds
            ``refs/fleet/rolled``, at whose commit the roster is read.

    Raises:
        AppError: As :func:`fleet.core.runner_reap_dropped.reap_pass` and
            :func:`fleet.core.runner_reap_dropped.read_rolled_roster` say. A
            roster that could not be read is logged with its code, every tick
            it happens, and nothing is reaped.
    """
    roster = runner_reap_dropped.read_rolled_roster(repo_root)
    if isinstance(roster, str):
        _log.warning("reaped nothing: %s", roster)
        return
    for line in runner_reap_dropped.reap_pass(roster):
        _log.info("%s", line)


__all__ = ["main_pass", "observe_pass", "publish_pass", "reap_pass"]
