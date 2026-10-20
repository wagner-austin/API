"""The three passes a hub tick runs after its claim: publish, observe and reap.

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
from fleet.core import declared, observe, runner_reap_dropped

_log = get_logger(__name__)


def publish_pass(
    credentials: McpCredentials, identity: JSONObject, *, repo_root: pathlib.Path
) -> None:
    """Publish the rolled registry to the queue, which refuses undeclared work.

    The publication names the roster's node runners too, and the queue
    retires every other runner's tick; the log line names each one retired.

    Args:
        credentials: The queue's endpoint and headers.
        identity: This runner's identity arguments; the publication is its.
        repo_root: The API checkout whose object store holds
            ``refs/fleet/rolled``.

    Raises:
        AppError: From the queue call. A read that could not list or read
            the registry sends nothing and is logged with its code, every
            tick it happens, since publishing a registry not read would
            refuse declared work.
        InvalidJsonError: From the read, when a project or machine file is
            not JSON.
        JSONTypeError: From the read, when a project or machine file does
            not decode.
    """
    read = declared.read_declared(repo_root)
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


def observe_pass(machines: Mapping[str, Machine], identity: JSONObject) -> None:
    """Record every enabled worker's Claude Code sessions in the board's ledger.

    The board's credentials are read HERE, not at the top of the tick, and
    only when the roster names a machine to observe: they are the
    taskboard's (``TASKBOARD_MCP_API_KEY``), a different secret from the
    queue's, which the claim and publish passes need without them.

    Args:
        machines: The roster's machines, from the loaded workspace.
        identity: This runner's identity arguments.

    Raises:
        AppError: The board-watch credential codes when the taskboard's
            variables are unset, and the contract faults
            :func:`fleet.core.observe.observe_node` names. A node that does
            not answer is logged, not raised.
    """
    observed = observe.observable(machines)
    if not observed:
        _log.info(
            "observed no session: none of the roster's %d machine(s) is an enabled worker, "
            "vpn-jump or session host",
            len(machines),
        )
        return
    board = board_config.load_credentials()
    for outcome in observe.observe_pass(board, observed, identity):
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


__all__ = ["observe_pass", "publish_pass", "reap_pass"]
