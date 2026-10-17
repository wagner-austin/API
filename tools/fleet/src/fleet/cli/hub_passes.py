"""The two passes a hub tick runs after its claim: publish and observe.

Both are the control duty's (MCPs board task c39ee056), so only
``fleet-agent --lane hub`` runs them (:mod:`fleet.cli.agent`); a sessions
tick claims and stops. The module docstring of :mod:`fleet.cli.agent` says
why each comes in the order it does.
"""

from __future__ import annotations

import pathlib

from board_watch import config as board_config
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.core import _test_hooks, declared, observe, registry

_log = get_logger(__name__)


def publish_pass(
    credentials: McpCredentials, identity: JSONObject, *, repo_root: pathlib.Path
) -> None:
    """Publish the rolled registry to the queue, which refuses undeclared work.

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
        InvalidJsonError: From the read, when a project file is not JSON.
        JSONTypeError: From the read, when a project file does not decode.
    """
    read = declared.read_declared(repo_root)
    if isinstance(read, str):
        _log.warning("published nothing: %s", read)
        return
    publication = declared.publish_registry(credentials, read, identity=identity)
    _log.info(
        "published %d project(s) from the registry at %s (%d dropped since the last "
        "publication) and refused %d queued job(s) naming a project it does not declare",
        publication["published"],
        read["commit"],
        publication["removed"],
        publication["refused"],
    )


def observe_pass(registry_path: pathlib.Path, identity: JSONObject) -> None:
    """Record every enabled worker's Claude Code sessions in the board's ledger.

    The board's credentials are read HERE, not at the top of the tick: they
    are the taskboard's (``TASKBOARD_MCP_API_KEY``), a different secret from
    the queue's, and a runner started without ``--registry`` has no use for
    them and must not be refused for lacking them.

    Args:
        registry_path: The identity registry, ``fleet-mcp/fleet-nodes.json``.
        identity: This runner's identity arguments.

    Raises:
        AppError: ``NODE_REGISTRY_UNREADABLE`` for a registry this cannot
            read, the board-watch credential codes when the taskboard's
            variables are unset, and the contract faults
            :func:`fleet.core.observe.observe_node` names. A node that does
            not answer is logged, not raised.
        OSError: When the registry file cannot be read.
    """
    nodes = registry.decode_registry_nodes(_test_hooks.read_text(registry_path))
    board = board_config.load_credentials()
    for outcome in observe.observe_pass(board, nodes, identity):
        _log.info("%s", observe.render_outcome(outcome))


__all__ = ["observe_pass", "publish_pass"]
