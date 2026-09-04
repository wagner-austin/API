"""The one document every fleet command reads.

Where the machines are, where the three append-only files live, and which
projects may be dispatched. Decoded strictly and completely: a command that
accepted a partial workspace would be a command that behaves differently
depending on which fields somebody remembered.

WHY PROJECTS ARE ENUMERATED AND NOT DISCOVERED. It would be easy to glob for
Makefiles and call the result the project list. That would make the set of
dispatchable work a property of the working tree at the moment somebody ran
the command, so two sessions could disagree about what exists, and a
half-finished directory would become dispatchable by existing. The workspace
names them, and :func:`require_project` refuses anything else.

WHY THREE FILES AND NOT ONE. The ledger is the durable record, the feed is the
subscribable stream, and the lease file is live mutable state. They have
different lifetimes and different readers -- a subscriber tails the feed
forever and never reads the ledger; a capacity check reads leases and never
reads either. One file would make every reader parse everything and would put
the mutable half in the append-only one.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
    require_str,
)
from typing_extensions import TypedDict

from fleet.contracts.node import NodeConfig, decode_node_config, encode_node_config
from fleet.contracts.project import ProjectConfig, decode_project_config, encode_project_config


class FleetWorkspace(TypedDict):
    """Everything a fleet command needs that is not on the command line.

    Attributes:
        nodes: The machines that may be dispatched to, keyed by the name a
            command's ``--node`` takes. The key is the workspace's name for
            the machine and ``NodeConfig.host`` is its SSH alias; they are
            usually the same string and are deliberately separate fields, so
            renaming a node in the workspace does not require the ssh config
            to agree.
        projects: The work that may be dispatched, keyed by repo-relative
            path.
        ledger: Path to the append-only dispatch record. Relative paths
            resolve against the workspace document's own directory, so a
            workspace can be moved without editing it.
        feed: Path to the append-only event stream subscribers tail.
        leases: Path to the live lease file. The only mutable one.
    """

    nodes: dict[str, NodeConfig]
    projects: dict[str, ProjectConfig]
    ledger: str
    feed: str
    leases: str


def require_node(workspace: FleetWorkspace, name: str) -> NodeConfig:
    """Look up a node, refusing an unknown name.

    Args:
        workspace: The decoded workspace.
        name: The node name a command was given.

    Returns:
        That node's declaration.

    Raises:
        AppError: With ``WORKSPACE_NODE_UNKNOWN`` when the name is not
            declared, naming the ones that are. Naming them is the whole
            value of the refusal: the alternative failure is an ssh attempt
            to a host that does not resolve, several seconds later, with a
            message about DNS.
    """
    found = workspace["nodes"].get(name)
    if found is None:
        raise AppError(
            FleetErrorCode.WORKSPACE_NODE_UNKNOWN,
            f"no node named {name!r} in this workspace; it declares "
            f"{', '.join(sorted(workspace['nodes'])) or '<none>'}",
        )
    return found


def require_project(workspace: FleetWorkspace, path: str) -> ProjectConfig:
    """Look up a project, refusing an undeclared one.

    Args:
        workspace: The decoded workspace.
        path: Repo-relative project path a command was given.

    Returns:
        That project's declaration.

    Raises:
        AppError: With ``WORKSPACE_PROJECT_UNKNOWN`` when the path is not
            declared, naming the ones that are. Refused rather than inferred
            from the filesystem -- see the module docstring.
    """
    found = workspace["projects"].get(path)
    if found is None:
        raise AppError(
            FleetErrorCode.WORKSPACE_PROJECT_UNKNOWN,
            f"no project {path!r} in this workspace; it declares "
            f"{', '.join(sorted(workspace['projects'])) or '<none>'}",
        )
    return found


def encode_fleet_workspace(workspace: FleetWorkspace) -> JSONObject:
    """Encode a workspace.

    Args:
        workspace: The workspace to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "nodes": {name: encode_node_config(node) for name, node in workspace["nodes"].items()},
        "projects": {
            path: encode_project_config(project) for path, project in workspace["projects"].items()
        },
        "ledger": workspace["ledger"],
        "feed": workspace["feed"],
        "leases": workspace["leases"],
    }


def _decode_named(value: JSONObject, field: str) -> dict[str, JSONValue]:
    """Read a required object-of-objects field.

    Args:
        value: The workspace object.
        field: The field name.

    Returns:
        The mapping, with its values still undecoded.

    Raises:
        JSONTypeError: If the field is missing, is not an object, or is
            empty. Empty is refused because a workspace declaring no nodes
            can dispatch nothing, and the failure it produces otherwise is
            "unknown node" against an empty list, which reads as a typo
            rather than as an unconfigured workspace.
    """
    found = require_dict(value, field)
    if not found:
        raise JSONTypeError(
            f"workspace declares no {field}; it would be able to dispatch nothing, and every "
            f"command would fail naming an empty list as though the caller had made a typo"
        )
    return found


def decode_fleet_workspace(value: JSONValue) -> FleetWorkspace:
    """Decode and validate a workspace.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated workspace.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, either mapping is empty, or a node or project fails its
            own decoder.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"workspace must be a JSON object, got {type(value).__name__}")
    nodes = {name: decode_node_config(node) for name, node in _decode_named(value, "nodes").items()}
    projects = {
        path: decode_project_config(project)
        for path, project in _decode_named(value, "projects").items()
    }
    return FleetWorkspace(
        nodes=nodes,
        projects=projects,
        ledger=require_str(value, "ledger"),
        feed=require_str(value, "feed"),
        leases=require_str(value, "leases"),
    )


__all__ = [
    "FleetWorkspace",
    "decode_fleet_workspace",
    "encode_fleet_workspace",
    "require_node",
    "require_project",
]
