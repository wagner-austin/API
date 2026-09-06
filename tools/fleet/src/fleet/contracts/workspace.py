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
        not_dispatchable: Machines this workspace deliberately will NOT
            dispatch to, keyed by their registry name, each carrying why.
            SILENCE IS NOT A DECISION, which is the whole reason this field
            exists: a machine simply absent from ``nodes`` is
            indistinguishable from one nobody has got round to adding, and
            ``austinpc`` -- the hub the dispatcher itself runs on, enabled
            and 24-core in the identity registry -- had been reading as both
            for weeks. Named here, its absence becomes a claim the drift
            check can honour; absent from both, it is drift.
        projects: The work that may be dispatched, keyed by repo-relative
            path.
        ledger: Path to the append-only dispatch record. Relative paths
            resolve against the workspace document's own directory, so a
            workspace can be moved without editing it.
        feed: Path to the append-only event stream subscribers tail.
        leases: Path to the live lease file. The only mutable one.
    """

    nodes: dict[str, NodeConfig]
    not_dispatchable: dict[str, str]
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
        "not_dispatchable": dict(workspace["not_dispatchable"]),
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


def _decode_not_dispatchable(value: JSONObject, nodes: dict[str, NodeConfig]) -> dict[str, str]:
    """Read the deliberate exclusions, refusing an empty or contradicted one.

    THE REASON IS REQUIRED AND MAY NOT BE BLANK. An exclusion whose reason is
    an empty string carries no more than absence did, which is the thing this
    field exists to stop being ambiguous. The next reader has to be able to
    tell "we decided against this machine, here is why" from "nobody has got
    round to it".

    Args:
        value: The workspace object.
        nodes: The already-decoded dispatch targets.

    Returns:
        Machine name to why this workspace will not dispatch to it. Empty is
        allowed -- a fleet that excludes nothing is a real answer, unlike a
        fleet that declares no nodes.

    Raises:
        JSONTypeError: If the field is missing, is not an object, a reason is
            not a non-empty string, or a name is BOTH declared as a node and
            excluded. That last one is a document that says two opposite
            things, and picking either would be a guess.
    """
    declared = require_dict(value, "not_dispatchable")
    excluded: dict[str, str] = {}
    for name in sorted(declared):
        reason = require_str(declared, name)
        if not reason:
            raise JSONTypeError(
                f"not_dispatchable[{name!r}] has an empty reason. An exclusion with no reason "
                "says no more than leaving the machine out entirely, which is the ambiguity "
                "this field exists to remove."
            )
        if name in nodes:
            raise JSONTypeError(
                f"{name!r} is declared both as a dispatchable node and as not_dispatchable. "
                "The workspace says two opposite things about the same machine and neither "
                "reading is safe to pick."
            )
        excluded[name] = reason
    return excluded


def decode_fleet_workspace(value: JSONValue) -> FleetWorkspace:
    """Decode and validate a workspace.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated workspace.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the node or project mapping is empty, a machine is both
            declared and excluded, or a node or project fails its own
            decoder.
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
        not_dispatchable=_decode_not_dispatchable(value, nodes),
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
