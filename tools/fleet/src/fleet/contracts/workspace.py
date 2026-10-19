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
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from fleet.contracts.machine import Machine, decode_machine, encode_machine
from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.contracts.pins import check_pinned_nodes
from fleet.contracts.project import ProjectConfig, decode_project_config, encode_project_config
from fleet.contracts.resources import decode_names, encode_names
from fleet.contracts.source import decode_path, decode_remote

#: The longest serve a workspace may declare (MCPs board task 8993c306). A
#: node runner hands over at the first fire boundary after its serve at
#: which it is idle (:mod:`fleet.cli.node_serve`), and its launcher ends it
#: at 39 minutes (:data:`fleet.cli.rolled.AGENT_WALL_SECONDS`), so 30 minutes
#: leaves nine for the drain: a serve past its time claims nothing more,
#: and loki's longest launch measured on 2026-10-04 took 155 s.
NODE_SERVE_CEILING_SECONDS = 1800

#: The slowest poll a workspace may declare: one start's repetition
#: (:data:`fleet.cli.node_serve.TICK_SECONDS`), since the collect pass at
#: every fire boundary already reads every held run at that rate.
NODE_POLL_CEILING_SECONDS = 180


class FleetWorkspace(TypedDict):
    """Everything a fleet command needs that is not on the command line.

    Attributes:
        machines: Every machine of the fleet, the roster's one file each
            (:mod:`fleet.contracts.machine`, MCPs board task e447de72),
            keyed by name.
        nodes: The machines that declare what dispatch may take from them,
            keyed by the name a command's ``--node`` takes, which is also
            ``NodeConfig.host``, the machine's SSH alias. Read from
            :attr:`machines`, never declared apart from them.
        not_dispatchable: The machines whose runners claim no work by
            decision, each carrying why: a machine whose ``dispatch`` is a
            reason, and a declared node whose dispatch is held. SILENCE IS
            NOT A DECISION: ``austinpc``, the hub the dispatcher ran on,
            enabled and 24-core, read for weeks as both excluded and
            forgotten, and a roster file cannot leave ``dispatch`` out, so
            every machine says which.
        projects: The work that may be dispatched, keyed by repo-relative
            path.
        data_paths: Per repository, keyed by the remote its projects declare,
            the directories holding DATA rather than code: corpora, binary
            fixtures, vendored tool distributions, recorded artifacts. An
            export of a project leaves out every one of them that does not
            lie inside the project being checked
            (:func:`~fleet.core.archive_scope.archive_pathspec`).

            KEYED BY REMOTE BECAUSE THIS WORKSPACE SPANS THREE REPOSITORIES
            and a repo-relative path means nothing outside its own. A flat
            list would let ``tests/fixtures`` declared for one repository
            silently strip a directory of that name from another.

            MEASURED, 2026-09-25, at ``f560de52`` of the API monorepo (board
            task 140e7042). The repository tracks 610,853,185 bytes and eight
            directories hold 521 MB of it, so every check of every project
            staged a 216,826,747-byte archive that took 43 seconds to build.
            Declaring those eight brings it to 27,462,327 bytes and 2.7
            seconds, 87.3% smaller, and the tar still holds all 51
            ``pyproject.toml``, all 51 ``scripts/guard.py`` and all 7
            workflow files.

            WHICH IS THE POINT, AND THE REASON THIS IS NOT AN INCLUDE LIST.
            ``libs/monorepo_guards`` audits the whole monorepo's SHAPE --
            its suite asserts at least forty package roots and reads every
            real workflow -- so an archive holding only that project and its
            poetry closure would carry one package root and fail. Every
            project's Makefile also reaches ``../../scripts/make/shell.mk``
            and ``../../tools/maketools/scripts/run.py``, which no
            dependency declaration names. The shape is cheap and the data is
            not, so the data is what goes.
        node_local_resources: The exclusive resource names a project may
            declare that each node runs its OWN copy of, so a lease on one
            node's copy never contends with a lease on another's
            (:func:`fleet.contracts.resources.scoped`). Every other declared
            name stays one thing in the whole fleet. ``corvis-fleet-testdb``
            is the first: a container every testdb node runs for itself,
            which read as fleet-wide until lavender-wsl became a second
            testdb node on 2026-09-29 and closed every testdb job it
            claimed as held (MCPs board task c4fc4f3e).
        node_serve_seconds: How long, from its start, a node runner serves
            before it hands over to the next scheduled start at an idle fire
            boundary (:mod:`fleet.cli.node_serve`, MCPs board task 8993c306),
            which bounds how long a runner reads the registry it started
            with. DECLARED HERE because this document is what the scheduled
            start runs beside the rolled agent, so the value and the code
            that reads it roll together. At most
            :data:`NODE_SERVE_CEILING_SECONDS`; zero hands over at the first
            boundary.
        node_poll_seconds: How often a serving runner's watch reads the
            runs it holds and its main thread lists the queued jobs
            (:mod:`fleet.cli.node_watch`, :mod:`fleet.cli.node_serve`), so
            how long an ended run or an arrived job can wait before the
            runner sees it. From 1 to :data:`NODE_POLL_CEILING_SECONDS`.
        ledger: Path to the append-only dispatch record. Relative paths
            resolve against the workspace document's own directory, so a
            workspace can be moved without editing it.
        feed: Path to the append-only event stream subscribers tail.
        leases: Path to the live lease file. The only mutable one.
    """

    machines: dict[str, Machine]
    nodes: dict[str, NodeConfig]
    not_dispatchable: dict[str, str]
    projects: dict[str, ProjectConfig]
    data_paths: dict[str, tuple[str, ...]]
    node_local_resources: tuple[str, ...]
    node_serve_seconds: int
    node_poll_seconds: int
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
        JSON-serialisable mapping carrying every field, the machines as
        :func:`fleet.contracts.machine.encode_machine` writes them.
    """
    return {
        "machines": {
            name: encode_machine(machine) for name, machine in workspace["machines"].items()
        },
        "projects": {
            path: encode_project_config(project) for path, project in workspace["projects"].items()
        },
        "data_paths": {remote: list(paths) for remote, paths in workspace["data_paths"].items()},
        "node_local_resources": encode_names(workspace["node_local_resources"]),
        "node_serve_seconds": workspace["node_serve_seconds"],
        "node_poll_seconds": workspace["node_poll_seconds"],
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


def _decode_node_local_resources(
    value: JSONObject, projects: dict[str, ProjectConfig]
) -> tuple[str, ...]:
    """Read the node-local resource names, refusing one no project declares.

    Args:
        value: The workspace object.
        projects: The decoded projects.

    Returns:
        The names, in declaration order.

    Raises:
        JSONTypeError: If the key is absent (a workspace written before the
            field existed is refused rather than read as "every resource is
            fleet-wide", which is the reading that serialised two testdb
            nodes), if it is not a list of non-empty strings, or if a name
            is exclusive to no project. That last one is a typo, and a typo
            here would leave the resource it meant fleet-wide while the
            workspace read as though it had been scoped.
    """
    if "node_local_resources" not in value:
        raise JSONTypeError(
            "workspace must declare node_local_resources, the exclusive resources each node "
            "runs its own copy of; [] says every declared resource is one thing in the fleet"
        )
    names = decode_names(value["node_local_resources"], field="node_local_resources")
    declared = {name for project in projects.values() for name in project["exclusive_resources"]}
    for name in names:
        if name not in declared:
            raise JSONTypeError(
                f"node_local_resources names {name!r}, which no project declares in "
                f"exclusive_resources; a misspelt name would leave the real one fleet-wide"
            )
    return names


def _decode_node_serve_seconds(value: JSONObject) -> int:
    """Read how long a node runner serves before it hands over.

    Args:
        value: The workspace object.

    Returns:
        The serve in seconds.

    Raises:
        JSONTypeError: If the field is missing, is not an integer, or lies
            outside zero to :data:`NODE_SERVE_CEILING_SECONDS`.
    """
    seconds = require_int(value, "node_serve_seconds")
    if not 0 <= seconds <= NODE_SERVE_CEILING_SECONDS:
        raise JSONTypeError(
            f"node_serve_seconds is {seconds}; it must lie between 0 and "
            f"{NODE_SERVE_CEILING_SECONDS}, or a runner still draining at its launcher's "
            "39-minute wall is ended mid-launch"
        )
    return seconds


def _decode_node_poll_seconds(value: JSONObject) -> int:
    """Read how often a serving node runner looks at its runs and the queue.

    Args:
        value: The workspace object.

    Returns:
        The poll in seconds.

    Raises:
        JSONTypeError: If the field is missing, is not an integer, or lies
            outside 1 to :data:`NODE_POLL_CEILING_SECONDS`.
    """
    seconds = require_int(value, "node_poll_seconds")
    if not 1 <= seconds <= NODE_POLL_CEILING_SECONDS:
        raise JSONTypeError(
            f"node_poll_seconds is {seconds}; it must lie between 1 and "
            f"{NODE_POLL_CEILING_SECONDS}: zero would read the node and the queue without "
            "pause, and a poll slower than a fire boundary sees nothing the boundary's "
            "collect pass does not"
        )
    return seconds


def _decode_data_paths(
    value: JSONObject, projects: dict[str, ProjectConfig]
) -> dict[str, tuple[str, ...]]:
    """Read the per-repository data directories an export leaves behind.

    Args:
        value: The workspace object.
        projects: The already-decoded projects, so a declaration that would
            strip one of them is refused here rather than discovered as a
            missing Makefile on a node.

    Returns:
        The declared paths, keyed by remote, in declaration order.

    Raises:
        JSONTypeError: If the field is missing, is not an object, a key is
            not a remote in the source grammar, a value is not a list of
            strings, a path is not a repo-relative directory in the path
            grammar, a path is the repository root, or a path ENCLOSES a
            project declared against that same remote.

            The enclosing case is checked against the projects for the same
            reason :func:`_decode_not_dispatchable` is checked against the
            nodes: the workspace would be saying two opposite things about
            one directory, that it is dispatchable work and that it is data
            no export carries, and the export would win silently by staging
            a tree with the project missing from it.

            REQUIRED, and empty spelled out as ``{}``. An absent key would
            read as "this fleet has no data directories" from a workspace
            where nobody had considered the question, and the difference
            between those two matters: the first stages 27 MB and the second
            stages 217 MB while looking identical.

            The root is refused because ``""`` names the whole repository,
            so declaring it would exclude every file from the export and
            produce an empty archive whose check fails on the node with a
            message about a missing Makefile.
    """
    declared = require_dict(value, "data_paths")
    paths: dict[str, tuple[str, ...]] = {}
    for remote in sorted(declared):
        entries = declared[remote]
        if not isinstance(entries, list):
            raise JSONTypeError(
                f"data_paths[{remote!r}] must be a list of repo-relative directories, got "
                f"{type(entries).__name__}"
            )
        decoded: list[str] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, str):
                raise JSONTypeError(
                    f"data_paths[{remote!r}][{index}] must be a string, got {type(entry).__name__}"
                )
            if not entry:
                raise JSONTypeError(
                    f"data_paths[{remote!r}][{index}] is the repository root, which would "
                    "exclude every file from the export and stage an empty archive"
                )
            decoded.append(decode_path(entry, field=f"data_paths[{remote!r}][{index}]"))
        checked = decode_remote(remote, field="data_paths key")
        for path in decoded:
            enclosed = [
                key
                for key, project in projects.items()
                if project["source"] is not None
                and project["source"]["remote"] == checked
                and _encloses(path, project["source"]["path"])
            ]
            if enclosed:
                raise JSONTypeError(
                    f"data_paths[{remote!r}] declares {path!r}, which contains the dispatchable "
                    f"project(s) {', '.join(sorted(enclosed))}. Excluding it would stage a tree "
                    "with the project itself missing, and the check would fail on the node with "
                    "a message about a missing Makefile rather than about this line"
                )
        paths[checked] = tuple(decoded)
    return paths


def _check_wsl_hosts(nodes: dict[str, NodeConfig]) -> dict[str, NodeConfig]:
    """Refuse a ``wsl_host`` that is not another Windows node of this workspace.

    Args:
        nodes: The decoded nodes.

    Returns:
        The same nodes.

    Raises:
        JSONTypeError: If a node's ``wsl_host`` names no declared node, or a
            node that is not Windows: the tick that asks the host what it sees
            would ask a machine it cannot reach, or one with no WSL to report.
    """
    for name in sorted(nodes):
        host = nodes[name]["wsl_host"]
        if host is None:
            continue
        declared = nodes.get(host)
        if declared is None:
            raise JSONTypeError(
                f"machines[{name!r}].guest names {host!r}, which declares no dispatch capacity"
            )
        if declared["platform"] is not NodePlatform.WINDOWS:
            raise JSONTypeError(
                f"machines[{name!r}].guest names {host!r}, a {declared['platform'].value} node; "
                "only a windows node runs WSL"
            )
    return nodes


def _encloses(data_path: str, project_path: str) -> bool:
    """Whether a data directory contains a project's own tree.

    Args:
        data_path: The declared data directory, never the repository root.
        project_path: The project's repo-relative path, ``""`` for a project
            that is the whole repository.

    Returns:
        True when excluding ``data_path`` would remove part or all of the
        project. A project that IS the repository is enclosed by every data
        path, since every one of them lies inside it.
    """
    if not project_path:
        return True
    return project_path == data_path or project_path.startswith(f"{data_path}/")


def decode_fleet_workspace(value: JSONValue) -> FleetWorkspace:
    """Decode and validate a workspace.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated workspace.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the machine or project mapping is empty, no machine
            declares capacity, a machine fails
            :func:`fleet.contracts.machine.decode_machine`, a node-local
            resource is declared by no project, a node's ``wsl_host`` is not
            another Windows node, a
            project is pinned to a node it cannot match, the
            node serve or poll lies outside its bounds, or a node or project fails
            its own decoder.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"workspace must be a JSON object, got {type(value).__name__}")
    machines = {
        name: decode_machine(name, machine)
        for name, machine in _decode_named(value, "machines").items()
    }
    nodes = _check_wsl_hosts(
        {
            name: node
            for name, machine in machines.items()
            if (node := machine["dispatch"]) is not None
        }
    )
    if not nodes:
        raise JSONTypeError(
            "no machine declares what dispatch may take from it; the workspace could dispatch "
            "nothing, and every command would fail naming an empty list"
        )
    projects = check_pinned_nodes(
        {
            path: decode_project_config(project)
            for path, project in _decode_named(value, "projects").items()
        },
        nodes,
    )
    return FleetWorkspace(
        machines=machines,
        nodes=nodes,
        not_dispatchable={
            name: held
            for name, machine in machines.items()
            if (held := machine["held"]) is not None
        },
        projects=projects,
        data_paths=_decode_data_paths(value, projects),
        node_local_resources=_decode_node_local_resources(value, projects),
        node_serve_seconds=_decode_node_serve_seconds(value),
        node_poll_seconds=_decode_node_poll_seconds(value),
        ledger=require_str(value, "ledger"),
        feed=require_str(value, "feed"),
        leases=require_str(value, "leases"),
    )


__all__ = [
    "NODE_POLL_CEILING_SECONDS",
    "NODE_SERVE_CEILING_SECONDS",
    "FleetWorkspace",
    "decode_fleet_workspace",
    "encode_fleet_workspace",
    "require_node",
    "require_project",
]
