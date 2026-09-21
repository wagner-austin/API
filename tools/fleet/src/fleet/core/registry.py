"""Reconciling this workspace against the fleet's identity registry.

THE FLEET IS WRITTEN DOWN TWICE, IN TWO REPOSITORIES, AND THE COLUMNS BARELY
OVERLAP:

  MCPs  fleet-mcp/fleet-nodes.json   name, role, user, PLATFORM, tailnetIp,
                                     ENABLED, tunnel, notes -- identity and
                                     reachability, every machine on the
                                     tailnet including ones nothing dispatches
                                     to (a phone, two boxes offline since
                                     August).
  API   tools/fleet/fleet.json       host, PLATFORM, stage_root,
                                     logical_cores, ram_gb, gpu, budget --
                                     what a dispatch may take from a machine.

Merging them would put a Cloudflare tunnel id beside a worker-RAM budget and
make one repo depend on the other. They are two VIEWS, and that is fine.

WHAT IS NOT FINE ARE THE TWO FACTS THEY SHARE. Whether a machine is expected
to answer: on 2026-09-05 the identity registry marked loki disabled for a trip
and this workspace never learned, because it had no field to learn it into.
Every auto-select dispatch then paid a ten-second ssh timeout rediscovering
it, and one was refused outright. And which platform it is: every script a
dispatch sends is rendered in the platform's dialect, so a workspace that
called a Linux box Windows would send it PowerShell and read the parse error
as the node's fault.

So the workspace now declares ``enabled`` and ``platform`` itself -- the API
repo must be able to dispatch with no MCPs checkout present -- and this
module makes the two CHECKABLE against each other. Detection, not prevention:
nothing here stops somebody adding a node to one file and not the other. It
stops that going unnoticed.

THE PATH IS PASSED IN, NEVER GUESSED. A reconciler that searched for the other
repo would silently report success on a machine where it simply failed to find
it, which is the failure mode this exists to end.
"""

from __future__ import annotations

import pathlib
from typing import TypedDict

from platform_core.error_codes_tooling import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONValue,
    load_json_str,
    narrow_json_to_str,
    require_bool,
    require_str,
)

from fleet.contracts.node import NodePlatform, decode_node_platform
from fleet.contracts.workspace import FleetWorkspace
from fleet.core import _test_hooks


class RegistryNode(TypedDict):
    """One machine as the identity registry declares it.

    A DELIBERATE SUBSET. That registry also carries tailnet addresses, tunnel
    ids and prose notes; none of it is this workspace's business, and
    decoding fields nothing reads would make the contract wider than the
    dependency. ``role`` and ``user`` joined ``name`` and ``enabled`` when
    the session-ledger observer started walking the registry (MCPs board
    task 5a3865bf): it must skip the hub (observed by pcsession-mcp) and
    clients (never provisioned), and it needs the account to write a script
    under.

    Attributes:
        name: The machine's registry name, which is also the workspace key
            and the hub's ssh alias for it.
        enabled: Whether it is expected to answer.
        role: ``hub``, ``vpn-jump``, ``worker`` or ``client``.
        user: The account the fleet provisioned on it, or None for a client.
        platform: ``windows`` or ``linux``, decoded into the closed set at
            the edge (board task cd5010c4): the session observer renders its
            script in this platform's dialect, so a value with no dialect is
            a registry this workspace cannot act on and is refused here, not
            discovered as a parse error on the far side. The registry's own
            vocabulary is the same two words (``fleet-mcp``'s
            ``FLEET_PLATFORMS``; a phone is ``linux`` there), so nothing it
            can carry is lost by the narrowing.
        cuda: Whether the registry's measured ``gpu`` column names a CUDA
            device, which is what this workspace's ``gpu`` capability tag
            means (:func:`fleet.contracts.tags.node_tags`): true when the
            column is an object whose ``probe`` is ``nvidia-smi``, false
            for ``null`` and for an adapter the Win32 video-controller probe
            found (integrated graphics, no CUDA). The column is a
            measurement (MCPs board task 41f45bd7) and this workspace's
            ``gpu`` is a declaration, and a job tagged ``gpu`` is matched on
            the declaration, so the two must agree (board task fd5cabfa, A1).
    """

    name: str
    enabled: bool
    role: str
    user: str | None
    platform: NodePlatform
    cuda: bool


class RegistryDrift(TypedDict):
    """How the two registries disagree.

    Every list holds workspace-or-registry NAMES, never prose, so a caller can
    act on them rather than parse a sentence.

    Attributes:
        enabled_here_disabled_there: Nodes this workspace will dispatch to
            that the registry says are off. THE EXPENSIVE ONE -- every
            dispatch pays an ssh timeout for each.
        disabled_here_enabled_there: Nodes this workspace is skipping that the
            registry says are live. Quiet, and worse in its way: capacity that
            exists and is not being used, with nothing to notice it.
        missing_from_registry: Nodes this workspace dispatches to that the
            registry has never heard of. Unprovisioned, or renamed.
        enabled_there_absent_here: Machines the registry says are live that
            this workspace neither dispatches to NOR declares
            ``not_dispatchable``. Capacity nobody has decided about --
            distinct from ``disabled_here_enabled_there``, where a decision
            was made and disagrees.
        platform_disagrees: Nodes the two files place on different
            platforms, as ``(name, here, there)``. The one that would be
            found by a dispatch rather than a reconciler: every script goes
            out in the platform's dialect, and the wrong one is a parse
            error on the far side that reads as the node's fault.
        gpu_disagrees: Nodes this workspace declares a CUDA device for that
            the registry measured none on, or the reverse, as
            ``(name, here, there)`` booleans. The node-lane runners claim
            by the tags this workspace derives, so a ``gpu`` job would land
            on a box whose measured adapter cannot run it, or never land on
            one that could.
    """

    enabled_here_disabled_there: tuple[str, ...]
    disabled_here_enabled_there: tuple[str, ...]
    missing_from_registry: tuple[str, ...]
    enabled_there_absent_here: tuple[str, ...]
    platform_disagrees: tuple[tuple[str, str, str], ...]
    gpu_disagrees: tuple[tuple[str, bool, bool], ...]


#: The registry's probe whose adapters are CUDA devices; the other probe,
#: ``win32-videocontroller``, reports integrated graphics.
CUDA_PROBE = "nvidia-smi"


def _decode_cuda(entry: dict[str, JSONValue]) -> bool:
    """Read whether a registry node's measured gpu is a CUDA device.

    Args:
        entry: The node's registry object.

    Returns:
        True iff ``gpu`` is an object whose ``probe`` is :data:`CUDA_PROBE`.

    Raises:
        AppError: ``NODE_REGISTRY_UNREADABLE`` when the ``gpu`` key is
            absent (the column was filled for every node on 2026-09-21 and
            an absent key is a shape change, not a machine without one) or
            is neither null nor an object with a string ``probe``.
    """
    if "gpu" not in entry:
        raise _unreadable(f"node {entry.get('name')!r} carries no 'gpu' key (null means none)")
    gpu = entry["gpu"]
    if gpu is None:
        return False
    if not isinstance(gpu, dict):
        raise _unreadable(
            f"node {entry.get('name')!r} 'gpu' is {type(gpu).__name__}, not an object"
        )
    return require_str(gpu, "probe") == CUDA_PROBE


def decode_registry_nodes(raw: str) -> dict[str, RegistryNode]:
    """Read the identity registry's node list.

    Args:
        raw: The registry document's text.

    Returns:
        Every declared node, keyed by name.

    Raises:
        AppError: ``NODE_REGISTRY_UNREADABLE`` when the document is not the
            shape the registry has always had -- an object with a ``nodes``
            array of objects carrying ``name``, ``enabled``, ``role`` and
            ``platform``. Raised rather
            than skipped: a reconciler that shrugged at an unreadable registry
            would report agreement it never established.
    """
    document: JSONValue = load_json_str(raw)
    if not isinstance(document, dict):
        raise _unreadable(f"the registry is {type(document).__name__}, not an object")
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise _unreadable(f"'nodes' is {type(nodes).__name__}, not an array")
    declared: dict[str, RegistryNode] = {}
    for entry in nodes:
        if not isinstance(entry, dict):
            raise _unreadable(f"a node is {type(entry).__name__}, not an object")
        name = require_str(entry, "name")
        raw_user = entry.get("user")
        declared[name] = RegistryNode(
            name=name,
            enabled=require_bool(entry, "enabled"),
            role=require_str(entry, "role"),
            # ``null`` for a client such as the phone, which is never
            # provisioned and has no account of ours to ssh in as.
            user=None if raw_user is None else narrow_json_to_str(raw_user),
            platform=decode_node_platform(require_str(entry, "platform")),
            cuda=_decode_cuda(entry),
        )
    return declared


def _unreadable(detail: str) -> AppError[FleetErrorCode]:
    """Build the refusal for a registry this cannot read.

    Args:
        detail: What was wrong with it.

    Returns:
        The error to raise.
    """
    return AppError(
        FleetErrorCode.NODE_REGISTRY_UNREADABLE,
        f"the fleet identity registry cannot be read: {detail}. Expected the shape "
        "fleet-mcp/fleet-nodes.json has always had -- an object with a 'nodes' array, "
        "each entry carrying 'name', 'enabled', 'role', 'platform' and 'gpu' (null or an "
        "object with a 'probe').",
    )


def compare(workspace: FleetWorkspace, registry: dict[str, RegistryNode]) -> RegistryDrift:
    """Compare this workspace's nodes against the identity registry.

    MOSTLY ONE-DIRECTIONAL. A node in the registry that this workspace does
    not declare is usually not drift: the registry holds every machine on the
    tailnet, including a phone and boxes offline since August, and nothing
    dispatches to those. Reporting them would be the noise that gets a check
    switched off.

    THE ONE EXCEPTION IS A MACHINE THE REGISTRY SAYS IS LIVE. Silence about a
    box that is off is a safe silence; silence about one that is running is
    capacity nobody has decided about, and it hides in exactly the direction
    that never announces itself. ``austinpc`` sat there for weeks: enabled,
    24 logical cores, and invisible to the scheduler, with no way to tell a
    deliberate exclusion from an oversight. So the workspace's
    ``not_dispatchable`` map answers it, and only an undecided machine is
    reported.

    Args:
        workspace: The dispatch workspace.
        registry: The identity registry's nodes, from
            :func:`decode_registry_nodes`.

    Returns:
        The four ways they can disagree, each a sorted tuple of node names.
    """
    enabled_here_disabled_there: list[str] = []
    disabled_here_enabled_there: list[str] = []
    missing: list[str] = []
    undecided: list[str] = []
    platform_disagrees: list[tuple[str, str, str]] = []
    gpu_disagrees: list[tuple[str, bool, bool]] = []
    for name, node in sorted(workspace["nodes"].items()):
        declared = registry.get(name)
        if declared is None:
            missing.append(name)
            continue
        if node["enabled"] and not declared["enabled"]:
            enabled_here_disabled_there.append(name)
        if not node["enabled"] and declared["enabled"]:
            disabled_here_enabled_there.append(name)
        if node["platform"] != declared["platform"]:
            platform_disagrees.append((name, node["platform"], declared["platform"]))
        here_cuda = node["gpu"] is not None
        if here_cuda != declared["cuda"]:
            gpu_disagrees.append((name, here_cuda, declared["cuda"]))
    for name, declared in sorted(registry.items()):
        unmentioned = name not in workspace["nodes"] and name not in workspace["not_dispatchable"]
        if declared["enabled"] and unmentioned:
            undecided.append(name)
    return RegistryDrift(
        enabled_here_disabled_there=tuple(enabled_here_disabled_there),
        disabled_here_enabled_there=tuple(disabled_here_enabled_there),
        missing_from_registry=tuple(missing),
        enabled_there_absent_here=tuple(undecided),
        platform_disagrees=tuple(platform_disagrees),
        gpu_disagrees=tuple(gpu_disagrees),
    )


def has_drifted(drift: RegistryDrift) -> bool:
    """Whether the two registries disagree at all.

    Args:
        drift: What :func:`compare` found.

    Returns:
        True when any of the six disagreements is non-empty.
    """
    return bool(
        drift["enabled_here_disabled_there"]
        or drift["disabled_here_enabled_there"]
        or drift["missing_from_registry"]
        or drift["enabled_there_absent_here"]
        or drift["platform_disagrees"]
        or drift["gpu_disagrees"]
    )


def describe(drift: RegistryDrift, *, registry_path: str) -> tuple[str, ...]:
    """Render the drift as lines, one per disagreement.

    Each line says which way the disagreement runs and what it costs, because
    the two directions call for opposite fixes and a reader who gets them
    backwards edits the wrong file.

    Args:
        drift: What :func:`compare` found.
        registry_path: Where the identity registry was read from, so the
            reader knows which two files to reconcile.

    Returns:
        The lines, empty when the two agree.
    """
    lines: list[str] = []
    for name in drift["enabled_here_disabled_there"]:
        lines.append(
            f"{name}: this workspace dispatches to it, {registry_path} says it is "
            "disabled. Every auto-select dispatch pays an ssh timeout for it."
        )
    for name in drift["disabled_here_enabled_there"]:
        lines.append(
            f"{name}: this workspace skips it, {registry_path} says it is enabled. "
            "Capacity that exists and is not being used."
        )
    for name in drift["missing_from_registry"]:
        lines.append(
            f"{name}: this workspace dispatches to it and {registry_path} has never "
            "heard of it. Unprovisioned, or renamed on one side only."
        )
    for name in drift["enabled_there_absent_here"]:
        lines.append(
            f"{name}: {registry_path} says it is enabled and this workspace says nothing "
            "at all. Declare it as a node, or as not_dispatchable with the reason -- "
            "silence cannot be told apart from an oversight."
        )
    for name, here, there in drift["platform_disagrees"]:
        lines.append(
            f"{name}: this workspace says {here}, {registry_path} says {there}. Every "
            "script a dispatch sends is rendered for the declared platform, so the wrong "
            "one is a parse error on the node that reads as the node's fault."
        )
    for name, here_cuda, there_cuda in drift["gpu_disagrees"]:
        here = "declares a CUDA device" if here_cuda else "declares no gpu"
        there = (
            f"measured one with {CUDA_PROBE}"
            if there_cuda
            else "measured none (null, or an adapter the Win32 probe found)"
        )
        lines.append(
            f"{name}: this workspace {here}, {registry_path} {there}. The node runners "
            "claim by the tags derived here, so a gpu job is matched on the declaration "
            "and runs, or never runs, on the wrong box."
        )
    return tuple(lines)


def reconcile(workspace: FleetWorkspace, *, registry_path: str) -> tuple[str, ...]:
    """Read the identity registry and report how it disagrees with this one.

    Args:
        workspace: The dispatch workspace.
        registry_path: Absolute path to ``fleet-mcp/fleet-nodes.json``.

    Returns:
        One line per disagreement, empty when the two agree.

    Raises:
        AppError: ``NODE_REGISTRY_UNREADABLE`` when the path does not hold a
            registry this can read. Note the path itself not existing raises
            from the reader rather than here -- a reconciler pointed at
            nothing has not established agreement and must not say it has.
    """
    registry = decode_registry_nodes(_test_hooks.read_text(pathlib.Path(registry_path)))
    return describe(compare(workspace, registry), registry_path=registry_path)


__all__ = [
    "RegistryDrift",
    "RegistryNode",
    "compare",
    "decode_registry_nodes",
    "describe",
    "has_drifted",
    "reconcile",
]
