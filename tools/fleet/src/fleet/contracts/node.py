"""One machine the fleet may dispatch to, and what it is currently holding.

A node is declared once, in the workspace, and probed live before every
dispatch. The two are different types on purpose: :class:`NodeConfig` is what
somebody wrote down and :class:`NodeState` is what the machine just said, and
conflating them is how a dispatcher ends up trusting a free-memory figure that
was true last week.

WHAT A NODE IS NOT ALLOWED TO OMIT. Every field below is required, including
``gpu`` -- which is spelled as an explicit ``None`` for a CPU-only node rather
than left out. Absence is not a state: a missing key is indistinguishable from
a key nobody has filled in yet, and the one time that matters is when a
measurement is about to be pinned to a card.
"""

from __future__ import annotations

from enum import StrEnum

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_str,
)
from platform_core.members import find_member
from typing_extensions import TypedDict

from fleet.contracts.budget import NodeBudget, decode_node_budget, encode_node_budget
from fleet.contracts.capability import Capability, decode_capability


#: The operating-system family a node runs, which decides every script a
#: dispatch sends it: how a file is written over ssh, how a script is run by
#: path, how the suite is detached from the connection (Task Scheduler on
#: one, a transient systemd user unit on the other), and how capacity is read.
#:
#: A closed set rather than a free string because each value names a whole
#: dialect in :mod:`fleet.core.dialect`, and a third value would need a third
#: one before it could dispatch anything. The identity registry
#: (``fleet-mcp/fleet-nodes.json``) declares the same field under the same
#: name, and ``fleet-nodes --registry`` reports the two disagreeing. Each
#: member's value is the word the workspace and the registry spell it with.
class NodePlatform(StrEnum):
    """The operating-system family a node runs."""

    WINDOWS = "windows"
    LINUX = "linux"


class NodeGpu(TypedDict):
    """A CUDA device on a node, as its driver reports it.

    Attributes:
        model: The device name, e.g. ``NVIDIA GeForce GTX 1630``.
        vram_mib: Total device memory in mebibytes, the unit ``nvidia-smi``
            reports so that no conversion sits between the tool and the
            record.
        compute_capability: The architecture, e.g. ``7.5``. Held as a STRING
            because it is a version rather than a quantity: ``8.0`` and
            ``8.10`` do not order as floats, and nothing here does arithmetic
            on it. This is the field that decides whether two nodes can be
            asked the same question -- measured 2026-09-04, the fleet holds
            7.5 and 8.6, which is two architectures rather than one.
        driver_version: The driver, e.g. ``591.86``. Recorded beside the
            architecture because a cross-node comparison that holds the
            driver constant answers a sharper question than one that does
            not.
    """

    model: str
    vram_mib: int
    compute_capability: str
    driver_version: str


class NodeConfig(TypedDict):
    """A machine the fleet may dispatch to, as declared in the workspace.

    Attributes:
        host: SSH destination -- an alias from ``~/.ssh/config``, not a
            hostname or an address. The alias carries the user and the key,
            and a tailnet address changes without warning while an alias does
            not.
        platform: Which :data:`NodePlatform` the node is, and so which script
            dialect every remote act uses. Declared rather than probed
            because it is needed to write the very first probe, and declared
            rather than read from the identity registry because the API repo
            must dispatch with no MCPs checkout present.
        stage_root: Absolute directory on the node holding staged working
            trees, one per run. Declared rather than derived from a home
            directory, because the three live nodes disagree about where a
            writable directory lives.
        logical_cores: The node's total logical processors. Written down as
            well as probed so a preflight can be reasoned about offline, and
            so a node that silently changes shape shows up as a mismatch
            rather than as a different answer.
        ram_gb: Total physical memory.
        gpu: The node's CUDA device, or None for a CPU-only node. Explicitly
            None rather than absent -- see the module docstring.
        enabled: Whether this machine is expected to answer at all. False for
            one that is deliberately off -- travelling, unprovisioned,
            retired -- and it is the difference between "did not answer" and
            "was never asked".

            THIS FIELD IS THE HALF OF THE FLEET REGISTRY THIS FILE WAS
            MISSING. ``fleet-mcp/fleet-nodes.json`` in the MCPs repo has
            carried ``enabled`` since the fleet was first written down; this
            workspace had no way to say it. On 2026-09-05 that registry marked
            loki off for a trip, this one never learned, and every auto-select
            dispatch paid a ten-second ssh timeout rediscovering it -- one of
            which was refused outright. Declared here rather than derived,
            because the API repo must be able to dispatch without an MCPs
            checkout; ``fleet-nodes --registry`` reconciles the two.
        test_database: Whether the node runs the fleet test database: the
            loopback postgres container ``corvis-fleet-testdb`` that MCPs
            ``scripts/testdb-setup.sh --container`` restarts empty and
            migrates before each run, provisioned on diphtheria by MCPs
            ``scripts/host/diphtheria/provision.sh``. True gives the node
            the ``testdb`` tag (:mod:`fleet.contracts.tags`), which every
            project whose suite needs a migrated ``corvis_test`` requires.

            DECLARED, AND REQUIRED LIKE ``enabled`` (MCPs board task
            6bbfd171). No Windows node can reach a test database (measured
            2026-09-26: sedona and serendipity both refuse the cluster's
            6432), so a Postgres-backed package handed to one fails its
            global setup; a default of false would hide a provisioned node
            from those packages and a default of true would send them to
            nodes with no database, so the workspace says which.
        rust: The version ``cargo --version`` printed on this node, e.g.
            ``1.98.1``, or None for a node with no Rust toolchain. Non-null
            gives the node the ``rust`` tag a crate-building project
            requires, and the runner's probe re-measures it every tick
            (:mod:`fleet.contracts.capability`, MCPs board task 1e2da299).
            REQUIRED like ``gpu``, null spelled out, for the same reason.
        cxx: The version of the C++ toolchain node-gyp would build with,
            as the probe reads it (vswhere's VC tools installationVersion
            on Windows, ``g++ -dumpfullversion`` on Linux), or None. Gives
            the ``cxx`` tag every project whose ``npm ci`` rebuilds a native
            module requires; required and re-measured like ``rust`` (MCPs
            board task 3f19c136).
        docker: The ServerVersion the execdocker user's ROOTLESS daemon
            reports, read only when that daemon also says it is rootless, or
            None. Gives the ``docker`` tag the MCPs deploy suite requires,
            so it runs on a daemon that cannot reach the stack's; required
            and re-measured like ``rust`` (MCPs board task 6c4516af).
        stack: The ServerVersion of the daemon the node's own account
            reaches, read only while that daemon holds the corvis stack's
            network and images (:data:`fleet.contracts.capability.STACK_IMAGES`),
            or None. Gives the ``stack`` tag the suites that start those
            images require; required and re-measured like ``rust`` (MCPs
            board task 554bffc1).
        elevated: Whether the node runs a second, ELEVATED runner, whose
            builds launch at RunLevel Highest for suites that register what
            only an administrator may (MCPs board task a98d7083). True gives
            the ``elevated`` tag and a second hub task for the node
            (``scripts/register-node-agents.ps1``); that runner re-reads its
            ssh session's token every tick and claims nothing unless the
            token is an administrator's (:mod:`fleet.contracts.elevation`).
            REQUIRED like ``test_database``, and refused on a linux node,
            where the equivalent would be a root build and no suite asks
            for one.
        wsl_host: The workspace name of the Windows node whose WSL runs
            this node's distro, or None for a machine of its own. A tick
            that cannot reach such a node asks that host what it sees (MCPs
            board task 45a4f22b: lavender-wsl read 'did not answer' for
            hours while lavender itself answered with 0.3 GB free). REQUIRED,
            null spelled out, and refused on a Windows node.
        budget: What share of this machine a dispatch may take.
    """

    host: str
    platform: NodePlatform
    stage_root: str
    logical_cores: int
    ram_gb: float
    gpu: NodeGpu | None
    enabled: bool
    test_database: bool
    rust: str | None
    cxx: str | None
    docker: str | None
    stack: str | None
    elevated: bool
    wsl_host: str | None
    budget: NodeBudget


class NodeState(TypedDict):
    """What a node reported when it was last probed.

    Separate from :class:`NodeConfig` because these are perishable. A
    dispatcher that cached them would be making the mistake this package
    exists to prevent one layer up: acting on a capacity reading that was
    true when somebody wrote it down.

    Attributes:
        host: The alias that was probed, so a state cannot be attributed to
            the wrong node after being passed around.
        free_ram_gb: Physical memory free at probe time.
        free_disk_gb: Free space on the staging drive at probe time.
        live_runs: Fleet dispatches currently live on this node, counted from
            the ledger rather than from the process table -- a run is ours
            because we recorded it, not because a process looks like one.
    """

    host: str
    free_ram_gb: float
    free_disk_gb: float
    live_runs: int


def encode_node_gpu(gpu: NodeGpu) -> JSONObject:
    """Encode a node's CUDA device.

    Args:
        gpu: The device to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "model": gpu["model"],
        "vram_mib": gpu["vram_mib"],
        "compute_capability": gpu["compute_capability"],
        "driver_version": gpu["driver_version"],
    }


def decode_node_gpu(value: JSONValue) -> NodeGpu:
    """Decode and validate a node's CUDA device.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated device.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or ``vram_mib`` is not positive. A card reporting no
            memory is a probe that failed rather than a card, and recording
            it would let a capacity check believe a device is present that
            nothing can run on.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"node gpu must be a JSON object, got {type(value).__name__}")
    vram_mib = require_int(value, "vram_mib")
    if vram_mib <= 0:
        raise JSONTypeError(
            f"vram_mib must be positive, got {vram_mib}; a card reporting no memory is a "
            "failed probe rather than a device"
        )
    return NodeGpu(
        model=require_str(value, "model"),
        vram_mib=vram_mib,
        compute_capability=require_str(value, "compute_capability"),
        driver_version=require_str(value, "driver_version"),
    )


def encode_node_config(node: NodeConfig) -> JSONObject:
    """Encode one node's declaration.

    Args:
        node: The node to encode.

    Returns:
        JSON-serialisable mapping carrying every field, with ``gpu`` present
        and null for a CPU-only node rather than omitted.
    """
    gpu = node["gpu"]
    return {
        "host": node["host"],
        "platform": node["platform"].value,
        "stage_root": node["stage_root"],
        "logical_cores": node["logical_cores"],
        "ram_gb": node["ram_gb"],
        "gpu": None if gpu is None else encode_node_gpu(gpu),
        "enabled": node["enabled"],
        "test_database": node["test_database"],
        "rust": node["rust"],
        "cxx": node["cxx"],
        "docker": node["docker"],
        "stack": node["stack"],
        "elevated": node["elevated"],
        "wsl_host": node["wsl_host"],
        "budget": encode_node_budget(node["budget"]),
    }


def decode_node_config(value: JSONValue) -> NodeConfig:
    """Decode and validate one node's declaration.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated node.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, ``gpu`` is absent rather than explicitly null,
            ``platform`` is not one of :class:`NodePlatform`'s words, or the
            machine's own numbers are not positive.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"node must be a JSON object, got {type(value).__name__}")
    platform = decode_node_platform(require_str(value, "platform"))
    if "gpu" not in value:
        raise JSONTypeError(
            "node must declare 'gpu', using null for a CPU-only machine. An absent key is "
            "indistinguishable from one nobody has filled in, and the difference matters "
            "the moment a measurement is pinned to a card."
        )
    if "enabled" not in value:
        raise JSONTypeError(
            "node must declare 'enabled'. Defaulting it to true is what let a machine that "
            "was deliberately powered off keep being dispatched to, ten seconds of ssh "
            "timeout at a time; defaulting it to false would silently shrink the fleet. "
            "Neither guess is safe, so the workspace says which."
        )
    if "test_database" not in value:
        raise JSONTypeError(
            "node must declare 'test_database': true only for a node that runs the fleet test "
            "database (corvis-fleet-testdb), false otherwise. A Postgres-backed package handed "
            "to a node without one fails its global setup, so neither default is safe."
        )
    for capability in Capability:
        if capability.value not in value:
            raise JSONTypeError(
                f"node must declare {capability.value!r}: the version of that toolchain its "
                "probe reports, or null for a node without one. A build handed to a node "
                "without it fails during its install, so an absent key is not a safe way to "
                "say none."
            )
    elevated = _decode_elevated(value, platform)
    logical_cores = require_int(value, "logical_cores")
    if logical_cores < 1:
        raise JSONTypeError(f"logical_cores must be at least 1, got {logical_cores}")
    stage_root = require_str(value, "stage_root")
    if not stage_root:
        raise JSONTypeError("stage_root must not be empty; it is where the working tree lands")
    gpu_value = value["gpu"]
    return NodeConfig(
        host=require_str(value, "host"),
        platform=platform,
        stage_root=stage_root,
        logical_cores=logical_cores,
        ram_gb=_positive_float(value, "ram_gb"),
        gpu=None if gpu_value is None else decode_node_gpu(gpu_value),
        enabled=require_bool(value, "enabled"),
        test_database=require_bool(value, "test_database"),
        rust=decode_capability(Capability.RUST, value["rust"]),
        cxx=decode_capability(Capability.CXX, value["cxx"]),
        docker=decode_capability(Capability.DOCKER, value["docker"]),
        stack=decode_capability(Capability.STACK, value["stack"]),
        elevated=elevated,
        wsl_host=_decode_wsl_host(value, platform),
        budget=decode_node_budget(require_dict(value, "budget")),
    )


def declared_capability(node: NodeConfig, capability: Capability) -> str | None:
    """Read a node's declaration of one toolchain.

    Args:
        node: The node's declaration.
        capability: The toolchain.

    Returns:
        The declared version, or None.
    """
    if capability is Capability.RUST:
        return node["rust"]
    if capability is Capability.CXX:
        return node["cxx"]
    if capability is Capability.DOCKER:
        return node["docker"]
    return node["stack"]


def decode_node_platform(value: str) -> NodePlatform:
    """Read a platform name into the closed set.

    Args:
        value: The declared name.

    Returns:
        The platform.

    Raises:
        JSONTypeError: If it is not one of :class:`NodePlatform`'s words. A
            value outside the set has no dialect, so nothing could be sent
            to the node; refusing here is what keeps that from surfacing as
            a script the far side cannot parse.
    """
    platform = find_member(value, NodePlatform)
    if platform is not None:
        return platform
    raise JSONTypeError(
        f"platform must be one of {', '.join(NodePlatform)}, got {value!r}; each names the "
        "script dialect every remote act uses, and a value outside the set has none"
    )


def _decode_elevated(obj: JSONObject, platform: NodePlatform) -> bool:
    """Read whether a node runs a second, elevated runner.

    Args:
        obj: The node's declaration.
        platform: Its already decoded platform.

    Returns:
        The declared value.

    Raises:
        JSONTypeError: If the key is absent or not a bool, or is true on a
            node that is not Windows.
    """
    if "elevated" not in obj:
        raise JSONTypeError(
            "node must declare 'elevated': true only for a Windows node whose ssh account is an "
            "administrator and which should run a second, elevated runner, false otherwise. "
            "True registers a runner that launches builds as an administrator, so it is never a "
            "default."
        )
    elevated = require_bool(obj, "elevated")
    if elevated and platform is not NodePlatform.WINDOWS:
        raise JSONTypeError(
            f"elevated is true on a {platform.value} node; an elevated runner launches its "
            "builds as a Task Scheduler task at RunLevel Highest, which only a Windows node has"
        )
    return elevated


def _decode_wsl_host(obj: JSONObject, platform: NodePlatform) -> str | None:
    """Read which Windows node runs this node's distro.

    Whether the name is another Windows node of the same workspace is the
    workspace decoder's check, since one node cannot see the others.

    Args:
        obj: The node's declaration.
        platform: Its already decoded platform.

    Returns:
        The host's workspace name, or None.

    Raises:
        JSONTypeError: If the key is absent, is neither a non-empty string
            nor null, or names a host on a Windows node, which runs no
            distro of its own that the fleet dispatches to.
    """
    if "wsl_host" not in obj:
        raise JSONTypeError(
            "node must declare 'wsl_host': the workspace name of the Windows node whose WSL runs "
            "this node's distro, or null for a machine of its own"
        )
    value = obj["wsl_host"]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise JSONTypeError(f"wsl_host must be a non-empty string or null, got {value!r}")
    if platform is NodePlatform.WINDOWS:
        raise JSONTypeError(
            f"wsl_host {value!r} is set on a windows node; only a linux node runs inside another "
            "machine's WSL"
        )
    return value


def _positive_float(obj: JSONObject, key: str) -> float:
    """Read a float that describes a physical quantity a machine has.

    Args:
        obj: The object to read from.
        key: The field name.

    Returns:
        The value.

    Raises:
        JSONTypeError: If the field is missing, mistyped, or not positive.
    """
    found = require_float(obj, key)
    if found <= 0.0:
        raise JSONTypeError(f"{key} must be positive, got {found}")
    return found


def describe_node(node: NodeConfig, state: NodeState) -> str:
    """Render a node's declaration against what it just reported.

    Args:
        node: The declared node.
        state: What it reported when last probed.

    Returns:
        One line naming the host, its architecture if it has a card, and what
        is free right now -- the ``sinfo`` line for a machine with an owner.
    """
    gpu = node["gpu"]
    card = "cpu-only" if gpu is None else f"{gpu['model']} sm_{gpu['compute_capability']}"
    return (
        f"{node['host']}: {card}, {state['free_ram_gb']:.1f}/{node['ram_gb']:.1f} GB RAM free, "
        f"{state['free_disk_gb']:.0f} GB disk free, {state['live_runs']} live run(s)"
    )


__all__ = [
    "NodeConfig",
    "NodeGpu",
    "NodePlatform",
    "NodeState",
    "declared_capability",
    "decode_node_config",
    "decode_node_gpu",
    "decode_node_platform",
    "describe_node",
    "encode_node_config",
    "encode_node_gpu",
]
