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
from typing_extensions import TypedDict

from fleet.contracts.action_cache import ActionCacheCounts
from fleet.contracts.budget import NodeBudget, decode_node_budget, encode_node_budget
from fleet.contracts.capability import Capability, decode_capability


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
    """A machine the fleet may dispatch to: its roster file's ``dispatch`` object.

    ``host`` and ``enabled`` are the machine's own facts, said once in its
    file (:mod:`fleet.contracts.machine`) and handed to
    :func:`decode_node_config`; the rest is the ``dispatch`` object's. Every
    node is a Linux machine, which its machine file is refused for unless it
    says (:func:`fleet.contracts.machine.decode_machine`).

    Attributes:
        host: SSH destination: the machine's roster name, which is its alias
            in ``~/.ssh/config``, not a hostname or an address. The alias
            carries the user and the key, and a tailnet address changes
            without warning while an alias does not.
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
        enabled: Whether the node's runners claim work: the machine is
            expected to answer and its dispatch is not held. False for one
            that is deliberately off -- travelling, unprovisioned, retired --
            and it is the difference between "did not answer" and "was never
            asked": on 2026-09-05 loki was off for a trip and every
            auto-select dispatch paid a ten-second ssh timeout rediscovering
            it, one of which was refused outright.
        test_database: Whether the node runs the fleet test database: the
            loopback postgres container ``corvis-fleet-testdb`` that MCPs
            ``scripts/testdb-setup.sh --container`` restarts empty and
            migrates before each run, provisioned on diphtheria by MCPs
            ``scripts/host/diphtheria/provision.sh``. True gives the node
            the ``testdb`` tag (:mod:`fleet.contracts.tags`), which every
            project whose suite needs a migrated ``corvis_test`` requires.

            DECLARED, AND REQUIRED LIKE ``enabled`` (MCPs board task
            6bbfd171). A node without the container cannot reach a test
            database (measured 2026-09-26: sedona and serendipity both
            refuse the cluster's 6432), so a Postgres-backed package handed
            to one fails its global setup; a default of false would hide a provisioned node
            from those packages and a default of true would send them to
            nodes with no database, so the workspace says which.
        rust: The version ``cargo --version`` printed on this node, e.g.
            ``1.98.1``, or None for a node with no Rust toolchain. Non-null
            gives the node the ``rust`` tag a crate-building project
            requires, and the runner's probe re-measures it every tick
            (:mod:`fleet.contracts.capability`, MCPs board task 1e2da299).
            REQUIRED like ``gpu``, null spelled out, for the same reason.
        cxx: The version of the C++ toolchain node-gyp would build with,
            as the probe reads it (``g++ -dumpfullversion``), or None. Gives
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
        budget: What share of this machine a dispatch may take.
    """

    host: str
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
    budget: NodeBudget


class SliceMemory(TypedDict):
    """A node's ``runners.slice`` as its capacity probe read it.

    Attributes:
        current_gb: ``memory.current``: what the slice's CI jobs hold now.
        high_gb: ``memory.high``: where the kernel starts throttling them.
    """

    current_gb: float
    high_gb: float


class LiveLoad(TypedDict):
    """What a node's live fleet runs were granted, read from the ledger.

    The share of the node a capacity check must treat as taken before it
    grants anything more (MCPs board task 939ec5c7). A run's workers spawn
    after its probe, so the probe's free memory cannot be trusted to show
    them; charging every live run its whole grant is what lets several runs
    share a node without three that each fit a snapshot overloading it
    together, the hazard a fixed one-run limit used to guard.

    Attributes:
        runs: Fleet dispatches currently live on the node, counted from the
            ledger rather than from the process table -- a run is ours
            because we recorded it, not because a process looks like one.
        workers: The test workers those runs were granted, in total.
        ram_gb: The memory those workers may hold: each run's workers times
            its project's ``worker_ram_gb``.
    """

    runs: int
    workers: int
    ram_gb: float


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
        live: What this node's live fleet runs were granted (:class:`LiveLoad`).
        ci_slice: The node's ``runners.slice`` memory, or None where the
            node hosts no CI runners and so has no such cgroup. Read so a refusal can say CI
            holds the lane rather than that somebody is on the machine
            (MCPs board task 5d6e57e7).
        reserved_cores: The cores the node's live reservations hold: work
            the fleet does not run, such as a clean-VM smoke on diphtheria,
            declared by its own ``.reservation`` file
            (:mod:`fleet.core.linux_capacity_probe`, MCPs board task
            84a2b690). Charged against the node's cores beside its live
            fleet runs.
        action_cache: The host's CI runners and the action archive cache
            (:mod:`fleet.contracts.action_cache`, MCPs board task
            ce2b92e0), or None where the node publishes no fresh counts.
            Carried to the tick, never weighed.
        cpu_pressure_pct: The share of the last minute some runnable task
            waited for a CPU, ``/proc/pressure/cpu``'s ``some avg60``
            (:mod:`fleet.core.linux_capacity_probe`, MCPs board task
            2ce5dc13). Every probe prints it or fails.
        io_pressure_pct: The share of the last minute some task stalled on
            I/O, ``/proc/pressure/io``'s ``some avg60`` (MCPs board task
            814c4555), printed as :attr:`cpu_pressure_pct` is.
    """

    host: str
    free_ram_gb: float
    free_disk_gb: float
    live: LiveLoad
    ci_slice: SliceMemory | None
    reserved_cores: int
    action_cache: ActionCacheCounts | None
    cpu_pressure_pct: float
    io_pressure_pct: float


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
    """Encode the ``dispatch`` object of one node's machine file.

    Args:
        node: The node to encode.

    Returns:
        JSON-serialisable mapping carrying every field the ``dispatch``
        object declares, with ``gpu`` present and null for a CPU-only node
        rather than omitted. ``held`` is the machine's
        (:func:`fleet.contracts.machine.encode_machine`), and ``host`` and
        ``enabled`` are the machine's own facts, so neither is here.
    """
    gpu = node["gpu"]
    return {
        "stage_root": node["stage_root"],
        "logical_cores": node["logical_cores"],
        "ram_gb": node["ram_gb"],
        "gpu": None if gpu is None else encode_node_gpu(gpu),
        "test_database": node["test_database"],
        "rust": node["rust"],
        "cxx": node["cxx"],
        "docker": node["docker"],
        "stack": node["stack"],
        "budget": encode_node_budget(node["budget"]),
    }


def decode_node_config(value: JSONValue, *, host: str, enabled: bool) -> NodeConfig:
    """Decode and validate one machine file's ``dispatch`` object.

    Args:
        value: The ``dispatch`` object, as the JSON loader produced it.
        host: The machine's name.
        enabled: Whether the machine answers and its dispatch is not held.

    Returns:
        The validated node.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, ``gpu`` is absent rather than explicitly null, or the
            machine's own numbers are not positive.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"node must be a JSON object, got {type(value).__name__}")
    if "gpu" not in value:
        raise JSONTypeError(
            "node must declare 'gpu', using null for a CPU-only machine. An absent key is "
            "indistinguishable from one nobody has filled in, and the difference matters "
            "the moment a measurement is pinned to a card."
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
    logical_cores = require_int(value, "logical_cores")
    if logical_cores < 1:
        raise JSONTypeError(f"logical_cores must be at least 1, got {logical_cores}")
    stage_root = require_str(value, "stage_root")
    if not stage_root:
        raise JSONTypeError("stage_root must not be empty; it is where the working tree lands")
    gpu_value = value["gpu"]
    return NodeConfig(
        host=host,
        stage_root=stage_root,
        logical_cores=logical_cores,
        ram_gb=_positive_float(value, "ram_gb"),
        gpu=None if gpu_value is None else decode_node_gpu(gpu_value),
        enabled=enabled,
        test_database=require_bool(value, "test_database"),
        rust=decode_capability(Capability.RUST, value["rust"]),
        cxx=decode_capability(Capability.CXX, value["cxx"]),
        docker=decode_capability(Capability.DOCKER, value["docker"]),
        stack=decode_capability(Capability.STACK, value["stack"]),
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
        f"{state['free_disk_gb']:.0f} GB disk free, {state['live']['runs']} live run(s) "
        f"holding {state['live']['workers']} worker(s)"
    )


__all__ = [
    "LiveLoad",
    "NodeConfig",
    "NodeGpu",
    "NodeState",
    "SliceMemory",
    "declared_capability",
    "decode_node_config",
    "decode_node_gpu",
    "describe_node",
    "encode_node_config",
    "encode_node_gpu",
]
