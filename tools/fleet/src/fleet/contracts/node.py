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
    require_dict,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from fleet.contracts.budget import NodeBudget, decode_node_budget, encode_node_budget


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
        budget: What share of this machine a dispatch may take.
    """

    host: str
    stage_root: str
    logical_cores: int
    ram_gb: float
    gpu: NodeGpu | None
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
        "stage_root": node["stage_root"],
        "logical_cores": node["logical_cores"],
        "ram_gb": node["ram_gb"],
        "gpu": None if gpu is None else encode_node_gpu(gpu),
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
    logical_cores = require_int(value, "logical_cores")
    if logical_cores < 1:
        raise JSONTypeError(f"logical_cores must be at least 1, got {logical_cores}")
    stage_root = require_str(value, "stage_root")
    if not stage_root:
        raise JSONTypeError("stage_root must not be empty; it is where the working tree lands")
    gpu_value = value["gpu"]
    return NodeConfig(
        host=require_str(value, "host"),
        stage_root=stage_root,
        logical_cores=logical_cores,
        ram_gb=_positive_float(value, "ram_gb"),
        gpu=None if gpu_value is None else decode_node_gpu(gpu_value),
        budget=decode_node_budget(require_dict(value, "budget")),
    )


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
    "NodeState",
    "decode_node_config",
    "decode_node_gpu",
    "describe_node",
    "encode_node_config",
    "encode_node_gpu",
]
