"""The node and machine declarations the contract tests build, shared by their files."""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import JSONValue

from fleet.contracts.budget import NodeBudget
from fleet.contracts.machine import LedgerObserver, Machine, MachineRole
from fleet.contracts.node import NodeConfig, NodeGpu, decode_node_config

BUDGET: Final = NodeBudget(
    reserved_cores=2,
    reserved_ram_gb=4.0,
    worker_ram_gb=1.1,
    max_disk_gb=20.0,
    checks_at_once=None,
    pinned_checks_only=False,
)

GPU: Final = NodeGpu(
    model="NVIDIA GeForce GTX 1630",
    vram_mib=4096,
    compute_capability="7.5",
    driver_version="591.86",
)


def node(*, gpu: NodeGpu | None = GPU, cores: int = 16, host: str = "lavender") -> NodeConfig:
    """Build a node declaration.

    Args:
        gpu: Its CUDA device, or None for a CPU-only machine.
        cores: Logical processors.
        host: The machine's name.

    Returns:
        The node.
    """
    return NodeConfig(
        host=host,
        stage_root="/home/corvis/fleet/stage",
        logical_cores=cores,
        ram_gb=32.0,
        gpu=gpu,
        enabled=True,
        test_database=False,
        rust=None,
        cxx=None,
        docker=None,
        stack=None,
        budget=BUDGET,
    )


def decode_like(value: JSONValue, declared: NodeConfig) -> NodeConfig:
    """Decode a ``dispatch`` object with the machine facts of a declared node.

    Args:
        value: The ``dispatch`` object.
        declared: The node whose ``host`` and ``enabled`` the machine's file
            would hand the decoder.

    Returns:
        The decoded node.
    """
    return decode_node_config(value, host=declared["host"], enabled=declared["enabled"])


def machine(name: str, declared: NodeConfig | None, *, held: str | None = None) -> Machine:
    """Build one roster machine, a worker that answers, observed by the fleet arm.

    Args:
        name: Its name.
        declared: Its capacity, or None for a machine declaring none.
        held: Why it takes no dispatched work, or None.

    Returns:
        The machine.
    """
    return Machine(
        name=name,
        role=MachineRole.WORKER,
        user="corvis",
        enabled=True,
        ledger_observer=LedgerObserver.FLEET_ARM,
        dispatch=declared,
        held=held,
    )


__all__ = ["BUDGET", "GPU", "decode_like", "machine", "node"]
