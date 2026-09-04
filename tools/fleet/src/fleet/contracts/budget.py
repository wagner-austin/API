"""What share of a machine somebody else is sitting at we are entitled to.

THE DIFFERENCE FROM ITS HPC3 SIBLING, because the two look alike and are not.
:mod:`hpc3.contracts.budget` caps our share of a cluster 102 other people use,
in GPU-hours and service units -- quantities a scheduler bills and an operator
never feels. Here the other user is one person, at the keyboard, and what they
feel is the machine becoming unusable. So nothing here is denominated in time
or money. Every cap is an instantaneous resource the owner would notice losing.

WHY MEMORY IS THE CAP AND CORES ARE ONLY A CEILING. ``pytest -n auto`` spawns
one worker per core and in the torch projects every worker RESERVES about
1.1 GB of commit on import, while its working set stays at a few megabytes.
Task Manager therefore shows a harmless-looking run holding tens of gigabytes.
Measured on two nodes on 2026-09-04:

* ``austinpc``: two overlapping suites left 66 processes holding 77.9 GB of
  commit with an aggregate CPU delta of 0.016 s over 5 s -- doing nothing, and
  leaving 22 GB of 179 GB free. Nothing else on the box could run.
* ``sedona``: 20 logical cores and 11.4 GB of free RAM. ``-n auto`` there asks
  for roughly 22 GB and would wedge the same way on a smaller machine.

A dispatcher that chose its worker count from the core count would have
reproduced the first incident on the second node. So :func:`admissible_workers`
divides memory and lets cores bound the answer, never the other way round.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
)
from typing_extensions import TypedDict


class NodeBudget(TypedDict):
    """Instantaneous caps on what a dispatch may take from one node.

    Attributes:
        reserved_cores: Logical cores left for whoever is using the machine.
            Subtracted from the node's total before any worker count is
            computed, so a dispatch cannot take the last core. This is the
            cap with no cluster analogue at all: Slurm never has to leave a
            core for the person sitting at the node.
        reserved_ram_gb: Gigabytes of physical memory left alone, for the same
            reason and measured the same way.
        worker_ram_gb: Memory to assume each test worker will hold. Declared
            rather than measured, because it has to be known BEFORE the
            workers exist -- and a projection that guessed low is exactly how
            a box ends up holding 77.9 GB. ~1.1 GB is the measured figure for
            a worker that imports torch; a project whose suite does not is
            entitled to declare less and get more workers.
        max_concurrent_runs: How many fleet dispatches may be live on this
            node at once, across all projects. A second bound on top of the
            memory arithmetic, because memory is checked against a snapshot
            and three dispatches that each fit alone do not fit together.
        max_disk_gb: Gigabytes the staged working trees may occupy in total.
            Counted because the first dispatch to a node is a cold stage of a
            whole monorepo, and a node that fills its system drive is a node
            its owner has to fix by hand.
    """

    reserved_cores: int
    reserved_ram_gb: float
    worker_ram_gb: float
    max_concurrent_runs: int
    max_disk_gb: float


def admissible_workers(budget: NodeBudget, *, logical_cores: int, free_ram_gb: float) -> int:
    """How many test workers this node can hold right now.

    MEMORY DIVIDES, CORES BOUND. The count is what the free memory affords
    after the owner's reservation, then clipped to the cores left after
    theirs. Doing it the other way -- cores, clipped by memory -- reads the
    same and is not: it makes the core count the number a caller sees when
    memory is plentiful, and the whole point is that memory is the quantity
    that ran out.

    Returns zero rather than raising. Zero is a real answer that
    :mod:`fleet.core.capacity` turns into a refusal naming a node with room;
    raising here would make "this node is busy" indistinguishable from "this
    budget is malformed", which is a decode-time question and already
    answered by :func:`decode_node_budget`.

    Args:
        budget: The node's declared caps.
        logical_cores: The node's total logical processors.
        free_ram_gb: Physical memory free on the node right now.

    Returns:
        Workers the node can take, at least zero.
    """
    spare_ram = free_ram_gb - budget["reserved_ram_gb"]
    spare_cores = logical_cores - budget["reserved_cores"]
    if spare_ram <= 0.0 or spare_cores <= 0:
        return 0
    return min(int(spare_ram / budget["worker_ram_gb"]), spare_cores)


def encode_node_budget(budget: NodeBudget) -> JSONObject:
    """Encode a node's caps.

    Args:
        budget: The budget to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "reserved_cores": budget["reserved_cores"],
        "reserved_ram_gb": budget["reserved_ram_gb"],
        "worker_ram_gb": budget["worker_ram_gb"],
        "max_concurrent_runs": budget["max_concurrent_runs"],
        "max_disk_gb": budget["max_disk_gb"],
    }


def decode_node_budget(value: JSONValue) -> NodeBudget:
    """Decode and validate a node's caps.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated budget.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or a field holds a number that cannot describe a
            machine. ``worker_ram_gb`` must be positive because it is a
            divisor and a zero would make every node look infinite; the
            reservations must not be negative because a negative reservation
            reads as permission to take more than the machine has, which is
            the opposite of what this file is for.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"node budget must be a JSON object, got {type(value).__name__}")
    worker_ram_gb = require_float(value, "worker_ram_gb")
    if worker_ram_gb <= 0.0:
        raise JSONTypeError(
            f"worker_ram_gb must be positive, got {worker_ram_gb}; it divides free memory to "
            "give a worker count, and a zero would report every node as having room for "
            "unlimited workers"
        )
    reserved_cores = require_int(value, "reserved_cores")
    reserved_ram_gb = require_float(value, "reserved_ram_gb")
    if reserved_cores < 0 or reserved_ram_gb < 0.0:
        raise JSONTypeError(
            f"reservations must not be negative, got reserved_cores={reserved_cores} and "
            f"reserved_ram_gb={reserved_ram_gb}; a negative reservation reads as permission "
            "to take more of the machine than it has"
        )
    max_concurrent_runs = require_int(value, "max_concurrent_runs")
    if max_concurrent_runs < 1:
        raise JSONTypeError(
            f"max_concurrent_runs must be at least 1, got {max_concurrent_runs}; a node that "
            "may hold no runs is spelled by leaving it out of the workspace"
        )
    max_disk_gb = require_float(value, "max_disk_gb")
    if max_disk_gb <= 0.0:
        raise JSONTypeError(f"max_disk_gb must be positive, got {max_disk_gb}")
    return NodeBudget(
        reserved_cores=reserved_cores,
        reserved_ram_gb=reserved_ram_gb,
        worker_ram_gb=worker_ram_gb,
        max_concurrent_runs=max_concurrent_runs,
        max_disk_gb=max_disk_gb,
    )


__all__ = [
    "NodeBudget",
    "admissible_workers",
    "decode_node_budget",
    "encode_node_budget",
]
