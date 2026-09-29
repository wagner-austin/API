"""The share of a runner host its GitHub Actions jobs may take.

MCPs board task 45a4f22b. Lavender's WSL instance carries eight self-hosted
runners and, beside them, the fleet's second testdb lane (lavender-wsl) with
its ``corvis-fleet-testdb`` container. On 2026-09-29 at 11:36Z the runners
were running an API training job, four MCPs ``tsc`` jobs and four pgvector
test databases at once: load 132 on 16 cores, 3.4 GB of 26 GB available,
and sshd stopped answering, so every fleet tick for lavender-wsl read
``did not answer`` and the testdb queue waited on a node that was up. The
runners and sshd shared ``system.slice``, which carried no ceiling at all.

This block is the ceiling, declared once per host. The provision renders it
as one systemd slice holding every WSL runner on the host, so CI jobs
throttle, and at worst are killed, inside their own share, while sshd, the
test database and the fleet's own checks keep the rest of the VM.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONTypeError, require_int
from typing_extensions import TypedDict

#: The slice every WSL runner on a host is placed in. One name for every
#: host: the budget differs per host, the unit it is enforced by does not.
#: No dash: systemd reads ``a-b.slice`` as a child of ``a.slice``.
CI_SLICE_NAME = "runners.slice"

#: systemd's CPUWeight range. 100 is every unit's default, so a weight below
#: it yields the processor to sshd and the fleet under contention and takes
#: all of it when nothing else wants it.
CPU_WEIGHT_MIN = 1
CPU_WEIGHT_MAX = 10000


class CiSlice(TypedDict):
    """The memory and CPU the host's runners may take together.

    Attributes:
        memory_high_gb: Where systemd starts reclaiming from the runners and
            throttling them (``MemoryHigh``), in GB.
        memory_max_gb: The hard ceiling (``MemoryMax``), in GB; a job that
            passes it is killed by the kernel inside the slice.
        cpu_weight: The slice's ``CPUWeight`` against every other unit's
            default of 100.
    """

    memory_high_gb: int
    memory_max_gb: int
    cpu_weight: int


def encode_ci_slice(budget: CiSlice) -> JSONObject:
    """Encode one host's CI budget.

    Args:
        budget: The budget to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "memory_high_gb": budget["memory_high_gb"],
        "memory_max_gb": budget["memory_max_gb"],
        "cpu_weight": budget["cpu_weight"],
    }


def decode_ci_slice(value: JSONObject, *, vm_memory_gb: int | None) -> CiSlice:
    """Decode and validate one host's CI budget.

    Args:
        value: The ``ci_slice`` object from the host's roster entry.
        vm_memory_gb: The host's ``wslconfig_min_memory_gb``, or None when
            the host keeps the Windows default and no size is declared.

    Returns:
        The validated budget.

    Raises:
        JSONTypeError: If a field is missing or mistyped, a memory figure is
            not positive, the throttle point sits above the ceiling, the
            weight is outside systemd's range, or the ceiling leaves the VM
            nothing beside the runners.
    """
    memory_high = require_int(value, "memory_high_gb")
    memory_max = require_int(value, "memory_max_gb")
    cpu_weight = require_int(value, "cpu_weight")
    if memory_high <= 0:
        raise JSONTypeError(f"ci_slice.memory_high_gb must be positive, got {memory_high}")
    if memory_max < memory_high:
        raise JSONTypeError(
            f"ci_slice.memory_max_gb {memory_max} is below memory_high_gb {memory_high}; "
            "the kernel would kill jobs before systemd ever throttled them"
        )
    if not CPU_WEIGHT_MIN <= cpu_weight <= CPU_WEIGHT_MAX:
        raise JSONTypeError(
            f"ci_slice.cpu_weight must be between {CPU_WEIGHT_MIN} and {CPU_WEIGHT_MAX}, "
            f"got {cpu_weight}"
        )
    if vm_memory_gb is not None and memory_max >= vm_memory_gb:
        raise JSONTypeError(
            f"ci_slice.memory_max_gb {memory_max} leaves nothing of the VM's "
            f"{vm_memory_gb} GB for sshd and the fleet's own checks, which is the "
            "starvation this ceiling exists to prevent"
        )
    return CiSlice(
        memory_high_gb=memory_high,
        memory_max_gb=memory_max,
        cpu_weight=cpu_weight,
    )


__all__ = [
    "CI_SLICE_NAME",
    "CPU_WEIGHT_MAX",
    "CPU_WEIGHT_MIN",
    "CiSlice",
    "decode_ci_slice",
    "encode_ci_slice",
]
