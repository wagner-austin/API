"""The budget contract: our own share of a machine 102 other people use.

HPC3's own QOS ceilings bound how much can run *at once*. They say nothing
about how much is consumed *in total*, and on the free partitions nothing
bills, so there is no external mechanism that ever says stop. A sweep of 24
GPUs for three days is 1,728 GPU-hours, entirely within every limit the
cluster enforces, and it is not a reasonable share of a shared machine.

So the cap is ours to declare and ours to enforce. Two of them, because they
answer different questions:

* ``self_imposed_gpu_hours`` is the courtesy limit. Nothing grants it and
  nothing but this package tracks it; it exists on the free partitions
  where nothing else counts.
* ``max_service_units`` is the spending limit, and it only binds on billing
  partitions -- which is exactly where it matters, since the personal balance
  is 1,000 SU and one three-day job on eleven cores would spend 792 of them.

Both are checked twice: projected before submission, so a flood never starts,
and observed while running, so a wrong projection is caught rather than
discovered afterwards.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue, require_float, require_str
from typing_extensions import TypedDict


class Budget(TypedDict):
    """Self-imposed caps on what a run may consume.

    Attributes:
        self_imposed_gpu_hours: GPUs multiplied by wall-clock hours, summed
            over every job. Nothing grants this and nothing tracks it but us:
            the free partitions have no GPU-hour allocation to draw down, so
            this is restraint on a shared machine rather than a balance.

            Named at length for that reason. It was ``max_gpu_hours``,
            which sat beside ``max_service_units`` and read as its
            matching half --
            two allowances, one for GPUs and one for money. It cost real time:
            an operator was told a GPU job could be paid for out of a
            service-unit balance that turned out to be a CPU allocation, and
            the adjacency of the two names is what made that reading natural.
        max_service_units: Service units, summed over every job. Unlike the
            line above this one has something behind it -- HPC3 grants an
            allocation and Slurm charges against it -- so this is a self-cap
            on spending something real. Zero on the free partitions however
            long they run, so it binds only where spending happens at all.
        charge_account: The Slurm account a billed job draws from. Empty when
            the workspace does not spend, which is the default posture.

            Part of the budget rather than a separate workspace field because
            it is the same declaration: a cap without an account names no
            money, and an account without a cap is an unbounded one. Slurm
            enforces the pairing too -- a billed partition refuses a job that
            names no account, with "please specify charge account", so a
            workspace that raised its cap and stopped there would be refused
            by the cluster rather than by this package.
    """

    self_imposed_gpu_hours: float
    max_service_units: float
    charge_account: str


class Consumption(TypedDict):
    """What a set of jobs actually holds or has held.

    Attributes:
        gpu_hours: Total GPU-hours across the jobs.
        service_units: Total service units across the jobs.
        jobs: How many jobs the totals cover.
    """

    gpu_hours: float
    service_units: float
    jobs: int


def _require_nonnegative_float(obj: dict[str, JSONValue], key: str) -> float:
    """Read a required float field that cannot be negative.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a number, or negative. A
            negative cap would admit everything, which is the opposite of
            what declaring a cap means.
    """
    value = require_float(obj, key)
    if value < 0.0:
        raise JSONTypeError(f"Field '{key}' must not be negative, got {value}")
    return value


def encode_budget(budget: Budget) -> dict[str, JSONValue]:
    """Encode a budget to a JSON object.

    Args:
        budget: Budget to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "self_imposed_gpu_hours": budget["self_imposed_gpu_hours"],
        "max_service_units": budget["max_service_units"],
        "charge_account": budget["charge_account"],
    }


def decode_budget(value: JSONValue) -> Budget:
    """Decode and validate a JSON value into a budget.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated budget.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or a cap is negative.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"budget must be a JSON object, got {type(value).__name__}")
    return Budget(
        self_imposed_gpu_hours=_require_nonnegative_float(value, "self_imposed_gpu_hours"),
        max_service_units=_require_nonnegative_float(value, "max_service_units"),
        charge_account=require_str(value, "charge_account"),
    )


def encode_consumption(consumption: Consumption) -> dict[str, JSONValue]:
    """Encode a consumption total to a JSON object.

    Args:
        consumption: Total to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "gpu_hours": consumption["gpu_hours"],
        "service_units": consumption["service_units"],
        "jobs": consumption["jobs"],
    }


__all__ = [
    "Budget",
    "Consumption",
    "decode_budget",
    "encode_budget",
    "encode_consumption",
]
