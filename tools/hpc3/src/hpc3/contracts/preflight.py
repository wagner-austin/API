"""The preflight contract: what the scheduler says would happen.

Unit tests prove this package builds the script it meant to. They cannot
prove the cluster will accept it, because acceptance depends on state that
lives only on the cluster: the user's associations, the QOS the partition
maps to, whether the requested GPU model exists on nodes that are currently
up. A spec can satisfy every rule in :mod:`hpc3.contracts.job` and still be
rejected at submission for a reason no local check can see.

``sbatch --test-only`` is Slurm's answer to exactly that. It runs the real
admission path -- account, QOS, partition, resource feasibility -- and then
allocates nothing. What comes back is the scheduler's own verdict plus an
estimated start, which is the only honest answer to "will this work" short of
running it.

The estimate is a snapshot, not a reservation. It is reported as such.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.cluster import ClusterFacts, require_partition


class PreflightResult(TypedDict):
    """The scheduler's verdict on a job it did not run.

    Attributes:
        start_estimate: When Slurm currently believes the job would start,
            in its own format. A snapshot of the queue at one instant, not a
            promise -- observed to be badly pessimistic on this cluster,
            where a job estimated at 3.4 hours started in under 5 seconds.
        processors: Cores the allocation would take.
        node_list: Nodes it would land on, as Slurm names them.
        partition: Partition it would run in, echoed back so a caller can
            confirm the scheduler read the request it thought it sent.
    """

    start_estimate: str
    processors: int
    node_list: str
    partition: str


def _require_nonempty_str(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required string field that must not be empty.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    return value


def encode_preflight_result(result: PreflightResult) -> dict[str, JSONValue]:
    """Encode a preflight result to a JSON object.

    Args:
        result: Result to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "start_estimate": result["start_estimate"],
        "processors": result["processors"],
        "node_list": result["node_list"],
        "partition": result["partition"],
    }


def decode_preflight_result(value: JSONValue, cluster: ClusterFacts) -> PreflightResult:
    """Decode and validate a JSON value into a preflight result.

    Args:
        value: Value produced by the JSON loader.
        cluster: The cluster whose partitions the verdict is checked against.

    Returns:
        Validated result.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, a string field is empty, or the processor count is below
            one.
        AppError: With ``PARTITION_UNKNOWN`` if the scheduler echoed back a
            partition this cluster does not have -- which would mean the
            workspace names the wrong machine.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"preflight result must be a JSON object, got {type(value).__name__}")

    processors = require_int(value, "processors")
    if processors < 1:
        raise JSONTypeError(f"Field 'processors' must be at least 1, got {processors}")

    return PreflightResult(
        start_estimate=_require_nonempty_str(value, "start_estimate"),
        processors=processors,
        node_list=_require_nonempty_str(value, "node_list"),
        partition=require_partition(cluster, value, "partition"),
    )


__all__ = [
    "PreflightResult",
    "decode_preflight_result",
    "encode_preflight_result",
]
