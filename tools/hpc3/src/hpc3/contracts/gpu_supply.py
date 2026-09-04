"""How many GPUs of each model a partition holds, and how many are free.

WHY THIS EXISTS, AND IT IS AN INCIDENT RATHER THAN A FEATURE. On 2026-09-04 a
ten-minute job pinned ``gres/gpu:A100:1`` on ``free-gpu`` and sat PENDING for
five hours with ``Reason=ReqNodeNotAvail``. Preflight had said
``OK ... would start 2026-09-13`` and exited zero. Every one of that
partition's eight A100s was allocated; the same partition held 72 A30 and 56
V100 GPUs, several of them free with idle cores beside them. Resubmitted
against a V100, the job started in about a hundred seconds.

TWO THINGS FAILED, and neither was Slurm.

The first is that ``sbatch --test-only`` answers "would this be admitted",
and preflight printed that as ``OK``. Admissible and going-to-start are
different questions, and the estimated start it printed alongside -- ten days
out for a thirty-minute job -- was the answer to the second one, sitting in
plain sight next to the word OK.

The second is that the GPU model was inherited rather than chosen. The
project's default is an A100 because some of its work needs one; this job ran
a 124-million-parameter model. Nothing asked whether the scarcest resource in
the partition was required, because nothing knew which resource was scarcest.

THE ESTIMATE IS NOT THE FIX. Slurm's ``StartTime`` on a preemptible partition
is when a slot can be GUARANTEED, not when backfill will find one: the
replacement V100 run was estimated at 2026-09-13 and started immediately, and
an image build the same day was estimated conservatively and preempted after
two minutes -- meaning it, too, had started at once. Both directions of that
error have now been observed. What is reliable is the supply itself, which is
what this module reads.
"""

from __future__ import annotations

import re

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

#: One ``sinfo`` GRES field: ``gpu:<model>:<count>`` with an optional index
#: list that only the used-column carries, e.g. ``gpu:A30:3(IDX:0-1,3)``.
_GRES = re.compile(r"gpu:(?P<model>[^:()\s]+):(?P<count>\d+)")


class GpuSupply(TypedDict):
    """One GPU model's inventory across a partition.

    Attributes:
        model: The model as Slurm spells it, e.g. ``A100``. Compared against
            the job's own request, which
            :mod:`hpc3.contracts.job` already requires to name a model rather
            than a bare count.
        total: Cards of this model the partition holds.
        used: Cards currently allocated.
        free: ``total - used``. Stored rather than derived at the call site so
            a reader of a recorded supply cannot subtract it differently.
    """

    model: str
    total: int
    used: int
    free: int


def parse_gpu_supply(text: str) -> tuple[GpuSupply, ...]:
    """Read ``sinfo -O "Gres,GresUsed"`` output into a per-model inventory.

    One line per node, two GRES fields per line: what the node has and what is
    allocated on it. Nodes are summed by model, so the result describes the
    partition rather than any node in it.

    A node whose line names no GPU is skipped rather than failing. A CPU-only
    node in a GPU partition is ordinary, and refusing to read the partition
    because one node has no card would make this unusable exactly where it is
    most needed.

    Args:
        text: Raw ``sinfo`` output, one node per line.

    Returns:
        One entry per model, ordered by model name so two reads of one
        partition compare equal.
    """
    totals: dict[str, int] = {}
    used: dict[str, int] = {}
    for line in text.splitlines():
        found = list(_GRES.finditer(line))
        if not found:
            continue
        # The first match is the node's inventory and the second what is
        # allocated on it; `sinfo` emits them in that column order. A node
        # with nothing allocated still prints a used-field, so a line with
        # only one match is a node that reports no usage column at all.
        head = found[0]
        totals[head["model"]] = totals.get(head["model"], 0) + int(head["count"])
        for match in found[1:]:
            used[match["model"]] = used.get(match["model"], 0) + int(match["count"])
    return tuple(
        GpuSupply(
            model=model,
            total=totals[model],
            used=used.get(model, 0),
            free=totals[model] - used.get(model, 0),
        )
        for model in sorted(totals)
    )


def free_of(supply: tuple[GpuSupply, ...], model: str) -> int:
    """How many cards of one model are free.

    Args:
        supply: The partition's inventory.
        model: The model to look up.

    Returns:
        Free cards, or zero when the partition holds none of that model. The
        two cases are deliberately the same number here; the caller that needs
        to tell them apart reads ``total`` as well, and
        :func:`~hpc3.core.gpu_supply.check_requested_gpu_available` does.
    """
    for entry in supply:
        if entry["model"] == model:
            return entry["free"]
    return 0


def describe_supply(supply: tuple[GpuSupply, ...]) -> str:
    """Render an inventory for an error message or a log line.

    Args:
        supply: The partition's inventory.

    Returns:
        One comma-separated summary, e.g. ``A100 0/8 free, A30 9/72 free``.
        Empty when the partition holds no GPUs at all.
    """
    return ", ".join(f"{e['model']} {e['free']}/{e['total']} free" for e in supply)


def encode_gpu_supply(entry: GpuSupply) -> JSONObject:
    """Encode one model's inventory.

    Args:
        entry: The entry to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "model": entry["model"],
        "total": entry["total"],
        "used": entry["used"],
        "free": entry["free"],
    }


def decode_gpu_supply(value: JSONValue) -> GpuSupply:
    """Decode and validate one model's inventory.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated entry.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or ``free`` disagrees with ``total - used``. The last is
            checked because the three are recorded together and a reader is
            entitled to use any of them.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"gpu supply must be a JSON object, got {type(value).__name__}")
    total = require_int(value, "total")
    used = require_int(value, "used")
    free = require_int(value, "free")
    if free != total - used:
        raise JSONTypeError(
            f"gpu supply reports {free} free against {total} total and {used} used, "
            f"which do not reconcile; one of the three was written by a different rule"
        )
    return GpuSupply(model=require_str(value, "model"), total=total, used=used, free=free)


__all__ = [
    "GpuSupply",
    "decode_gpu_supply",
    "describe_supply",
    "encode_gpu_supply",
    "free_of",
    "parse_gpu_supply",
]
