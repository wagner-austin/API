"""Refuse a job pinned to a GPU model the partition has none of free.

THE INCIDENT IS IN :mod:`hpc3.contracts.gpu_supply`. In one line: a
ten-minute job pinned an A100 on a partition whose eight A100s were all
allocated, waited five hours, and started in a hundred seconds once
resubmitted against a V100 the same partition had free.

WHY A REFUSAL AND NOT A WARNING. Preflight already printed the evidence -- an
estimated start ten days out -- next to the word ``OK``, and it was read as
noise, because everything preflight prints that is not an error is noise by
the time you have run it twice. A rule that fires only when the requested
model has NOTHING free and another model does has no false-positive space to
speak of: at that moment the job cannot start on what it asked for, and can
start on something else.

WHAT IT DOES NOT DO is choose for you. Some work genuinely needs the card it
names -- 80 GB of memory, or bf16, or a compute capability the pinned torch
was built for. So the message names what is free and leaves the decision to
whoever knows why the model was pinned, which is never this function.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.logging import get_logger

from hpc3.contracts.gpu_supply import (
    GpuSupply,
    describe_supply,
    free_of,
    parse_gpu_supply,
)
from hpc3.contracts.job import JobSpec
from hpc3.core import remote

_log = get_logger("hpc3.gpu_supply")

#: Reads each node's GPU inventory and what is allocated on it, in one call.
#:
#: ``GresUsed`` rather than a per-node ``scontrol show node`` loop: a partition
#: here is thirty-six nodes, and preflight already pays for several round
#: trips. One command keeps this cheap enough to run every time.
SUPPLY_COMMAND = 'sinfo -p {partition} -O "Gres:64,GresUsed:64" --noheader'


def read_gpu_supply(host: str, partition: str) -> tuple[GpuSupply, ...]:
    """Ask the scheduler what GPUs a partition holds and how many are free.

    Args:
        host: SSH destination.
        partition: Partition to inspect.

    Returns:
        One entry per GPU model, ordered by model name. Empty when the
        partition holds no GPUs, which is the ordinary answer for a CPU
        partition and is not an error.
    """
    return parse_gpu_supply(remote.run_remote(host, SUPPLY_COMMAND.format(partition=partition)))


def check_requested_gpu_available(spec: JobSpec, supply: tuple[GpuSupply, ...]) -> None:
    """Refuse a job whose GPU model has no free cards while another model does.

    Args:
        spec: The job about to be flown. A CPU-only job passes untouched.
        supply: The partition's inventory, from :func:`read_gpu_supply`.

    Raises:
        AppError: With ``GPU_MODEL_EXHAUSTED`` when the requested model has no
            free cards and some other model in the same partition does.

            The two conditions are both required. If nothing at all is free
            the partition is simply busy and waiting is the only option, which
            is not a mistake and not this rule's business. If the requested
            model is free the job starts. It is the combination -- queueing
            for the one model that is exhausted while others idle -- that is
            almost always an inherited default rather than a decision.
    """
    request = spec["gpu"]
    if request is None or supply == ():
        return
    if free_of(supply, request["model"]) >= request["count"]:
        return
    alternatives = tuple(
        entry for entry in supply if entry["model"] != request["model"] and entry["free"] > 0
    )
    if alternatives == ():
        return
    reason = spec["gpu_pinned_because"]
    if reason is not None:
        # The run declared its card pin IS the measurement, so queueing for
        # the exhausted model is the decision the refusal below exists to
        # force. Logged rather than silent: a deliberate wait should still
        # say what it is waiting behind.
        _log.info(
            "%r queues for exhausted %s x%d deliberately (%s) -- supply: %s",
            spec["name"],
            request["model"],
            request["count"],
            reason,
            describe_supply(supply),
        )
        return
    offered = ", ".join(f"{entry['model']} ({entry['free']} free)" for entry in alternatives)
    raise AppError(
        Hpc3ErrorCode.GPU_MODEL_EXHAUSTED,
        (
            f"{spec['name']!r} pins {request['model']} x{request['count']} on partition "
            f"{spec['partition']!r}, which has none free right now: {describe_supply(supply)}. "
            f"These are free and would start it: {offered}. "
            f"Slurm will still ADMIT this job -- `sbatch --test-only` says yes and prints an "
            f"estimated start, which on a preemptible partition is when a slot can be "
            f"guaranteed rather than when backfill will find one. Measured 2026-09-04: an "
            f"A100-pinned job estimated at nine days waited five hours and was cancelled, "
            f"and the same work on a V100 the same partition had free started in about a "
            f"hundred seconds. Either pin a model that is free, or declare "
            f"'gpu_pinned_because' in the run document -- the card is the measurement, say "
            f"so -- and wait deliberately."
        ),
    )


__all__ = ["SUPPLY_COMMAND", "check_requested_gpu_available", "read_gpu_supply"]
