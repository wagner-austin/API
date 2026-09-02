"""Array-task identity: one sbatch call, many jobs, ids the parsers must expand.

A job array is one submission that Slurm runs as many tasks, and the whole
reason this package uses one is throughput: submitting a 96-member sweep one
``sbatch`` at a time is three SSH round trips per member, ~13 seconds each,
measured while the cluster itself scheduled the jobs instantly (rusted's
ab48 waves, 2026-09-01). One array call submits the same 96 in one trip.

The price is identity, and it was MEASURED before it was coded (probe job
55678543 on the free partition, 2026-09-01, throttled ``--array=0-3%2``):

* A RUNNING task appears everywhere as ``55678543_0`` -- base id,
  underscore, task index.
* PENDING tasks appear as ONE aggregate row -- ``55678543_[2-3%2]`` -- in
  ``squeue`` AND in ``sacct -X``. Both. The sacct half was the surprise, and
  a parser that had assumed per-task accounting rows would silently treat
  every pending member as absent.
* ``sacct -j 55678543_2`` returns NOTHING while task 2 is still inside the
  pending aggregate. Absent from accounting therefore still means "not
  finished", never "safe to resubmit".

So every reader that matches recorded task ids against cluster output must
expand aggregates first, and this module is the one place that knows how.
"""

from __future__ import annotations

import itertools

from platform_core.errors import AppError, Hpc3ErrorCode

_THROTTLE_SEPARATOR = "%"


def array_task_id(base_id: str, index: int) -> str:
    """Name one task of a submitted array, the way the cluster names it.

    Args:
        base_id: The id ``sbatch`` announced for the whole array.
        index: The task's array index.

    Returns:
        ``"55678543_2"`` for base ``"55678543"`` and index ``2``.
    """
    return f"{base_id}_{index}"


def format_array_indices(indices: tuple[int, ...]) -> str:
    """Render array indices as the ``--array`` argument, ranges compressed.

    Compressed for legibility, not necessity: ``0-47`` and the 48-term list
    mean the same thing to Slurm, but only one of them is readable in a
    batch script header or a ledger line. A campaign resubmitting a sparse
    gap emits exactly that gap -- ``3,17-19`` -- against the same script,
    which is what keeps the task-to-member mapping identical across
    convergence passes.

    Args:
        indices: The document positions to run. Order and duplicates are the
            caller's bug to avoid, not this function's to repair: indices
            must be strictly increasing.

    Returns:
        The comma-joined, range-compressed index expression.

    Raises:
        AppError: With ``ARRAY_INDICES_EMPTY`` when no index is given -- an
            array of nothing is a submission of nothing, and Slurm's own
            answer to ``--array=`` is a usage error three layers later.
            With ``ARRAY_ID_UNPARSABLE`` when indices are not strictly
            increasing, because a shuffled or duplicated index list means
            the caller's member bookkeeping is already wrong and the
            submission built from it would run the wrong set.
    """
    if len(indices) == 0:
        raise AppError(
            Hpc3ErrorCode.ARRAY_INDICES_EMPTY,
            "an array submission needs at least one task index; refusing to "
            "render --array= for an empty set.",
        )
    for earlier, later in itertools.pairwise(indices):
        if later <= earlier:
            raise AppError(
                Hpc3ErrorCode.ARRAY_ID_UNPARSABLE,
                f"array indices must be strictly increasing; got {later} after "
                f"{earlier}. A shuffled index list means the member bookkeeping "
                "behind it is already wrong.",
            )
    parts: list[str] = []
    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = index
        previous = index
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def _expand_bracket(base_id: str, expression: str, raw: str) -> tuple[str, ...]:
    """Expand one bracketed index expression into task ids.

    Args:
        base_id: The array's base job id.
        expression: The text between ``[`` and ``]``, throttle stripped.
        raw: The whole id being parsed, for error messages.

    Returns:
        One task id per index, in the order the expression lists them.

    Raises:
        AppError: With ``ARRAY_ID_UNPARSABLE`` on an empty term, a
            non-numeric bound, or a reversed range.
    """
    ids: list[str] = []
    for term in expression.split(","):
        if term == "":
            raise AppError(
                Hpc3ErrorCode.ARRAY_ID_UNPARSABLE,
                f"array id {raw!r} carries an empty index term.",
            )
        low, dash, high = term.partition("-")
        if not low.isdigit() or (dash == "-" and not high.isdigit()):
            raise AppError(
                Hpc3ErrorCode.ARRAY_ID_UNPARSABLE,
                f"array id {raw!r} carries a non-numeric index term {term!r}.",
            )
        if dash == "":
            ids.append(array_task_id(base_id, int(low)))
            continue
        first, last = int(low), int(high)
        if last < first:
            raise AppError(
                Hpc3ErrorCode.ARRAY_ID_UNPARSABLE,
                f"array id {raw!r} carries a reversed range {term!r}.",
            )
        ids.extend(array_task_id(base_id, index) for index in range(first, last + 1))
    return tuple(ids)


def expand_job_id(job_id: str) -> tuple[str, ...]:
    """Expand one cluster-reported job id into the task ids it stands for.

    The three shapes the cluster actually emits (probe job 55678543):

    * ``"55678543"`` -- a plain job, returned as itself.
    * ``"55678543_0"`` -- one array task, returned as itself.
    * ``"55678543_[2-3%2]"`` -- a pending aggregate, expanded to
      ``("55678543_2", "55678543_3")`` with the ``%`` throttle discarded:
      the throttle says how fast tasks may start, not which tasks exist.

    Args:
        job_id: The id as ``squeue`` or ``sacct`` printed it.

    Returns:
        Every individual task id the row stands for, in index order.

    Raises:
        AppError: With ``ARRAY_ID_UNPARSABLE`` when a bracket does not
            close, carries an empty or non-numeric term, or a reversed
            range. Raised rather than passed through, because an id this
            cannot read is a set of tasks of unknown membership -- and every
            caller uses the result to decide what is LIVE, where a silent
            miss becomes a double submission racing on one artifact.
    """
    base, underscore, suffix = job_id.partition("_")
    if underscore == "":
        return (job_id,)
    if not suffix.startswith("["):
        return (job_id,)
    if not suffix.endswith("]"):
        raise AppError(
            Hpc3ErrorCode.ARRAY_ID_UNPARSABLE,
            f"array id {job_id!r} opens a bracket it never closes.",
        )
    expression = suffix[1:-1].partition(_THROTTLE_SEPARATOR)[0]
    return _expand_bracket(base, expression, job_id)


__all__ = [
    "array_task_id",
    "expand_job_id",
    "format_array_indices",
]
