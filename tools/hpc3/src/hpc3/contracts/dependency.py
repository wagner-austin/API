"""What a job waits for, and what happens when the wait can never end.

A staged pipeline -- extract then evaluate, SIRIUS then ZODIAC -- is two jobs
where the second must not start until the first has finished. Slurm expresses
that with ``--dependency``, and the hazard is specific and measured: when a
dependency can never be satisfied, Slurm does NOT reject the dependent job. It
queues it forever with reason ``DependencyNeverSatisfied``. On HPC3 that was
261 of 621 pending GPU jobs in a single sample -- a queue mostly full of work
that will never run, indistinguishable in ``squeue``'s state column from work
that is merely waiting.

So every dependency this package emits is paired with
``--kill-on-invalid-dep=yes``. A stage whose predecessor failed is cancelled
rather than parked: it frees the QOS slot it was holding, it stops counting
against the concurrent-job ceiling, and it turns a silent forever-pend into a
terminal state the ledger can close. The reason it was cancelled is not lost --
it is what the predecessor's own accounting row says.

``afterok`` is the default and the one worth wanting. ``afterany`` runs the
next stage whether or not the previous one worked, which for a pipeline means
computing a second result on top of a failed first.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

AFTER_OK = "afterok"
"""Run only if the predecessor succeeded. The pipeline default."""

AFTER_ANY = "afterany"
"""Run once the predecessor finished, successfully or not."""

AFTER_NOT_OK = "afternotok"
"""Run only if the predecessor failed -- a cleanup or salvage stage."""

DEPENDENCY_KINDS = (AFTER_OK, AFTER_ANY, AFTER_NOT_OK)
"""Every kind this package emits.

Slurm also has ``after`` (start once the predecessor STARTS), ``singleton``
and ``expand``. They are absent because none of them expresses "this stage
consumes the previous stage's output", which is the only thing a pipeline
needs, and ``after`` in particular reads as if it did and does not.
"""


class Dependency(TypedDict):
    """What a job waits on before it may start.

    Attributes:
        kind: One of :data:`DEPENDENCY_KINDS`.
        job_ids: Ids that must reach the required state first. Never empty --
            a dependency on nothing is not a dependency, and Slurm would take
            ``--dependency=afterok:`` as a malformed argument rather than as
            an absent one.
    """

    kind: str
    job_ids: list[str]


def decode_dependency(value: JSONValue, key: str) -> Dependency | None:
    """Decode a job's dependency, which may be absent.

    Args:
        value: The field's value. ``None`` means the job waits for nothing.
        key: Field name, used in error messages.

    Returns:
        The validated dependency, or None.

    Raises:
        JSONTypeError: If the value is neither null nor an object, the kind is
            not one this package emits, the id list is missing or empty, an id
            is not a positive integer written as a string, or an id repeats.
            Ids are checked because a typo'd one is not a job that will never
            finish -- it is a job that never existed, which under
            ``kill-on-invalid-dep`` cancels the dependent stage immediately
            and reads like the pipeline failed.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"Field {key!r} must be a dependency object or null, got {type(value).__name__}"
        )

    kind = require_str(value, "kind")
    if kind not in DEPENDENCY_KINDS:
        raise JSONTypeError(
            f"Field {key!r} kind must be one of {list(DEPENDENCY_KINDS)}, got {kind!r}"
        )

    raw = require_list(value, "job_ids")
    if raw == []:
        raise JSONTypeError(f"Field {key!r} must name at least one job id")

    job_ids: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise JSONTypeError(f"Field {key!r} job_ids must be strings, got {type(item).__name__}")
        if item == "" or not item.isdigit():
            raise JSONTypeError(f"Field {key!r} job_ids must be numeric Slurm ids, got {item!r}")
        job_ids.append(item)

    if len(set(job_ids)) != len(job_ids):
        raise JSONTypeError(f"Field {key!r} must not repeat a job id, got {job_ids}")

    return Dependency(kind=kind, job_ids=job_ids)


def encode_dependency(dependency: Dependency | None) -> JSONValue:
    """Encode a dependency back to JSON.

    Args:
        dependency: The dependency, or None.

    Returns:
        An object carrying the kind and ids, or null.
    """
    if dependency is None:
        return None
    ids: list[JSONValue] = list(dependency["job_ids"])
    return {"kind": dependency["kind"], "job_ids": ids}


def dependency_argument(dependency: Dependency) -> str:
    """Render the value of ``--dependency``.

    Args:
        dependency: What the job waits on.

    Returns:
        ``<kind>:<id>[:<id>...]`` -- colon-joined, which is Slurm's AND form.
        The comma form is OR, and a pipeline that started when *any* of its
        inputs finished would read the others mid-write.
    """
    return dependency["kind"] + ":" + ":".join(dependency["job_ids"])


def describe_dependency(dependency: Dependency | None) -> str:
    """Render a dependency for a human reading a console line.

    Args:
        dependency: What the job waits on, or None.

    Returns:
        A short phrase, or ``"nothing"`` when the job is unblocked -- never an
        empty string, which in a status line reads as a missing value rather
        than as an absence.
    """
    if dependency is None:
        return "nothing"
    return f"{dependency['kind']} {','.join(dependency['job_ids'])}"


__all__ = [
    "AFTER_ANY",
    "AFTER_NOT_OK",
    "AFTER_OK",
    "DEPENDENCY_KINDS",
    "Dependency",
    "decode_dependency",
    "dependency_argument",
    "describe_dependency",
    "encode_dependency",
]
