"""Stages that run in order, each waiting on the one before it.

A chain is the shape a pipeline actually has: SIRIUS then ZODIAC, extract then
evaluate. Stage two must not start until stage one has *succeeded*, and it
usually wants different resources -- a training stage holds a GPU, the
evaluation that reads its checkpoints often does not.

So a chain differs from a sweep in both directions. A sweep is one template run
several ways, concurrently, and its size is bounded by how many of your jobs
may RUN at once. A chain is several different jobs run one after another, and
only ever one of them is runnable, so the running ceiling cannot bind it. What
can bind a chain is the submit ceiling -- every stage is queued immediately,
just blocked -- and on HPC3 that is 3500 on ``free-part``, far above any
pipeline worth writing. No ceiling check is applied here for that reason; a
check that cannot fire is worse than none, because it reads as protection.

Every stage is validated up front, before anything is submitted. That matters:
the dependency ids are not known until the previous stage has been queued, so
it would be easy to build stage three only after stage one is already running
and discover its partition was misspelled then. Here a broken final stage stops
the chain before the first job exists.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue
from typing_extensions import TypedDict

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.job import JobSpec, decode_job_spec, encode_job_spec

MINIMUM_STAGES = 2
"""A chain of one is a run.

Refused rather than accepted-and-degenerate: a one-stage chain submitted
through this path would be an ordinary job carrying a dependency on nothing,
and the author almost certainly meant to write the second stage.
"""


class ChainSpec(TypedDict):
    """An ordered pipeline, fully validated before any of it is submitted.

    Attributes:
        stages: The jobs, in execution order. Each is a complete spec whose
            ``depends_on`` is None -- the real dependency cannot exist yet,
            because it names an id Slurm has not issued. Submission supplies
            it from the previous stage's actual id.
    """

    stages: list[JobSpec]


def decode_chain_spec(value: JSONValue, cluster: ClusterFacts) -> ChainSpec:
    """Decode and validate a JSON value into a chain.

    Args:
        value: Value produced by the JSON loader, carrying a ``stages`` list
            of complete job objects.
        cluster: The cluster whose measured limits every stage is checked
            against.

    Returns:
        A chain whose every stage satisfies every submission rule.

    Raises:
        JSONTypeError: If the value is not an object, ``stages`` is missing or
            holds fewer than :data:`MINIMUM_STAGES`, two stages share a name,
            or a stage already declares ``depends_on`` -- the chain supplies
            that, and a hand-written one would be silently replaced.
        AppError: If any stage breaks a submission rule.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"chain spec must be a JSON object, got {type(value).__name__}")

    raw = value.get("stages")
    if not isinstance(raw, list):
        raise JSONTypeError("Field 'stages' must be a list of job objects")
    if len(raw) < MINIMUM_STAGES:
        raise JSONTypeError(
            f"Field 'stages' must hold at least {MINIMUM_STAGES} stages, got {len(raw)}. "
            "A single stage is a run, not a chain."
        )

    for index, item in enumerate(raw):
        if isinstance(item, dict) and item.get("depends_on") is not None:
            raise JSONTypeError(
                f"Stage {index} declares 'depends_on'. A chain wires its own "
                "dependencies from the ids Slurm issues, so this would be replaced."
            )

    stages = [decode_job_spec(item, cluster) for item in raw]

    names = [stage["name"] for stage in stages]
    if len(set(names)) != len(names):
        raise JSONTypeError(f"Field 'stages' must not repeat a name, got {names}")

    return ChainSpec(stages=stages)


def encode_chain_spec(spec: ChainSpec) -> dict[str, JSONValue]:
    """Encode a chain to a JSON object.

    Args:
        spec: Chain to encode.

    Returns:
        JSON-serialisable mapping carrying every stage.
    """
    stages: list[JSONValue] = [encode_job_spec(stage) for stage in spec["stages"]]
    return {"stages": stages}


__all__ = ["MINIMUM_STAGES", "ChainSpec", "decode_chain_spec", "encode_chain_spec"]
