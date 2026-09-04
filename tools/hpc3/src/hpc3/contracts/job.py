"""The job contract: what a submission must state before it can be rendered.

The five rules below cost real time to learn, so they are enforced when a spec
is decoded rather than documented for a reader to remember. A spec that
violates one cannot be constructed from JSON at all, which means an invalid
job is caught at author time instead of an hour into the queue or ten hours
into a run.

1. A GPU request, where there is one, names its model, and the cluster
   carries it. A bare ``--gres=gpu:1`` on a mixed partition is a coin flip
   over generations; where the pinned torch does not target the card that
   comes up, the failure reads like a bug in the training code. A job may
   also ask for no GPU at all -- that is how CPU-only work is expressed --
   and the partition must agree in BOTH directions: a GPU on a CPU partition
   pends forever, and no GPU on a GPU partition runs happily while occupying
   a card it never touches.
2. The partition does not charge, unless the workspace has declared a
   service-unit budget to charge it against. Not a per-run consent flag -- a
   flag would be a limit a run could switch off; the allowance is a declared
   cap that also binds how much may be spent. A partition's name is not
   evidence either way: HPC3 has a QOS named ``free-gpu`` that charges full
   rate, and a default partition named ``standard`` that charges too. Only the
   measured usage factor decides.
3. A preemptible run long enough to matter carries requeue and checkpointing.
   Under ``PreemptMode=CANCEL`` an eviction destroys unsaved work.
4. The wall clock fits the partition. Slurm rejects the rest at submission.
5. The partition exists on the cluster the workspace selected.
6. An imaged payload does not need a shell it will not get. The batch script
   interpolates the command into an ``apptainer exec`` line, so an unquoted
   ``&&`` splits that line rather than reaching the container -- see
   :mod:`hpc3.contracts.payload`, which seven dead jobs paid for.

Every rule is asked of a :class:`~hpc3.contracts.cluster.ClusterFacts` rather
than of a constant, so the same code enforces a different machine's real
limits without a branch anywhere in it.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from hpc3.contracts.cluster import (
    ClusterFacts,
    GpuRequest,
    decode_gpu_request,
    encode_gpu_request,
    require_partition,
)
from hpc3.contracts.dependency import Dependency, decode_dependency, encode_dependency
from hpc3.contracts.experiment import encode_experiment, require_experiment
from hpc3.contracts.image import (
    HOST_BOUND_ROOTS,
    ImageReference,
    decode_image_reference,
    encode_image_reference,
)
from hpc3.contracts.job_rules import (
    _check_partition_carries_gpu,
    _check_partition_is_funded,
    _check_preemption_protection,
    _check_time_limit,
)
from hpc3.contracts.layout import require_project
from hpc3.contracts.payload import check_imaged_command_can_run
from hpc3.contracts.pins import encode_pinned_packages, require_pinned_packages


class JobSpec(TypedDict):
    """One submission, fully specified.

    Attributes:
        project: Which body of work this belongs to. Prefixes the job name so
            ``squeue`` is self-describing on a shared machine, and names the
            directory the job's scripts and logs live in so two projects
            cannot scatter into each other.
        name: The job's own name within its project.
        partition: Partition to submit to. Validated against the selected
            cluster, not against a fixed list.
        gpu: GPUs to pin, or None for a CPU-only job. Never generic when
            present -- see rule 1.
        cpus: CPU cores requested. Where a partition bills, this is usually
            the whole charge: billing tracks cores, not GPUs or memory.
        mem_gb: Host memory requested, in GiB.
        minutes: Wall-clock limit.
        requeue: Whether the script carries ``--requeue``. What that buys
            depends on the partition's ``PreemptMode``: under ``REQUEUE``
            Slurm resubmits a preempted job itself; under ``CANCEL`` the
            flag is INERT for preemption -- measured on HPC3's free
            partition 2026-09-02, when a wave took 22 array tasks carrying
            ``#SBATCH --requeue`` straight to terminal PREEMPTED with
            nothing left in the queue, and one ``hpc3-campaign`` converge
            pass was the thing that actually resubmitted them. On a
            CANCEL partition the campaign is the requeue.
        checkpoint_steps: Training steps between checkpoints; 0 means none.
        depends_on: Jobs that must finish before this one starts, or None for
            a job that waits on nothing. Emitted alongside
            ``--kill-on-invalid-dep=yes`` so an unsatisfiable wait cancels
            rather than parking forever. See
            :mod:`hpc3.contracts.dependency`.
        image: Image the payload runs inside, or None to run directly on the
            host. When present, ``env_path`` names a directory INSIDE that
            image rather than on the cluster, and the batch script wraps the
            payload in ``apptainer exec``. The reference carries a digest, so
            a queued job says which image it is running rather than merely
            where the file was read from -- see
            :mod:`hpc3.contracts.image`.
        env_path: Absolute path to a directory with a ``bin`` holding the
            payload's interpreter or binary. On the cluster when ``image`` is
            None; inside the image when it is not. A host-bound path is
            refused in the second case: HPC3 mounts ``/pub`` into every
            container, so an in-image environment there would be shadowed at
            runtime by the host directory and the interpreter would vanish.
        pinned_packages: Distribution versions that environment must actually
            contain. Checked at preflight against the environment's own
            report, because a path proves the directory exists and nothing
            about what is in it. Empty means the project declared no pins.
        deterministic: Whether kernel-level numerical determinism is
            configured for this run. Rendered into the batch script and
            recorded in the ledger, because it partitions results rather
            than improving them -- see
            :mod:`hpc3.contracts.workspace`.
        experiment: What this run IS, as free-form key/value pairs -- the
            corpus digest it trains on, its seed, the model it starts from.
            Required and never empty, and carried into the ledger, because a
            job id and a name say which row in ``squeue`` this was and nothing
            about which result it produced. See
            :mod:`hpc3.contracts.experiment`.
        command: Payload to run, executed with that ``bin`` already on PATH.
        gpu_pinned_because: Why the pinned GPU model must hold even while
            exhausted, or None when no such claim is made. The gpu-supply
            rule refuses a pin for an exhausted model while other models
            idle, because that combination is almost always an inherited
            default; a per-card measurement is the exception -- the card IS
            the experiment -- and this field is how a run says so and queues
            deliberately. Refused on a CPU-only job (no pin to justify) and
            refused blank (a reason nobody wrote is not a reason).
    """

    project: str
    name: str
    partition: str
    gpu: GpuRequest | None
    gpu_pinned_because: str | None
    cpus: int
    mem_gb: int
    minutes: int
    requeue: bool
    checkpoint_steps: int
    depends_on: Dependency | None
    image: ImageReference | None
    env_path: str
    pinned_packages: dict[str, str]
    deterministic: bool
    experiment: dict[str, str]
    command: str
    artifact: str | None


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


def require_artifact_in_command(obj: dict[str, JSONValue], command: str) -> str | None:
    """Read the declared artifact path, and refuse one the command never writes.

    The ledger is an index: job -> image -> artifact. A declaration nobody
    checks turns that into a confident wrong answer -- a reader follows the
    path, finds nothing, and cannot tell whether the run failed, wrote
    somewhere else, or was never going to write at all.

    The check is deliberately a substring test rather than a parse. This
    contract does not know any payload's flags, and it should not: the claim
    being verified is only that the path the ledger will publish is a path
    this command mentions. That catches the failure that actually happens --
    an output path edited in one place and not the other -- without pretending
    to understand what the command does with it.

    The key is REQUIRED; only its value may be null. Absent and null used to
    read alike, and the result was an index with its answer column empty: of
    130 recorded runs, 8 carried an image digest -- which fills itself in from
    the spec -- and ONE named where its result went. Every probe run in
    ``runs/`` writes ``--out /pub/wagnera3/probe/<name>.json`` and none of them
    said so, so `hpc3-trace` could reach the job and the image and then stop.

    Requiring the key does not force a fiction on a run that produces nothing.
    It forces the author to SAY which of the two they mean, once, in the spec
    -- and writing ``"artifact": null`` next to a command with an ``--out``
    flag is a claim somebody has to make on purpose rather than a field they
    never noticed.

    Args:
        obj: The run document being decoded.
        command: The command this run will execute, already validated.

    Returns:
        The declared path, or None when the run states it produces nothing
        durable. A directory is a legitimate answer where a run writes several
        files into one place: the reader follows it and finds them.

    Raises:
        JSONTypeError: If ``artifact`` is absent, is present but is not a
            non-empty string or null, or names a path its own command does
            not contain.
    """
    if "artifact" not in obj:
        raise JSONTypeError(
            "Field 'artifact' is required. Name the path this run writes its result to -- "
            "the ledger publishes it, so `hpc3-trace` can answer 'which file holds this "
            "run's answer'. Write null to state that the run produces nothing durable."
        )
    artifact = obj["artifact"]
    if artifact is None:
        return None
    if not isinstance(artifact, str):
        raise JSONTypeError(
            f"Field 'artifact' must be a string or null, got {type(artifact).__name__}"
        )
    if artifact == "":
        raise JSONTypeError("Field 'artifact' must name a path or be null, not an empty string")
    if artifact not in command:
        raise JSONTypeError(
            f"Field 'artifact' names {artifact!r}, which does not appear in this run's "
            "command. The ledger publishes this path as where the result will be, so a "
            "declaration the command does not honour would index a file nobody writes."
        )
    return artifact


def _require_env_path(obj: dict[str, JSONValue], image: ImageReference | None) -> str:
    """Read the interpreter directory, checked against where it will resolve.

    The same field names two different filesystems depending on ``image``,
    and getting that wrong fails in a way nothing reports. HPC3 bind-mounts
    ``/pub`` into every container, so an in-image environment under it is
    replaced at runtime by the host directory: the batch script would put a
    path on PATH that exists on both sides and holds the wrong interpreter,
    or none.

    Args:
        obj: Object being decoded.
        image: The image the job runs inside, or None for a host run.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, or empty; or,
            when an image is present, if the path sits under a root the
            cluster bind-mounts over.
    """
    value = _require_nonempty_str(obj, "env_path")
    if image is None:
        return value
    first_segment = value.strip("/").split("/")[0]
    if first_segment in HOST_BOUND_ROOTS:
        raise JSONTypeError(
            f"Field 'env_path' is inside an image but sits under /{first_segment}, "
            f"which the cluster bind-mounts over -- the host directory would shadow "
            f"the image's own environment at runtime, got {value!r}"
        )
    return value


def _require_positive(obj: dict[str, JSONValue], key: str) -> int:
    """Read a required integer field that must be at least one.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not an integer, or below one.
            A job asking for zero cores or zero minutes describes no work.
    """
    value = require_int(obj, key)
    if value < 1:
        raise JSONTypeError(f"Field '{key}' must be at least 1, got {value}")
    return value


def encode_job_spec(spec: JobSpec) -> dict[str, JSONValue]:
    """Encode a job spec to a JSON object.

    Args:
        spec: Spec to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "project": spec["project"],
        "name": spec["name"],
        "partition": spec["partition"],
        "gpu": encode_gpu_request(spec["gpu"]),
        "gpu_pinned_because": spec["gpu_pinned_because"],
        "cpus": spec["cpus"],
        "mem_gb": spec["mem_gb"],
        "minutes": spec["minutes"],
        "requeue": spec["requeue"],
        "checkpoint_steps": spec["checkpoint_steps"],
        "depends_on": encode_dependency(spec["depends_on"]),
        "artifact": spec["artifact"],
        "image": encode_image_reference(spec["image"]),
        "env_path": spec["env_path"],
        "pinned_packages": encode_pinned_packages(spec["pinned_packages"]),
        "deterministic": spec["deterministic"],
        "experiment": encode_experiment(spec["experiment"]),
        "command": spec["command"],
    }


def decode_job_spec(
    value: JSONValue, cluster: ClusterFacts, *, max_service_units: float
) -> JobSpec:
    """Decode and validate a JSON value into a job spec.

    Args:
        value: Value produced by the JSON loader.
        cluster: The cluster whose measured limits the rules are checked
            against.
        max_service_units: The workspace's declared service-unit budget.
            Keyword-only and required: a default would answer the billing
            question on behalf of a caller that never asked, and the safe
            answer and the intended answer differ between call sites.

    Returns:
        A spec that satisfies every submission rule on that cluster.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing,
            mistyped, empty, or non-positive.
        AppError: If the spec names no GPU the cluster carries, targets a
            partition it does not have, targets one that does not carry the
            model, bills without consent, leaves a long preemptible run
            unprotected, or exceeds the partition's ceiling. The code
            identifies which.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"job spec must be a JSON object, got {type(value).__name__}")

    partition = require_partition(cluster, value, "partition")
    gpu = decode_gpu_request(cluster, value.get("gpu"), "gpu")
    minutes = _require_positive(value, "minutes")
    requeue = require_bool(value, "requeue")

    checkpoint_steps = require_int(value, "checkpoint_steps")
    if checkpoint_steps < 0:
        raise JSONTypeError(
            f"Field 'checkpoint_steps' must not be negative, got {checkpoint_steps}"
        )

    deterministic = require_bool(value, "deterministic")
    _check_partition_carries_gpu(cluster, partition, gpu)
    _check_partition_is_funded(cluster, partition, max_service_units)
    _check_time_limit(cluster, partition, minutes)
    _check_preemption_protection(
        cluster, partition, minutes, requeue, checkpoint_steps, deterministic
    )

    image = decode_image_reference(value.get("image"), "image")
    command = _require_nonempty_str(value, "command")
    check_imaged_command_can_run(image, command)

    gpu_pinned_because = value.get("gpu_pinned_because")
    if gpu_pinned_because is not None:
        if not isinstance(gpu_pinned_because, str) or gpu_pinned_because.strip() == "":
            raise JSONTypeError(
                "Field 'gpu_pinned_because' must be a non-empty string when present; "
                "a blank reason is not a reason"
            )
        if gpu is None:
            raise JSONTypeError(
                "Field 'gpu_pinned_because' is declared on a CPU-only job -- "
                "there is no GPU pin to justify"
            )

    return JobSpec(
        project=require_project(value, "project"),
        name=_require_nonempty_str(value, "name"),
        partition=partition,
        gpu=gpu,
        gpu_pinned_because=gpu_pinned_because,
        cpus=_require_positive(value, "cpus"),
        mem_gb=_require_positive(value, "mem_gb"),
        minutes=minutes,
        requeue=requeue,
        checkpoint_steps=checkpoint_steps,
        depends_on=decode_dependency(value.get("depends_on"), "depends_on"),
        image=image,
        env_path=_require_env_path(value, image),
        pinned_packages=require_pinned_packages(value, "pinned_packages"),
        deterministic=deterministic,
        experiment=require_experiment(value, "experiment"),
        command=command,
        artifact=require_artifact_in_command(value, command),
    )


__all__ = [
    "JobSpec",
    "decode_job_spec",
    "encode_job_spec",
    "require_artifact_in_command",
]
