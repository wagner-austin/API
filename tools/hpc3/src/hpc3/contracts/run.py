"""Resolving a run document against a workspace into a full job spec.

A run document says only what is specific to this run::

    {"project": "abl", "name": "armB-s42", "command": "python train.py --arm B"}

Everything else -- partition, GPU, cores, memory, wall clock, environment --
comes from the project's entry in the workspace. Any of those may be restated
in the run document to override it for this run alone::

    {"project": "abl", "name": "armC-full", "command": "...",
     "minutes": 600, "requeue": true, "checkpoint_steps": 500}

Overriding is not a way around validation. The merge produces an ordinary job
object and hands it to :func:`~hpc3.contracts.job.decode_job_spec`, which
applies every rule it would apply to a hand-authored spec -- so an override
that lengthens a preemptible run past an hour must also carry requeue and
checkpointing, exactly as it would have to if the whole spec were written out.

An unrecognised field is refused rather than ignored. ``"minute": 600`` is a
run the author believes is capped at ten hours and that Slurm will kill at the
project default; silently taking the default is how that happens.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, require_list

from hpc3.contracts.chain import ChainSpec, decode_chain_spec
from hpc3.contracts.job import JobSpec, decode_job_spec
from hpc3.contracts.layout import require_project
from hpc3.contracts.sweep import SweepSpec, decode_sweep_spec
from hpc3.contracts.workspace import (
    PROJECT_FIELDS,
    ProjectConfig,
    Workspace,
    encode_project_config,
    require_project_config,
    workspace_cluster,
)

RUN_IDENTITY_FIELDS = ("project", "name", "command", "experiment", "depends_on")
"""What only a run can say. Never inherited, never optional.

``experiment`` is here rather than among the project defaults because what a
run IS differs per run -- the corpus digest and the seed are the whole point of
running it twice.

``depends_on`` is here rather than among the defaults for a harder reason: it
names Slurm job ids, which do not exist until something has been submitted. A
project default could only ever name ids from a previous session, and a stale
one is not an error -- it is a wait on a job that finished last week, which
``afterok`` satisfies instantly and silently.
"""

SWEEP_IDENTITY_FIELDS = ("project", "name", "members", "experiment", "depends_on")

CHAIN_IDENTITY_FIELDS = ("project", "name", "stages", "experiment")
"""What a chain document may say for itself.

``depends_on`` is absent deliberately: a chain wires its stages to each other
from the ids Slurm issues, and an outer dependency stated here would have to be
merged with those. That is expressible and not yet needed, and a field that
half-works is worse than one that is not there.
"""

STAGE_IDENTITY_FIELDS = ("suffix", "command")
"""What one stage says for itself, on top of anything it overrides.

No ``name``: a stage is named by the chain plus its suffix, so the two stages
of one pipeline sort together in ``squeue`` and share a log directory prefix.
"""
"""The sweep equivalent; the command comes from each member instead.

The sweep's ``experiment`` describes what the members share; expansion adds
each member's suffix so the rows stay distinguishable in the ledger.
"""


def _check_known_fields(document: dict[str, JSONValue], identity: tuple[str, ...]) -> None:
    """Reject any field the merge would silently drop.

    Args:
        document: The run or sweep document as read.
        identity: The fields specific to this document kind.

    Raises:
        AppError: With ``RUN_FIELD_UNKNOWN`` naming the offending fields and
            what is accepted. A misspelled override is a run that does not do
            what its author wrote down, and ignoring it makes the document and
            the job disagree with nothing to show for it.
    """
    allowed = {*identity, *PROJECT_FIELDS}
    unknown = sorted(set(document) - allowed)
    if unknown != []:
        raise AppError(
            Hpc3ErrorCode.RUN_FIELD_UNKNOWN,
            f"Document carries unknown field(s) {unknown}. "
            f"A run may state {list(identity)} and may override "
            f"{list(PROJECT_FIELDS)}; anything else would be ignored.",
        )


def _merged(
    document: dict[str, JSONValue], defaults: ProjectConfig, project: str
) -> dict[str, JSONValue]:
    """Overlay a document's overrides onto a project's defaults.

    Args:
        document: The run or sweep document as read.
        defaults: The project's declared resource settings.
        project: The validated project name.

    Returns:
        A complete job object, ready for the job contract to validate.
    """
    merged = encode_project_config(defaults)
    for field in PROJECT_FIELDS:
        if field in document:
            merged[field] = document[field]
    merged["project"] = project
    return merged


def _require_document(value: JSONValue, kind: str) -> dict[str, JSONValue]:
    """Narrow a loaded value to an object.

    Args:
        value: Value produced by the JSON loader.
        kind: Word for the document, used in the message.

    Returns:
        The document.

    Raises:
        JSONTypeError: If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{kind} must be a JSON object, got {type(value).__name__}")
    return value


def resolve_run(workspace: Workspace, value: JSONValue) -> JobSpec:
    """Resolve one run document into a fully specified job.

    Args:
        workspace: The decoded workspace supplying the defaults.
        value: The run document, as produced by the JSON loader.

    Returns:
        A spec satisfying every submission rule, with the project's defaults
        filled in and the document's overrides applied.

    Raises:
        JSONTypeError: If the document is not an object, or a field is
            missing, mistyped or empty.
        AppError: With ``RUN_FIELD_UNKNOWN`` for an unrecognised field,
            ``WORKSPACE_PROJECT_UNKNOWN`` if the workspace declares no such
            project, or one of the job contract's codes if the merged result
            breaks a submission rule.
    """
    document = _require_document(value, "run")
    _check_known_fields(document, RUN_IDENTITY_FIELDS)

    project = require_project(document, "project")
    merged = _merged(document, require_project_config(workspace, project), project)
    merged["name"] = document.get("name")
    merged["command"] = document.get("command")
    merged["experiment"] = document.get("experiment")
    merged["depends_on"] = document.get("depends_on")
    return decode_job_spec(merged, workspace_cluster(workspace))


def resolve_sweep(workspace: Workspace, value: JSONValue) -> SweepSpec:
    """Resolve one sweep document into a validated sweep.

    The template's command is taken from the first member rather than stated
    separately: a sweep whose members all replace the command has no use for a
    template command, and leaving a stale one in the document invites reading
    it as what runs.

    Args:
        workspace: The decoded workspace supplying the defaults.
        value: The sweep document, as produced by the JSON loader.

    Returns:
        A sweep whose template satisfies every submission rule and whose size
        fits the partition's per-user ceilings.

    Raises:
        JSONTypeError: If the document is not an object, the member list is
            missing or empty, or a member is invalid.
        AppError: With ``RUN_FIELD_UNKNOWN`` for an unrecognised field,
            ``WORKSPACE_PROJECT_UNKNOWN`` for an undeclared project, a job
            code if the template breaks a submission rule, or a sweep code if
            the set exceeds the QOS.
    """
    document = _require_document(value, "sweep")
    _check_known_fields(document, SWEEP_IDENTITY_FIELDS)

    project = require_project(document, "project")
    base = _merged(document, require_project_config(workspace, project), project)
    base["name"] = document.get("name")
    base["experiment"] = document.get("experiment")
    base["depends_on"] = document.get("depends_on")

    members = require_list(document, "members")
    if members == []:
        raise JSONTypeError("Field 'members' must not be empty")
    first = members[0]
    base["command"] = first.get("command") if isinstance(first, dict) else None

    return decode_sweep_spec({"base": base, "members": members}, workspace_cluster(workspace))


def _resolve_stage(
    stage: JSONValue, chain: dict[str, JSONValue], defaults: ProjectConfig, project: str
) -> dict[str, JSONValue]:
    """Overlay one stage's overrides onto the chain's, onto the project's.

    Three layers rather than two, because the stages of a pipeline genuinely
    differ: a training stage holds a GPU and the evaluation reading its
    checkpoints often does not, while both share the environment and the
    wall-clock policy the chain set once.

    Args:
        stage: The stage document.
        chain: The chain document, supplying the middle layer.
        defaults: The project's declared settings, supplying the base layer.
        project: The validated project name.

    Returns:
        A complete job object, ready for the job contract to validate.

    Raises:
        JSONTypeError: If the stage is not an object or names no suffix.
        AppError: With ``RUN_FIELD_UNKNOWN`` if the stage carries a field
            neither identity nor overridable.
    """
    document = _require_document(stage, "chain stage")
    _check_known_fields(document, STAGE_IDENTITY_FIELDS)

    suffix = document.get("suffix")
    if not isinstance(suffix, str) or suffix == "":
        raise JSONTypeError(f"Stage field 'suffix' must be a non-empty string, got {suffix!r}")
    if "/" in suffix or "\\" in suffix:
        raise JSONTypeError(
            f"Stage field 'suffix' must not contain a path separator, got {suffix!r}"
        )

    merged = _merged(chain, defaults, project)
    for field in PROJECT_FIELDS:
        if field in document:
            merged[field] = document[field]

    merged["name"] = f"{chain.get('name')}-{suffix}"
    merged["command"] = document.get("command")
    # The chain's experiment plus this stage's own label, so two stages of one
    # pipeline are two distinguishable rows in the ledger rather than two
    # records claiming to be the same thing.
    chain_experiment = chain.get("experiment")
    experiment = dict(chain_experiment) if isinstance(chain_experiment, dict) else {}
    experiment["stage"] = suffix
    merged["experiment"] = experiment
    merged["depends_on"] = None
    return merged


def resolve_chain(workspace: Workspace, value: JSONValue) -> ChainSpec:
    """Resolve one chain document into an ordered list of validated jobs.

    Args:
        workspace: The decoded workspace supplying the defaults.
        value: The chain document, as produced by the JSON loader.

    Returns:
        A chain whose every stage satisfies every submission rule. No stage
        carries a dependency yet: those name ids Slurm has not issued, and
        submission supplies them.

    Raises:
        JSONTypeError: If the document is not an object, the stage list is
            missing or too short, or a stage is invalid.
        AppError: With ``RUN_FIELD_UNKNOWN`` for an unrecognised field,
            ``WORKSPACE_PROJECT_UNKNOWN`` for an undeclared project, or a job
            code if any stage breaks a submission rule.
    """
    document = _require_document(value, "chain")
    _check_known_fields(document, CHAIN_IDENTITY_FIELDS)

    project = require_project(document, "project")
    defaults = require_project_config(workspace, project)

    stages = require_list(document, "stages")
    resolved: list[JSONValue] = [
        _resolve_stage(stage, document, defaults, project) for stage in stages
    ]
    return decode_chain_spec({"stages": resolved}, workspace_cluster(workspace))


__all__ = [
    "CHAIN_IDENTITY_FIELDS",
    "RUN_IDENTITY_FIELDS",
    "STAGE_IDENTITY_FIELDS",
    "SWEEP_IDENTITY_FIELDS",
    "resolve_chain",
    "resolve_run",
    "resolve_sweep",
]
