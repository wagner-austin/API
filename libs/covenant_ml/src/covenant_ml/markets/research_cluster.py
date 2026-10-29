"""Evaluator-owned HPC3 paths, frozen inputs and bounded attempt recipes."""

from __future__ import annotations

import shlex
from hashlib import sha256
from pathlib import Path, PurePosixPath

from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.stage_manifest import StagedFile, StageManifest, encode_stage_manifest
from typing_extensions import ReadOnly, TypedDict

from .research_journal import research_policy_digest
from .research_trial_records import ResearchTrial, encode_research_trial
from .validation import require_nonempty


class ResearchCluster(TypedDict):
    """Operator-owned transport and runtime paths; never supplied by a proposal."""

    python: ReadOnly[str]
    workspace: ReadOnly[str]
    project: ReadOnly[str]
    root: ReadOnly[str]
    entrypoint: ReadOnly[str]
    image: ReadOnly[str]
    checksums: ReadOnly[str]
    report_task: ReadOnly[str]


def encode_research_cluster(value: ResearchCluster) -> JSONObject:
    """Encode all execution settings.

    Args:
        value: Validated operator settings.

    Returns:
        JSON execution settings.
    """
    return {
        "python": value["python"],
        "workspace": value["workspace"],
        "project": value["project"],
        "root": value["root"],
        "entrypoint": value["entrypoint"],
        "image": value["image"],
        "checksums": value["checksums"],
        "report_task": value["report_task"],
    }


def decode_research_cluster(document: JSONObject) -> ResearchCluster:
    """Require native controller paths and POSIX cluster paths without traversal.

    Args:
        document: Operator configuration.

    Returns:
        Immutable validated settings.

    Raises:
        ValueError: RESEARCH_CLUSTER_CONFIG for invalid fields or paths.
    """
    if set(document) != set(ResearchCluster.__annotations__):
        raise ValueError("RESEARCH_CLUSTER_CONFIG: exact fields required")
    value = ResearchCluster(
        python=require_nonempty(document, "python"),
        workspace=require_nonempty(document, "workspace"),
        project=require_nonempty(document, "project"),
        root=require_nonempty(document, "root"),
        entrypoint=require_nonempty(document, "entrypoint"),
        image=require_nonempty(document, "image"),
        checksums=require_nonempty(document, "checksums"),
        report_task=require_nonempty(document, "report_task"),
    )
    paths = (
        Path(value["python"]),
        Path(value["workspace"]),
        PurePosixPath(value["root"]),
        PurePosixPath(value["entrypoint"]),
        PurePosixPath(value["image"]),
    )
    for path in paths:
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("RESEARCH_CLUSTER_CONFIG: absolute non-traversing paths required")
    return value


def attempt_identity(trial: ResearchTrial) -> str:
    """Bind one attempt number to its entire frozen policy.

    Args:
        trial: Admitted trial request.

    Returns:
        Unique experiment identity within this study's journal.
    """
    return f"{research_policy_digest(trial['policy'])}:{trial['number']}"


def prepare_bundle(directory: Path, cluster: ResearchCluster, trial: ResearchTrial) -> None:
    """Freeze a new request, checksum file, staging manifest and bounded recipe.

    Args:
        directory: Exclusive new local attempt directory.
        cluster: Operator-owned runtime and pre-staged input identities.
        trial: Request reconstructed from the admission journal.

    Raises:
        ValueError: RESEARCH_CLUSTER_INPUTS for unpinned code, runtime or manifest.
        FileExistsError: Attempt preparation already exists; never overwrite it.
    """
    policy = trial["policy"]
    lines = set(cluster["checksums"].splitlines())
    required = {
        f"{policy['image_digest']}  {cluster['image']}",
        f"{policy['dataset_digest']}  {trial['manifest']}",
        f"{policy['code_digest']}  {cluster['entrypoint']}/scripts/research_trial.py",
    }
    if not required <= lines:
        raise ValueError("RESEARCH_CLUSTER_INPUTS: require frozen runtime, manifest and entrypoint")
    directory.mkdir(parents=True, exist_ok=False)
    remote = PurePosixPath(cluster["root"]) / research_policy_digest(policy) / str(trial["number"])
    request = dump_json_str(encode_research_trial(trial))
    request_digest = sha256(request.encode()).hexdigest()
    checksums = cluster["checksums"].rstrip("\n") + f"\n{request_digest}  {remote}/request.json\n"
    files: list[StagedFile] = []
    for name, text in (("request.json", request), ("checksums.txt", checksums)):
        payload = text.encode("utf-8")
        (directory / name).write_bytes(payload)
        files.append(
            StagedFile(name=name, sha256=sha256(payload).hexdigest(), size_bytes=len(payload))
        )
    manifest = StageManifest(
        destination=str(remote),
        files=files,
        provenance={
            "research_attempt": attempt_identity(trial),
            "policy": research_policy_digest(policy),
        },
    )
    (directory / "stage.json").write_text(dump_json_str(encode_stage_manifest(manifest)))
    (directory / "expected.txt").write_text("\n".join(item["sha256"] for item in files))
    output = str(remote / "result" / "evaluation.json")
    seconds = trial["proposal"]["seconds"]
    invocation = shlex.join(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=5s",
            f"{seconds}s",
            "python",
            "-m",
            "scripts.research_trial",
            "--request",
            str(remote / "request.json"),
            "--out",
            output,
        ]
    )
    command = (
        shlex.join(["sha256sum", "-c", str(remote / "checksums.txt")])
        + " && cd "
        + shlex.quote(cluster["entrypoint"])
        + " && "
        + invocation
    )
    recipe: JSONObject = {
        "project": cluster["project"],
        "name": "research-" + research_policy_digest(policy)[:12] + f"-{trial['number']}",
        "command": shlex.join(["bash", "-c", command]),
        "artifact": output,
        "experiment": {
            "research_attempt": attempt_identity(trial),
            "request_sha256": request_digest,
        },
        "cpus": 1,
        "mem_gb": 4,
        "minutes": (seconds + 64) // 60,
        "partition": "free",
        "gpu": None,
        "requeue": False,
        "image": {"path": cluster["image"], "sha256": policy["image_digest"], "binds": ["/pub"]},
    }
    (directory / "run.json").write_text(dump_json_str(recipe))


__all__ = [
    "ResearchCluster",
    "attempt_identity",
    "decode_research_cluster",
    "encode_research_cluster",
    "prepare_bundle",
]
