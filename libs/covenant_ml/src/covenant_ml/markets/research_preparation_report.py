"""Publish the evaluator's own reading of one privately prepared native attempt.

A prepared synthetic attempt lives only under the evaluator's private root: the
frozen policy, the admission journal and the bounded scheduler recipe that
``research_cluster.prepare_bundle`` wrote. None of it is readable by an audit,
so a session restating it on a wiki page is a copy nobody re-measures. This
reporter reads those three files where they are, refuses a recipe that does
not belong to the policy or the journal, and posts its reading under the
task's HPC evidence identity. It registers, stages and dispatches nothing.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_int,
    require_str,
)
from platform_core.load_report import post_load_report, report_identity
from typing_extensions import ReadOnly, TypedDict

from . import _test_hooks
from .research_journal import research_journal, research_policy_digest
from .research_records import decode_research_policy, decode_research_proposal
from .validation import read_document


class PreparedRecipe(TypedDict):
    """The scheduler request ``prepare_bundle`` froze for one attempt."""

    name: ReadOnly[str]
    research_attempt: ReadOnly[str]
    request_sha256: ReadOnly[str]
    cpus: ReadOnly[int]
    mem_gb: ReadOnly[int]
    minutes: ReadOnly[int]
    partition: ReadOnly[str]
    gpu_requested: ReadOnly[bool]
    requeue: ReadOnly[bool]


def decode_prepared_recipe(document: JSONObject) -> PreparedRecipe:
    """Validate the fields of a prepared recipe this reading reports.

    Args:
        document: Parsed ``run.json`` written by ``prepare_bundle``.

    Returns:
        The recipe's identity and scheduler request.

    Raises:
        ValueError: A required field is absent or has the wrong type.
    """
    experiment = require_dict(document, "experiment")
    if "gpu" not in document:
        raise ValueError("RESEARCH_PREPARATION_RECIPE: gpu must be declared, null for none")
    return PreparedRecipe(
        name=require_str(document, "name"),
        research_attempt=require_str(experiment, "research_attempt"),
        request_sha256=require_str(experiment, "request_sha256"),
        cpus=require_int(document, "cpus"),
        mem_gb=require_int(document, "mem_gb"),
        minutes=require_int(document, "minutes"),
        partition=require_str(document, "partition"),
        gpu_requested=document["gpu"] is not None,
        requeue=require_bool(document, "requeue"),
    )


def encode_prepared_recipe(recipe: PreparedRecipe) -> JSONObject:
    """Encode a prepared recipe reading for publication.

    Args:
        recipe: Validated recipe fields.

    Returns:
        JSON object with every field, in declaration order.
    """
    return {
        "name": recipe["name"],
        "research_attempt": recipe["research_attempt"],
        "request_sha256": recipe["request_sha256"],
        "cpus": recipe["cpus"],
        "mem_gb": recipe["mem_gb"],
        "minutes": recipe["minutes"],
        "partition": recipe["partition"],
        "gpu_requested": recipe["gpu_requested"],
        "requeue": recipe["requeue"],
    }


def report_preparation(
    policy_path: Path, ledger: Path, recipe_path: Path, task: str
) -> JSONObject:
    """Measure the policy, journal and recipe, then publish the reading as the program.

    Args:
        policy_path: The evaluator's frozen policy document.
        ledger: The admission journal initialized under that policy; only read.
        recipe_path: The ``run.json`` ``prepare_bundle`` wrote for one attempt.
        task: Full board task id receiving the ``hpc-report-<task>`` reading.

    Returns:
        The published measurement.

    Raises:
        ValueError: WRDS_REPORT_TASK for a malformed task, before any file is
            read; JOURNAL errors for a journal of another policy;
            RESEARCH_PREPARATION_POLICY, _ATTEMPT or _RESOURCES when the recipe
            names another policy, an attempt the journal never admitted, or an
            allocation the policy does not freeze.
        OSError: An input cannot be read.
        AppError: The board refused the reading.
    """
    report_identity(task, "hpc")
    policy = decode_research_policy(read_document(policy_path))
    digest = research_policy_digest(policy)
    attempts = research_journal(policy).read(ledger)
    recipe_bytes = recipe_path.read_bytes()
    recipe_text = recipe_bytes.decode("utf-8")
    recipe = decode_prepared_recipe(narrow_json_to_dict(load_json_str(recipe_text)))
    owner, _, number = recipe["research_attempt"].partition(":")
    if owner != digest:
        raise ValueError(
            f"RESEARCH_PREPARATION_POLICY: recipe names policy {owner}, journal holds {digest}"
        )
    admitted = {str(row["number"]) for row in attempts if row["status"] == "admitted"}
    if number.removesuffix(":final") not in admitted:
        raise ValueError(f"RESEARCH_PREPARATION_ATTEMPT: attempt {number} was not admitted")
    if (recipe["cpus"], recipe["mem_gb"]) != (policy["cpus"], policy["mem_gb"]):
        raise ValueError("RESEARCH_PREPARATION_RESOURCES: recipe allocation differs from policy")
    rows: list[JSONValue] = [
        {
            "number": row["number"],
            "status": row["status"],
            "seed": decode_research_proposal(narrow_json_to_dict(load_json_str(row["request"])))[
                "seed"
            ],
            "candidate": row["candidate"],
            "job_id": row["job_id"],
        }
        for row in attempts
    ]
    report: JSONObject = {
        "kind": "RESEARCH_PREPARATION_EVIDENCE",
        "observed_at": _test_hooks.clock(),
        "reporter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy_digest": digest,
        "seeds": list(policy["seeds"]),
        "max_candidates": policy["max_candidates"],
        "cpus": policy["cpus"],
        "mem_gb": policy["mem_gb"],
        "max_seconds": policy["max_seconds"],
        "ledger": str(ledger.resolve()),
        "journal_sha256": sha256(ledger.read_bytes()).hexdigest(),
        "attempts": rows,
        "recipe_sha256": sha256(recipe_bytes).hexdigest(),
        "recipe": encode_prepared_recipe(recipe),
    }
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_test_hooks.stack_text,
        task_id=task,
        body="RESEARCH PREPARATION LOAD-REPORT\n" + dump_json_str(report),
        family="hpc",
    )
    return report


__all__ = [
    "PreparedRecipe",
    "decode_prepared_recipe",
    "encode_prepared_recipe",
    "report_preparation",
]
