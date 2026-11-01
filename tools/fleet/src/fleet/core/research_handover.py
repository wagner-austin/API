"""Verify completed research through a recorded replacement without changing its request.

The private handover plan is content-addressed like the original request. It
can change only the interpreter, never the admission, journal, artifact or
native attempt. The existing controller inspector verifies the replacement;
the existing HPC poll verifies the completed allocation without submitting.
"""

from __future__ import annotations

import os
import pathlib

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_str,
)
from platform_core.load_report import post_load_report

from fleet.contracts.research import ResearchRequest, require_digest
from fleet.core import _test_hooks, research_control, research_hpc, staging


def poll(
    root: pathlib.Path,
    request: ResearchRequest,
    document: JSONObject,
    digest: str,
    task: str,
    environment: tuple[tuple[str, str], ...],
) -> JSONObject:
    """Re-measure the frozen replacement and the original completed allocation.

    Args:
        root: Evaluator-owned specification root.
        request: Original shared-queue reference.
        document: Original verified request, preserved byte for byte.
        digest: Content identity of the evaluator-authorized handover plan.
        task: Queue's reporting task.
        environment: Private controller environment, never published.

    Returns:
        Existing native observation plus the exact handover plan identity.

    Raises:
        ValueError: Any identity drift, incomplete attempt or verification failure.
        OSError: Missing private inputs or unavailable controller.
        AppError: The measured report could not be published.
    """
    plan = research_control.read_pinned(root, require_digest(digest), "handover.json")
    if set(plan) != {
        "request_sha256",
        "replacement_sha256",
        "provenance",
        "selfcheck",
        "wheels_dir",
        "native_job_id",
        "record_sha256",
        "recipe",
        "checksums",
    }:
        raise ValueError("RESEARCH_HANDOVER_FIELDS: unexpected handover plan fields")
    if (
        require_str(plan, "request_sha256") != request["spec_sha256"]
        or require_str(document, "backend") != "hpc3"
        or require_str(document, "phase") != "validation"
    ):
        raise ValueError("RESEARCH_HANDOVER_OWNER: require the original validation request")
    replacement = ResearchRequest(
        backend=request["backend"],
        attempt=request["attempt"],
        spec_sha256=require_digest(require_str(plan, "replacement_sha256")),
    )
    target = research_control.read_specification(root, replacement, require_str(document, "tenant"))
    python = research_control.require_path(target, "python")
    if {**target, "python": document["python"]} != document or target["python"] == document[
        "python"
    ]:
        raise ValueError("RESEARCH_HANDOVER_SCOPE: only a different interpreter may be substituted")
    directory = research_control.require_path(document, "directory")
    for field, filename in (("recipe", "run.json"), ("checksums", "checksums.txt")):
        if research_control.verified_input(plan, field) != directory / filename:
            raise ValueError(
                "RESEARCH_HANDOVER_ARTIFACT: pins must name the original attempt files"
            )
    provenance = research_control.verified_input(plan, "provenance")
    frozen = narrow_json_to_dict(load_json_str(_test_hooks.read_text(provenance)))
    if require_str(frozen, "controller_python") != str(python):
        raise ValueError("RESEARCH_HANDOVER_CONTROLLER: provenance names another interpreter")
    selfcheck = research_control.verified_input(plan, "selfcheck")
    wheels = research_control.require_path(plan, "wheels_dir")
    log = root / request["spec_sha256"] / "controller.jsonl"
    measured = research_control.invoke(
        (
            str(python),
            "-m",
            "hpc3.cli.controller_inspect",
            "--provenance",
            str(provenance),
            "--wheels-dir",
            str(wheels),
            "--selfcheck",
            str(selfcheck),
            "--selfcheck-sha256",
            staging.digest(_test_hooks.read_bytes(selfcheck)),
            "--report-task",
            task,
        ),
        log,
        300,
        environment,
    )
    if (
        require_str(measured, "controller_python") != str(python)
        or require_str(measured, "controller_python_sha256")
        != require_str(frozen, "controller_python_sha256")
        or require_str(measured, "provenance_sha256")
        != staging.digest(_test_hooks.read_bytes(provenance))
    ):
        raise ValueError("RESEARCH_HANDOVER_CONTROLLER: measured runtime differs from frozen input")
    result = research_hpc.poll(target, log, task, environment, completed_only=True)
    if require_str(result, "native_job_id") != require_str(plan, "native_job_id") or require_str(
        result, "record_sha256"
    ) != require_digest(require_str(plan, "record_sha256")):
        raise ValueError("RESEARCH_HANDOVER_RESULT: original allocation or verified result changed")
    report: JSONObject = {
        "kind": "RESEARCH_HANDOVER_EVIDENCE",
        "handover_sha256": digest,
        "request_sha256": request["spec_sha256"],
        "replacement_sha256": replacement["spec_sha256"],
        "original_python": require_str(document, "python"),
        "original_python_sha256": staging.digest(
            _test_hooks.read_bytes(research_control.require_path(document, "python"))
        ),
        "replacement_controller": measured,
        "native_observation": result,
        "measured_at_unix": _test_hooks.now(),
    }
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_test_hooks.read_text,
        task_id=task,
        body="RESEARCH HANDOVER LOAD-REPORT\n" + dump_json_str(report),
        family="hpc",
    )
    return {**result, "handover_sha256": digest}


def record(path: pathlib.Path, digest: str) -> None:
    """Exclusively record a verified handover while the existing poll lock is held.

    Args:
        path: Original request's private handover marker.
        digest: Verified content-addressed plan identity.

    Raises:
        OSError: An existing marker or failed durable write; never overwrite it.
        ValueError: Invalid digest or failed readback.
    """
    payload = require_digest(digest) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if _test_hooks.read_text(path) != payload:
        raise ValueError("RESEARCH_HANDOVER_WRITE: durable marker differs from the verified plan")


__all__ = ["poll", "record"]
