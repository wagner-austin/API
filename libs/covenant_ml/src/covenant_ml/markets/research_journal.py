"""Research admission and lifecycle on the shared transactional event journal.

Persist the raw request before decoding it. Malformed requests therefore remain
visible as not-admitted attempts even when their decoder raises. An admitted
candidate/seed is consumed permanently, including when execution fails.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.event_journal import EventJournal
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_int,
    require_str,
)
from platform_core.run_record import decode_run_record
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .research_limits import policy_refusal
from .research_records import (
    ResearchPolicy,
    ResearchProposal,
    candidate_digest,
    decode_research_policy,
    decode_research_proposal,
    encode_research_policy,
)
from .validation import require_bounded_int, require_nonempty


class ResearchAttempt(TypedDict):
    """Durable proposal, disposition and scheduler/result references."""

    number: ReadOnly[int]
    request: ReadOnly[str]
    status: ReadOnly[str]
    reason: ReadOnly[str]
    candidate: ReadOnly[str]
    job_id: ReadOnly[str]
    record: ReadOnly[str]
    observed_ms: ReadOnly[int]
    final_manifest_digest: ReadOnly[str]
    selection: ReadOnly[str]


def encode_research_attempt(attempt: ResearchAttempt) -> JSONObject:
    """Encode lifecycle evidence without introducing a numeric result format.

    Args:
        attempt: Validated state.

    Returns:
        JSON object; record contains the existing RunRecord encoding.
    """
    return {
        "number": attempt["number"],
        "request": attempt["request"],
        "status": attempt["status"],
        "reason": attempt["reason"],
        "candidate": attempt["candidate"],
        "job_id": attempt["job_id"],
        "record": attempt["record"],
        "observed_ms": attempt["observed_ms"],
        "final_manifest_digest": attempt["final_manifest_digest"],
        "selection": attempt["selection"],
    }


def decode_research_attempt(document: JSONObject) -> ResearchAttempt:
    """Validate a serialized lifecycle row.

    Args:
        document: Encoded attempt.

    Returns:
        Immutable row.

    Raises:
        ValueError: RESEARCH-004 for unknown fields or status.
    """
    if set(document) != set(ResearchAttempt.__annotations__):
        raise ValueError("[RESEARCH-004] require exact attempt fields")
    status = require_str(document, "status")
    if status not in {
        "refused",
        "admitted",
        "submitting",
        "running",
        "complete",
        "failed",
        "timeout",
        "interrupted",
    }:
        raise ValueError("[RESEARCH-004] unknown attempt status")
    selection = require_str(document, "selection")
    if selection not in {"", "selected", "released"}:
        raise ValueError("[RESEARCH-004] unknown selection state")
    final_digest = require_str(document, "final_manifest_digest")
    if selection:
        if status != "complete":
            raise ValueError("[RESEARCH-004] selection requires completed validation")
        final_digest = require_digest(document, "final_manifest_digest")
    elif final_digest:
        raise ValueError("[RESEARCH-004] only a selected attempt can bind final evidence")
    return ResearchAttempt(
        number=require_bounded_int(document, "number", 1, 2**31 - 1),
        request=require_nonempty(document, "request"),
        status=status,
        reason=require_str(document, "reason"),
        candidate=require_str(document, "candidate"),
        job_id=require_str(document, "job_id"),
        record=require_str(document, "record"),
        observed_ms=require_bounded_int(document, "observed_ms", 0, 2**63 - 1),
        final_manifest_digest=final_digest,
        selection=selection,
    )


def _refusal(
    policy: ResearchPolicy, proposal: ResearchProposal, history: tuple[ResearchAttempt, ...]
) -> str:
    """Check policy and consumed attempts while holding the journal writer lock.

    Args:
        policy: Frozen evaluator limits.
        proposal: Decoded request.
        history: Complete journal history.

    Returns:
        Named refusal, or empty for an admissible proposal.
    """
    if any(row["selection"] for row in history):
        return "study_sealed"
    reason = policy_refusal(policy, proposal)
    if reason:
        return reason
    return _history_refusal(policy, proposal, history)


def _history_refusal(
    policy: ResearchPolicy, proposal: ResearchProposal, history: tuple[ResearchAttempt, ...]
) -> str:
    """Reserve only unseen candidate/seed pairs within the total trial ceiling.

    Args:
        policy: Frozen trial ceiling.
        proposal: Decoded candidate.
        history: Locked prior attempts.

    Returns:
        Named refusal or empty string.
    """
    candidate = candidate_digest(proposal)
    consumed = tuple(row for row in history if row["candidate"])
    candidates = {row["candidate"] for row in consumed}
    if proposal["parent"] != "root" and proposal["parent"] not in candidates:
        return "parent_unknown"
    if candidate not in candidates and len(candidates) >= policy["max_candidates"]:
        return "candidate_budget"
    for row in consumed:
        prior = decode_research_proposal(narrow_json_to_dict(load_json_str(row["request"])))
        if row["candidate"] == candidate and prior["seed"] == proposal["seed"]:
            return "candidate_seed_consumed"
    return ""


def _complete(policy: ResearchPolicy, event: JSONObject, row: ResearchAttempt) -> str:
    """Validate a completion's pinned lineage and existing RunRecord.

    Args:
        policy: Frozen identities.
        event: Completion event.
        row: Running attempt.

    Returns:
        Canonical shared RunRecord text.

    Raises:
        ValueError: RESEARCH-007 for mismatched lineage or image.
    """
    record_document = require_dict(event, "record")
    record = decode_run_record(record_document)
    if (
        require_nonempty(event, "job_id") != row["job_id"]
        or require_digest(event, "dataset_digest") != policy["dataset_digest"]
        or require_digest(event, "code_digest") != policy["code_digest"]
        or record["fingerprint"]["image_digest"] != policy["image_digest"]
        or record["label"] != str(row["number"])
        or record["experiment"] != policy["protocol_digest"]
        or not record["payload_digest"]
    ):
        raise ValueError("[RESEARCH-007] completion does not match admitted lineage")
    return dump_json_str(record_document)


def _transition(
    policy: ResearchPolicy, history: tuple[ResearchAttempt, ...], event: JSONObject
) -> tuple[ResearchAttempt, ...]:
    """Replay a request, admission or scheduler lifecycle transition.

    Args:
        policy: Immutable policy.
        history: Prior state.
        event: Strict journal event.

    Returns:
        Updated immutable state.

    Raises:
        ValueError: RESEARCH-005/006 for invalid event ordering or transition.
    """
    action = require_str(event, "action")
    observed = require_bounded_int(event, "observed_ms", 0, 2**63 - 1)
    if action == "receive":
        request = dump_json_str(require_dict(event, "request"))
        return (
            *history,
            ResearchAttempt(
                number=len(history) + 1,
                request=request,
                status="refused",
                reason="not_admitted",
                candidate="",
                job_id="",
                record="",
                observed_ms=observed,
                final_manifest_digest="",
                selection="",
            ),
        )
    row = _event_row(history, event)
    number = row["number"]
    updated = encode_research_attempt(row)
    updated["observed_ms"] = observed
    if action == "admit" and row["reason"] == "not_admitted":
        proposal = decode_research_proposal(narrow_json_to_dict(load_json_str(row["request"])))
        reason = _refusal(policy, proposal, history)
        updated["reason"] = reason
        updated["status"] = "refused" if reason else "admitted"
        updated["candidate"] = "" if reason else candidate_digest(proposal)
    elif action == "submit" and row["status"] == "admitted":
        updated["status"] = "submitting"
    elif action == "scheduled" and row["status"] == "submitting":
        job_id = require_nonempty(event, "job_id")
        if any(prior["job_id"] == job_id for prior in history):
            raise ValueError("[RESEARCH-006] scheduler job already belongs to an attempt")
        updated["status"], updated["job_id"] = "running", job_id
    elif action == "complete" and row["status"] == "running":
        updated["record"] = _complete(policy, event, row)
        updated["status"] = "complete"
    elif action in {"select", "release_final"}:
        updated["selection"], updated["final_manifest_digest"] = _selection(history, event, row)
    elif action in {"failed", "timeout", "interrupted"} and row["status"] in {
        "admitted",
        "submitting",
        "running",
    }:
        updated["status"], updated["reason"] = action, require_nonempty(event, "reason")
    else:
        raise ValueError("[RESEARCH-006] invalid attempt transition")
    return (*history[: number - 1], decode_research_attempt(updated), *history[number:])


def _event_row(history: tuple[ResearchAttempt, ...], event: JSONObject) -> ResearchAttempt:
    """Resolve an existing attempt and refuse backwards event time.

    Args:
        history: Locked journal state.
        event: Event carrying the attempt number and observation time.

    Returns:
        Existing attempt without changing its execution status.

    Raises:
        ValueError: RESEARCH-005 for an absent attempt or backwards time.
    """
    number = require_int(event, "number")
    if not 1 <= number <= len(history):
        raise ValueError("[RESEARCH-005] unknown attempt number")
    row = history[number - 1]
    if require_int(event, "observed_ms") < row["observed_ms"]:
        raise ValueError("[RESEARCH-005] observation time moved backwards")
    return row


def _selection(
    history: tuple[ResearchAttempt, ...], event: JSONObject, row: ResearchAttempt
) -> tuple[str, str]:
    """Advance final selection while retaining the completed validation receipt.

    Args:
        history: Locked study attempts, including every active reservation.
        event: Evaluator-owned select or release_final event.
        row: Target validation attempt.

    Returns:
        New selection state and its frozen final-manifest digest.

    Raises:
        ValueError: RESEARCH-008 for unfinished or sealed validation;
            RESEARCH-009 for release before selection or changed final evidence.
    """
    if row["status"] != "complete":
        raise ValueError("[RESEARCH-008] selection requires completed validation")
    digest = _final_identity(event, row)
    if require_str(event, "action") == "select":
        if any(
            prior["selection"] or prior["status"] in {"admitted", "submitting", "running"}
            for prior in history
        ):
            raise ValueError("[RESEARCH-008] selection requires settled, unsealed validation")
        return "selected", digest
    if row["selection"] != "selected" or digest != row["final_manifest_digest"]:
        raise ValueError("[RESEARCH-009] release requires the selected manifest, once")
    return "released", digest


def _final_identity(event: JSONObject, row: ResearchAttempt) -> str:
    """Bind evaluator-only selection/release to completed validation evidence.

    The journal must be inaccessible to proposers: this validates state and
    identity, not caller authority. Release is consumed before external work;
    a crash cannot turn it back into permission to inspect the final data again.

    Args:
        event: Evaluator-owned selection or release, never a proposal.
        row: Completed or selected validation attempt with its original record.

    Returns:
        Exact final-manifest digest to bind before any final execution.

    Raises:
        ValueError: RESEARCH-009 for extra fields or mismatched lineage.
    """
    if set(event) != {
        "action",
        "number",
        "observed_ms",
        "candidate",
        "validation_payload_digest",
        "final_manifest_digest",
    }:
        raise ValueError("[RESEARCH-009] require exact evaluator selection/release fields")
    candidate = require_digest(event, "candidate")
    payload = require_digest(event, "validation_payload_digest")
    final_digest = require_digest(event, "final_manifest_digest")
    record = decode_run_record(narrow_json_to_dict(load_json_str(row["record"])))
    if (candidate, payload) != (row["candidate"], record["payload_digest"]):
        raise ValueError("[RESEARCH-009] selected candidate or validation evidence changed")
    return final_digest


def research_journal(policy: ResearchPolicy) -> EventJournal[tuple[ResearchAttempt, ...]]:
    """Bind a validated immutable policy to transactional admission and replay.

    Args:
        policy: Evaluator-owned policy; never proposer-controlled.

    Returns:
        Shared journal configured for this study.
    """
    encoded = encode_research_policy(policy)
    frozen = decode_research_policy(encoded)
    return EventJournal(
        {"schema": 1, "mode": "research_validation", "policy": encoded},
        lambda: (),
        lambda history, event: _transition(frozen, history, event),
    )


def admit_proposal(
    path: Path, policy: ResearchPolicy, document: JSONObject, observed_ms: int
) -> ResearchAttempt:
    """Persist input before validation, then atomically reserve a candidate/seed.

    Args:
        path: Initialized journal.
        policy: Frozen evaluator policy.
        document: Untrusted proposal object.
        observed_ms: Actual observation time.

    Returns:
        Admitted or policy-refused attempt.

    Raises:
        ValueError: Malformed proposal; its raw request remains not admitted.
        TypeError: Wrong proposal field type; raw evidence remains.
        sqlite3.OperationalError: Contended or inaccessible journal.
    """
    journal = research_journal(policy)
    history = journal.transact(
        path,
        lambda _: {
            "action": "receive",
            "request": document,
            "observed_ms": observed_ms,
        },
    )
    number = history[-1]["number"]
    updated = journal.transact(
        path,
        lambda _: {
            "action": "admit",
            "number": number,
            "observed_ms": observed_ms,
        },
    )
    return updated[number - 1]


def research_policy_digest(policy: ResearchPolicy) -> str:
    """Hash all evaluator-owned rules for run manifests.

    Args:
        policy: Frozen policy.

    Returns:
        Canonical policy SHA-256.
    """
    return sha256(dump_json_str(encode_research_policy(policy)).encode()).hexdigest()


__all__ = [
    "ResearchAttempt",
    "admit_proposal",
    "decode_research_attempt",
    "encode_research_attempt",
    "research_journal",
    "research_policy_digest",
]
