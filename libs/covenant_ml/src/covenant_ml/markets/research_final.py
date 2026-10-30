"""Evaluator-owned final selection, its single release and the one final-period score.

The controller consumes the research journal's release before creating a private
grant bound to exact request and final-bundle bytes. The compute worker consumes
that grant once without a journal copy. Failures cannot restore either permission.

Proposers never reach this module. Their programs run in the isolated process
boundary (``fleet.core.research_proposal``), whose output only enters
``admit_proposal``; the journal, the policy and the final manifest stay with
the evaluator's account.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record

from ..benchmarking import make_baseline_trainers
from .crsp_final import run_crsp_final
from .encrypted_storage import decrypted_bundle, read_encrypted_bundle
from .research_execution import require_trial_attempt, trial_event
from .research_journal import ResearchAttempt
from .research_keys import EvaluatorKeys
from .research_release import FinalGrant, consume_final_grant, write_final_grant
from .research_trial import load_trial_sources, publish_monthly_record, require_trial_lineage
from .research_trial_records import ResearchTrial, encode_research_trial, evaluation_digest


def _final_identity(row: ResearchAttempt, final_digest: str) -> JSONObject:
    """Bind the journal's own candidate and validation evidence to exact final bytes.

    Args:
        row: The admitted attempt the trial names, as the journal holds it.
        final_digest: SHA-256 of the staged final-period manifest.

    Returns:
        The evaluator-owned selection or release fields for the journal.

    Raises:
        ValueError: RESEARCH-008 when the attempt has no completed validation record.
    """
    if row["status"] != "complete":
        raise ValueError("[RESEARCH-008] selection requires completed validation")
    record = decode_run_record(narrow_json_to_dict(load_json_str(row["record"])))
    return {
        "candidate": row["candidate"],
        "validation_payload_digest": record["payload_digest"],
        "final_manifest_digest": final_digest,
    }


def select_candidate(path: Path, trial: ResearchTrial, final_manifest: Path) -> ResearchAttempt:
    """Seal the study on one completed validation attempt and bind its final manifest.

    Args:
        path: Evaluator-owned admission journal.
        trial: The selected attempt's frozen request.
        final_manifest: Staged CRSP manifest covering the final period.

    Returns:
        The selected attempt; every later admission is refused as ``study_sealed``.

    Raises:
        ValueError: RESEARCH_EXECUTION_REQUEST when the request differs from
            admission; RESEARCH-008 for unfinished, unsettled or sealed validation.
        OSError: The journal or final manifest cannot be read.
    """
    row = require_trial_attempt(path, trial)
    digest = sha256(final_manifest.read_bytes()).hexdigest()
    return trial_event(path, trial, "select", _final_identity(row, digest))


def release_final(
    path: Path,
    trial: ResearchTrial,
    final_manifest: Path,
    directory: Path,
    *,
    keys: EvaluatorKeys,
) -> ResearchAttempt:
    """Commit the original journal's release before creating its private native grant.

    Args:
        path: Evaluator-owned admission journal.
        trial: The selected attempt's frozen request.
        final_manifest: The staged manifest bound at selection.
        directory: New evaluator-private grant directory for trusted delivery.
        keys: Evaluator key custody, consulted only after the one-shot release commits.

    Returns:
        Released journal row, containing no key material.

    Raises:
        ValueError: Altered admission, selection or final metadata before release;
            key or grant validation after release leaves it consumed.
        OSError: Unreadable journal or manifest before release; key or grant
            storage failure after release leaves it consumed.
    """
    row = require_trial_attempt(path, trial)
    if row["selection"] != "selected":
        raise ValueError("[RESEARCH-009] release requires the selected manifest, once")
    digest = row["final_manifest_digest"]
    if (
        sha256(final_manifest.read_bytes()).hexdigest() != digest
        or evaluation_digest(trial["evaluation"]) != trial["policy"]["protocol_digest"]
    ):
        raise ValueError("RESEARCH_TRIAL_LINEAGE: released protocol or final manifest changed")
    released = trial_event(path, trial, "release_final", _final_identity(row, digest))
    grant = FinalGrant(
        request_digest=sha256(dump_json_str(encode_research_trial(trial)).encode()).hexdigest(),
        manifest_digest=digest,
        recipient=keys.access["final_recipient"],
        max_plaintext_bytes=keys.access["max_plaintext_bytes"],
    )
    write_final_grant(directory, grant, keys.acquire("final"))
    return released


def execute_final(
    trial: ResearchTrial,
    final_manifest: Path,
    fingerprint: RunFingerprint,
    code_digest: str,
    output: Path,
    *,
    grant_directory: Path,
    scratch: Path,
) -> Path:
    """Consume a private released grant, then score once without accessing a journal.

    Args:
        trial: Exact request bound by the controller's private grant.
        final_manifest: Encrypted final manifest bound at selection.
        fingerprint: Measured native runtime.
        code_digest: Measured entrypoint source identity.
        output: Result path in a new directory.
        grant_directory: Private one-use grant, absent from proposer mounts.
        scratch: Scheduler-owned private job storage.

    Returns:
        Shared numerical record sidecar labelled final-<attempt>.

    Raises:
        ValueError: Grant, runtime, data or numerical validation fails.
        OSError: Missing or consumed grant, or inaccessible evidence.
    """
    grant, identity = consume_final_grant(grant_directory, trial, final_manifest)
    digest = grant["manifest_digest"]
    require_trial_lineage(trial, fingerprint, code_digest, final_manifest, digest)
    bundle = read_encrypted_bundle(final_manifest, digest)
    with decrypted_bundle(
        final_manifest,
        digest,
        partition="final",
        identity=identity,
        scratch=scratch,
        max_plaintext_bytes=grant["max_plaintext_bytes"],
    ) as manifest:
        sources = load_trial_sources(trial, manifest)
        output.parent.mkdir(parents=True, exist_ok=False)
        proposal, evaluation = trial["proposal"], trial["evaluation"]
        final = run_crsp_final(
            sources,
            make_baseline_trainers(proposal["config"]),
            proposal["seed"],
            features=proposal["features"],
            stopping_months=evaluation["stopping_months"],
            confidence=evaluation["confidence"],
        )
        return publish_monthly_record(
            trial,
            fingerprint,
            sources,
            output,
            partition="final",
            label=f"final-{trial['number']}",
            study=final,
            bundle=bundle,
            bundle_digest=digest,
        )


__all__ = ["execute_final", "release_final", "select_candidate"]
