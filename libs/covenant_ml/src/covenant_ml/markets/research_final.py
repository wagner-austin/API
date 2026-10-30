"""Evaluator-owned final selection, its single release and the one final-period score.

Final scoring is authorized by the research journal, never by a request field:
the candidate, seed and features come from the admitted request the journal
holds, and the run proceeds only when the journal consumes that attempt's one
``release_final`` against the final manifest bound at selection. The release is
committed before any final-period relation is read, so a crash or a refusal
afterwards cannot turn it back into permission to look at the data again.

Proposers never reach this module. Their programs run in the isolated process
boundary (``fleet.core.research_proposal``), whose output only enters
``admit_proposal``; the journal, the policy and the final manifest stay with
the evaluator's account.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint
from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record

from ..benchmarking import make_baseline_trainers
from .crsp_final import run_crsp_final
from .encrypted_storage import decrypted_bundle, read_encrypted_bundle
from .research_execution import require_trial_attempt, trial_event
from .research_journal import ResearchAttempt
from .research_keys import EvaluatorKeys
from .research_trial import load_trial_sources, publish_monthly_record, require_trial_lineage
from .research_trial_records import ResearchTrial


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


def execute_final(
    path: Path,
    trial: ResearchTrial,
    final_manifest: Path,
    fingerprint: RunFingerprint,
    code_digest: str,
    output: Path,
    *,
    keys: EvaluatorKeys,
    scratch: Path,
) -> Path:
    """Consume the selected attempt's one release, then score the final period once.

    Args:
        path: Evaluator-owned admission journal.
        trial: The selected attempt's frozen request.
        final_manifest: The staged manifest bound at selection.
        fingerprint: Runtime measured by the existing benchmark fingerprint builder.
        code_digest: Digest of the actual entrypoint source, measured by that entrypoint.
        output: Result path in a new directory for this run.
        keys: Evaluator key custody, consulted only after the one-shot release commits.
        scratch: Scheduler-owned job-local temporary storage.

    Returns:
        Shared numerical record sidecar, labelled ``final-<attempt>``.

    Raises:
        ValueError: RESEARCH_EXECUTION_REQUEST, RESEARCH-009 (not selected, or
            its release already consumed), RESEARCH_TRIAL_REFUSED or
            RESEARCH_TRIAL_LINEAGE (including final manifest bytes other than
            those bound at selection), all before release; CRSP and evaluation
            refusals after release, which stays consumed.
        FileExistsError: The output directory exists; the release stays consumed.
        OSError: Journal, final data or result storage is unavailable.
    """
    row = require_trial_attempt(path, trial)
    if row["selection"] != "selected":
        raise ValueError("[RESEARCH-009] release requires the selected manifest, once")
    digest = row["final_manifest_digest"]
    require_trial_lineage(trial, fingerprint, code_digest, final_manifest, digest)
    trial_event(path, trial, "release_final", _final_identity(row, digest))
    bundle = read_encrypted_bundle(final_manifest, digest)
    with decrypted_bundle(
        final_manifest,
        digest,
        partition="final",
        identity=keys.acquire("final"),
        scratch=scratch,
        max_plaintext_bytes=keys.access["max_plaintext_bytes"],
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


__all__ = ["execute_final", "select_candidate"]
