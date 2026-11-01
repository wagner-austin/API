"""Publish the evaluator's re-measurement of one written validation seed summary.

``research_proposals summarize`` writes its payload and shared RunRecord under
the evaluator's private root, where no audit can read them, so a session that
restates the numbers on a wiki page leaves a copy nobody re-measures. This
reporter reads the written summary, its sidecar and the journal where they
are, recomputes the summary through the same ``validation_summary`` owner from
the journal's completed records, and refuses anything that does not match
before posting its reading under the task's HPC evidence identity. It writes,
stages and submits nothing, and never computes a second aggregation.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)
from platform_core.load_report import post_load_report, report_identity
from platform_core.run_record import (
    decode_run_record,
    encode_run_record,
    run_record,
    run_record_sidecar,
)

from . import _test_hooks, research_summary
from .research_execution import require_trial_attempt
from .research_journal import (
    completed_results,
    encode_research_attempt,
    research_journal,
    research_policy_digest,
)
from .research_records import decode_research_proposal
from .research_trial_records import decode_research_trial
from .validation import read_document

SUMMARY_LABEL = "validation-seed-summary"


def _digest(text: str) -> str:
    """Return the SHA-256 of one canonical JSON text."""
    return sha256(text.encode("utf-8")).hexdigest()


def _result_reading(item: JSONValue) -> JSONObject:
    """Reduce one verified summary result to its attempts, seeds and interval."""
    result = narrow_json_to_dict(item)
    inputs = [narrow_json_to_dict(entry) for entry in require_list(result, "inputs")]
    return {
        "candidate": result["candidate"],
        "arm": result["arm"],
        "reference": result["reference"],
        "n_seeds": result["n_seeds"],
        "n_months": result["n_months"],
        "attempts": [entry["attempt"] for entry in inputs],
        "seeds": [narrow_json_to_dict(entry["proposal"])["seed"] for entry in inputs],
        "interval": result["interval"],
    }


def report_summary(ledger: Path, trial_path: Path, output: Path, task: str) -> JSONObject:
    """Re-measure a written seed summary against its journal and publish the reading.

    Args:
        ledger: The evaluator's validation journal the summary consumed; only read.
        trial_path: The admitted trial document ``summarize`` was given.
        output: The summary payload ``summarize`` wrote; its RunRecord is the sidecar.
        task: Full board task id receiving the ``hpc-report-<task>`` reading.

    Returns:
        The published measurement.

    Raises:
        ValueError: WRDS_REPORT_TASK for a malformed task, before any file is
            read; RESEARCH_EXECUTION_REQUEST for a trial the journal never
            admitted; RESEARCH_SUMMARY_CODE when the summary was written by
            other summary code; _JOURNAL when the journal's rows changed since
            the summary was written; _POLICY, _RESULTS or _RECORD when the
            written payload or RunRecord differs from the recomputation; the
            ranking owner's RESEARCH_SELECTION errors for missing seeds.
        OSError: An input cannot be read.
        AppError: The board refused the reading.
    """
    report_identity(task, "hpc")
    trial = decode_research_trial(read_document(trial_path))
    policy = trial["policy"]
    require_trial_attempt(ledger, trial)
    journal_bytes = ledger.read_bytes()
    rows = research_journal(policy).read(ledger)
    payload_text = output.read_text(encoding="utf-8")
    document = narrow_json_to_dict(load_json_str(payload_text))
    record_text = run_record_sidecar(output).read_text(encoding="utf-8")
    record = decode_run_record(load_json_str(record_text))
    owner = sha256(Path(research_summary.__file__).read_bytes()).hexdigest()
    if require_str(document, "reporter_sha256") != owner:
        raise ValueError("RESEARCH_SUMMARY_CODE: summary was written by other summary code")
    snapshot = _digest(dump_json_str([encode_research_attempt(row) for row in rows]))
    if require_str(document, "journal_snapshot_sha256") != snapshot:
        raise ValueError("RESEARCH_SUMMARY_JOURNAL: journal rows changed since the summary")
    digest = research_policy_digest(policy)
    if require_str(document, "policy_digest") != digest:
        raise ValueError("RESEARCH_SUMMARY_POLICY: summary binds another policy")
    expected, observations = research_summary.validation_summary(
        policy, trial["evaluation"], completed_results(rows)
    )
    results = require_list(document, "results")
    if dump_json_str(results) != dump_json_str(expected["results"]):
        raise ValueError("RESEARCH_SUMMARY_RESULTS: written results differ from recomputation")
    # The shared record owner rebuilds what ``summarize`` should have written,
    # under the fingerprint it measured, so the comparison reorders nothing.
    rebuilt = run_record(
        experiment=_digest(dump_json_str(document)),
        label=SUMMARY_LABEL,
        fingerprint=record["fingerprint"],
        observations=observations,
        payload_digest=_digest(dump_json_str(results)),
    )
    if dump_json_str(encode_run_record(rebuilt)) != dump_json_str(encode_run_record(record)):
        raise ValueError("RESEARCH_SUMMARY_RECORD: RunRecord does not bind this summary")
    attempts: list[JSONValue] = [
        {
            "number": row["number"],
            "status": row["status"],
            "seed": decode_research_proposal(narrow_json_to_dict(load_json_str(row["request"])))[
                "seed"
            ],
            "candidate": row["candidate"],
            "job_id": row["job_id"],
            "record_sha256": _digest(row["record"]),
        }
        for row in rows
    ]
    report: JSONObject = {
        "kind": "RESEARCH_SUMMARY_EVIDENCE",
        "observed_at": _test_hooks.clock(),
        "reporter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary_code_sha256": owner,
        "numpy": np.__version__,
        "policy_digest": digest,
        "required_seeds": list(policy["seeds"]),
        "ledger": str(ledger.resolve()),
        "journal_sha256": sha256(journal_bytes).hexdigest(),
        "journal_sha256_after": sha256(ledger.read_bytes()).hexdigest(),
        "journal_snapshot_sha256": snapshot,
        "attempts": attempts,
        "output": str(output.resolve()),
        "payload_sha256": _digest(payload_text),
        "record_sha256": _digest(record_text),
        "experiment": record["experiment"],
        "payload_digest": record["payload_digest"],
        "interpretation": document["interpretation"],
        "final_scoring": document["final_scoring"],
        "recomputed_matches": True,
        "results": [_result_reading(item) for item in results],
    }
    post_load_report(
        post=_test_hooks.http_post,
        read_text=_test_hooks.stack_text,
        task_id=task,
        body="RESEARCH SUMMARY LOAD-REPORT\n" + dump_json_str(report),
        family="hpc",
    )
    return report


__all__ = ["SUMMARY_LABEL", "report_summary"]
