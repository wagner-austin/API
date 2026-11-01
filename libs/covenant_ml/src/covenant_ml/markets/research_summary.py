"""Descriptive moving-block uncertainty over the same seed means used for ranking.

The journal supplies completed validation records. No ciphertext, key or final
manifest is read. Each arm's paired difference is averaged across fixed seeds
before resampling consecutive months; seeds never become extra time observations.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from platform_core.run_record import Observation, encode_run_record

from ..benchmarking.harness import write_observed_manifest
from ..benchmarking.types import BenchmarkModelName
from ..evaluation.codec_records import encode_interval
from ..evaluation.paired import compare_moving_block
from .crsp_calendar import crsp_month
from .crsp_study import EQUAL_WEIGHT
from .research_execution import require_trial_attempt
from .research_journal import (
    completed_results,
    encode_research_attempt,
    research_journal,
    research_policy_digest,
)
from .research_ranking import CandidateMean, SeedResult, candidate_means
from .research_records import ResearchPolicy, encode_research_policy, encode_research_proposal
from .research_trial_records import (
    ResearchEvaluation,
    ResearchTrial,
    encode_research_evaluation,
    evaluation_digest,
)


def _result(
    item: CandidateMean, arm: str, policy: ResearchPolicy, evaluation: ResearchEvaluation
) -> tuple[JSONObject, tuple[Observation, ...]]:
    """Resample one candidate's exact averaged paired differences against zero."""
    design = evaluation["design"]
    values: NDArray[np.float64] = np.asarray(item.differences, dtype=np.float64)
    months: NDArray[np.int64] = np.asarray(
        [crsp_month(window.decision) for window in item.windows], dtype=np.int64
    )
    # Adding the baseline back and subtracting it again loses the exact paired
    # differences through rounding. The estimand is the increment against zero.
    comparison = compare_moving_block(
        f"{arm}_minus_{policy['selection']['reference']}",
        "zero",
        values,
        np.zeros(len(values), dtype=np.float64),
        months,
        months,
        block_length=design["bootstrap_block_months"],
        n_resamples=design["bootstrap_draws"],
        seed=design["bootstrap_seed"],
        confidence=evaluation["confidence"],
    )
    interval = comparison["difference"]
    inputs: list[JSONValue] = [
        {
            "attempt": result.number,
            "proposal": encode_research_proposal(result.proposal),
            "record_sha256": sha256(
                dump_json_str(encode_run_record(result.record)).encode()
            ).hexdigest(),
            "payload_sha256": result.record["payload_digest"],
        }
        for result in item.results
    ]
    windows: list[JSONValue] = [
        {"decision": window.decision, "entry": window.entry, "exit": window.exit}
        for window in item.windows
    ]
    result: JSONObject = {
        "candidate": item.candidate,
        "arm": arm,
        "reference": policy["selection"]["reference"],
        "cost_bps": policy["selection"]["cost_bps"],
        "inputs": inputs,
        "calendar": windows,
        "mean_paired_differences": list(item.differences),
        "n_months": len(item.windows),
        "n_seeds": len(item.results),
        "interval": encode_interval(interval),
    }
    prefix = f"{item.candidate}.{arm}_vs_{policy['selection']['reference']}"
    observations = tuple(
        Observation(name=f"{prefix}.{name}", value=value)
        for name, value in (
            ("estimate", interval["estimate"]),
            ("std_error", interval["std_error"]),
            ("ci_low", interval["ci_low"]),
            ("ci_high", interval["ci_high"]),
            ("p_value", interval["p_value"]),
        )
    )
    return result, observations


def validation_summary(
    policy: ResearchPolicy, evaluation: ResearchEvaluation, completed: Sequence[SeedResult]
) -> tuple[JSONObject, tuple[Observation, ...]]:
    """Summarize every complete candidate and arm using frozen evaluation settings.

    Args:
        policy: Immutable admission and selection policy.
        evaluation: Exact evaluation whose digest the policy binds.
        completed: Completed records from the existing journal.

    Returns:
        Provenance-bearing results and shared numerical observations.

    Raises:
        ValueError: Changed protocol, incomplete required seeds, misaligned or
            nonfinite evidence, or invalid bootstrap settings.
        OverflowError: Unrepresentable finite-input sum.
    """
    if evaluation_digest(evaluation) != policy["protocol_digest"]:
        raise ValueError("RESEARCH_SUMMARY_PROTOCOL: evaluation differs from frozen policy")
    results: list[JSONValue] = []
    observations: list[Observation] = []
    for arm in (
        BenchmarkModelName.CLEARGBM,
        BenchmarkModelName.LIGHTGBM,
        BenchmarkModelName.XGBOOST,
        EQUAL_WEIGHT,
    ):
        for item in candidate_means(policy, completed, arm):
            result, measured = _result(item, arm, policy, evaluation)
            results.append(result)
            observations.extend(measured)
    return {
        "kind": "research-validation-seed-summary-v1",
        "interpretation": (
            "Descriptive after-search validation uncertainty; no multiplicity adjustment."
        ),
        "selection_uses_interval": False,
        "final_scoring": "The selected candidate uses its fixed final seed, not this seed average.",
        "policy": encode_research_policy(policy),
        "evaluation": encode_research_evaluation(evaluation),
        "results": results,
    }, tuple(observations)


def write_validation_summary(
    ledger: Path, trial: ResearchTrial, output: Path, fingerprint: RunFingerprint
) -> Path:
    """Read a settled journal and publish the summary through the shared record owner.

    Args:
        ledger: Evaluator-owned validation journal.
        trial: Admitted request carrying the frozen study configuration.
        output: Payload path inside a new, exclusively created directory.
        fingerprint: Measured summarizing process, not an originating compute job.

    Returns:
        Shared RunRecord sidecar path.

    Raises:
        ValueError: Unsettled journal, altered trial or invalid evidence.
        FileExistsError: Output directory already exists.
        OSError: Journal or output cannot be read or written.
    """
    require_trial_attempt(ledger, trial)
    rows = research_journal(trial["policy"]).read(ledger)
    if any(row["status"] in {"admitted", "submitting", "running"} for row in rows):
        raise ValueError("RESEARCH_SUMMARY_UNSETTLED: every admitted attempt must settle")
    document, observations = validation_summary(
        trial["policy"], trial["evaluation"], completed_results(rows)
    )
    snapshot: list[JSONValue] = [encode_research_attempt(row) for row in rows]
    document["journal_snapshot_sha256"] = sha256(dump_json_str(snapshot).encode()).hexdigest()
    document["policy_digest"] = research_policy_digest(trial["policy"])
    document["reporter_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    document["attempts"] = [
        {"number": row["number"], "candidate": row["candidate"], "status": row["status"]}
        for row in rows
    ]
    # Bind provenance as well as results: the shared payload digest covers the
    # results list, while this experiment identity covers the entire document.
    experiment = sha256(dump_json_str(document).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=False)
    return write_observed_manifest(
        document, fingerprint, output, experiment, "validation-seed-summary", observations
    )


__all__ = ["validation_summary", "write_validation_summary"]
