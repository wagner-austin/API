"""Measure every frozen panel path without fitting or changing failed outcomes.

ValueError and JSONTypeError are observed outcomes at this diagnostic boundary
only. A refused path contributes no return, label or score. Other types propagate.
Only aggregate counts and source flags are published, never row identifiers,
dates, prices or returns. The first refused path is retained as a row digest.
"""

import os
import sys
from bisect import bisect_left
from hashlib import sha256
from math import isfinite
from pathlib import Path

from platform_core.determinism_cpu import pin_single_thread

determinism = pin_single_thread()

from platform_core.comparability import encode_run_fingerprint
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, dump_json_str, require_float, require_str
from covenant_ml.benchmarking.harness import write_observed_manifest
from covenant_ml.markets.crsp_calendar import CrspWindow
from covenant_ml.markets.crsp_holdings import CrspOutcomes, crsp_window_path, index_crsp_outcomes
from covenant_ml.markets.crsp_study import crsp_known_panel
from covenant_ml.markets.encrypted_storage import decrypted_bundle
from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
from covenant_ml.markets.research_trial import load_trial_sources
from covenant_ml.markets.research_trial_records import decode_research_trial
from covenant_ml.markets.validation import read_document


def row_metadata(row: JSONObject | None) -> JSONObject:
    """Describe source flags and price validity without disclosing observations."""
    if row is None:
        return {"present": False}
    price = row["dlyprc"]
    finite_positive = False
    if price is not None:
        amount = require_float(row, "dlyprc")
        finite_positive = isfinite(amount) and amount > 0
    return {
        "present": True,
        "row_sha256": sha256(dump_json_str(row).encode()).hexdigest(),
        "price_present": price is not None,
        "finite_positive_price": finite_positive,
        "price_flag": require_str(row, "dlyprcflg"),
        "return_missing_flag": require_str(row, "dlyretmissflg"),
        "return_duration_flag": require_str(row, "dlyretdurflg"),
        "delisting_flag": require_str(row, "dlydelflg"),
    }


def failure_context(outcomes: CrspOutcomes, security: int, window: CrspWindow) -> JSONObject:
    """Read the exact boundary and adjacent flags from the verified source index."""
    position = bisect_left(outcomes.sessions, window.exit)
    assert outcomes.sessions[position] == window.exit
    neighbors: list[JSONValue] = []
    for offset in (-2, -1, 0, 1, 2):
        at = position + offset
        if 0 <= at < len(outcomes.sessions):
            neighbors.append({
                "session_offset": offset,
                "row": row_metadata(outcomes.returns.get((security, outcomes.sessions[at]))),
            })
    dates = [require_str(row, "delistingdt") for row in outcomes.delists.get(security, ())]
    return {
        "period": "training" if window.decision < outcomes.protocol["validation_start"] else "validation",
        "entry": row_metadata(outcomes.returns.get((security, window.entry))),
        "exit": row_metadata(outcomes.returns.get((security, window.exit))),
        "exit_neighbors": neighbors,
        "terminal_on_decision": sum(day == window.decision for day in dates),
        "terminal_inside_window": sum(window.entry <= day < window.exit for day in dates),
        "terminal_on_exit": sum(day == window.exit for day in dates),
        "terminal_after_exit": sum(day > window.exit for day in dates),
    }


def error_identity(error: ValueError | JSONTypeError) -> tuple[str, str]:
    """Classify the observed refusal and stack, excluding identifier-bearing text."""
    message = str(error)
    code = message[1:9] if message.startswith("[CRSP-") and message[9:10] == "]" else "OTHER_VALUE_ERROR"
    if isinstance(error, JSONTypeError):
        code = "JSON_TYPE_ERROR"
    names: list[str] = []
    trace = error.__traceback__
    while trace is not None:
        names.append(trace.tb_frame.f_code.co_name)
        trace = trace.tb_next
    if "_entry_filled" in names:
        boundary = "entry"
    elif "_require_fill" in names:
        boundary = "ordinary_exit"
    elif "crsp_terminal_value" in names:
        boundary = "terminal_value"
    else:
        boundary = "held_path"
    return code, boundary


request = read_document(Path(sys.argv[1]))
trial = decode_research_trial(read_document(Path(require_str(request, "trial"))))
keys = EvaluatorKeys(decode_research_access(read_document(Path(require_str(request, "access")))))
output = Path(sys.argv[2])
assert not output.exists() and not Path(str(output) + ".runrecord.json").exists()
counts: dict[str, int] = {}
examples: dict[str, JSONObject] = {}
first_failure: JSONObject | None = None
checked = passed = rejected = terminal = empty_decisions = 0
bundle_digest = trial["policy"]["dataset_digest"]
with decrypted_bundle(
    Path(trial["manifest"]), bundle_digest, partition="train-validation",
    identity=keys.acquire("train-validation"), scratch=Path(os.environ["TMPDIR"]),
    max_plaintext_bytes=keys.access["max_plaintext_bytes"],
) as manifest:
    private = manifest.parent
    sources = load_trial_sources(trial, manifest)
    print("Verified frozen sources loaded.", flush=True)
    panel = crsp_known_panel(sources)
    outcomes = index_crsp_outcomes(sources)
    print(f"Frozen panel has {len(panel)} decisions; auditing every eligible path.", flush=True)
    for decision_number, decision in enumerate(panel):
        assert decision.window.exit < sources.protocol["holdout_start"]
        if not decision.features:
            empty_decisions += 1
        for feature in decision.features:
            checked += 1
            try:
                path = crsp_window_path(outcomes, feature.permno, decision.window)
            except (ValueError, JSONTypeError) as error:
                code, boundary = error_identity(error)
                entry = outcomes.returns.get((feature.permno, decision.window.entry))
                exit_row = outcomes.returns.get((feature.permno, decision.window.exit))
                entry_flag = "ABSENT" if entry is None else require_str(entry, "dlyprcflg")
                exit_flag = "ABSENT" if exit_row is None else require_str(exit_row, "dlyprcflg")
                period = "training" if decision.window.decision < sources.protocol["validation_start"] else "validation"
                key = "/".join((code, boundary, period, entry_flag, exit_flag))
                counts[key] = counts.get(key, 0) + 1
                if key not in examples:
                    example = failure_context(outcomes, feature.permno, decision.window)
                    example.update({"code": code, "boundary": boundary, "path_ordinal": checked})
                    examples[key] = example
                    print(dump_json_str({"refusal_signature": key, "first_example": example}), flush=True)
                    if first_failure is None:
                        first_failure = example
            else:
                passed += 1
                rejected += not path.filled
                terminal += path.terminal is not None
        if (decision_number + 1) % 12 == 0:
            print(f"Audited {decision_number + 1} decisions and {checked} paths.", flush=True)
    assert checked == passed + sum(counts.values())
assert not private.exists()
fingerprint = market_fingerprint(determinism, config_test_hooks.get_env)
encoded_counts: JSONObject = dict(counts)
encoded_examples: JSONObject = dict(examples)
measurement: JSONObject = {
    "kind": "licensed-v3-execution-path-audit",
    "native_job_id": os.environ["SLURM_JOB_ID"],
    "source_commit": "087f67b4c4aae9a731c913330840dcc665b51701",
    "program_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    "bundle_sha256": bundle_digest,
    "decisions": len(panel), "eligible_paths": checked,
    "accepted_paths": passed, "confirmed_rejected_entries": rejected,
    "accepted_terminal_paths": terminal, "empty_decisions": empty_decisions,
    "refused_paths": sum(counts.values()), "refusal_counts": encoded_counts,
    "first_failure": first_failure, "first_examples_by_signature": encoded_examples,
    "all_paths_accounted": checked == passed + sum(counts.values()),
    "plaintext_removed": True,
    "scope": "First refusal per eligible path across the frozen train-validation panel. No fitting, scores, candidate selection or final access. Refused paths receive no imputed return or label. Public examples expose flags and row digests only, not identifiers, dates, prices or returns.",
}
document: JSONObject = {
    "results": [measurement], "diagnosis_of": "58749373",
    "fingerprint": encode_run_fingerprint(fingerprint), "dataset": {"sha256": bundle_digest},
}
write_observed_manifest(document, fingerprint, output,
                        sha256(dump_json_str(document).encode()).hexdigest(),
                        "licensed-v3-execution-path-audit", ())
print("Full frozen-panel path audit complete; private plaintext removed.", flush=True)
