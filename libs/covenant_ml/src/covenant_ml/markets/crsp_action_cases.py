"""Match captured actions to independently rederived refused holding windows.

This measures evidence coverage, never returns, consideration completeness or
historical publication availability. All source values remain in the encrypted
inputs; public metadata contains no security identifiers or absolute dates.
"""

from __future__ import annotations

from datetime import date
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_dict,
    require_int,
    require_str,
)

from ..evaluation.evidence import read_evidence
from .campaign_records import require_digest
from .crsp_action_records import decode_crsp_action_evidence
from .crsp_action_schema import CRSP_ACTIONS, CrspActionRelation
from .crsp_actions import CrspActionRows, check_crsp_action_rows
from .crsp_calendar import CrspWindow, crsp_windows
from .crsp_cohort import crsp_audit_cases, load_crsp_cohort_sources, resolve_crsp_anchors
from .crsp_cohort_records import CrspCohortPlan
from .crsp_extract import CrspSources, read_crsp_evidence
from .crsp_queries import crsp_scopes
from .crsp_rows import crsp_row_key
from .encrypted_storage import decrypted_bundle, read_encrypted_bundle
from .research_keys import EvaluatorKeys


def _detail(row: JSONObject, relation: CrspActionRelation, window: CrspWindow) -> JSONObject:
    """Describe presence, zero and date positions without copying free-form flag text."""
    origin = date.fromisoformat(window.entry)
    offsets: JSONObject = {
        field: (
            None
            if row[field] is None
            else (date.fromisoformat(require_str(row, field)) - origin).days
        )
        for field in relation.dates
    }
    present: JSONObject = {
        field: row[field] is not None for field in relation.columns if field not in relation.key
    }
    zeros: JSONObject = {
        field: None if row[field] is None else row[field] == 0
        for field in relation.columns
        if field in {"delret", "deldivamt", "disdivamt", "disfacpr", "disfacshr"}
    }
    return {
        "row_sha256": sha256(dump_json_str(row).encode()).hexdigest(),
        "day_offsets": offsets,
        "present": present,
        "zero": zeros,
        "any_date_after_exit": any(
            row[field] is not None and require_str(row, field) > window.exit
            for field in relation.dates
        ),
    }


def _events(checked: CrspActionRows, security: int, window: CrspWindow) -> JSONObject:
    """Keep all captured events for the security, including entry/exit boundaries.

    Event identity remains visible when the capture excluded its detail. Linked
    security fields are described only by presence; they never expand this join.
    """
    output: JSONObject = {}
    origin = date.fromisoformat(window.entry)
    for event_role, detail_role in ((0, 1), (2, 3)):
        events, details = CRSP_ACTIONS[event_role], CRSP_ACTIONS[detail_role]
        by_key = {crsp_row_key(row, details.role): row for row in checked.rows[details.role]}
        joined: list[JSONValue] = []
        by_event_key = {crsp_row_key(row, events.role): row for row in checked.rows[events.role]}
        for key in sorted(by_event_key):
            event = by_event_key[key]
            if require_int(event, "permno") != security:
                continue
            day = require_str(event, events.dates[0])
            detail = by_key.get(crsp_row_key(event, events.role))
            joined.append(
                {
                    "event_sha256": sha256(dump_json_str(event).encode()).hexdigest(),
                    "event_day_offset": (date.fromisoformat(day) - origin).days,
                    "before_entry": day < window.entry,
                    "on_entry": day == window.entry,
                    "inside_window": window.entry <= day < window.exit,
                    "on_exit": day == window.exit,
                    "after_exit": day > window.exit,
                    "detail": None if detail is None else _detail(detail, details, window),
                }
            )
        output[events.role] = joined
    return output


def crsp_action_cases(
    sources: CrspSources, cases: tuple[JSONObject, ...], checked: CrspActionRows
) -> tuple[JSONObject, ...]:
    """Join already validated evidence using the canonical calendar and anchor owner.

    Args:
        sources: Authenticated frozen diagnostic cohort.
        cases: Ordered audit cases validated by crsp_audit_cases.
        checked: Action rows independently checked against this cohort's scope.

    Returns:
        One metadata-only record per original case, in its original order.

    Raises:
        ValueError: CRSP_ACTION_WINDOW if an anchor is not a permitted monthly
            entry, its period differs, or the original exit offset disagrees.
        ValueError: Existing anchor or calendar validation errors propagate.
    """
    protocol = sources.protocol
    anchors = resolve_crsp_anchors(sources, cases)
    calendar = tuple(
        (require_str(row, "caldt"), require_str(row, "tradingflg")) for row in sources.calendar
    )
    windows = {
        window.entry: window
        for window in crsp_windows(
            calendar, protocol["train_start"], crsp_scopes(protocol)[0].last_decision
        )
        if window.exit < protocol["holdout_start"]
    }
    results: list[JSONObject] = []
    for case in cases:
        digest = require_digest(require_dict(case, "entry"), "row_sha256")
        anchor = anchors[digest]
        window = windows.get(require_str(anchor, "dlycaldt"))
        if window is None:
            raise ValueError("CRSP_ACTION_WINDOW: anchor is not a non-final monthly entry")
        period = "training" if window.decision < protocol["validation_start"] else "validation"
        exit_offset = (date.fromisoformat(window.exit) - date.fromisoformat(window.entry)).days
        if (
            require_str(case, "period") != period
            or require_int(require_dict(case, "source_window"), "exit_day_offset") != exit_offset
        ):
            raise ValueError("CRSP_ACTION_WINDOW: audit period or exit differs from the calendar")
        results.append(
            {
                "path_ordinal": require_int(case, "path_ordinal"),
                "period": period,
                "entry_row_sha256": digest,
                "exit_day_offset": exit_offset,
                "actions": _events(checked, require_int(anchor, "permno"), window),
            }
        )
    return tuple(results)


def inspect_crsp_actions(
    cohort_bundle: Path,
    cohort_sha256: str,
    action_bundle: Path,
    action_sha256: str,
    plan: CrspCohortPlan,
    keys: EvaluatorKeys,
    *,
    scratch: Path,
) -> JSONObject:
    """Authenticate both bundles, join every case, then remove all private plaintext.

    Args:
        cohort_bundle: Previously derived validation-only frozen observations.
        cohort_sha256: Independent expected cohort bundle digest.
        action_bundle: Existing encrypted corporate-action capture.
        action_sha256: Independent expected action bundle digest.
        plan: Independent original study, audit and protocol identities.
        keys: Evaluator custody; only the train-validation identity is acquired.
        scratch: Private scheduler storage managed by the existing native launcher.

    Returns:
        Input identities, row counts, case metadata and normal-unwind cleanup.
        No case is declared repaired and no return, label or fitted score is made.

    Raises:
        ValueError: CRSP_ACTION_LINEAGE for mismatched source provenance, or
            existing ciphertext, file, scope, budget, anchor and window refusals.
        OSError: Missing evidence or inaccessible private storage.
        Exception: Cryptographic errors propagate after scratch cleanup.
    """
    cohort_metadata = read_encrypted_bundle(cohort_bundle, cohort_sha256)
    action_metadata = read_encrypted_bundle(action_bundle, action_sha256)
    total = sum(
        item["plaintext"]["size_bytes"]
        for bundle in (cohort_metadata, action_metadata)
        for item in bundle["files"]
    )
    if total > keys.access["max_plaintext_bytes"]:
        raise ValueError("CRSP_ACTION_LIMIT: both bundles exceed the job's plaintext budget")
    with (
        decrypted_bundle(
            cohort_bundle,
            cohort_sha256,
            partition="train-validation",
            identity=keys.acquire("train-validation"),
            scratch=scratch,
            max_plaintext_bytes=min(plan["max_cohort_bytes"], keys.access["max_plaintext_bytes"]),
        ) as cohort_path,
        decrypted_bundle(
            action_bundle,
            action_sha256,
            partition="train-validation",
            identity=keys.acquire("train-validation"),
            scratch=scratch,
            max_plaintext_bytes=keys.access["max_plaintext_bytes"],
        ) as action_path,
    ):
        cohort, sources = load_crsp_cohort_sources(cohort_path, plan)
        record = decode_crsp_action_evidence(load_json_str(action_path.read_text("utf-8")))
        if record["cohort_bundle_sha256"] != cohort_sha256 or record["cohort"] != cohort:
            raise ValueError("CRSP_ACTION_LINEAGE: action capture belongs to another cohort")
        payloads = {
            item["name"]: read_crsp_evidence(action_path.parent, item) for item in record["files"]
        }
        expected = {item["name"] for item in record["files"]} | {action_path.name}
        if {item["plaintext"]["name"] for item in action_metadata["files"]} != expected:
            raise ValueError("CRSP_ACTION_LINEAGE: action bundle has undeclared evidence")
        complete_bytes = sum(map(len, payloads.values())) + action_path.stat().st_size
        if complete_bytes > record["limits"]["max_partition_bytes"]:
            raise ValueError("CRSP_ACTION_LIMIT: complete capture exceeds its byte budget")
        checked = check_crsp_action_rows(
            {item.role: payloads[item.role + ".csv"] for item in CRSP_ACTIONS},
            crsp_scopes(sources.protocol)[0],
            tuple(sorted({require_int(row, "permno") for row in sources.returns})),
            record["started_at"],
            record["limits"],
        )
        cases = crsp_audit_cases(
            read_evidence(cohort_path.parent, "audit.json", plan["audit_sha256"]), plan
        )
        results = crsp_action_cases(sources, cases, checked)
    return {
        "kind": "CRSP_ACTION_CASE_DIAGNOSTIC",
        "cohort_bundle_sha256": cohort_sha256,
        "action_bundle_sha256": action_sha256,
        "audit_sha256": plan["audit_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "rows": checked.counts,
        "cases": list(results),
        "plaintext_removed": not cohort_path.parent.exists() and not action_path.parent.exists(),
        "outcomes_qualified": False,
    }


__all__ = ["crsp_action_cases", "inspect_crsp_actions"]
