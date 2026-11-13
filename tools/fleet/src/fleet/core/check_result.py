"""The typed result a runner sends with a check's close (MCPs board task 601eef87).

MCPs stores a check's result beside its job (migration 957,
``recordDispatchResult``) when ``dispatch_report``'s close carries one, so
the review, the release cut and phase 6's breakage rules read typed fields
rather than parsing the FLEET-CHECK line back. The line stays the rendering
a person reads; both are read off the same tail by the same readers
(:mod:`fleet.core.check_figures`).

The result carries the run and node the start reported, which the queue
compares with the claim, so a restarted runner's report of an earlier run is
refused by its run id. A figure the tail does not print is ``None`` or
absent, never zero. The failed cases are the names MCPs' maketools wrote
beside the kept reports, which only its ``check_reports`` parses, read off
the tail (:func:`fleet.core.check_results.read_failed_cases`, MCPs board
task d26709c0). A check that wrote no such file, one whose project's check
runs no maketools lock or one stopped before its runners returned, sends
them as ``None``, unread (MCPs migration 959), and the kept reports' path
and digest let a reader find and verify them.
"""

from __future__ import annotations

import datetime

from platform_core.json_utils import JSONObject, JSONValue

from fleet.core.check_figures import read_budget_reading, read_coverage_reading, read_test_counts
from fleet.core.check_results import read_failed_cases, read_results_digest
from fleet.core.run_summary import phase_seconds, read_run_summary
from fleet.core.verdict import CHECK_BANNER


def _instant(unix: int) -> str:
    """An instant as the queue takes it.

    Args:
        unix: Whole seconds since the epoch.

    Returns:
        ISO-8601 in UTC with its zone.
    """
    return datetime.datetime.fromtimestamp(unix, tz=datetime.UTC).isoformat()


def encode_check_result(
    *,
    tail: str,
    run_id: str,
    node: str,
    started_unix: int,
    ended_unix: int,
    results_path: str,
) -> JSONObject:
    """Build a check's typed result from its tail and its run.

    Args:
        tail: What the tail read printed (:func:`fleet.core.run_registry.log_tail_script`).
        run_id: The run the start reported.
        node: The node the start reported.
        started_unix: When the dispatch began.
        ended_unix: When the check ended, as the verdict reads it.
        results_path: Where the retire keeps the run's reports
            (:func:`fleet.core.check_results.retained_results_path`).

    Returns:
        The ``result`` argument of ``dispatch_report``'s close.

    Raises:
        ValueError: From :func:`fleet.core.run_summary.phase_seconds` on a
            summary line no build prints, and from
            :func:`fleet.core.check_results.read_failed_cases` on a failures
            line that is not JSON.
        TypeError: From :func:`fleet.core.check_results.read_failed_cases`
            on a failures line that is JSON but not an array of strings.
    """
    counts = read_test_counts(tail)
    coverage = read_coverage_reading(tail)
    budget = read_budget_reading(tail)
    digest = read_results_digest(tail)
    failed_cases = read_failed_cases(tail)
    budget_value: JSONObject = (
        {"outcome": "unavailable"}
        if budget is None
        else {
            "outcome": budget["outcome"].value,
            "seconds": budget["seconds"],
            "bound": budget["bound"],
        }
    )
    coverage_value: list[JSONValue] = (
        []
        if coverage is None
        else [
            {"measure": entry["measure"].value, "percent": entry["percent"]}
            for entry in coverage["measures"]
        ]
    )
    phases_value: list[JSONValue] = [
        {"phase": entry["phase"], "seconds": entry["seconds"]}
        for entry in phase_seconds(read_run_summary(tail))
    ]
    return {
        "runId": run_id,
        "node": node,
        "banner": CHECK_BANNER in tail,
        "testsPassed": None if counts is None else counts["passed"],
        "testsFailed": None if counts is None else counts["failed"],
        "testsSkipped": None,
        "beganAt": _instant(started_unix),
        "endedAt": _instant(ended_unix),
        "budget": budget_value,
        "artifact": None if digest is None else {"path": results_path, "digest": digest},
        "coverage": coverage_value,
        "phases": phases_value,
        "failedCases": None if failed_cases is None else list[JSONValue](failed_cases),
    }


__all__ = ["encode_check_result"]
