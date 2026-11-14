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
absent, never zero. The test totals are the counts MCPs' maketools wrote
beside the kept reports (:func:`fleet.core.check_results.read_test_totals`,
MCPs board task 000f9302), never the line's ``N passed``, which a case's
own name can print: fleet job ea1dd2c0 stored 22 passed and 0 failed for a
run of 5024 passed and 105 failed. A check that kept no JUnit report sends
them as ``None``, unread, though the line may still render a printed
count. The failed cases are the names MCPs' maketools wrote
beside the kept reports, which only its ``check_reports`` parses, read off
the tail (:func:`fleet.core.check_results.read_failed_cases`, MCPs board
task d26709c0). A check that wrote no such file, one whose project's check
runs no maketools lock or one stopped before its runners returned, sends
them as ``None``, unread (MCPs migration 959), and the kept reports' path
and digest let a reader find and verify them.

The registry commit is the one the launch recorded beside the run
(:func:`fleet.core.run_registry.read_registry`, MCPs board task 1fd01f09),
which can be newer than the one the queue admitted the job under: MCPs
matches check reuse on it (migration 975, board task 2b1e55de), so a run
from a working tree or one launched before the record sends ``None`` and is
never reused.
"""

from __future__ import annotations

import datetime

from platform_core.json_utils import JSONObject, JSONValue

from fleet.core.check_figures import read_budget_reading, read_coverage_reading
from fleet.core.check_results import read_failed_cases, read_results_digest, read_test_totals
from fleet.core.run_registry import WORKING_TREE, read_registry
from fleet.core.run_summary import UNREAD, phase_seconds, read_run_summary
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
    transcript_path: str,
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
        transcript_path: Where the retire keeps the run's transcript
            (:func:`fleet.core.names.retained_log_path`), the path the
            verdict line names as ``log=``; MCPs stores it as the result's
            ``transcript_path`` (MCPs migration 960, board task d9872e99).

    Returns:
        The ``result`` argument of ``dispatch_report``'s close.

    Raises:
        ValueError: From :func:`fleet.core.run_summary.phase_seconds` on a
            summary line no build prints, and from
            :func:`fleet.core.check_results.read_failed_cases` and
            :func:`fleet.core.check_results.read_test_totals` on a failures
            or totals line that is not JSON.
        TypeError: From :func:`fleet.core.check_results.read_failed_cases`
            on a failures line that is JSON but not an array of strings, and
            from :func:`fleet.core.check_results.read_test_totals` on a
            totals line that is not an object of three counts.
    """
    totals = read_test_totals(tail)
    coverage = read_coverage_reading(tail)
    budget = read_budget_reading(tail)
    digest = read_results_digest(tail)
    failed_cases = read_failed_cases(tail)
    registry = read_registry(tail)
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
        "testsPassed": None if totals is None else totals["passed"],
        "testsFailed": None if totals is None else totals["failed"],
        "testsSkipped": None if totals is None else totals["skipped"],
        "beganAt": _instant(started_unix),
        "endedAt": _instant(ended_unix),
        "budget": budget_value,
        "artifact": None if digest is None else {"path": results_path, "digest": digest},
        "coverage": coverage_value,
        "phases": phases_value,
        "failedCases": None if failed_cases is None else list[JSONValue](failed_cases),
        "transcriptPath": transcript_path,
        "registryCommit": None if registry in (WORKING_TREE, UNREAD) else registry,
    }


__all__ = ["encode_check_result"]
