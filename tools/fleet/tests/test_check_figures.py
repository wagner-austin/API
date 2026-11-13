"""A check tail's typed figures, and the result its close carries (MCPs board task 601eef87).

The verdict line renders these readers' values (``test_export_verdict.py``,
``test_verdict_node_test.py`` and ``test_verdict_budget.py`` hold the
rendering); these cases hold the typed values the queue stores, one per
family of runner output and one per figure the tail leaves unread.
"""

from __future__ import annotations

import pytest

from fleet.core.check_figures import (
    NODE_TEST_SIGN,
    BudgetOutcome,
    Coverage,
    CoverageMeasure,
    read_budget_reading,
    read_coverage_reading,
    read_test_counts,
)
from fleet.core.check_result import encode_check_result
from fleet.core.check_results import RESULTS_DIGEST_MARKER
from fleet.core.run_summary import PhaseSeconds, RunSummary, phase_seconds


class TestTheCounts:
    def test_node_s_own_totals_come_first(self) -> None:
        tail = f"Tests  5 passed\n{NODE_TEST_SIGN} pass 58\n{NODE_TEST_SIGN} fail 2\n"
        assert read_test_counts(tail) == {"passed": 58, "failed": 2}

    def test_the_failed_count_is_read_off_the_passed_line_only(self) -> None:
        assert read_test_counts("attempt 1/3 failed\n      Tests  2312 passed (2312)\n") == {
            "passed": 2312,
            "failed": 0,
        }
        assert read_test_counts("Tests  2 failed | 885 passed (887)") == {
            "passed": 885,
            "failed": 2,
        }

    def test_the_last_total_on_the_last_line_carrying_one_is_the_suite_s(self) -> None:
        tail = (
            "Test Files  2 failed | 60 passed (62)\n"
            "case says 1 passed, 2 passed and 3 passed\n"
            "trailing log line"
        )
        assert read_test_counts(tail) == {"passed": 3, "failed": 0}
        assert read_test_counts("\n7 failed | 9 passed") == {"passed": 9, "failed": 7}

    def test_a_tail_with_no_passed_total_is_unread(self) -> None:
        assert read_test_counts(f"{NODE_TEST_SIGN} pass 60\n") is None


class TestTheCoverage:
    @pytest.mark.parametrize(
        ("tail", "rendered", "measures"),
        [
            (
                "All files |   99.71 |    98.42 |  100 |\n",
                "statements=99.71% branches=98.42%",
                [
                    Coverage(measure=CoverageMeasure.STATEMENTS, percent=99.71),
                    Coverage(measure=CoverageMeasure.BRANCHES, percent=98.42),
                ],
            ),
            (
                "TOTAL  5756  0  1526  0  97%\nTOTAL  5756  0  100%\n",
                "total=100%",
                [Coverage(measure=CoverageMeasure.TOTAL, percent=100.0)],
            ),
            (
                "COVERAGE TOTAL: lines 591/592 branches 0/0\n",
                "lines=591/592 branches=0/0",
                [Coverage(measure=CoverageMeasure.LINES, percent=99.83)],
            ),
            (
                f"{NODE_TEST_SIGN} all files | 97.30 | 88.89 | 100.00 |\n",
                "lines=97.30% branches=88.89%",
                [
                    Coverage(measure=CoverageMeasure.LINES, percent=97.3),
                    Coverage(measure=CoverageMeasure.BRANCHES, percent=88.89),
                ],
            ),
            (
                "ERROR: Coverage for branches (97.5%) does not meet global threshold (100%)\n",
                "branches=97.5% below threshold",
                [Coverage(measure=CoverageMeasure.BRANCHES, percent=97.5)],
            ),
            (
                "ERROR: Coverage for functions (90%) does not meet global threshold (100%)\n",
                "functions=90% below threshold",
                [],
            ),
        ],
    )
    def test_each_family_renders_and_types_from_one_match(
        self, tail: str, rendered: str, measures: list[Coverage]
    ) -> None:
        assert read_coverage_reading(tail) == {"rendered": rendered, "measures": measures}

    def test_a_gradle_branch_total_over_something_is_typed(self) -> None:
        reading = read_coverage_reading("COVERAGE TOTAL: lines 0/0 branches 378/378\n")
        assert reading == {
            "rendered": "lines=0/0 branches=378/378",
            "measures": [Coverage(measure=CoverageMeasure.BRANCHES, percent=100.0)],
        }

    def test_a_tail_with_no_coverage_is_unread(self) -> None:
        assert read_coverage_reading("Tests  1 passed\n") is None


class TestTheBudget:
    def test_over_is_read_before_within(self) -> None:
        tail = (
            "CHECK BUDGET: a took 10s of 300s\n"
            "CHECK OVER BUDGET: a took 1261s, over the 300s budget by 961s.\n"
        )
        assert read_budget_reading(tail) == {
            "outcome": BudgetOutcome.EXCEEDED,
            "seconds": 1261,
            "bound": 300,
        }

    def test_within(self) -> None:
        assert read_budget_reading("CHECK BUDGET: fleet-mcp took 30s of 300s (setup 1s).\n") == {
            "outcome": BudgetOutcome.WITHIN,
            "seconds": 30,
            "bound": 300,
        }

    def test_no_budget_line_is_unread(self) -> None:
        assert read_budget_reading("make: *** [Makefile:12: check] Error 2\n") is None


class TestThePhases:
    def test_a_phase_run_twice_is_one_entry_in_first_run_order(self) -> None:
        summary = RunSummary(generation="1", began="yes", phases="deps:7s,check:214s,deps:46s")
        assert phase_seconds(summary) == [
            PhaseSeconds(phase="deps", seconds=53),
            PhaseSeconds(phase="check", seconds=214),
        ]

    @pytest.mark.parametrize("phases", ["none", "unread"])
    def test_a_build_with_no_phases_or_no_summary_has_none(self, phases: str) -> None:
        assert phase_seconds(RunSummary(generation="1", began="no", phases=phases)) == []

    def test_an_entry_no_build_prints_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"phase entry 'Check:3s' is not <phase>:<seconds>s"):
            phase_seconds(RunSummary(generation="1", began="no", phases="install:1s,Check:3s"))


def test_the_result_carries_every_figure_the_tail_printed() -> None:
    tail = (
        "      Tests  2 failed | 2385 passed (2387)\n"
        "All files |     100 |      100 |     100 |     100 |\n"
        "CHECK OVER BUDGET: fleet-mcp took 320s, over the 300s budget by 20s.\n"
        "FLEET_RUN: generation=1793732844 began=no "
        "phases=install:1s,workspace-build:20s,check:30s\n"
        f"{RESULTS_DIGEST_MARKER}: {'e' * 64}\n"
    )

    assert encode_check_result(
        tail=tail,
        run_id="MCPs-fleet-mcp-lavender-1794578820",
        node="lavender",
        started_unix=1757000000,
        ended_unix=1757000060,
        results_path="/home/corvis/fleet/stage/logs/MCPs-fleet-mcp-lavender-1794578820.results",
    ) == {
        "runId": "MCPs-fleet-mcp-lavender-1794578820",
        "node": "lavender",
        "banner": False,
        "testsPassed": 2385,
        "testsFailed": 2,
        "testsSkipped": None,
        "beganAt": "2025-09-04T15:33:20+00:00",
        "endedAt": "2025-09-04T15:34:20+00:00",
        "budget": {"outcome": "exceeded", "seconds": 320, "bound": 300},
        "artifact": {
            "path": "/home/corvis/fleet/stage/logs/MCPs-fleet-mcp-lavender-1794578820.results",
            "digest": "e" * 64,
        },
        "coverage": [
            {"measure": "statements", "percent": 100.0},
            {"measure": "branches", "percent": 100.0},
        ],
        "phases": [
            {"phase": "install", "seconds": 1},
            {"phase": "workspace-build", "seconds": 20},
            {"phase": "check", "seconds": 30},
        ],
        "failedCases": None,
    }
