"""The check budget field of a fleet verdict (MCPs board task 1b152218).

A check over maketools' five-minute budget exits non-zero after every suite
passed, so without this field its row read ``exit=2 banner=no`` like any
other red one. The tails here are the real closing lines of fleet logs:
grandma-api's on loki (over), NavProbe's on loki (within) and MCPs
ps-harness's on serendipity (over, through check-lock).
"""

from __future__ import annotations

from fleet.core import verdict

SHA = "ffedae57a35a2bb2b45a1243107641148caca939"

OVER_TAIL = (
    "make[1]: Leaving directory 'C:/fleet/stage/services-grandma-api-loki-1791168143/"
    "services/grandma-api'\n"
    "\n"
    "========================================================================\n"
    "CHECK OVER BUDGET: services/grandma-api took 347s, over the 300s budget by 47s.\n"
    "  split : the whole make check, run by check-budget with no lock to queue on\n"
    "  fix   : make the suite lighter; nothing raises the budget.\n"
    "========================================================================\n"
    "make: *** [Makefile:27: check] Error 1\n"
)

WITHIN_TAIL = (
    "================= 13 failed, 700 passed in 144.48s (0:02:24) ==================\n"
    "make[1]: *** [Makefile:48: test] Error 1\n"
    "CHECK BUDGET: clients/NavProbe took 228s of 300s (the whole make check, run by "
    "check-budget with no lock to queue on).\n"
    "make: *** [Makefile:52: check] Error 1\n"
)

LOCKED_OVER_TAIL = (
    "Tests Passed: 979, Failed: 0, Skipped: 0\n"
    "CHECK OVER BUDGET: scripts/ps-harness took 1261s, over the 300s budget by 961s.\n"
    "make: *** [test] Error 3\n"
)


class TestReadingTheBudget:
    def test_an_over_budget_check_reads_over_with_its_time_and_budget(self) -> None:
        assert verdict.read_budget(OVER_TAIL) == "over:347s/300s"

    def test_a_check_within_budget_reads_within_even_when_its_suite_failed(self) -> None:
        assert verdict.read_budget(WITHIN_TAIL) == "within:228s/300s"

    def test_the_locked_form_of_the_rule_reads_the_same_way(self) -> None:
        assert verdict.read_budget(LOCKED_OVER_TAIL) == "over:1261s/300s"

    def test_a_tail_without_the_rules_line_is_unread(self) -> None:
        assert verdict.read_budget("make: *** [Makefile:12: check] Error 2\n") == "unread"


class TestTheBudgetOnTheLine:
    def test_an_over_budget_row_says_so_between_coverage_and_log(self) -> None:
        judged = verdict.judge(
            job_id="56979410-6d70-4aa9-9dc6-95873d0025cf",
            project="services/grandma-api",
            sha=SHA,
            node="loki",
            exit_code=2,
            tail=OVER_TAIL,
            log_path="C:/fleet/stage/logs/services-grandma-api-loki-1791168143.log",
            run_id="services-grandma-api-loki-1791168143",
        )

        assert judged["budget"] == "over:347s/300s"
        assert verdict.render_verdict(judged) == (
            f"FLEET-CHECK 56979410 services/grandma-api sha={SHA} node=loki exit=2 "
            "banner=no tests=unread coverage=unread budget=over:347s/300s "
            "log=loki:C:/fleet/stage/logs/services-grandma-api-loki-1791168143.log "
            "run=services-grandma-api-loki-1791168143"
        )
