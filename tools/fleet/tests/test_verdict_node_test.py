"""node's built-in runner's totals in a check's verdict (MCPs board task ed8e76a7).

coax-drone/site runs ``node --test --experimental-test-coverage``, which
closes with its own totals behind U+2139 rather than vitest's or pytest's
``N passed``, so every verdict of that project read ``tests=unread
coverage=unread`` over green runs. The tails here are the last lines of two
real runs of it on loki, byte for byte: fleet job 5c236243 at coax-drone
4484f0c, which passed, and f7eddcb9 at 35e46ce, which failed two cases and
printed no coverage table.
"""

from __future__ import annotations

from fleet.core import check_figures, verdict

INFO = check_figures.NODE_TEST_SIGN

#: The end of fleet job 5c236243's transcript, from its last case to the
#: banner (log loki:/home/corvis/fleet/stage/logs/coax-drone-site-loki-1794573532.log).
PASSED_TAIL = "".join(
    [
        "  ✔ refuses anything outside the wiki's markdown (0.170852ms)\n",
        "✔ wikiPath (0.496211ms)\n",
        f"{INFO} tests 60\n",
        f"{INFO} suites 20\n",
        f"{INFO} pass 60\n",
        f"{INFO} fail 0\n",
        f"{INFO} cancelled 0\n",
        f"{INFO} skipped 0\n",
        f"{INFO} todo 0\n",
        f"{INFO} duration_ms 205408.192485\n",
        f"{INFO} start of coverage report\n",
        f"{INFO} --------------------------------------------------------------\n",
        f"{INFO} file          | line % | branch % | funcs % | uncovered lines\n",
        f"{INFO} --------------------------------------------------------------\n",
        f"{INFO} js            |        |          |         | \n",
        f"{INFO}  data.mjs     | 100.00 |   100.00 |  100.00 | \n",
        f"{INFO}  markdown.mjs | 100.00 |   100.00 |  100.00 | \n",
        f"{INFO}  studies.mjs  | 100.00 |   100.00 |  100.00 | \n",
        f"{INFO}  views.mjs    | 100.00 |   100.00 |  100.00 | \n",
        f"{INFO} --------------------------------------------------------------\n",
        f"{INFO} all files     | 100.00 |   100.00 |  100.00 | \n",
        f"{INFO} --------------------------------------------------------------\n",
        f"{INFO} end of coverage report\n",
        "=== ALL CHECKS PASSED ===\n",
        "fleet-phase check ended 2026-11-13T12:42:24Z after 209 s, exit 0\n",
    ]
)

#: The totals of fleet job f7eddcb9 (log
#: loki:/home/corvis/fleet/stage/logs/coax-drone-site-loki-1794572975.log),
#: whose two failed cases left node no coverage table to print.
FAILED_TAIL = "".join(
    [
        f"{INFO} tests 60\n",
        f"{INFO} suites 20\n",
        f"{INFO} pass 58\n",
        f"{INFO} fail 2\n",
        f"{INFO} cancelled 0\n",
        f"{INFO} skipped 0\n",
        f"{INFO} todo 0\n",
        "✖ failing tests:\n",
        "make: *** [Makefile:20: check] Error 1\n",
    ]
)


class TestNodeTestTotals:
    def test_a_passing_run_reads_its_counts_and_its_tree_row(self) -> None:
        assert verdict.read_tests(PASSED_TAIL) == "60p/0f"
        assert verdict.read_coverage(PASSED_TAIL) == "lines=100.00% branches=100.00%"

    def test_a_failing_run_reads_its_failed_count_and_no_coverage(self) -> None:
        assert verdict.read_tests(FAILED_TAIL) == "58p/2f"
        assert verdict.read_coverage(FAILED_TAIL) == "unread"

    def test_its_totals_win_over_a_case_titled_with_a_passed_count(self) -> None:
        tail = f"  ✔ says 3 passed when three cases pass (0.2ms)\n{PASSED_TAIL}"

        assert verdict.read_tests(tail) == "60p/0f"

    def test_the_last_totals_are_the_runs_own(self) -> None:
        nested = f"{INFO} pass 4\n{INFO} fail 1\n{INFO} all files | 50.00 | 25.00 | 100.00 |\n"

        assert verdict.read_tests(nested + PASSED_TAIL) == "60p/0f"
        assert verdict.read_coverage(nested + PASSED_TAIL) == "lines=100.00% branches=100.00%"

    def test_lines_and_branches_are_read_from_their_own_columns(self) -> None:
        row = f"{INFO} all files     |  97.30 |    88.89 |  100.00 | \n"

        assert verdict.read_coverage(row) == "lines=97.30% branches=88.89%"

    def test_a_pass_line_without_its_fail_line_is_not_a_total(self) -> None:
        assert verdict.read_tests(f"{INFO} pass 60\n") == "unread"
        assert verdict.read_tests(f"{INFO} fail 0\n") == "unread"

    def test_the_sign_must_open_the_line(self) -> None:
        assert verdict.read_tests(f"log: {INFO} pass 60\nlog: {INFO} fail 0\n") == "unread"
        row = f"log: {INFO} all files | 100.00 | 100.00 | 100.00 |\n"

        assert verdict.read_coverage(row) == "unread"

    def test_the_verdict_line_carries_both(self) -> None:
        logs = "/home/corvis/fleet/stage/logs/coax-drone-site-loki-1794573532"
        line = verdict.render_verdict(
            verdict.judge(
                job_id="5c236243-19f5-4c1e-bdc3-0a24077feec1",
                project="coax-drone/site",
                sha="4484f0c1867bd853e9aac47fa2334b802a6e6e10",
                node="loki",
                exit_code=0,
                tail=PASSED_TAIL,
                ended_unix=1794573744,
                log_path=f"{logs}.log",
                results_path=f"{logs}.results",
                run_id="coax-drone-site-loki-1794573532",
                workers=1,
            )
        )

        assert line.startswith(
            "FLEET-CHECK 5c236243 coax-drone/site sha=4484f0c1867bd853e9aac47fa2334b802a6e6e10 "
            "node=loki exit=0 banner=yes tests=60p/0f coverage=lines=100.00% branches=100.00% "
            "budget=unread ended=2026-11-13T12:42:24Z "
            "log=loki:/home/corvis/fleet/stage/logs/coax-drone-site-loki-1794573532.log "
            "run=coax-drone-site-loki-1794573532 workers=1 "
        )
