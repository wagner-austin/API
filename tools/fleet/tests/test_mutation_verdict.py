"""The mutation step's outcome on the verdict line (MCPs board task 637f648f).

Every tail here is lines copied from a real check's transcript on its node,
colours stripped, each named by its fleet job and log; long paths inside a
stack trace are shortened and nothing else is changed. Each expected field
is worked by hand from those lines: the summary's two counts, the error's
code and the check budget line's span.
"""

from __future__ import annotations

from fleet.core import verdict
from fleet.core.mutation_verdict import read_mutation, read_outcome, read_span

#: Fleet job a5feffe9, packages/db at 626fa0d7 on pendragon
#: (MCPs-packages-db-pendragon-1794104941.log): every test passed at 100
#: percent coverage and the step found 8 of 453 mutants surviving.
FAILED_TAIL = (
    "stryker: 18:43:59 (713756) INFO MutationTestExecutor Done in 13 minutes and 19 seconds.\n"
    'MUTANT SURVIVED src/agent-tasks/definition-references.ts:65:7 ConditionalExpression -> "true"\n'
    'MUTANT SURVIVED src/types/agent-task-definition.ts:64:50 ArrowFunction -> "() => undefined"\n'
    "mutation: FAILED: 8 of 453 mutant(s) made in 18 src file(s) changed against origin/main "
    "(merge base 603a6306cac7020691024f8d486748d5c024ad79) survived. Kill each with a test, or "
    "rewrite the code so the mutation cannot change what it does.\n"
    "make[1]: *** [Makefile:46: _test-unlocked] Error 1\n"
    "fleet-phase test ended 2026-11-08T02:43:59Z after 868 s, exit 2\n"
    "CHECK BUDGET: packages/db took 68s of 300s (setup 0s, suite 868s, queued 0s (not counted); "
    "mutation testing 801s of its own 900s bound (not counted)).\n"
    "make: *** [Makefile:49: check] Error 2\n"
)

#: Fleet job 5c879052, packages/db at 1505fa77 on sedona
#: (MCPs-packages-db-sedona-1794105035.log): 275 mutants, none survived.
PASSED_TAIL = (
    "mutation: 275 mutant(s) in 40 src file(s); StrykerJS runs 56 of the package's 852 test "
    "file(s) (31 the change adds or changes, 25 beside a changed source)\n"
    "stryker: 18:40:40 (1158606) INFO MutationTestExecutor Done in 8 minutes and 27 seconds.\n"
    "mutation: passed: 0 of 275 mutant(s) made in 40 src file(s) changed against origin/main "
    "(merge base 5b9123a1e40f90f0f4d8b7505995137050414ad7) survived\n"
    "CHECK BUDGET: packages/db took 51s of 300s (setup 1s, suite 559s, queued 0s (not counted); "
    "mutation testing 509s of its own 900s bound (not counted)).\n"
)

#: tools/fleet's own check on colossus (tools-fleet-colossus-1794090910.log):
#: its commit added no src line, and API's step records a 0 s span.
NOT_APPLICABLE_TAIL = (
    "mutation: not applicable: 0 lines added to src files against origin/main (merge base "
    "c6b228f01419c5bc7e12f663621ac788fc7b8e6e), so 0 mutants were made\n"
    "mutation: 0s, recorded as the check's mutation testing\n"
    "CHECK BUDGET: tools/fleet took 112s of 300s (the whole make check, run by check-budget with "
    "no lock to queue on; mutation testing 0s of its own 900s bound (not counted)).\n"
)

#: Fleet job 0a1c6bce, packages/db at 4c9ff953 on sedona
#: (MCPs-packages-db-sedona-1794087831.log): StrykerJS's initial test run
#: timed out and the TypeScript step raised its named error.
STRYKER_FAILED_TAIL = (
    "stryker: 13:51:24 (3438049) ERROR DryRunExecutor Initial test run timed out!\n"
    "stryker: Error: Something went wrong in the initial test run\n"
    "\n"
    "MutationStepError: MUTATION_STRYKER_FAILED: stryker run exited 1; its log is the stryker: "
    "lines above\n"
    "    at runMutationStep (file:///.../mcp-shared/dist/src/mutation/step.js:373:15)\n"
    "fleet-phase lint ended 2026-11-07T21:45:02Z after 32 s, exit 0\n"
    "CHECK BUDGET: packages/db took 62s of 300s (setup 1s, suite 415s, queued 0s (not counted); "
    "mutation testing 355s of its own 900s bound (not counted)).\n"
)

#: packages/maketools on lacrymal (MCPs-packages-maketools-lacrymal-1794108046.log):
#: the Python step's unmutated baseline failed, and the step recorded no
#: span, so the budget line names none.
BASELINE_FAILED_TAIL = (
    '  File ".../mcp-shared-py/src/mcp_shared_py/mutation/trials.py", line 142, in run_baseline\n'
    "    raise MutationStepError(\n"
    "mcp_shared_py.mutation.context.MutationStepError: MUTATION_BASELINE_FAILED: "
    "tests/test_fleet_proof_routing.py::test_each_named_task_is_told_once_with_its_commits\n"
    "fleet-phase lint ended 2026-11-08T03:21:30Z after 31 s, exit 0\n"
    "CHECK BUDGET: packages/maketools took 97s of 300s (setup 7s, suite 90s, queued 0s (not "
    "counted)).\n"
)

#: Fleet job 217b306c, packages/db at 9124b39d on serendipity
#: (MCPs-packages-db-serendipity-1794106470.log): a test of the step failed
#: and vitest's diff printed the summary it expected; the suite failed before
#: any step ran.
TEST_DIFF_TAIL = (
    '+     "mutation: FAILED: 8 of 15 mutant(s) made in 3 src file(s) changed against origin/main '
    '(merge base 02b866a00d3470b1700300eff855d3d96341c64a) survived. Kill each with a test, or '
    'rewrite the code so the mutation cannot change what it does.",\n'
    "CHECK BUDGET: packages/db took 86s of 300s (setup 0s, suite 85s, queued 0s (not counted)).\n"
)


def test_a_step_that_found_survivors_reads_failed_with_its_count_and_span() -> None:
    assert read_mutation(FAILED_TAIL) == "failed:8/453,801s/900s"


def test_a_step_that_killed_every_mutant_reads_passed() -> None:
    assert read_mutation(PASSED_TAIL) == "passed:0/275,509s/900s"


def test_a_change_with_nothing_to_mutate_reads_not_applicable() -> None:
    assert read_mutation(NOT_APPLICABLE_TAIL) == "not-applicable,0s/900s"


def test_a_step_that_raised_reads_its_code_with_the_span_it_recorded() -> None:
    assert read_mutation(STRYKER_FAILED_TAIL) == "error:MUTATION_STRYKER_FAILED,355s/900s"


def test_a_python_step_s_error_is_read_through_its_module_path_and_a_missing_span_is_unread() -> None:
    assert read_mutation(BASELINE_FAILED_TAIL) == "error:MUTATION_BASELINE_FAILED,unread"


def test_a_summary_inside_a_test_s_diff_is_not_the_step_s_outcome() -> None:
    assert read_outcome(TEST_DIFF_TAIL) == "unread"
    assert read_mutation(TEST_DIFF_TAIL) == "unread"


def test_a_tail_with_neither_line_reads_unread() -> None:
    assert read_mutation("=== ALL CHECKS PASSED ===\n") == "unread"


def test_a_span_with_no_outcome_keeps_the_span() -> None:
    # A step stopped after it recorded its span and before its summary: the
    # budget line alone is left of PASSED_TAIL's step.
    budget = PASSED_TAIL.splitlines()[-1]
    assert read_span(budget) == "509s/900s"
    assert read_mutation(budget) == "unread,509s/900s"


def test_the_last_of_the_step_s_lines_is_the_outcome() -> None:
    assert read_outcome(FAILED_TAIL + PASSED_TAIL) == "passed:0/275"
    assert read_outcome(PASSED_TAIL + FAILED_TAIL) == "failed:8/453"
    assert read_outcome(PASSED_TAIL + STRYKER_FAILED_TAIL) == "error:MUTATION_STRYKER_FAILED"
    assert read_outcome(STRYKER_FAILED_TAIL + NOT_APPLICABLE_TAIL) == "not-applicable"
    assert read_span(FAILED_TAIL + PASSED_TAIL) == "509s/900s"


def test_the_verdict_line_carries_the_field_before_the_results_path() -> None:
    judged = verdict.judge(
        job_id="a5feffe9-9a34-4b0e-8f62-1c3f4b2b9f10",
        project="MCPs/packages/db",
        sha="626fa0d7" + "0" * 32,
        node="pendragon",
        exit_code=2,
        tail="\x1b[32m" + FAILED_TAIL + "\x1b[39m",
        ended_unix=1794104639,
        log_path="/home/corvis/fleet/stage/logs/MCPs-packages-db-pendragon-1794104941.log",
        results_path="/home/corvis/fleet/stage/logs/MCPs-packages-db-pendragon-1794104941.results",
        run_id="MCPs-packages-db-pendragon-1794104941",
        workers=3,
    )
    assert judged["mutation"] == "failed:8/453,801s/900s"
    assert verdict.render_verdict(judged).endswith(
        " artifact=unread mutation=failed:8/453,801s/900s "
        "results=pendragon:/home/corvis/fleet/stage/logs/MCPs-packages-db-pendragon-1794104941.results"
    )
