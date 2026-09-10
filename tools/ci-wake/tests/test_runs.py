"""The GitHub boundary: what ``gh`` is asked, and what its answers mean.

THE PAYLOADS BELOW ARE THE REAL SHAPES, not a convenient reduction of them.
``conclusion`` is ``null`` while a run is in flight and a string afterwards,
and ``total_count`` is a sibling of ``jobs`` rather than its length -- both
facts decide behaviour this module owns, so both appear in the fixtures
exactly as the API sends them.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
)

from ci_wake import _test_hooks
from ci_wake.runs import (
    GH_TIMEOUT_SECONDS,
    WorkflowRun,
    decode_jobs,
    decode_runs,
    gh_json,
    is_terminal,
    jobs_argv,
    runs_argv,
)
from tests.conftest import REPO, SHA, FakeCompleted, FakeGh


def _run(
    *,
    run_id: int = 34397357156,
    name: str = "Check",
    status: str = "completed",
    conclusion: JSONValue = "success",
) -> JSONObject:
    """Build one ``workflow_runs`` element.

    Args:
        run_id: The run's numeric id.
        name: The workflow's display name.
        status: The run's status.
        conclusion: Its conclusion, or None while it is in flight.

    Returns:
        The element.
    """
    return {
        "id": run_id,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


def _listing(*runs: JSONObject) -> JSONObject:
    """Wrap runs in the shape ``actions/runs`` returns.

    Args:
        runs: The elements.

    Returns:
        The listing.
    """
    return {"total_count": len(runs), "workflow_runs": list(runs)}


def _job(name: str, conclusion: JSONValue) -> JSONObject:
    """Build one ``jobs`` element.

    Args:
        name: The job's display name.
        conclusion: Its conclusion, or None.

    Returns:
        The element.
    """
    return {"name": name, "conclusion": conclusion}


class TestArgv:
    def test_the_runs_query_is_by_head_sha(self) -> None:
        """By sha rather than by branch: the bridge asks about the exact
        commit somebody enrolled, so a busy branch cannot answer for it."""
        assert runs_argv(REPO, SHA) == (
            "gh",
            "api",
            f"repos/{REPO}/actions/runs?head_sha={SHA}&per_page=100",
        )

    def test_the_jobs_query_names_the_run(self) -> None:
        assert jobs_argv(REPO, 7) == ("gh", "api", f"repos/{REPO}/actions/runs/7/jobs?per_page=100")

    def test_both_are_pasteable(self) -> None:
        """The whole reason this package speaks ``gh`` rather than raw HTTPS
        is that its questions can be re-asked by hand. A vector carrying a
        shell-quoted or url-encoded fragment would not survive that."""
        for vector in (runs_argv(REPO, SHA), jobs_argv(REPO, 7)):
            assert vector[0] == "gh"
            assert all("'" not in token and '"' not in token for token in vector)


class TestGhJson:
    def test_a_clean_exit_parses_stdout(self) -> None:
        argv = runs_argv(REPO, SHA)
        _test_hooks.run_process = FakeGh(
            {argv: FakeCompleted(stdout=dump_json_str(_listing(_run())))}
        )

        assert decode_runs(gh_json(argv))[0]["workflow"] == "Check"

    def test_it_asks_for_captured_text_within_the_timeout(self) -> None:
        """The three arguments are the contract with :mod:`subprocess`, and
        a cycle that forgot ``capture_output`` would read an empty stdout and
        report every push as having no runs."""
        argv = runs_argv(REPO, SHA)
        recorded: list[tuple[bool, bool, int]] = []

        def _run_process(
            args: Sequence[str], *, capture_output: bool, text: bool, timeout: int
        ) -> FakeCompleted:
            recorded.append((capture_output, text, timeout))
            return FakeCompleted(stdout=dump_json_str(_listing()))

        _test_hooks.run_process = _run_process
        gh_json(argv)

        assert recorded == [(True, True, GH_TIMEOUT_SECONDS)]

    def test_a_non_zero_exit_refuses_and_carries_the_cli_s_own_words(self) -> None:
        """Not caught, not retried. ``gh`` reports absence, being logged out
        and an API refusal on stderr in words, and the operator's first step
        is the same for all three."""
        argv = runs_argv(REPO, SHA)
        _test_hooks.run_process = FakeGh(
            {argv: FakeCompleted(returncode=4, stderr="gh: not logged in\n")}
        )

        with pytest.raises(AppError) as caught:
            gh_json(argv)

        assert caught.value.code is CiWakeErrorCode.GH_COMMAND_FAILED
        assert "not logged in" in caught.value.message
        assert "exited 4" in caught.value.message

    def test_a_clean_exit_with_unparseable_output_is_a_different_fault(self) -> None:
        """Reporting it as GH_COMMAND_FAILED would send the reader to the
        authentication that was working."""
        argv = runs_argv(REPO, SHA)
        _test_hooks.run_process = FakeGh({argv: FakeCompleted(stdout="<html>rate limited</html>")})

        with pytest.raises(InvalidJsonError):
            gh_json(argv)


class TestDecodeRuns:
    def test_every_run_is_returned_in_github_s_order(self) -> None:
        runs = decode_runs(_listing(_run(run_id=1), _run(run_id=2, name="packages")))

        assert [run["run_id"] for run in runs] == [1, 2]
        assert runs[1]["workflow"] == "packages"

    def test_an_in_flight_run_s_null_conclusion_becomes_the_empty_string(self) -> None:
        """Narrowed at the edge so nothing downstream has to hold the
        difference between "no conclusion yet" and "a conclusion that is
        missing"."""
        runs = decode_runs(_listing(_run(status="in_progress", conclusion=None)))

        assert runs[0]["conclusion"] == ""

    def test_an_empty_listing_is_not_an_error(self) -> None:
        """A push whose runs have not been created yet is the ordinary state
        for the first minute of that push's life."""
        assert decode_runs(_listing()) == ()

    def test_a_payload_missing_a_field_this_package_reads_refuses(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_runs({"workflow_runs": [{"id": 1, "name": "Check"}]})

    def test_a_payload_that_is_not_a_listing_refuses(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_runs({"message": "Not Found"})


class TestDecodeJobs:
    def test_the_total_comes_from_total_count_not_the_array_length(self) -> None:
        """They differ past one page, and the count is what the eviction
        test depends on."""
        tally = decode_jobs({"total_count": 137, "jobs": [_job("check (db)", "success")]})

        assert tally["total"] == 137
        assert tally["listed"] == 1

    def test_a_run_with_no_jobs_tallies_to_zero(self) -> None:
        """This is the evicted-from-the-queue case, and the whole reason the
        jobs call is made at all."""
        tally = decode_jobs({"total_count": 0, "jobs": []})

        assert tally["total"] == 0
        assert tally["failed"] == ()

    def test_failures_are_named_in_the_order_the_payload_returned_them(self) -> None:
        tally = decode_jobs(
            {
                "total_count": 3,
                "jobs": [
                    _job("audit", "failure"),
                    _job("check (mcp-shared)", "success"),
                    _job("check (packages/db)", "timed_out"),
                ],
            }
        )

        assert tally["failed"] == ("audit", "check (packages/db)")

    def test_a_cancelled_job_lands_in_cancelled_not_in_failed(self) -> None:
        """THEY ARE DIFFERENT EVENTS AND LUMPING THEM IS THE DEFECT THIS SPLIT
        EXISTS TO REMOVE. A cancelled job did not fail -- it was stopped,
        usually by a superseding push -- and its package's changes are the
        ones that may go unchecked, which is a different thing to tell
        somebody than "your job failed".

        Shaped from run 34418498808 in wagner-austin/API, 2026-09-09: 42 jobs,
        3 cancelled by a superseding push, the rest completed.
        """
        tally = decode_jobs(
            {
                "total_count": 3,
                "jobs": [
                    _job("check (services/handwriting-ai)", "cancelled"),
                    _job("audit", "failure"),
                    _job("check (mcp-shared)", "success"),
                ],
            }
        )

        assert tally["cancelled"] == ("check (services/handwriting-ai)",)
        assert tally["failed"] == ("audit",)

    def test_skipped_is_not_a_failure(self) -> None:
        tally = decode_jobs({"total_count": 1, "jobs": [_job("check (search)", "skipped")]})

        assert tally["failed"] == ()

    def test_a_conclusion_this_package_has_never_seen_counts_as_failed(self) -> None:
        """The drift-safe direction for FAILURE reporting: a conclusion
        GitHub adds later is NAMED rather than silently omitted, so a post
        can never under-report how much is broken."""
        tally = decode_jobs({"total_count": 1, "jobs": [_job("check (new)", "quarantined")]})

        assert tally["failed"] == ("check (new)",)

    def test_a_null_conclusion_counts_as_failed_too(self) -> None:
        tally = decode_jobs({"total_count": 1, "jobs": [_job("check (hung)", None)]})

        assert tally["failed"] == ("check (hung)",)

    def test_a_payload_that_is_not_a_job_listing_refuses(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_jobs({"total_count": 1})


class TestIsTerminal:
    def test_only_completed_means_over(self) -> None:
        """Written as an equality rather than as a list of non-terminal
        statuses: a status GitHub adds later must read as NOT terminal, or
        this bridge announces a verdict for a run still executing."""
        for status in ("queued", "in_progress", "waiting", "requested", "pending"):
            assert is_terminal(_terminal_probe(status)) is False
        assert is_terminal(_terminal_probe("completed")) is True


def _terminal_probe(status: str) -> WorkflowRun:
    """Build a run carrying one status, for the terminality table.

    Args:
        status: The status under test.

    Returns:
        The run.
    """
    return WorkflowRun(
        run_id=1, workflow="Check", status=status, conclusion="", html_url="https://example.test"
    )
