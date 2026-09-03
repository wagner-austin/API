"""Tests for the check that compares a request against a measurement.

THE MEASURED CASE. ``turkic-lstm`` declared ``minutes: 720``; its members
finished in 1618 and 1585 seconds -- 27 minutes, 3.7% of the request. Five
more sat unschedulable for hours, because Slurm backfills a job into a hole
its own size and twelve-hour holes are rare on a busy free partition.

HOW IT PASSED EVERYTHING. It was never measured -- the project was created
before LSTM had ever run on the cluster. It was inherited from the README's
``abl`` example, which says 720 while the README's own ``turkic-lstm`` example
says 240. And the budget was fitted to it: cap 84.0 GPU-hours, 7 members x 12
hours = 84.0 exactly, so the number positioned to contradict the request had
been derived from it.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.closure import Closure, decode_closure, encode_closure
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.project import ProjectConfig, decode_project_config
from hpc3.core.rightsize import (
    HEADROOM,
    MINIMUM_OBSERVATIONS,
    describe,
    observed_runtimes,
    oversized_projects,
)
from tests.against_hpc3 import decode_ledger_entry
from tests.conftest import cluster, ledger_row, project_config

_AT = "2026-08-28T21:00:00+00:00"


def _closure(job_id: str, elapsed: int | None, state: str = "COMPLETED") -> Closure:
    """Build a closure carrying a runtime.

    Args:
        job_id: Job id.
        elapsed: Seconds the job ran, or None for a pre-field closure.
        state: Terminal state accounting reported.

    Returns:
        A validated closure.
    """
    return decode_closure(
        {
            "job_id": job_id,
            "state": state,
            "closed_at": _AT,
            "elapsed_seconds": elapsed,
        }
    )


def _entry(job_id: str, project: str = "turkic-lstm", partition: str = "free-gpu") -> LedgerEntry:
    """Build a ledger entry for a project.

    Args:
        job_id: Job id.
        project: Project the job belongs to.
        partition: Partition the job went to.

    Returns:
        A validated entry.
    """
    return decode_ledger_entry(
        ledger_row(
            job_id=job_id,
            name=f"{project}.bases-{job_id}",
            project=project,
            partition=partition,
        )
    )


def _project(minutes: int) -> ProjectConfig:
    """Build a project declaring a time limit.

    Args:
        minutes: The declared wall-clock request.

    Returns:
        A validated project config.
    """
    config: JSONValue = project_config(minutes=minutes)
    return decode_project_config(config, cluster(), config_dir=pathlib.Path.cwd())


class TestTheClosureCarriesTheRuntime:
    def test_it_round_trips(self) -> None:
        closure = _closure("55645374", 1618)
        assert decode_closure(encode_closure(closure)) == closure

    def test_a_pre_field_closure_states_its_absence(self) -> None:
        """272 closures on this machine were written before the field
        existed. null is what they honestly carry."""
        assert _closure("55539909", None)["elapsed_seconds"] is None

    def test_a_missing_key_is_refused_rather_than_defaulted(self) -> None:
        """A closure that never carried a runtime and one whose runtime was
        dropped by a bad write are different records."""
        with pytest.raises(JSONTypeError, match="required; write null if unknown"):
            decode_closure({"job_id": "1", "state": "COMPLETED", "closed_at": _AT})

    def test_a_negative_runtime_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be negative"):
            decode_closure(
                {"job_id": "1", "state": "COMPLETED", "closed_at": _AT, "elapsed_seconds": -1}
            )

    def test_a_mistyped_runtime_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be an integer or null"):
            decode_closure(
                {"job_id": "1", "state": "COMPLETED", "closed_at": _AT, "elapsed_seconds": "1618"}
            )

    def test_a_boolean_is_not_an_integer_here(self) -> None:
        """True is an int in Python and is not a duration."""
        with pytest.raises(JSONTypeError, match="must be an integer or null"):
            decode_closure(
                {"job_id": "1", "state": "COMPLETED", "closed_at": _AT, "elapsed_seconds": True}
            )


class TestCollectingHistory:
    def test_runtimes_are_grouped_by_project(self) -> None:
        entries = [_entry("1"), _entry("2", project="mi")]
        closures = {"1": _closure("1", 1618), "2": _closure("2", 60)}
        assert observed_runtimes(entries, closures) == {
            "turkic-lstm": [(1618, "1", "free-gpu")],
            "mi": [(60, "2", "free-gpu")],
        }

    def test_a_job_with_no_closure_is_skipped(self) -> None:
        assert observed_runtimes([_entry("1")], {}) == {}

    def test_a_closure_with_no_runtime_is_skipped_not_counted_as_zero(self) -> None:
        """Counting null as zero would make an unmeasured history look
        instantaneous and turn every project into a finding."""
        assert observed_runtimes([_entry("1")], {"1": _closure("1", None)}) == {}

    def test_a_cancelled_job_is_not_evidence(self) -> None:
        """Found on this check's first live run. `bases-uz-r2` was cancelled
        before it started and carries elapsed_seconds=0; counted, it drags the
        observed maximum toward zero and makes any request look oversized."""
        closures = {"1": _closure("1", 0, state="CANCELLED")}
        assert observed_runtimes([_entry("1")], closures) == {}

    def test_a_preempted_job_is_not_evidence_either(self) -> None:
        """It measures when the run was killed, not what it needed -- and on
        free-gpu that is most of them."""
        closures = {"1": _closure("1", 1189, state="PREEMPTED")}
        assert observed_runtimes([_entry("1")], closures) == {}


class TestTheFinding:
    def test_the_measured_over_request_is_found(self) -> None:
        entries = [_entry("55645374"), _entry("55645385")]
        closures = {"55645374": _closure("55645374", 1618), "55645385": _closure("55645385", 1585)}
        found = oversized_projects({"turkic-lstm": _project(720)}, entries, closures)
        assert [f["project"] for f in found] == ["turkic-lstm"]
        assert found[0]["longest_seconds"] == 1618
        assert found[0]["evidence"] == "55645374"

    def test_a_request_within_headroom_is_not_remarked_on(self) -> None:
        """Headroom is correct: a resume replays a partial epoch, a V100 is
        not an A100, a corpus grows."""
        entries = [_entry("1"), _entry("2")]
        closures = {"1": _closure("1", 1618), "2": _closure("2", 1585)}
        minutes = 1618 * HEADROOM // 60
        assert oversized_projects({"turkic-lstm": _project(minutes)}, entries, closures) == []

    def test_one_finished_run_is_an_anecdote(self) -> None:
        """The first run of a new experiment is when a generous limit is most
        defensible."""
        assert MINIMUM_OBSERVATIONS == 2
        found = oversized_projects(
            {"turkic-lstm": _project(720)}, [_entry("1")], {"1": _closure("1", 60)}
        )
        assert found == []

    def test_a_project_with_no_history_says_nothing(self) -> None:
        assert oversized_projects({"turkic-lstm": _project(720)}, [], {}) == []

    def test_a_run_on_another_partition_is_not_evidence(self) -> None:
        """The other thing this check's first live run got wrong.
        `turkic-lstm.image-v2` ran on `free` for 885s under build.sbatch's own
        two-hour limit, never under `minutes` -- and became the evidence for a
        claim about a resource line it had never used."""
        entries = [_entry("1", partition="free"), _entry("2", partition="free")]
        closures = {"1": _closure("1", 885), "2": _closure("2", 900)}
        assert oversized_projects({"turkic-lstm": _project(720)}, entries, closures) == []

    def test_only_the_matching_partitions_runs_count(self) -> None:
        entries = [
            _entry("build", partition="free"),
            _entry("1"),
            _entry("2"),
        ]
        closures = {
            "build": _closure("build", 88000),
            "1": _closure("1", 1618),
            "2": _closure("2", 1585),
        }
        found = oversized_projects({"turkic-lstm": _project(720)}, entries, closures)
        assert [f["evidence"] for f in found] == ["1"]
        assert found[0]["observations"] == 2

    def test_the_longest_run_is_the_one_compared_not_the_mean(self) -> None:
        """The limit has to hold the worst case, so one long run justifies a
        long limit however many short ones surround it.

        Chosen to discriminate: against a 720 min request, the longest run
        (11000s x 4 = 733 min) clears it and the mean (5530s x 4 = 369 min)
        would not, so a mean-based check would report this and be wrong.
        """
        entries = [_entry("1"), _entry("2")]
        closures = {"1": _closure("1", 60), "2": _closure("2", 11000)}
        assert oversized_projects({"turkic-lstm": _project(720)}, entries, closures) == []


class TestTheMessage:
    def _described(self) -> str:
        """Render the measured finding.

        Returns:
            The operator-facing line.
        """
        entries = [_entry("55645374"), _entry("55645385")]
        closures = {"55645374": _closure("55645374", 1618), "55645385": _closure("55645385", 1585)}
        found = oversized_projects({"turkic-lstm": _project(720)}, entries, closures)
        return describe(found[0])

    def test_it_states_both_numbers_and_the_ratio(self) -> None:
        line = self._described()
        assert "requests 720 min" in line
        assert "took 27 min" in line
        assert "3.7% of the request" in line

    def test_it_names_the_job_the_claim_rests_on(self) -> None:
        """So the claim can be checked against the cluster, not trusted."""
        assert "job 55645374" in self._described()

    def test_it_says_why_it_matters(self) -> None:
        assert "backfills" in self._described()

    def test_the_suggested_ceiling_comes_from_the_measurement(self) -> None:
        """Never from the request: scaling the request down produces a
        smaller number with the same defect, since the request is the thing
        under suspicion. 1618s x 4 = 107 min."""
        assert "107 min or less" in self._described()
