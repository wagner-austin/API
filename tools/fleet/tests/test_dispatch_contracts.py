"""Decoding the dispatch queue's answers, including every way they can be wrong.

JSON removes the PARSING failure class, not the CONTRACT one. A tool that
renamed a field would hand back perfectly well-formed JSON with the wrong keys
in it, so every field is checked and every check is exercised here -- a
decoder whose validation is only ever run on valid input is a decoder nobody
has tested.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.contracts.dispatch import (
    DISPATCH_COMMANDS,
    DISPATCH_STATUSES,
    DispatchJob,
    decode_claim,
    decode_listing,
    decode_reported,
    decode_submitted,
    encode_job_line,
)
from tests._queue_fakes import queue_job


def answer(body: JSONObject) -> str:
    """Render a tool answer.

    Args:
        body: The envelope object.

    Returns:
        Its JSON text.
    """
    return dump_json_str(body)


def claimed_job(answer: str) -> DispatchJob:
    """Decode a claim answer that must have produced a job.

    Raising rather than asserting ``is not None``: a bare null check is a
    weak assertion -- it passes for any object at all -- and the guard bans
    it for that reason. This narrows AND says what was expected.

    Args:
        answer: The tool's text.

    Returns:
        The job.

    Raises:
        AssertionError: When the answer carried no job.
    """
    job = decode_claim(answer)
    if job is None:
        raise AssertionError(f"expected a claimed job, got an empty queue: {answer}")
    return job


class TestDecodeClaim:
    def test_an_empty_queue_is_none_and_not_an_error(self) -> None:
        """The outcome of most polls. Making it an error would make the
        normal case indistinguishable from a fault in every runner log."""
        assert decode_claim(answer({"claimed": None})) is None

    def test_a_claimed_job_carries_every_field_the_runner_acts_on(self) -> None:
        job = claimed_job(
            answer(
                {
                    "claimed": queue_job(
                        status="claimed",
                        requestedNode="lavender",
                        claimedBy="fleet-runner-austinpc",
                    )
                }
            )
        )

        assert job["job_id"] == "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
        assert job["project"] == "libs/demo"
        assert job["command"] == "check"
        assert job["status"] == "claimed"
        assert job["requested_node"] == "lavender"
        assert job["node"] is None
        assert job["run_id"] == ""
        assert job["claimed_by"] == "fleet-runner-austinpc"
        assert job["submitted_by"] == "opus-dispatch-0905"
        assert job["session_id"] == "11111111-aaaa-4aaa-8aaa-111111111111"

    def test_the_submitter_is_carried_because_the_ledger_row_needs_it(self) -> None:
        """A dispatch whose provenance was the RUNNER's label would record
        only that the runner ran something, which is the one fact nobody
        needs. The queue job is the only place the asking session is named."""
        job = claimed_job(answer({"claimed": queue_job(submittedBy="opus-weight-injection-0902")}))

        assert job["submitted_by"] == "opus-weight-injection-0902"


class TestMalformedAnswers:
    def test_an_answer_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(AppError) as raised:
            decode_claim("[1, 2]")

        assert raised.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED
        assert "not an object" in raised.value.message

    def test_a_missing_envelope_member_is_refused_by_name(self) -> None:
        with pytest.raises(AppError) as raised:
            decode_claim(answer({"something-else": None}))

        assert "no 'claimed' member" in raised.value.message

    def test_a_job_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(AppError) as raised:
            decode_claim(answer({"claimed": "a string"}))

        assert "a job is str, not an object" in raised.value.message

    def test_a_missing_string_field_is_refused_by_name(self) -> None:
        row = queue_job()
        del row["project"]

        with pytest.raises(AppError) as raised:
            decode_claim(answer({"claimed": row}))

        assert "field 'project' is NoneType, not a string" in raised.value.message

    def test_a_missing_nullable_field_is_refused_rather_than_read_as_null(self) -> None:
        """The difference this checks is real: the tool renders every absent
        value as an explicit null precisely so a consumer can tell "no node
        yet" from "the field is gone". Treating them alike would let a
        renamed field read as an un-started job forever."""
        row = queue_job()
        del row["node"]

        with pytest.raises(AppError) as raised:
            decode_claim(answer({"claimed": row}))

        assert "field 'node' is missing" in raised.value.message

    def test_a_nullable_field_of_the_wrong_type_is_refused(self) -> None:
        with pytest.raises(AppError) as raised:
            decode_claim(answer({"claimed": queue_job(node=7)}))

        assert "not a string or null" in raised.value.message

    def test_a_status_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(AppError) as raised:
            decode_claim(answer({"claimed": queue_job(status="exploded")}))

        assert "status 'exploded' is not one of" in raised.value.message

    def test_a_command_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(AppError) as raised:
            decode_claim(answer({"claimed": queue_job(command="deploy")}))

        assert "command 'deploy' is not one of" in raised.value.message

    def test_the_refusal_names_the_repository_the_fix_belongs_in(self) -> None:
        """A malformed answer means the TOOL moved. A reader who has to work
        that out is a reader who starts in the wrong repository.

        Note the OTHER failure -- a tool that refuses the call -- never
        reaches this decoder at all: the MCP SDK answers a refusal with
        ``isError`` and the transport raises ``RPC_ERROR`` carrying the
        tool's own message. Only a SUCCESSFUL answer of the wrong shape gets
        here, which is what makes this code mean "the contract moved".
        """
        with pytest.raises(AppError) as raised:
            decode_claim('"a bare json string"')

        assert raised.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED
        assert "MCPs repo" in raised.value.message

    def test_every_declared_status_and_command_decodes(self) -> None:
        for status in DISPATCH_STATUSES:
            assert claimed_job(answer({"claimed": queue_job(status=status)}))["status"] == status
        for command in DISPATCH_COMMANDS:
            decoded = claimed_job(answer({"claimed": queue_job(command=command)}))
            assert decoded["command"] == command


class TestOtherEnvelopes:
    def test_a_report_answer_is_read_from_its_job_member(self) -> None:
        job = decode_reported(answer({"job": queue_job(status="running")}))

        assert job["status"] == "running"

    def test_a_submit_answer_is_read_from_its_submitted_member(self) -> None:
        job = decode_submitted(answer({"submitted": queue_job()}))

        assert job["status"] == "queued"

    def test_a_listing_decodes_every_row(self) -> None:
        jobs = decode_listing(
            answer(
                {
                    "jobs": [queue_job(), queue_job(id="bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb")],
                    "pagination": {"total": 2},
                }
            )
        )

        assert [job["job_id"] for job in jobs] == [
            "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
            "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb",
        ]

    def test_an_empty_listing_decodes_to_nothing(self) -> None:
        assert decode_listing(answer({"jobs": []})) == ()

    def test_a_listing_that_is_not_an_array_is_refused(self) -> None:
        with pytest.raises(AppError) as raised:
            decode_listing(answer({"jobs": {"not": "an array"}}))

        assert "'jobs' is dict, not an array" in raised.value.message


class TestRendering:
    def test_a_started_job_names_its_node_and_run(self) -> None:
        line = encode_job_line(
            decode_reported(
                answer(
                    {
                        "job": queue_job(
                            status="running", node="lavender", runId="libs-demo-1757000000"
                        )
                    }
                )
            )
        )

        assert line == (
            "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa running make check libs/demo "
            "@lavender run=libs-demo-1757000000"
        )

    def test_a_queued_job_names_the_node_it_asked_for(self) -> None:
        line = encode_job_line(claimed_job(answer({"claimed": queue_job(requestedNode="sedona")})))

        assert "@sedona" in line
        assert "run=" not in line

    def test_a_queued_job_that_named_no_node_says_so(self) -> None:
        assert "@any node" in encode_job_line(claimed_job(answer({"claimed": queue_job()})))
