"""Research rows remain readable by existing fleet consumers during rollout."""

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, dump_json_str

from fleet.contracts.dispatch import (
    DispatchCommand,
    DispatchLane,
    decode_listing,
    encode_job_line,
)
from fleet.contracts.research import decode_research_request, encode_research_request
from tests._queue_fakes import queue_job


def test_existing_listing_decodes_research_without_treating_it_as_a_build() -> None:
    """A new command must not poison a listing containing unrelated jobs."""
    rows = decode_listing(
        dump_json_str(
            {
                "jobs": [
                    queue_job(),
                    queue_job(
                        command="research",
                        project="research/wrds",
                        requestedNode="ziggurat",
                        sha=None,
                        researchRequest={
                            "backend": "wrds",
                            "specSha256": "a" * 64,
                            "attempt": "pilot-1",
                        },
                    ),
                ]
            }
        )
    )
    assert tuple(row["command"] for row in rows) == (
        DispatchCommand.CHECK,
        DispatchCommand.RESEARCH,
    )
    assert encode_job_line(rows[1]) == (
        "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa queued research request research/wrds @ziggurat"
    )
    assert DispatchLane.RESEARCH.value == "research"


@pytest.mark.parametrize("backend", ["hpc3", "wrds"])
def test_reference_roundtrip(backend: str) -> None:
    """The private reference has one canonical wire shape on both backends."""
    reference: JSONObject = {"backend": backend, "specSha256": "b" * 64, "attempt": "trial-2"}
    assert encode_research_request(decode_research_request(reference)) == reference


@pytest.mark.parametrize(
    "value, error",
    [
        (None, "Expected JSON object"),
        ({}, "RESEARCH_REFERENCE_FIELDS"),
        (
            {"backend": "wrds", "specSha256": "a" * 64, "attempt": "../bad"},
            "RESEARCH_REFERENCE_ATTEMPT",
        ),
        (
            {"backend": "wrds", "specSha256": "bad", "attempt": "trial"},
            "RESEARCH_REFERENCE_DIGEST",
        ),
        (
            {"backend": "other", "specSha256": "a" * 64, "attempt": "trial"},
            "Invalid backend",
        ),
    ],
)
def test_malformed_reference_is_refused(value: JSONValue, error: str) -> None:
    """No invalid reference can reach a private filesystem lookup."""
    with pytest.raises((ValueError, JSONTypeError), match=error):
        decode_research_request(value)
