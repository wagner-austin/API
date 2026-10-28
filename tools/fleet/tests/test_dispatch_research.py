"""Research rows remain readable by existing fleet consumers during rollout."""

from platform_core.json_utils import dump_json_str

from fleet.contracts.dispatch import (
    DispatchCommand,
    DispatchLane,
    decode_listing,
    encode_job_line,
)
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
