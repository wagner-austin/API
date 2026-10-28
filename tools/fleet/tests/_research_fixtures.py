"""Shared research queue metadata, without credentials or payload data."""

from platform_core.json_utils import JSONObject, JSONValue

from fleet.contracts.research_worker import ResearchWorker
from tests._queue_fakes import QUEUE_CREDENTIALS, queue_job


def worker() -> ResearchWorker:
    """Return an explicit tenant-authorized controller placement.

    Returns:
        Placement used by polling and CLI tests.
    """
    return ResearchWorker(
        tenant=QUEUE_CREDENTIALS["tenant_id"],
        node="ziggurat",
        host="ziggurat",
        python="/private/runtime/bin/python",
        root="/private/requests",
        env_file="/private/env.ps1",
        script="/private/poll.sh",
    )


def research_job(**overrides: JSONValue) -> JSONObject:
    """Construct the canonical queue wire row with optional lifecycle changes.

    Args:
        **overrides: Fields varied for the case under test.

    Returns:
        A complete queue row.
    """
    row = queue_job(
        command="research",
        project="research/wrds",
        status="claimed",
        requestedNode="ziggurat",
        taskId="33333333-cccc-4ccc-8ccc-333333333333",
        sha=None,
        researchRequest={"backend": "wrds", "specSha256": "a" * 64, "attempt": "pilot"},
    )
    row.update(overrides)
    return row


__all__ = ["research_job", "worker"]
