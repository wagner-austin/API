"""The queue's listings: what each asks ``dispatch_list`` for, and what it reads.

Every case pins the whole argument object the listing sends, so a listing
that stops naming ``full`` fails here: fleet-mcp's default is to become the
compact row (MCPs board task a8d3a736), which the runner's decoder cannot
read (board task beae2ca4).
"""

from __future__ import annotations

from platform_core.json_utils import dump_json_str

from fleet.core import _test_hooks, queue_listing
from tests._queue_fakes import DEFAULT_JOB_ID, QUEUE_CREDENTIALS, FakeQueue, queue_job


class TestHeldBy:
    def test_asks_the_queue_which_live_jobs_this_runner_holds(self) -> None:
        """Asked of the QUEUE rather than remembered locally: a runner that
        crashed between launching and writing a note to itself would strand a
        job on a node with nothing left that knows to collect it."""
        endpoint = FakeQueue(
            [dump_json_str({"jobs": [queue_job(status="running")], "pagination": {}})]
        )
        _test_hooks.http_post = endpoint

        jobs = queue_listing.held_by(QUEUE_CREDENTIALS, agent="fleet-runner-austinpc")

        assert len(jobs) == 1
        assert endpoint.tools == ["dispatch_list"]
        assert endpoint.arguments[0] == {
            "claimedBy": "fleet-runner-austinpc",
            "status": "live",
            "full": True,
        }

    def test_holding_nothing_is_an_empty_tuple(self) -> None:
        _test_hooks.http_post = FakeQueue([dump_json_str({"jobs": []})])

        assert queue_listing.held_by(QUEUE_CREDENTIALS, agent="fleet-runner-austinpc") == ()


class TestQueuedFor:
    def test_every_projects_waiting_jobs_are_one_page_of_queued_rows(self) -> None:
        endpoint = FakeQueue([dump_json_str({"jobs": [queue_job()]})])
        _test_hooks.http_post = endpoint

        jobs = queue_listing.queued_for(QUEUE_CREDENTIALS, project=None)

        assert [job["job_id"] for job in jobs] == [DEFAULT_JOB_ID]
        assert endpoint.tools == ["dispatch_list"]
        assert endpoint.arguments[0] == {
            "status": "queued",
            "limit": queue_listing.LISTING_PAGE_LIMIT,
            "full": True,
        }

    def test_one_projects_waiting_jobs_name_that_project(self) -> None:
        endpoint = FakeQueue([dump_json_str({"jobs": []})])
        _test_hooks.http_post = endpoint

        assert queue_listing.queued_for(QUEUE_CREDENTIALS, project="libs/demo") == ()
        assert endpoint.arguments[0] == {
            "status": "queued",
            "limit": queue_listing.LISTING_PAGE_LIMIT,
            "full": True,
            "project": "libs/demo",
        }


class TestPages:
    def test_a_cancelled_page_is_this_runners_cancelled_jobs_from_an_offset(self) -> None:
        endpoint = FakeQueue(
            [
                dump_json_str(
                    {"jobs": [queue_job(status="cancelled")], "pagination": {"nextOffset": 100}}
                )
            ]
        )
        _test_hooks.http_post = endpoint

        page = queue_listing.cancelled_page(
            QUEUE_CREDENTIALS, agent="fleet-node-lavender", offset=0
        )

        assert [job["status"] for job in page["jobs"]] == ["cancelled"]
        assert page["next_offset"] == 100
        assert endpoint.arguments[0] == {
            "claimedBy": "fleet-node-lavender",
            "status": "cancelled",
            "offset": 0,
            "limit": queue_listing.LISTING_PAGE_LIMIT,
            "full": True,
        }

    def test_a_submitted_page_is_one_submitters_jobs_for_one_project(self) -> None:
        endpoint = FakeQueue([dump_json_str({"jobs": [], "pagination": {"nextOffset": None}})])
        _test_hooks.http_post = endpoint

        page = queue_listing.submitted_page(
            QUEUE_CREDENTIALS, project="libs/demo", submitted_by="opus-dispatch-0905", offset=100
        )

        assert page["jobs"] == ()
        assert page["next_offset"] is None
        assert endpoint.arguments[0] == {
            "project": "libs/demo",
            "submittedBy": "opus-dispatch-0905",
            "offset": 100,
            "limit": queue_listing.LISTING_PAGE_LIMIT,
            "full": True,
        }
