"""The elevated runner takes its project's job however it was tagged (MCPs board task 939ec5c7).

MCPs/scripts/ps-harness job 7c16305c was submitted on 2026-10-03 17:16Z with
``[windows]`` for a project the registry declares ``[windows, elevated]``. The
queue's exclusive rule hands the elevated runner only jobs requiring the tag,
and an ordinary runner's fits leave the project out, so it waited for hours
while serendipity had room in both lanes. These drive whole ticks of lavender,
declared elevated, against the real queue client and a fake endpoint: the
elevated runner asks the lane a second time, as the ordinary runner would, for
only the projects that declare ``elevated``, and the job it gets passes the
tag gate, because what a job needs is its project's declaration.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONValue, dump_json_str, narrow_json_to_str

from fleet.cli import node_agent, node_claim
from fleet.contracts.dispatch import DispatchJob, decode_job
from fleet.core import _test_hooks
from tests._node_agent_fixtures import _credentials_in_env, node_argv, sourced_document
from tests._queue_fakes import FakeQueue, queue_job
from tests.conftest import DEMO_PROJECT, PROBE_OK, FakeRun, ok
from tests.test_node_agent_elevated import ADMINISTRATOR_TOKEN

__all__ = ["_credentials_in_env"]

#: A project only an elevated runner may run, as the real registry has them.
ELEVATED_PROJECT = "MCPs/scripts/ps-harness"

#: The tags 7c16305c was submitted with: its project's, less ``elevated``.
SUBMITTED_TAGS: list[JSONValue] = ["windows"]

#: What the elevated runner's verdict adds for such a job.
UNTAGGED_NOTE = ", submitted without the elevated tag its project declares"


@pytest.fixture(name="untagged_config")
def _untagged_config(config_path: pathlib.Path) -> pathlib.Path:
    """The shared workspace with lavender elevated and an elevated project beside the demo.

    The elevated project declares no source, so a job of it that passes the
    tag gate is refused next by ``PROJECT_REMOTE_MISSING``, which is how a
    case reads that the gate let it through.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    document = sourced_document((("npm", "ci"),))
    nodes = document["nodes"]
    projects = document["projects"]
    assert isinstance(nodes, dict)
    assert isinstance(projects, dict)
    lavender = nodes["lavender"]
    assert isinstance(lavender, dict)
    lavender["elevated"] = True
    projects[ELEVATED_PROJECT] = {
        "worker_ram_gb": 0.5,
        "minimum_workers": 1,
        "expected_minutes": 10,
        "required_tags": ["windows", "elevated"],
        "source": None,
    }
    config_path.write_text(dump_json_str(document), encoding="utf-8")
    return config_path


def _tick(config_path: pathlib.Path, *, elevated: bool, replies: list[str]) -> FakeQueue:
    """Run one tick of lavender's runner whose node has room and an admin's token.

    Args:
        config_path: The workspace document.
        elevated: Whether this is the elevated runner.
        replies: What the queue answers, in order.

    Returns:
        The queue it spoke to.
    """
    _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK), ok(""), ok(ADMINISTRATOR_TOKEN)])
    endpoint = FakeQueue(replies)
    _test_hooks.http_post = endpoint
    argv = node_argv(config_path) + ([node_agent.ELEVATED_FLAG] if elevated else [])
    assert node_agent.main(argv) == 0
    return endpoint


def _verdict(endpoint: FakeQueue) -> str:
    """The verdict of the tick's one recorded tick report.

    Args:
        endpoint: The queue the tick spoke to.

    Returns:
        The report's verdict.
    """
    (tick,) = endpoint.ticks
    return narrow_json_to_str(tick["verdict"])


class TestTheElevatedRunnersSecondAsk:
    def test_it_takes_a_job_of_its_project_submitted_without_the_tag(
        self, untagged_config: pathlib.Path
    ) -> None:
        """7c16305c's shape: the lane's own ask matches nothing, the second
        takes it, and the job reaches the project's next refusal rather than
        ``PROJECT_TAGS_MISMATCH``."""
        job = queue_job(status="claimed", project=ELEVATED_PROJECT, requiredTags=SUBMITTED_TAGS)
        endpoint = _tick(
            untagged_config,
            elevated=True,
            replies=[
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": None}),
                dump_json_str({"claimed": job}),
                dump_json_str({"job": queue_job(status="refused")}),
            ],
        )

        assert endpoint.tools == [
            "dispatch_list",
            "dispatch_claim",
            "dispatch_claim",
            "dispatch_report",
        ]
        _, lane, untagged, closed = endpoint.arguments
        assert (lane["tags"], lane["projects"]) == (
            ["elevated", "windows"],
            [ELEVATED_PROJECT, DEMO_PROJECT],
        )
        assert (untagged["tags"], untagged["projects"]) == (SUBMITTED_TAGS, [ELEVATED_PROJECT])
        assert (untagged["lane"], untagged["node"]) == ("node", "lavender")
        assert untagged["agent"] == "fleet-node-lavender-elevated"
        assert narrow_json_to_str(closed["detail"]).startswith(
            f"PROJECT_REMOTE_MISSING: project '{ELEVATED_PROJECT}' declares no source"
        )
        verdict = _verdict(endpoint)
        assert verdict.startswith("claimed ")
        assert verdict.endswith(UNTAGGED_NOTE)

    def test_nothing_waiting_in_either_ask_claims_nothing(
        self, untagged_config: pathlib.Path
    ) -> None:
        endpoint = _tick(
            untagged_config,
            elevated=True,
            replies=[
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": None}),
                dump_json_str({"claimed": None}),
            ],
        )

        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_claim"]
        assert _verdict(endpoint) == (
            "asked for 2 fitting project(s); nothing in the node lane matched"
        )

    def test_a_job_its_lane_hands_it_is_not_asked_for_again(
        self, untagged_config: pathlib.Path
    ) -> None:
        job = queue_job(
            status="claimed", project=ELEVATED_PROJECT, requiredTags=["elevated", "windows"]
        )
        endpoint = _tick(
            untagged_config,
            elevated=True,
            replies=[
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": job}),
                dump_json_str({"job": queue_job(status="refused")}),
            ],
        )

        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report"]
        assert not _verdict(endpoint).endswith(UNTAGGED_NOTE)

    def test_the_ordinary_runner_asks_its_lane_once(self, untagged_config: pathlib.Path) -> None:
        """After asking whether to yield, which finds nothing waiting."""
        endpoint = _tick(
            untagged_config,
            elevated=False,
            replies=[
                dump_json_str({"jobs": []}),
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": None}),
            ],
        )

        assert endpoint.tools == ["dispatch_list", "dispatch_list", "dispatch_claim"]
        assert endpoint.arguments[2]["projects"] == [DEMO_PROJECT]


def _decoded(tags: list[JSONValue]) -> DispatchJob:
    """A claimed job of the elevated project with these tags, decoded as the runner reads it.

    Args:
        tags: The tags it was submitted with.

    Returns:
        The job.
    """
    row = queue_job(status="claimed", project=ELEVATED_PROJECT, requiredTags=tags)
    return decode_job(row, answer=dump_json_str(row))


class TestTheVerdictNote:
    def test_the_elevated_runner_names_the_missing_tag(self) -> None:
        job = _decoded(SUBMITTED_TAGS)

        assert node_claim.untagged_note(job, elevated=True) == UNTAGGED_NOTE

    def test_a_job_carrying_the_tag_adds_nothing(self) -> None:
        job = _decoded(["elevated", "windows"])

        assert node_claim.untagged_note(job, elevated=True) == ""

    def test_the_ordinary_runner_adds_nothing(self) -> None:
        job = _decoded(SUBMITTED_TAGS)

        assert node_claim.untagged_note(job, elevated=False) == ""
