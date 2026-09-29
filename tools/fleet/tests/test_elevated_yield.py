"""The ordinary runner steps aside while the elevated one has work (MCPs a98d7083).

serendipity's two runners share one slot, and the ordinary runner claimed
five jobs back to back on 2026-09-29 while MCPs/execution-elevated 58feb9d1
stayed queued, because it frees and refills the slot inside one tick. These
drive :mod:`fleet.core.elevated_yield` against the real queue client and a
fake endpoint speaking the tool's wire shape, and one whole ordinary tick
through :mod:`fleet.cli.node_agent`.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import node_agent
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core import _test_hooks, elevated_yield
from tests._node_agent_fixtures import _credentials_in_env, node_argv, sourced_document
from tests._queue_fakes import QUEUE_CREDENTIALS, FakeQueue, queue_job
from tests._toolchain_fixtures import LAVENDER_2026_09_23
from tests.conftest import PROBE_OK, FakeRun, ok, workspace_document

__all__ = ["_credentials_in_env"]

#: The project that requires the tag, as the real registry names it.
ELEVATED_PROJECT = "MCPs/execution-elevated"


def _with_elevated(document: JSONObject, *, node_elevated: bool) -> JSONObject:
    """Give a workspace document an elevated project, and lavender its flag.

    Args:
        document: A workspace document from the shared fixtures.
        node_elevated: What lavender declares.

    Returns:
        The same document, changed in place.
    """
    projects = document["projects"]
    nodes = document["nodes"]
    assert isinstance(projects, dict)
    assert isinstance(nodes, dict)
    lavender = nodes["lavender"]
    assert isinstance(lavender, dict)
    lavender["elevated"] = node_elevated
    projects[ELEVATED_PROJECT] = {
        "worker_ram_gb": 0.5,
        "minimum_workers": 1,
        "expected_minutes": 10,
        "required_tags": ["windows", "elevated"],
        "source": None,
    }
    return document


def _workspace(*, node_elevated: bool) -> FleetWorkspace:
    """The shared one-node workspace with an elevated project.

    Args:
        node_elevated: What lavender declares.

    Returns:
        The decoded workspace.
    """
    return decode_fleet_workspace(_with_elevated(workspace_document(), node_elevated=node_elevated))


def _queued(*jobs: JSONObject) -> str:
    """A ``dispatch_list`` answer holding these jobs.

    Args:
        *jobs: The wire rows.

    Returns:
        The tool's text.
    """
    return dump_json_str({"jobs": list(jobs)})


def _ask(
    workspace: FleetWorkspace, *, elevated: bool, replies: list[str]
) -> tuple[bool, FakeQueue]:
    """Ask whether lavender's runner yields, against a scripted queue.

    Args:
        workspace: The decoded workspace.
        elevated: Whether the asking runner is the elevated one.
        replies: What the queue answers, in order.

    Returns:
        The answer, and the endpoint that recorded the calls.
    """
    endpoint = FakeQueue(replies)
    _test_hooks.http_post = endpoint
    answer = elevated_yield.yields_to_elevated(
        QUEUE_CREDENTIALS,
        workspace,
        alias="lavender",
        node=workspace["nodes"]["lavender"],
        elevated=elevated,
    )
    return answer, endpoint


class TestTheElevatedProjects:
    def test_only_projects_requiring_the_tag_are_asked_about(self) -> None:
        assert elevated_yield.elevated_projects(_workspace(node_elevated=True)) == (
            ELEVATED_PROJECT,
        )

    def test_a_registry_without_one_names_none(self) -> None:
        assert elevated_yield.elevated_projects(decode_fleet_workspace(workspace_document())) == ()


class TestWhoYields:
    def test_the_ordinary_runner_yields_to_a_job_naming_no_node(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        job = queue_job(project=ELEVATED_PROJECT, requiredTags=["elevated", "windows"])
        with caplog.at_level("INFO"):
            answer, endpoint = _ask(
                _workspace(node_elevated=True), elevated=False, replies=[_queued(job)]
            )

        assert answer is True
        assert endpoint.tools == ["dispatch_list"]
        assert endpoint.arguments == [
            {"project": ELEVATED_PROJECT, "status": "queued", "limit": 100}
        ]
        assert any(
            record.getMessage().startswith(
                "lavender claims nothing so its elevated runner takes the next slot: "
            )
            for record in caplog.records
        )

    def test_it_yields_to_a_job_pinned_to_this_node(self) -> None:
        job = queue_job(project=ELEVATED_PROJECT, requestedNode="lavender")
        answer, _ = _ask(_workspace(node_elevated=True), elevated=False, replies=[_queued(job)])

        assert answer is True

    def test_a_job_pinned_to_another_node_is_that_nodes_business(self) -> None:
        job = queue_job(project=ELEVATED_PROJECT, requestedNode="serendipity")
        answer, endpoint = _ask(
            _workspace(node_elevated=True), elevated=False, replies=[_queued(job)]
        )

        assert (answer, endpoint.tools) == (False, ["dispatch_list"])

    def test_an_empty_queue_leaves_the_ordinary_runner_to_claim(self) -> None:
        answer, endpoint = _ask(_workspace(node_elevated=True), elevated=False, replies=[_queued()])

        assert (answer, endpoint.tools) == (False, ["dispatch_list"])

    @pytest.mark.parametrize(
        ("node_elevated", "elevated"),
        [(True, True), (False, False)],
        ids=["the elevated runner itself", "a node with no elevated runner"],
    )
    def test_the_rest_never_ask_the_queue(self, node_elevated: bool, elevated: bool) -> None:
        workspace = _workspace(node_elevated=node_elevated)
        answer, endpoint = _ask(workspace, elevated=elevated, replies=[])

        assert (answer, endpoint.tools) == (False, [])


def test_a_whole_ordinary_tick_claims_nothing_while_an_elevated_job_waits(
    config_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The measured case: lavender has room and a ready toolchain, and still
    takes nothing, so its elevated runner gets the slot."""
    config_path.write_text(
        dump_json_str(_with_elevated(sourced_document((("npm", "ci"),)), node_elevated=True)),
        encoding="utf-8",
    )
    _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK), ok(""), ok(LAVENDER_2026_09_23)])
    endpoint = FakeQueue([_queued(), _queued(queue_job(project=ELEVATED_PROJECT))])
    _test_hooks.http_post = endpoint

    with caplog.at_level("INFO"):
        assert node_agent.main(node_argv(config_path)) == 0

    assert endpoint.tools == ["dispatch_list", "dispatch_list"]
    assert endpoint.arguments[1] == {"project": ELEVATED_PROJECT, "status": "queued", "limit": 100}
