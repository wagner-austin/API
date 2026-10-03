"""A node's elevated runner: its identity, its tags and its token gate (MCPs board task a98d7083).

MCPs' Task Scheduler installers register what only an administrator may, and
every build launched as an S4U task at RunLevel Limited, so no dispatched job
could run them. A node declaring ``elevated`` gets a second runner. These drive
whole ticks of it against lavender declared elevated: it claims only with the
``elevated`` tag, only once its probe has read an administrator's token, and
under an identity of its own, while the node's ordinary runner claims without
the tag.
"""

from __future__ import annotations

import pathlib
import sys
import uuid

import pytest
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import node_agent
from fleet.core import _test_hooks, queue
from tests._node_agent_fixtures import (
    _credentials_in_env,
    node_argv,
    sourced_document,
)
from tests._queue_fakes import FakeQueue
from tests._toolchain_fixtures import LAVENDER_2026_09_23
from tests.conftest import DEMO_PROJECT, PROBE_OK, FakeRun, ok

__all__ = ["_credentials_in_env"]

#: lavender's ready answer with its ssh session's token read as an
#: administrator's, the line the Windows probe now ends with.
ADMINISTRATOR_TOKEN = LAVENDER_2026_09_23 + "integrity=yes=administrator\n"

#: The same with a filtered token, an account outside Administrators.
FILTERED_TOKEN = LAVENDER_2026_09_23 + "integrity=no=limited\n"


@pytest.fixture(name="elevated_config")
def _elevated_config(config_path: pathlib.Path) -> pathlib.Path:
    """The shared workspace with a source, and lavender declaring an elevated runner.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    document = sourced_document((("npm", "ci"),))
    nodes = document["nodes"]
    assert isinstance(nodes, dict)
    lavender = nodes["lavender"]
    assert isinstance(lavender, dict)
    lavender["elevated"] = True
    config_path.write_text(dump_json_str(document), encoding="utf-8")
    return config_path


def _tick(
    config_path: pathlib.Path, toolchain: str, *, elevated: bool, caplog: pytest.LogCaptureFixture
) -> tuple[FakeQueue, list[str]]:
    """Run one tick of lavender's runner whose node has room and answers so.

    Args:
        config_path: The workspace document.
        toolchain: What the toolchain probe prints.
        elevated: Whether this is the elevated runner.
        caplog: The test's log capture.

    Returns:
        The queue it spoke to and every message it logged.
    """
    _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK), ok(""), ok(toolchain)])
    endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
    _test_hooks.http_post = endpoint
    argv = node_argv(config_path) + ([node_agent.ELEVATED_FLAG] if elevated else [])
    with caplog.at_level("INFO"):
        assert node_agent.main(argv) == 0
    return endpoint, [record.getMessage() for record in caplog.records]


class TestTheElevatedIdentity:
    def test_it_is_a_session_of_its_own_so_each_runner_collects_only_its_claims(self) -> None:
        label, session = node_agent.node_identity("lavender", elevated=True)

        assert label == "fleet-node-lavender-elevated"
        assert session == str(uuid.uuid5(uuid.NAMESPACE_URL, "fleet-node-agent/lavender/elevated"))
        assert (label, session) != node_agent.node_identity("lavender", elevated=False)


class TestClaiming:
    def test_the_elevated_runner_claims_with_the_elevated_tag_once_its_token_is_an_admins(
        self, elevated_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        endpoint, _ = _tick(elevated_config, ADMINISTRATOR_TOKEN, elevated=True, caplog=caplog)

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        held, claim = endpoint.arguments
        assert held["claimedBy"] == "fleet-node-lavender-elevated"
        assert claim["tags"] == ["elevated", "windows"]
        assert claim["agent"] == "fleet-node-lavender-elevated"

    def test_the_ordinary_runner_of_the_same_node_claims_without_it(
        self, elevated_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Its builds launch at Limited whatever the token, so the line is
        not its business and a filtered token claims all the same."""
        endpoint, _ = _tick(elevated_config, FILTERED_TOKEN, elevated=False, caplog=caplog)

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        assert endpoint.arguments[1]["tags"] == ["windows"]
        assert endpoint.arguments[1]["agent"] == "fleet-node-lavender"

    def test_a_filtered_token_claims_nothing_and_names_the_accounts_fix(
        self, elevated_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        endpoint, messages = _tick(elevated_config, FILTERED_TOKEN, elevated=True, caplog=caplog)

        assert endpoint.tools == ["dispatch_list"]
        (tick,) = endpoint.ticks
        assert (tick["agent"], tick["elevated"], tick["claiming"]) == (
            "fleet-node-lavender-elevated",
            True,
            False,
        )
        assert (tick["tags"], tick["fits"]) == (["elevated", "windows"], [DEMO_PROJECT])
        assert (
            "lavender cannot launch elevated; claiming nothing: NODE_NOT_ELEVATED: lavender "
            "(lavender) declares an elevated runner, but its ssh session does not hold an "
            "administrator's token, so every build it launched at RunLevel Highest would fail to "
            "register. Add the ssh account to Administrators on the node, or set elevated to false"
        ) in messages

    def test_a_probe_with_no_token_line_claims_nothing_either(
        self, elevated_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A node still running a probe older than the line is not trusted."""
        endpoint, messages = _tick(
            elevated_config, LAVENDER_2026_09_23, elevated=True, caplog=caplog
        )

        assert endpoint.tools == ["dispatch_list"]
        assert any("NODE_NOT_ELEVATED" in message for message in messages)


class TestRefusalsAndAnnounce:
    def test_an_elevated_runner_for_a_node_that_declares_none_is_refused_before_any_call(
        self, config_path: pathlib.Path
    ) -> None:
        config_path.write_text(dump_json_str(sourced_document((("npm", "ci"),))), encoding="utf-8")
        endpoint = FakeQueue([])
        _test_hooks.http_post = endpoint

        with pytest.raises(ValueError, match=r"^lavender declares no elevated runner"):
            node_agent.main([*node_argv(config_path), node_agent.ELEVATED_FLAG])

        assert endpoint.tools == []

    def test_announce_registers_the_elevated_label_with_its_tags(
        self, elevated_config: pathlib.Path
    ) -> None:
        _test_hooks.hostname = lambda: "austinpc"
        endpoint = FakeQueue(["checked in"])
        _test_hooks.http_post = endpoint

        argv = [*node_argv(elevated_config), node_agent.ELEVATED_FLAG, queue.ANNOUNCE_FLAG]
        assert node_agent.main(argv) == 0

        checkin = endpoint.arguments[0]
        assert checkin["agent"] == "fleet-node-lavender-elevated"
        assert checkin["machine"] == f"{sys.platform}:austinpc"
        assert narrow_json_to_str(checkin["body"]) == (
            "fleet-node-lavender-elevated for lavender: claims the queue's node lane for jobs "
            "naming lavender or no node, carrying elevated, windows"
        )
