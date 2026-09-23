"""The toolchain gate in a node runner's claim tick (MCPs board task bad56f65).

Lavender claimed slime jobs d515d038 and 7235d4c4 on 2026-09-22 and 09-23,
staged a whole export, ran npm ci and died at the Makefile's first python
call on the Microsoft Store alias (MCPs board task e62c8120). The runner now
asks the node's toolchain after its capacity and before the queue, so these
drive a whole tick against what lavender really answered that day and assert
that a node which cannot build takes nothing off the lane, and says what
would fix it.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import dump_json_str

from fleet.cli import node_agent
from fleet.contracts.toolchain import install_command
from fleet.core import _test_hooks
from tests._node_agent_fixtures import (
    PROBED,
    _credentials_in_env,
    _sourced_config,
    node_argv,
)
from tests._queue_fakes import FakeQueue
from tests._toolchain_fixtures import LAVENDER_STORE_STUB, WRONG_PYTHON
from tests.conftest import PROBE_OK, FakeRun, failed, ok

__all__ = ["_credentials_in_env", "_sourced_config"]


def _tick(
    config_path: pathlib.Path,
    toolchain_run: _test_hooks.CommandResult,
    caplog: pytest.LogCaptureFixture,
) -> list[str]:
    """Run one tick whose node has room and answers the toolchain probe so.

    Args:
        config_path: The workspace document.
        toolchain_run: What running the toolchain probe returns.
        caplog: The test's log capture.

    Returns:
        Every message the tick logged, after asserting it claimed nothing.
    """
    runner = FakeRun([ok(""), ok(PROBE_OK), ok(""), toolchain_run])
    _test_hooks.run = runner
    endpoint = FakeQueue([dump_json_str({"jobs": []})])
    _test_hooks.http_post = endpoint
    with caplog.at_level("INFO"):
        assert node_agent.main(node_argv(config_path)) == 0
    assert endpoint.tools == ["dispatch_list"]
    assert [call[0] for call in runner.calls] == ["ssh"] * 4
    return [record.getMessage() for record in caplog.records]


class TestAToolchainThatCanBuild:
    def test_a_ready_node_says_what_it_was_judged_on_and_asks_the_queue(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _test_hooks.run = FakeRun(PROBED)
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-2:] == [
            "lavender toolchain ready: python 3.11.9; poetry, git, make, node, tar present",
            "nothing in the node lane for lavender",
        ]


class TestAToolchainThatCannotBuild:
    def test_the_store_alias_claims_nothing_and_names_what_would_install_python(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = _tick(sourced_config, ok(LAVENDER_STORE_STUB), caplog)

        refusal = next(m for m in messages if m.startswith("lavender cannot build"))
        assert refusal.startswith(
            "lavender cannot build; claiming nothing: NODE_TOOL_MISSING: "
            "lavender (lavender) cannot run a build: python -- "
        )
        assert install_command("python", ("winget", "choco")) in refusal
        assert "poetry -- " in refusal
        assert "nothing in the node lane for lavender" in messages

    def test_a_wrong_minor_python_claims_nothing_with_its_own_code(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = _tick(sourced_config, ok(WRONG_PYTHON), caplog)

        assert any(
            m.startswith(
                "lavender cannot build; claiming nothing: NODE_PYTHON_MISMATCH: "
                "lavender (lavender) reports Python 'Python 3.12.4' where 3.11 is required"
            )
            for m in messages
        )


class TestAToolchainThatDidNotAnswer:
    def test_a_probe_that_fails_on_the_node_claims_nothing_with_the_transports_code(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = _tick(sourced_config, failed(1, "the term 'python' is not recognized"), caplog)

        refusal = next(m for m in messages if "did not answer the toolchain probe" in m)
        assert refusal.startswith(
            "lavender did not answer the toolchain probe; claiming nothing: DISPATCH_FAILED: "
        )
        assert "the term 'python' is not recognized" in refusal

    def test_an_answer_naming_no_tool_claims_nothing_as_an_unread_probe(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = _tick(sourced_config, ok("WARNING: profile banner\n"), caplog)

        assert (
            "lavender did not answer the toolchain probe; claiming nothing: NODE_TOOL_MISSING: "
            "lavender: a toolchain probe returned nothing recognisable, so the node was never "
            "asked: 'WARNING: profile banner'"
        ) in messages
