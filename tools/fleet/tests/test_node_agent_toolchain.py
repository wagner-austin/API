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
from tests._toolchain_fixtures import (
    LAVENDER_2026_09_23,
    LAVENDER_STORE_STUB,
    SERENDIPITY_2026_09_25,
    WRONG_PYTHON,
)
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


class TestTheToolTagsAClaimCarries:
    def test_a_node_whose_probe_found_ffmpeg_claims_with_its_tag(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """pendragon after its ffmpeg install, 2026-10-02 02:48Z (MCPs board
        task 939ec5c7): the runner claims with ffmpeg beside windows, so the
        queue may hand it grandma-api's check, and logs no tagged gap."""
        _test_hooks.run = FakeRun(
            [ok(""), ok(PROBE_OK), ok(""), ok(LAVENDER_2026_09_23 + "ffmpeg=yes=ffmpeg 7.1.1\n")]
        )
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        assert endpoint.arguments[1]["tags"] == ["ffmpeg", "windows"]
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-2:] == [
            "lavender toolchain ready: python 3.11.9; node v24.20.0; "
            "poetry, git, make, tar present; ffmpeg present",
            "nothing in the node lane for lavender",
        ]

    def test_a_node_without_ffmpeg_claims_without_it(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """pendragon before 02:48Z: windows alone, so the queue keeps
        grandma-api's check for another node and offers this one the rest."""
        _test_hooks.run = FakeRun(PROBED)
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["tags"] == ["windows"]

    def test_a_card_and_compiler_fleet_json_does_not_declare_are_claimed_and_logged(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MCPs board task 939ec5c7, A2: what the probe finds is what the
        runner claims with, so a node is eligible on the tick after an
        install with no file edited, and each difference from fleet.json is
        logged so the declaration can be corrected."""
        found = "cxx=yes=17.14.37710.0\ngpu=yes=NVIDIA GeForce GTX 1630, 7.5\ntestdb=no=\n"
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK), ok(""), ok(LAVENDER_2026_09_23 + found)])
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["tags"] == ["cxx", "gpu", "windows"]
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-3:] == [
            "lavender (lavender) declares gpu none but nvidia-smi reports 'NVIDIA GeForce GTX "
            "1630, 7.5', so it claims with the gpu tag; correct gpu in fleet.json",
            "lavender (lavender) declares cxx none but its probe reports cxx '17.14.37710.0', so "
            "it claims with the cxx tag; set cxx to '17.14.37710.0' in fleet.json",
            "nothing in the node lane for lavender",
        ]


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
        assert messages[-3:] == [
            "lavender toolchain ready: python 3.11.9; node v24.20.0; "
            "poetry, git, make, tar present; ffmpeg absent",
            "lavender claims without the tag of every tool it lacks: ffmpeg -- grandma-api's "
            "check converts real audio files through ffmpeg, so those jobs go to a node that has "
            "it -- winget install --id Gyan.FFmpeg.Essentials -e --source winget --silent "
            "--accept-package-agreements --accept-source-agreements --disable-interactivity",
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

    def test_node_18_claims_nothing_with_its_own_code(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Serendipity's real answer on 2026-09-25, when it took an MCPs check
        and failed at node-gyp: now the lane is left alone."""
        messages = _tick(sourced_config, ok(SERENDIPITY_2026_09_25), caplog)

        assert any(
            m.startswith(
                "lavender cannot build; claiming nothing: NODE_NODEJS_MISMATCH: "
                "lavender (lavender) reports Node.js 'v18.13.0' where 24 or newer is required"
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
