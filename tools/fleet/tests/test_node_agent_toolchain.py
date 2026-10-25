"""The toolchain gate in a node runner's claim tick (MCPs board task bad56f65).

Lavender claimed slime jobs d515d038 and 7235d4c4 on 2026-09-22 and 09-23,
staged a whole export, ran npm ci and died at the Makefile's first python
call, which found no interpreter (MCPs board task e62c8120). The runner now
asks the node's toolchain after its capacity and before the queue, so these
drive a whole tick against what diphtheria answered, and the answers derived
from it, and assert that a node which cannot build takes nothing off the
lane, and says what would fix it.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import dump_json_str

from fleet.cli import node_agent
from fleet.core import _test_hooks
from tests._node_agent_fixtures import (
    NOTHING_LAUNCHED,
    NOTHING_MATCHED,
    PROBED,
    SERVED_HOLDING_NOTHING,
    _credentials_in_env,
    _sourced_config,
    node_argv,
)
from tests._queue_fakes import FakeQueue
from tests._toolchain_fixtures import (
    CHROME_APT,
    CHROME_REASON,
    DIPHTHERIA_2026_09_23,
    GO_APT,
    GO_REASON,
    HOOKS_REASON,
    NO_PYTHON,
    OLD_NODE,
    PDFTOTEXT_APT,
    PDFTOTEXT_REASON,
    WRONG_PYTHON,
)
from tests.conftest import PROBE_OK, FakeRun, failed, ok

__all__ = ["_credentials_in_env", "_sourced_config"]

#: What every ready summary of :data:`DIPHTHERIA_2026_09_23` begins with.
READY_HEAD = (
    "lavender toolchain ready: python 3.11.15; node v24.21.0; poetry, git, make, tar present; "
)

#: The hooks entry of a gap on a node whose managers are apt-get and pipx:
#: the hooks check's tools install through pip alone.
HOOKS_GAP = (
    f"hooks -- {HOOKS_REASON}, so those jobs go to a node that has it -- no automatic install "
    "on this node"
)


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
        """A node after its ffmpeg install (MCPs board task 939ec5c7): the
        runner claims with ffmpeg beside linux, so the queue may hand it an
        audio check, and its gap no longer names ffmpeg."""
        _test_hooks.run = FakeRun(
            [
                ok(""),
                ok(PROBE_OK),
                ok(""),
                ok(DIPHTHERIA_2026_09_23 + "ffmpeg=yes=ffmpeg version 6.1.1-3ubuntu5\n"),
            ]
        )
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        assert endpoint.arguments[1]["tags"] == ["ffmpeg", "linux"]
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-5:] == [
            READY_HEAD + "ffmpeg present; hooks absent; go absent; chrome absent; pdftotext absent",
            f"lavender claims without the tag of every tool it lacks: {HOOKS_GAP}; go -- "
            f"{GO_REASON}, so those jobs go to a node that has it -- {GO_APT}; chrome -- "
            f"{CHROME_REASON}, so those jobs go to a node that has it -- {CHROME_APT}; "
            f"pdftotext -- {PDFTOTEXT_REASON}, so those jobs go to a node that has it -- "
            f"{PDFTOTEXT_APT}",
            NOTHING_MATCHED,
            NOTHING_LAUNCHED,
            SERVED_HOLDING_NOTHING,
        ]

    def test_a_node_whose_probe_found_every_tagged_tool_claims_with_each(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A node once its interpreter carries the claude-hooks check's tools
        (MCPs board task ec895824) claims with hooks, so that package's check
        may land here; with go found too (MCPs board task 1da15750) it claims
        with go, so rcs-bridge's check may land here, with Chrome found (MCPs
        board task 2f596185) it claims with chrome, so web-scraper's check may
        land here, with pdftotext found (MCPs board task 6dd73ada) it claims
        with pdftotext, so wiki-check's check may land here, and with every
        tagged tool present it logs no gap."""
        found = (
            "ffmpeg=yes=ffmpeg version 6.1.1-3ubuntu5\n"
            "hooks=yes=/home/corvis/.claude/corvis-hooks.json\n"
            "go=yes=go version go1.27.1 linux/amd64\nchrome=yes=155.0.8059.39\n"
            "pdftotext=yes=pdftotext version 24.02.0\n"
        )
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK), ok(""), ok(DIPHTHERIA_2026_09_23 + found)])
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["tags"] == [
            "chrome",
            "ffmpeg",
            "go",
            "hooks",
            "linux",
            "pdftotext",
        ]
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-4:] == [
            READY_HEAD
            + "ffmpeg present; hooks present; go present; chrome present; pdftotext present",
            NOTHING_MATCHED,
            NOTHING_LAUNCHED,
            SERVED_HOLDING_NOTHING,
        ]

    def test_a_node_without_ffmpeg_claims_without_it(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """linux alone, so the queue keeps an audio check for another node and
        offers this one the rest."""
        _test_hooks.run = FakeRun(PROBED)
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["tags"] == ["linux"]

    def test_a_card_and_compiler_the_registry_does_not_declare_are_claimed_and_logged(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MCPs board task 939ec5c7, A2: what the probe finds is what the
        runner claims with, so a node is eligible on the tick after an
        install with no file edited, and each difference from the registry is
        logged so the declaration can be corrected."""
        found = "cxx=yes=13.3.0\ngpu=yes=NVIDIA GeForce GTX 1630, 7.5\ntestdb=no=\n"
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK), ok(""), ok(DIPHTHERIA_2026_09_23 + found)])
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["tags"] == ["cxx", "gpu", "linux"]
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-5:] == [
            "lavender (lavender) declares gpu none but nvidia-smi reports 'NVIDIA GeForce GTX "
            "1630, 7.5', so it claims with the gpu tag; correct gpu in its registry/machines "
            "file",
            "lavender (lavender) declares cxx none but its probe reports cxx '13.3.0', so it "
            "claims with the cxx tag; set cxx to '13.3.0' in its registry/machines file",
            NOTHING_MATCHED,
            NOTHING_LAUNCHED,
            SERVED_HOLDING_NOTHING,
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
        assert messages[-5:] == [
            READY_HEAD + "ffmpeg absent; hooks absent; go absent; chrome absent; pdftotext absent",
            "lavender claims without the tag of every tool it lacks: ffmpeg -- grandma-api's "
            "check converts real audio files through ffmpeg, so those jobs go to a node that has "
            f"it -- sudo apt-get install -y ffmpeg; {HOOKS_GAP}; "
            f"go -- {GO_REASON}, so those jobs go to a node that has it -- {GO_APT}; "
            f"chrome -- {CHROME_REASON}, so those jobs go to a node that has it -- "
            f"{CHROME_APT}; pdftotext -- {PDFTOTEXT_REASON}, so those jobs go to a node that "
            f"has it -- {PDFTOTEXT_APT}",
            NOTHING_MATCHED,
            NOTHING_LAUNCHED,
            SERVED_HOLDING_NOTHING,
        ]


class TestAToolchainThatCannotBuild:
    def test_a_node_without_its_interpreter_claims_nothing_and_names_the_gap(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = _tick(sourced_config, ok(NO_PYTHON), caplog)

        refusal = next(m for m in messages if m.startswith("lavender cannot build"))
        assert refusal.startswith(
            "lavender cannot build; claiming nothing: NODE_TOOL_MISSING: "
            "lavender (lavender) cannot run a build: python -- "
        )
        assert "poetry -- " in refusal
        assert refusal.endswith("-- no automatic install on this node")
        assert messages[-3:] == [refusal, NOTHING_LAUNCHED, SERVED_HOLDING_NOTHING]

    def test_a_wrong_minor_python_claims_nothing_with_its_own_code(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = _tick(sourced_config, ok(WRONG_PYTHON), caplog)

        assert any(
            m.startswith(
                "lavender cannot build; claiming nothing: NODE_PYTHON_MISMATCH: "
                "lavender (lavender) reports Python 'Python 3.12.3' where 3.11 is required"
            )
            for m in messages
        )

    def test_node_18_claims_nothing_with_its_own_code(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The Node.js serendipity answered on 2026-09-25, when it took an MCPs
        check and failed at node-gyp: now the lane is left alone."""
        messages = _tick(sourced_config, ok(OLD_NODE), caplog)

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
        messages = _tick(sourced_config, failed(127, "sh: 1: python3: not found"), caplog)

        refusal = next(m for m in messages if "did not answer the toolchain probe" in m)
        assert refusal.startswith(
            "lavender did not answer the toolchain probe; claiming nothing: DISPATCH_FAILED: "
        )
        assert "sh: 1: python3: not found" in refusal

    def test_an_answer_naming_no_tool_claims_nothing_as_an_unread_probe(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = _tick(sourced_config, ok("WARNING: profile banner\n"), caplog)

        assert (
            "lavender did not answer the toolchain probe; claiming nothing: NODE_TOOL_MISSING: "
            "lavender: a toolchain probe returned nothing recognisable, so the node was never "
            "asked: 'WARNING: profile banner'"
        ) in messages
