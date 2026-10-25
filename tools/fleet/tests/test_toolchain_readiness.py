"""The toolchain's value forms, the ones a node runner reads without raising
(MCPs board task bad56f65): :func:`fleet.core.toolchain.attempt_toolchain`
and :func:`fleet.core.toolchain.readiness_gap`, against what diphtheria
answered on 2026-09-23 and the answers derived from it, and the Node.js floor
that serendipity's 2026-09-25 answer added.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.toolchain import ToolReport, describe_gap, is_ready, node_is_right
from fleet.core import _test_hooks, toolchain
from tests._toolchain_fixtures import (
    DIPHTHERIA_2026_09_23,
    NO_PYTHON,
    OLD_NODE,
    node,
)
from tests.conftest import FakeRun, failed, ok

#: What every summary of :data:`DIPHTHERIA_2026_09_23` begins with.
READY_HEAD = "python 3.11.15; node v24.21.0; poetry, git, make, tar present; "


class TestAttemptToolchain:
    def test_a_ready_answer_comes_back_as_its_reports(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(DIPHTHERIA_2026_09_23)])

        answered = toolchain.attempt_toolchain(node(), writer="lavender")

        assert answered == toolchain.read_reports(DIPHTHERIA_2026_09_23)
        assert [report["name"] for report in toolchain.read_reports(DIPHTHERIA_2026_09_23)] == [
            "python",
            "poetry",
            "git",
            "make",
            "node",
            "tar",
            "apt-get",
            "pipx",
        ]

    def test_a_node_that_cannot_be_reached_is_a_value_with_the_transport_code(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "ssh: connect to host lavender: timed out")])

        answered = toolchain.attempt_toolchain(node(), writer="lavender")

        assert not isinstance(answered, tuple)
        assert answered["code"] is FleetErrorCode.NODE_UNREACHABLE
        assert "timed out" in answered["message"]

    def test_an_answer_naming_no_tool_is_a_value_naming_the_host(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("nothing\n")])

        assert toolchain.attempt_toolchain(node(), writer="lavender") == {
            "code": FleetErrorCode.NODE_TOOL_MISSING,
            "message": "lavender: a toolchain probe returned nothing recognisable, so the "
            "node was never asked: 'nothing'",
        }

    def test_read_reports_of_an_answer_naming_no_tool_is_empty(self) -> None:
        assert toolchain.read_reports("WARNING: banner\n") == ()


class TestReadySummary:
    def test_it_names_what_a_ready_node_was_judged_on(self) -> None:
        assert toolchain.ready_summary(toolchain.read_reports(DIPHTHERIA_2026_09_23)) == (
            READY_HEAD + "ffmpeg absent; hooks absent; go absent; chrome absent; pdftotext absent"
        )

    def test_it_names_a_tagged_tool_the_probe_found(self) -> None:
        """A node once ffmpeg is installed (MCPs board task 939ec5c7): the
        tag its runner claims with is on the same line."""
        found = toolchain.read_reports(
            DIPHTHERIA_2026_09_23 + "ffmpeg=yes=ffmpeg version 6.1.1-3ubuntu5\n"
        )
        assert toolchain.ready_summary(found) == (
            READY_HEAD + "ffmpeg present; hooks absent; go absent; chrome absent; pdftotext absent"
        )

    def test_it_names_the_hooks_environment_the_probe_found(self) -> None:
        """A node once its interpreter carries the claude-hooks check's tools
        (MCPs board task ec895824)."""
        found = toolchain.read_reports(
            DIPHTHERIA_2026_09_23 + "hooks=yes=/home/corvis/.claude/corvis-hooks.json\n"
        )
        assert toolchain.ready_summary(found) == (
            READY_HEAD + "ffmpeg absent; hooks present; go absent; chrome absent; pdftotext absent"
        )

    def test_it_names_go_the_probe_found(self) -> None:
        """A node once go is installed (MCPs board task 1da15750): its runner
        claims with go, and the summary says so on the same line."""
        found = toolchain.read_reports(
            DIPHTHERIA_2026_09_23 + "go=yes=go version go1.27.1 linux/amd64\n"
        )
        assert toolchain.ready_summary(found) == (
            READY_HEAD + "ffmpeg absent; hooks absent; go present; chrome absent; pdftotext absent"
        )

    def test_it_names_chrome_the_probe_found(self) -> None:
        """diphtheria after its 2026-10-09 install (MCPs board task 2f596185):
        its runner claims with chrome, and the summary says so on the same
        line, so the tag web-scraper's check waits for is visible each tick."""
        found = toolchain.read_reports(
            DIPHTHERIA_2026_09_23 + "chrome=yes=Google Chrome 155.0.8059.39\n"
        )
        assert toolchain.ready_summary(found) == (
            READY_HEAD + "ffmpeg absent; hooks absent; go absent; chrome present; pdftotext absent"
        )

    def test_it_names_pdftotext_the_probe_found(self) -> None:
        """diphtheria on 2026-10-11 (MCPs board task 6dd73ada): its runner
        claims with pdftotext, and the summary says so on the same line, so
        the tag wiki-check's check waits for is visible each tick."""
        found = toolchain.read_reports(
            DIPHTHERIA_2026_09_23 + "pdftotext=yes=pdftotext version 24.02.0\n"
        )
        assert toolchain.ready_summary(found) == (
            READY_HEAD + "ffmpeg absent; hooks absent; go absent; chrome absent; pdftotext present"
        )


class TestReadinessGap:
    def test_the_node_that_answered_ready_has_no_gap(self) -> None:
        reports = toolchain.read_reports(DIPHTHERIA_2026_09_23)

        assert toolchain.readiness_gap("diphtheria", node("diphtheria"), reports) is None

    def test_an_absent_interpreter_is_a_gap_returned_not_raised(self) -> None:
        gap = toolchain.readiness_gap("lavender", node(), toolchain.read_reports(NO_PYTHON))

        stated = ("", "") if gap is None else (gap.code.value, gap.message.split(" -- ")[0])
        assert stated == (
            FleetErrorCode.NODE_TOOL_MISSING.value,
            "lavender (lavender) cannot run a build: python",
        )

    def test_node_18_is_its_own_code_naming_what_was_found(self) -> None:
        """serendipity's Node.js, which claimed MCPs/packages/wiki-search on
        2026-09-25 and failed at node-gyp: every tool present, Python right,
        Node.js 18."""
        gap = toolchain.readiness_gap(
            "serendipity", node("serendipity"), toolchain.read_reports(OLD_NODE)
        )

        stated = ("", "") if gap is None else (gap.code.value, gap.message)
        assert stated == (
            FleetErrorCode.NODE_NODEJS_MISMATCH.value,
            "serendipity (serendipity) reports Node.js 'v18.13.0' where 24 or newer is "
            "required; the TypeScript projects declare that engine, and native modules they "
            "install fail to build under an older one",
        )


class TestNodeIsRight:
    def test_the_floor_and_above_are_right_and_below_is_not(self) -> None:
        def judged(version: str) -> bool:
            return node_is_right((ToolReport(name="node", present=True, version=version),))

        assert (judged("v24.0.0"), judged("v25.1.0"), judged("24.20.0")) == (True, True, True)
        assert (judged("v23.11.1"), judged("v18.13.0")) == (False, False)

    def test_a_version_with_no_numeric_major_is_not_right(self) -> None:
        assert not node_is_right((ToolReport(name="node", present=True, version="vnext"),))

    def test_an_absent_or_unreported_node_is_not_right(self) -> None:
        assert not node_is_right((ToolReport(name="node", present=False, version=""),))
        assert not node_is_right(())

    def test_an_old_node_is_named_with_the_floor_and_what_it_found(self) -> None:
        described = describe_gap("serendipity", toolchain.read_reports(OLD_NODE))

        assert described == "serendipity: missing node 24+ (found v18.13.0)"
        assert not is_ready(toolchain.read_reports(OLD_NODE))

    def test_an_absent_node_is_named_once_by_its_install_not_again_by_the_floor(self) -> None:
        reports = toolchain.read_reports(
            DIPHTHERIA_2026_09_23.replace("node=yes=v24.21.0", "node=no=")
        )

        described = describe_gap("lavender", reports)

        assert described == "lavender: missing node (install by hand)"
        assert "node 24+" not in described
