"""The toolchain's value forms, the ones a node runner reads without raising
(MCPs board task bad56f65): :func:`fleet.core.toolchain.attempt_toolchain`
and :func:`fleet.core.toolchain.readiness_gap`, against what lavender,
sedona and diphtheria answered on 2026-09-23 and serendipity on
2026-09-25, and the Node.js floor that serendipity's answer added.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.toolchain import ToolReport, describe_gap, is_ready, node_is_right
from fleet.core import _test_hooks, toolchain
from tests._toolchain_fixtures import (
    DIPHTHERIA_2026_09_23,
    LAVENDER_2026_09_23,
    LAVENDER_STORE_STUB,
    SEDONA_2026_09_23,
    SERENDIPITY_2026_09_25,
    node,
)
from tests.conftest import FakeRun, failed, ok


class TestAttemptToolchain:
    def test_a_ready_answer_comes_back_as_its_reports(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(LAVENDER_2026_09_23)])

        answered = toolchain.attempt_toolchain(node())

        assert answered == toolchain.read_reports(LAVENDER_2026_09_23)
        assert [report["name"] for report in toolchain.read_reports(LAVENDER_2026_09_23)] == [
            "python",
            "poetry",
            "git",
            "make",
            "node",
            "tar",
            "winget",
            "choco",
            "pip",
        ]

    def test_a_node_that_cannot_be_reached_is_a_value_with_the_transport_code(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "ssh: connect to host lavender: timed out")])

        answered = toolchain.attempt_toolchain(node())

        assert not isinstance(answered, tuple)
        assert answered["code"] is FleetErrorCode.NODE_UNREACHABLE
        assert "timed out" in answered["message"]

    def test_an_answer_naming_no_tool_is_a_value_naming_the_host(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("nothing\n")])

        assert toolchain.attempt_toolchain(node()) == {
            "code": FleetErrorCode.NODE_TOOL_MISSING,
            "message": "lavender: a toolchain probe returned nothing recognisable, so the "
            "node was never asked: 'nothing'",
        }

    def test_read_reports_of_an_answer_naming_no_tool_is_empty(self) -> None:
        assert toolchain.read_reports("WARNING: banner\n") == ()


class TestReadySummary:
    def test_it_names_what_a_ready_node_was_judged_on(self) -> None:
        assert (
            toolchain.ready_summary(toolchain.read_reports(LAVENDER_2026_09_23))
            == "python 3.11.9; node v24.20.0; poetry, git, make, tar present"
        )
        assert (
            toolchain.ready_summary(toolchain.read_reports(DIPHTHERIA_2026_09_23))
            == "python 3.11.15; node v24.21.0; poetry, git, make, tar present"
        )


class TestReadinessGap:
    def test_the_three_nodes_that_answered_ready_have_no_gap(self) -> None:
        for name, answer in (
            ("lavender", LAVENDER_2026_09_23),
            ("sedona", SEDONA_2026_09_23),
            ("diphtheria", DIPHTHERIA_2026_09_23),
        ):
            assert toolchain.readiness_gap(name, node(name), toolchain.read_reports(answer)) is None

    def test_the_store_alias_is_a_gap_returned_not_raised(self) -> None:
        gap = toolchain.readiness_gap(
            "lavender", node(), toolchain.read_reports(LAVENDER_STORE_STUB)
        )

        stated = ("", "") if gap is None else (gap.code.value, gap.message.split(" -- ")[0])
        assert stated == (
            FleetErrorCode.NODE_TOOL_MISSING.value,
            "lavender (lavender) cannot run a build: python",
        )

    def test_serendipitys_node_18_is_its_own_code_naming_what_was_found(self) -> None:
        """The node that claimed MCPs/packages/wiki-search on 2026-09-25 and
        failed at node-gyp: every tool present, Python right, Node.js 18."""
        gap = toolchain.readiness_gap(
            "serendipity",
            node("serendipity"),
            toolchain.read_reports(SERENDIPITY_2026_09_25),
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

    def test_serendipity_is_named_with_the_floor_and_what_it_found(self) -> None:
        described = describe_gap("serendipity", toolchain.read_reports(SERENDIPITY_2026_09_25))

        assert described == "serendipity: missing node 24+ (found v18.13.0)"
        assert not is_ready(toolchain.read_reports(SERENDIPITY_2026_09_25))

    def test_an_absent_node_is_named_once_by_its_install_not_again_by_the_floor(self) -> None:
        reports = toolchain.read_reports(
            LAVENDER_2026_09_23.replace("node=yes=v24.20.0", "node=no=")
        )

        described = describe_gap("lavender", reports)

        assert described.startswith("lavender: missing node (winget install --id OpenJS.NodeJS.LTS")
        assert "node 24+" not in described
