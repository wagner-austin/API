"""The toolchain's value forms, the ones a node runner reads without raising
(MCPs board task bad56f65): :func:`fleet.core.toolchain.attempt_toolchain`
and :func:`fleet.core.toolchain.readiness_gap`, against what lavender,
sedona and diphtheria answered on 2026-09-23.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.core import _test_hooks, toolchain
from tests._toolchain_fixtures import (
    DIPHTHERIA_2026_09_23,
    LAVENDER_2026_09_23,
    LAVENDER_STORE_STUB,
    SEDONA_2026_09_23,
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
            == "python 3.11.9; poetry, git, make, node, tar present"
        )
        assert (
            toolchain.ready_summary(toolchain.read_reports(DIPHTHERIA_2026_09_23))
            == "python 3.11.15; poetry, git, make, node, tar present"
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
