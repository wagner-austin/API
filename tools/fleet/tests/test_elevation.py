"""The elevated runner's declaration and its token gate (MCPs board task a98d7083).

A node's ``elevated`` field registers a runner whose builds launch as an
administrator, so the decoder demands it be said aloud and refuses it where no
elevated launch exists; the gate reads the probe's ``integrity`` line and nothing
else, because a declaration alone outlives the account's membership in
Administrators.
"""

from __future__ import annotations

import pytest
from platform_core.errors import FleetErrorCode
from platform_core.json_utils import JSONTypeError

from fleet.contracts.elevation import (
    ADMINISTRATOR,
    INTEGRITY_PROBE,
    elevated_session,
    elevation_gap,
)
from fleet.contracts.node import decode_node_config, encode_node_config
from fleet.contracts.project import ProjectConfig
from fleet.contracts.tags import NodeTag
from fleet.contracts.toolchain import ToolReport
from fleet.core import dispatch
from fleet.core.toolchain import read_reports
from tests._toolchain_fixtures import LAVENDER_2026_09_23, node


def _integrity(present: bool, answer: str) -> ToolReport:
    """One ``integrity`` line, as the probe parser reads it.

    Args:
        present: The line's yes or no.
        answer: What it carried.

    Returns:
        The report.
    """
    return ToolReport(name=INTEGRITY_PROBE, present=present, version=answer)


@pytest.mark.parametrize(
    ("tags", "elevated"),
    [
        ((NodeTag.WINDOWS, NodeTag.ELEVATED), True),
        ((NodeTag.WINDOWS,), False),
        ((), False),
    ],
)
def test_the_recipe_is_elevated_exactly_when_the_project_requires_it(
    tags: tuple[NodeTag, ...], elevated: bool
) -> None:
    """The one place the tag becomes a Highest launch, for the queue's lane
    and the working-tree lane alike."""
    plan = ProjectConfig(
        worker_ram_gb=0.5,
        minimum_workers=1,
        expected_minutes=10,
        exclusive_resources=(),
        external_paths=(),
        required_tags=tags,
        source=None,
    )

    assert dispatch.recipe_for(plan, path="execution/elevated", install=())["elevated"] is elevated
    assert dispatch.working_tree_recipe("execution/elevated", plan)["elevated"] is elevated


class TestTheDeclaration:
    def test_an_absent_elevated_key_is_refused(self) -> None:
        """True registers a runner that builds as an administrator, so it is
        never a default, and false is said aloud."""
        encoded = encode_node_config(node())
        del encoded["elevated"]

        with pytest.raises(JSONTypeError, match="must declare 'elevated': true only for a Windows"):
            decode_node_config(encoded)

    def test_elevated_is_carried_both_ways_on_a_windows_node(self) -> None:
        encoded = encode_node_config(node())
        assert encoded["elevated"] is False

        assert decode_node_config({**encoded, "elevated": True})["elevated"] is True

    def test_an_elevated_linux_node_is_refused(self) -> None:
        with pytest.raises(
            JSONTypeError, match=r"^elevated is true on a linux node; an elevated runner launches"
        ):
            decode_node_config(
                {**encode_node_config(node()), "platform": "linux", "elevated": True}
            )


class TestTheTokenGate:
    def test_the_probe_parser_keeps_the_integrity_line(self) -> None:
        reports = read_reports(LAVENDER_2026_09_23 + "integrity=yes=administrator\n")

        assert reports[-1] == _integrity(True, ADMINISTRATOR)

    def test_an_administrators_token_opens_the_gate(self) -> None:
        reports = (_integrity(True, ADMINISTRATOR),)

        assert elevated_session(reports)
        assert elevation_gap("serendipity", node("serendipity"), reports) is None

    @pytest.mark.parametrize(
        "reports",
        [
            (_integrity(False, "limited"),),
            (_integrity(True, "limited"),),
            (ToolReport(name="docker", present=True, version=ADMINISTRATOR),),
            (),
        ],
    )
    def test_anything_else_closes_it_naming_the_accounts_fix(
        self, reports: tuple[ToolReport, ...]
    ) -> None:
        """A filtered token, a line in another shape, another tool's line
        that happens to say the word, and a probe with no line at all."""
        assert not elevated_session(reports)
        gap = elevation_gap("serendipity", node("serendipity"), reports)

        assert (None if gap is None else (gap.code, gap.message)) == (
            FleetErrorCode.NODE_NOT_ELEVATED,
            "serendipity (serendipity) declares an elevated runner, but its ssh session does not "
            "hold an administrator's token, so every build it launched at RunLevel Highest would "
            "fail to register. Add the ssh account to Administrators on the node, or set elevated "
            "to false",
        )
