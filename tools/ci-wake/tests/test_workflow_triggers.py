"""Which pushed refs start a workflow (MCPs board task 24547f50).

The reader runs over every workflow this repository carries and over MCPs's
two, verbatim, so a workflow written in a shape it does not read fails here
before it fails a push; then each rule GitHub states for ``on.push`` is read
against small workflows, and each shape the reader refuses is refused by
name with its line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError

from ci_wake.workflow_triggers import (
    EVERY_REF,
    PushTrigger,
    Workflow,
    admits,
    describe,
    read_workflow,
)
from tests.conftest import MCPS_CHECK, MCPS_CHECK_LIVE

#: This repository's root: tools/ci-wake/tests is three levels below it.
ROOT: Final = Path(__file__).resolve().parents[3]

MAIN: Final = "refs/heads/main"
DX_MOD: Final = "refs/tags/archive/dx-mod"


def _push(text: str) -> PushTrigger | None:
    """Read a workflow named ``w.yml`` and return its push trigger."""
    return read_workflow("w.yml", text).push


def _trigger(text: str) -> PushTrigger:
    """Read a workflow that must declare a push trigger, and return it."""
    push = _push(text)
    if push is None:
        raise AssertionError(f"no push trigger read from {text!r}")
    return push


def _refused(text: str) -> str:
    """Read a workflow the reader must refuse, returning the message."""
    with pytest.raises(AppError) as refused:
        read_workflow("w.yml", text)
    assert refused.value.code is CiWakeErrorCode.WORKFLOW_TRIGGER_UNREAD
    return refused.value.message


class TestTheWorkflowsThatExist:
    def test_mcps_runs_on_main_and_never_on_the_tag_that_was_announced(self) -> None:
        check = read_workflow("check.yml", MCPS_CHECK)
        live = read_workflow("check-live.yml", MCPS_CHECK_LIVE)

        main_only = PushTrigger(
            branches=("main",), branches_ignore=None, tags=None, tags_ignore=None
        )
        assert check == Workflow("check.yml", main_only)
        assert live == Workflow("check-live.yml", main_only)
        # Both read as exactly this trigger, so its answers are theirs.
        assert admits(main_only, MAIN)
        assert not admits(main_only, DX_MOD)
        assert not admits(main_only, "refs/heads/opus-mcps-1008-3057")

    def test_every_workflow_in_this_repository_reads_and_packages_runs_on_tags(self) -> None:
        files = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
        workflows = {
            path.name: read_workflow(path.name, path.read_text(encoding="utf-8")) for path in files
        }

        assert len(workflows) == len(files) > 0
        # API's bare ``push:``: GitHub runs it for every branch and tag, which
        # is why "never enrol a tag" would have been wrong here.
        assert workflows["packages.yml"].push == EVERY_REF
        assert admits(EVERY_REF, DX_MOD)
        assert workflows["runner-reaper-proof.yml"].push is None


class TestGitHubsRules:
    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            (MAIN, True),
            ("refs/heads/feature/x", True),
            (DX_MOD, True),
            ("refs/notes/commits", False),
        ],
    )
    def test_no_ref_filter_runs_for_every_branch_and_tag_and_nothing_else(
        self, ref: str, expected: bool
    ) -> None:
        assert admits(EVERY_REF, ref) is expected

    def test_branch_filters_alone_run_for_no_tag(self) -> None:
        trigger = _trigger("on:\n  push:\n    branches-ignore: [wip]\n")
        assert admits(trigger, MAIN)
        assert not admits(trigger, "refs/heads/wip")
        assert not admits(trigger, "refs/tags/v1")

    def test_tag_filters_alone_run_for_no_branch(self) -> None:
        trigger = _trigger('on:\n  push:\n    tags:\n      - "v*"\n')
        assert admits(trigger, "refs/tags/v1.2")
        assert not admits(trigger, "refs/tags/release/v1")
        assert not admits(trigger, MAIN)

    def test_star_stops_at_a_slash_and_double_star_does_not(self) -> None:
        trigger = _trigger("on:\n  push:\n    branches: ['release/*', 'hotfix/**']\n")
        assert admits(trigger, "refs/heads/release/1")
        assert not admits(trigger, "refs/heads/release/1/rc")
        assert admits(trigger, "refs/heads/hotfix/a/b")
        assert not admits(trigger, "refs/heads/hotfix")

    def test_an_ignore_filter_admits_what_matches_none_of_it(self) -> None:
        trigger = _trigger(
            "on:\n  push:\n    tags-ignore:\n      - archive/**\n    branches: [main]\n"
        )
        assert not admits(trigger, DX_MOD)
        assert admits(trigger, "refs/tags/v1")
        assert admits(trigger, MAIN)

    def test_path_filters_alone_leave_every_ref_admitted(self) -> None:
        trigger = _push('on:\n  push:\n    paths:\n      - "libs/a/**"\n      - "b/[c]?"\n')
        assert trigger == EVERY_REF


class TestTheShapesItReads:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("on: push\n", EVERY_REF),
            ("on: [pull_request, push]  # both\n", EVERY_REF),
            ("on: workflow_dispatch\n", None),
            ('"on":\n  push:\n', EVERY_REF),
            ("'on':\n  pull_request:\n  push:\n  workflow_dispatch:\n", EVERY_REF),
            ("on:\n  schedule:\n    - cron: '0 0 * * *'\n", None),
        ],
    )
    def test_the_event_forms(self, text: str, expected: PushTrigger | None) -> None:
        assert _push(text) == expected

    def test_comments_quotes_and_both_sequence_forms(self) -> None:
        text = (
            "# a workflow\n"
            "on:\n"
            "\n"
            "  # the push trigger\n"
            "  push:\n"
            "    branches:\n"
            '      - "main"  # the one branch\n'
            "      - 'release/#1'\n"
            '    tags: ["v*", x]  # and tags\n'
            "jobs:\n"
            "  push:\n"
            "    runs-on: x\n"
        )
        assert _push(text) == PushTrigger(
            branches=("main", "release/#1"),
            branches_ignore=None,
            tags=("v*", "x"),
            tags_ignore=None,
        )

    def test_an_empty_filter_admits_nothing_of_its_kind(self) -> None:
        trigger = _trigger("on:\n  push:\n    branches: []\n")
        assert trigger == PushTrigger(
            branches=(), branches_ignore=None, tags=None, tags_ignore=None
        )
        assert not admits(trigger, MAIN)


class TestTheShapesItRefuses:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("name: x\n", "w.yml line 1: no on: at column zero"),
            ("on: push\non: pull_request\n", "w.yml line 2: a second on: at column zero"),
            ("on: {push: {}}\n", "w.yml line 1: on: is a flow mapping"),
            ("on:\njobs:\n", "w.yml line 1: on: is empty"),
            (
                "on: [push\n",
                "w.yml line 1: '[push' is a flow sequence that does not close on its line",
            ),
            ("on: *anchor\n", "w.yml line 1: '*anchor' is not a quoted string or a plain word"),
            ('on: ["a"b"]\n', 'w.yml line 1: \'"a"b"\' is not a quoted string or a plain word'),
            ("on:\n  push:\n push:\n", "w.yml line 3: 'push:' is not an event of on:"),
            ("on:\n  - push\n", "w.yml line 2: '- push' is not an event of on:"),
            ("on:\n  push:\n  push:\n", "w.yml line 3: on: names push twice"),
            ("on:\n  push: {}\n", "w.yml line 2: push is '{}', not a mapping of filters or empty"),
            (
                "on:\n  push:\n    branch: [main]\n",
                "w.yml line 3: push has no filter named 'branch'",
            ),
            (
                "on:\n  push:\n    tags: [a]\n    tags: [b]\n",
                "w.yml line 4: push names 'tags' twice",
            ),
            (
                "on:\n  push:\n    branches: main\n",
                "w.yml line 3: branches is 'main', not a sequence",
            ),
            (
                "on:\n  push:\n    branches: [a]\n    branches-ignore: [b]\n",
                "w.yml line 2: push names both branches and branches-ignore",
            ),
            (
                "on:\n  push:\n    tags:\n      - a\n    tags-ignore:\n      - b\n",
                "w.yml line 2: push names both tags and tags-ignore",
            ),
            (
                "on:\n  push:\n    branches: [main]\n      - x\n",
                "w.yml line 4: '- x' is not a filter of push",
            ),
            (
                "on:\n  push:\n    branches:\n    - main\n",
                "w.yml line 4: '- main' is not a filter of push",
            ),
            (
                "on:\n  push:\n    branches:\n      - '!main'\n",
                "w.yml line 4: pattern '!main' negates, which is order-dependent",
            ),
            (
                "on:\n  push:\n    tags: ['v[0-9]']\n",
                "w.yml line 3: pattern 'v[0-9]' uses the operator '['",
            ),
        ],
    )
    def test_each_is_named_with_its_line(self, text: str, message: str) -> None:
        assert _refused(text) == (
            f"{message}; whether this push starts a workflow is read from every workflow's on: "
            "block, and one it cannot read cannot be judged"
        )


class TestDescribe:
    def test_each_kind_of_trigger_is_said_in_words(self) -> None:
        assert describe(read_workflow("check.yml", MCPS_CHECK)) == (
            "check.yml: push on branches main"
        )
        assert describe(Workflow("packages.yml", EVERY_REF)) == (
            "packages.yml: push on every branch and tag"
        )
        assert describe(Workflow("diag.yml", None)) == "diag.yml: no push trigger"
        both = PushTrigger(branches=(), branches_ignore=None, tags=None, tags_ignore=("a", "b/**"))
        assert describe(Workflow("w.yml", both)) == (
            "w.yml: push on branches (none); tags-ignore a, b/**"
        )
