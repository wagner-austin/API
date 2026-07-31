"""Tests for the wiki anchor-drift report.

The git resolver is injected through ``scripts._test_hooks`` with
save-and-restore, so no test shells out to git and no test depends on
the repository's actual anchor state.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.wiki_anchors import (
    STATUS_CURRENT,
    STATUS_STALE,
    STATUS_UNRESOLVED,
    AnchorStatus,
    collect_anchor_statuses,
    format_report,
    run_report,
)

from scripts import _test_hooks as script_hooks
from scripts import wiki_anchors
from tankpit_bot import _test_hooks as core_hooks

RECORDED = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"

PAGE = f"""---
title: Fixture
tags: [fixture]
related:
  - "[[other]]"
source_paths:
  - "src/fixture.py"
source_git_blobs:
  "src/fixture.py": "{RECORDED}"
fact_checked: "2026-07-31"
confidence: high
hubs: [things]
---

# Fixture
"""


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Save and restore the injected hooks around every test.

    Yields:
        None, with the original hook implementations restored after.
    """
    original_resolve = script_hooks.resolve_tree_hash
    original_argv = core_hooks.get_argv
    yield
    script_hooks.resolve_tree_hash = original_resolve
    core_hooks.get_argv = original_argv


def _install_resolver(result: str | None) -> None:
    """Inject a resolver returning one fixed answer.

    Args:
        result: Object id to return, or None to simulate an
            unresolvable path.
    """

    class _FakeResolver:
        def __call__(self, project_root: Path, repo_path: str) -> str | None:
            return result

    script_hooks.resolve_tree_hash = _FakeResolver()


def _install_argv(argv: list[str]) -> None:
    """Inject the argument list ``main`` reads.

    Args:
        argv: Arguments to present, without the program name. A
            placeholder program name is prepended, matching the real
            ``get_argv`` which returns the full ``sys.argv``.
    """

    class _FakeArgv:
        def __call__(self) -> list[str]:
            return ["tankpit-wiki-anchors", *argv]

    core_hooks.get_argv = _FakeArgv()


def _write_page(root: Path, name: str, text: str) -> None:
    """Create one wiki page in a fake tree.

    Args:
        root: Project root to build under.
        name: Page filename.
        text: Full markdown text.
    """
    pages_dir = root / "wiki" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / name).write_text(text, encoding="utf-8")


def _status(status_value: str) -> AnchorStatus:
    """Build one anchor status for formatter tests.

    Args:
        status_value: Status label to carry.

    Returns:
        A populated :class:`AnchorStatus`.
    """
    return AnchorStatus(
        page="alpha.md",
        path="src/fixture.py",
        recorded=RECORDED,
        current=OTHER,
        status=status_value,
        fact_checked="2026-07-31",
    )


class TestCollect:
    """Resolving anchors against HEAD."""

    def test_no_wiki_yields_no_statuses(self, tmp_path: Path) -> None:
        """A tree without ``wiki/pages`` reports nothing."""
        _install_resolver(RECORDED)
        assert collect_anchor_statuses(tmp_path) == []

    def test_matching_hash_is_current(self, tmp_path: Path) -> None:
        """An anchor equal to HEAD's id is CURRENT."""
        _write_page(tmp_path, "alpha.md", PAGE)
        _install_resolver(RECORDED)
        statuses = collect_anchor_statuses(tmp_path)
        assert [s["status"] for s in statuses] == [STATUS_CURRENT]

    def test_differing_hash_is_stale(self, tmp_path: Path) -> None:
        """An anchor behind HEAD's id is STALE."""
        _write_page(tmp_path, "alpha.md", PAGE)
        _install_resolver(OTHER)
        statuses = collect_anchor_statuses(tmp_path)
        assert [s["status"] for s in statuses] == [STATUS_STALE]
        assert statuses[0]["recorded"] == RECORDED
        assert statuses[0]["current"] == OTHER

    def test_unresolvable_path_is_unresolved(self, tmp_path: Path) -> None:
        """A path git cannot resolve is UNRESOLVED with empty current."""
        _write_page(tmp_path, "alpha.md", PAGE)
        _install_resolver(None)
        statuses = collect_anchor_statuses(tmp_path)
        assert [s["status"] for s in statuses] == [STATUS_UNRESOLVED]
        assert statuses[0]["current"] == ""

    def test_page_without_frontmatter_is_skipped(self, tmp_path: Path) -> None:
        """An unparseable page contributes no anchors and does not raise."""
        _write_page(tmp_path, "alpha.md", PAGE)
        _write_page(tmp_path, "bare.md", "# Bare\n")
        _install_resolver(RECORDED)
        statuses = collect_anchor_statuses(tmp_path)
        assert [s["page"] for s in statuses] == ["alpha.md"]

    def test_page_without_anchors_contributes_none(self, tmp_path: Path) -> None:
        """A page with no ``source_git_blobs`` yields no rows."""
        stripped = PAGE.replace(f'source_git_blobs:\n  "src/fixture.py": "{RECORDED}"\n', "")
        _write_page(tmp_path, "alpha.md", stripped)
        _install_resolver(RECORDED)
        assert collect_anchor_statuses(tmp_path) == []

    def test_fact_checked_is_carried_for_triage(self, tmp_path: Path) -> None:
        """The page's audit date rides along with each anchor."""
        _write_page(tmp_path, "alpha.md", PAGE)
        _install_resolver(OTHER)
        assert collect_anchor_statuses(tmp_path)[0]["fact_checked"] == "2026-07-31"


class TestFormat:
    """Rendering the report."""

    def test_stale_rows_are_listed_with_guidance(self) -> None:
        """A stale row prints, and the audit-not-bump guidance follows."""
        lines = format_report([_status(STATUS_STALE)], show_all=False)
        assert "STALE" in lines[0]
        assert "alpha.md" in lines[0]
        assert "1 anchors: 1 stale, 0 unresolved, 0 current" in lines[1]
        assert "owed an AUDIT, not a bump" in lines[2]

    def test_current_rows_are_hidden_by_default(self) -> None:
        """Without ``show_all`` a current anchor prints only in the summary."""
        lines = format_report([_status(STATUS_CURRENT)], show_all=False)
        assert lines == ["1 anchors: 0 stale, 0 unresolved, 1 current"]

    def test_show_all_includes_current_rows(self) -> None:
        """``show_all`` lists current anchors too."""
        lines = format_report([_status(STATUS_CURRENT)], show_all=True)
        assert "CURRENT" in lines[0]
        assert len(lines) == 2

    def test_no_guidance_line_when_nothing_is_stale(self) -> None:
        """The guidance line is withheld when there is no drift."""
        lines = format_report([_status(STATUS_UNRESOLVED)], show_all=False)
        assert not any("owed an AUDIT" in line for line in lines)

    def test_undated_page_renders_placeholder(self) -> None:
        """A page with no ``fact_checked`` renders ``(undated)``."""
        status = _status(STATUS_STALE)
        status["fact_checked"] = ""
        assert "(undated)" in format_report([status], show_all=False)[0]

    def test_empty_input_still_summarises(self) -> None:
        """Zero anchors produce a single zero summary line."""
        assert format_report([], show_all=True) == ["0 anchors: 0 stale, 0 unresolved, 0 current"]


class TestRunReport:
    """The printing wrapper."""

    def test_prints_and_returns_stale_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``run_report`` writes the table and returns the stale count."""
        _write_page(tmp_path, "alpha.md", PAGE)
        _install_resolver(OTHER)
        expected = format_report(collect_anchor_statuses(tmp_path), show_all=False)
        stale = run_report(tmp_path, show_all=False)
        assert stale == 1
        assert capsys.readouterr().out.splitlines() == expected


class TestMain:
    """CLI argument handling."""

    def test_plain_run_prints_the_default_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        """With no arguments main prints exactly the stale-only report."""
        _install_resolver(None)
        _install_argv([])
        expected = format_report(collect_anchor_statuses(Path.cwd()), show_all=False)
        wiki_anchors.main()
        assert capsys.readouterr().out.splitlines() == expected

    def test_all_flag_reaches_the_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``--all`` is threaded through to ``format_report``."""
        _install_resolver(None)
        _install_argv(["--all"])
        expected = format_report(collect_anchor_statuses(Path.cwd()), show_all=True)
        wiki_anchors.main()
        assert capsys.readouterr().out.splitlines() == expected

    def test_exit_code_flag_is_quiet_without_drift(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--exit-code`` does not raise when nothing is stale."""
        _install_resolver(None)
        _install_argv(["--exit-code"])
        expected = format_report(collect_anchor_statuses(Path.cwd()), show_all=False)
        wiki_anchors.main()
        assert capsys.readouterr().out.splitlines() == expected

    def test_exit_code_flag_raises_on_drift(self) -> None:
        """``--exit-code`` exits 1 when any anchor is stale."""
        _install_resolver(OTHER)
        _install_argv(["--exit-code"])
        with pytest.raises(SystemExit) as excinfo:
            wiki_anchors.main()
        assert excinfo.value.code == 1

    def test_help_prints_usage_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``--help`` prints usage and exits 0."""
        _install_resolver(None)
        _install_argv(["--help"])
        with pytest.raises(SystemExit) as excinfo:
            wiki_anchors.main()
        assert excinfo.value.code == 0
        assert capsys.readouterr().out == wiki_anchors._USAGE

    def test_unknown_argument_exits_two(self, capsys: pytest.CaptureFixture[str]) -> None:
        """An unrecognized flag prints usage and exits 2."""
        _install_resolver(None)
        _install_argv(["--nope"])
        with pytest.raises(SystemExit) as excinfo:
            wiki_anchors.main()
        assert excinfo.value.code == 2
        assert capsys.readouterr().out == wiki_anchors._USAGE


class TestRealResolver:
    """The production git resolver, exercised against this repository."""

    def test_resolves_a_tracked_path(self) -> None:
        """``pyproject.toml`` resolves to a 40-hex blob id in HEAD."""
        project_root = Path(__file__).resolve().parents[2]
        # ``or ""`` narrows the optional without an is-not-None or an
        # isinstance assertion (both guard-banned as weak); a None result
        # would collapse to "" and fail the length check below.
        blob = script_hooks._real_resolve_tree_hash(project_root, "pyproject.toml") or ""
        assert len(blob) == 40
        assert set(blob) <= set("0123456789abcdef")

    def test_untracked_path_resolves_to_none(self) -> None:
        """A path absent from HEAD resolves to None rather than raising."""
        project_root = Path(__file__).resolve().parents[2]
        assert script_hooks._real_resolve_tree_hash(project_root, "no/such/path.xyz") is None
