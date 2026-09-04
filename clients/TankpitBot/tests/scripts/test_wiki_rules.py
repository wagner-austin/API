"""Tests for the wiki structure guard rule.

Each test builds a fake wiki tree under ``tmp_path`` and asserts on the
exact violations the rule produces; the final tests run the rule
against the REAL repository, which must be green.

No fakes are needed beyond the trees themselves — the rule is pure
filesystem reads, so the real production code path is what runs here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.wiki_rules import (
    CONFIDENCE_VALUES,
    REQUIRED_FRONTMATTER_KEYS,
    run_wiki_rules,
)

GREEN_FRONTMATTER = """---
title: Fixture Page
tags: [fixture]
related:
  - "[[other]]"
source_paths:
  - "src/fixture.py"
source_git_blobs:
  "src/fixture.py": "0123456789abcdef0123456789abcdef01234567"
fact_checked: "2026-07-31"
confidence: high
hubs: [things]
---

# Fixture Page

Body text.
"""


def _write_source_fixture(root: Path) -> None:
    """Create the source file the fixture frontmatter declares.

    Provenance checks must not depend on the wiki files under test, so
    the fixture's ``source_paths`` points at its own throwaway module.

    Args:
        root: Project root to create ``src/fixture.py`` under.
    """
    src_dir = root / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "fixture.py").write_text('"""Fixture."""\n', encoding="utf-8")


def _build_wiki(
    root: Path,
    *,
    pages: dict[str, str],
    hub_links: dict[str, list[str]],
    index_text: str,
) -> None:
    """Create a complete fake wiki tree.

    Args:
        root: Project root to build the ``wiki/`` tree under.
        pages: Page filename -> full markdown text.
        hub_links: Hub stem -> page filenames that hub links to.
        index_text: Full text of ``wiki/index.md``.
    """
    pages_dir = root / "wiki" / "pages"
    hubs_dir = root / "wiki" / "hubs"
    pages_dir.mkdir(parents=True)
    hubs_dir.mkdir(parents=True)
    _write_source_fixture(root)
    for name, text in pages.items():
        (pages_dir / name).write_text(text, encoding="utf-8")
    for stem, targets in hub_links.items():
        lines = [f"# {stem}", ""]
        lines.extend(f"[{t}](../pages/{t}) -- description" for t in targets)
        (hubs_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "wiki" / "index.md").write_text(index_text, encoding="utf-8")


def _green_tree(root: Path) -> None:
    """Build a fake wiki with zero violations.

    Args:
        root: Project root to build under.
    """
    _build_wiki(
        root,
        pages={"alpha.md": GREEN_FRONTMATTER},
        hub_links={"things": ["alpha.md"]},
        index_text="Wiki. 1 hub, 1 content pages.\n\n[Things](hubs/things.md) -- x (1 pages)\n",
    )


def _capture(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Run the rule and return its count plus emitted stdout.

    Args:
        root: Project root to check.
        capsys: pytest stdout/stderr capture fixture.

    Returns:
        Pair of (violation count, captured stdout text).
    """
    count = run_wiki_rules(root)
    return count, capsys.readouterr().out


class TestOutOfScope:
    """A tree with no wiki is not this rule's business."""

    def test_missing_pages_dir_is_zero_violations(self, tmp_path: Path) -> None:
        """A project without ``wiki/pages`` returns zero."""
        assert run_wiki_rules(tmp_path) == 0

    def test_missing_hubs_dir_skips_navigation_checks(self, tmp_path: Path) -> None:
        """Pages without a hubs dir are still frontmatter-checked."""
        pages_dir = tmp_path / "wiki" / "pages"
        pages_dir.mkdir(parents=True)
        _write_source_fixture(tmp_path)
        (pages_dir / "alpha.md").write_text(GREEN_FRONTMATTER, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_missing_index_skips_count_checks(self, tmp_path: Path) -> None:
        """Hubs without an index produce no count violations."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "index.md").unlink()
        assert run_wiki_rules(tmp_path) == 0


class TestGreenTree:
    """The happy path emits nothing at all."""

    def test_green_tree_has_no_violations(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A conformant wiki produces zero violations and no output."""
        _green_tree(tmp_path)
        count, out = _capture(tmp_path, capsys)
        assert count == 0
        assert out == ""


class TestFrontmatter:
    """Every page must open with a parseable, complete block."""

    def test_missing_block_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A page with no frontmatter is reported once."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text("# No matter\n", encoding="utf-8")
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "alpha.md: no frontmatter block" in out

    def test_unclosed_block_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A block whose closing fence is absent is reported."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text("---\ntitle: X\n", encoding="utf-8")
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "alpha.md: unclosed frontmatter block" in out

    def test_each_required_key_is_reported_when_absent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Dropping every required key yields one violation each."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(
            "---\nhubs: [things]\n---\n\n# X\n", encoding="utf-8"
        )
        count, out = _capture(tmp_path, capsys)
        assert count == len(REQUIRED_FRONTMATTER_KEYS)
        for key in REQUIRED_FRONTMATTER_KEYS:
            assert f"missing required key '{key}'" in out

    def test_malformed_fact_checked_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non ``YYYY-MM-DD`` date is rejected."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('"2026-07-31"', '"July 2026"')
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "fact_checked 'July 2026' is not YYYY-MM-DD" in out

    def test_impossible_date_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A well-shaped but non-existent date is rejected."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('"2026-07-31"', '"2026-02-30"')
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "is not YYYY-MM-DD" in out

    def test_unknown_confidence_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A confidence outside the SCHEMA's three levels is rejected."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace("confidence: high", "confidence: certain")
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "confidence 'certain' not one of" in out

    def test_every_declared_confidence_level_passes(self, tmp_path: Path) -> None:
        """Each SCHEMA confidence level is accepted."""
        for level in CONFIDENCE_VALUES:
            root = tmp_path / level
            root.mkdir()
            _green_tree(root)
            page = GREEN_FRONTMATTER.replace("confidence: high", f"confidence: {level}")
            (root / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
            assert run_wiki_rules(root) == 0

    def test_blank_and_unparseable_lines_are_ignored(self, tmp_path: Path) -> None:
        """Blank lines and stray prose inside the block do not crash."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace("title: Fixture Page", "title: Fixture Page\n\n  stray")
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_inline_flow_sequence_satisfies_a_required_key(self, tmp_path: Path) -> None:
        """``related: [a, b]`` counts as present, like the block form."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('related:\n  - "[[other]]"', "related: [a, b]")
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_empty_flow_sequence_is_accepted(self, tmp_path: Path) -> None:
        """An empty ``[]`` sequence parses to no items."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('related:\n  - "[[other]]"', "related: []")
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_single_quoted_scalar_is_unwrapped(self, tmp_path: Path) -> None:
        """Single quotes strip exactly like double quotes."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('"2026-07-31"', "'2026-07-31'")
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0


class TestProvenance:
    """Declared sources must exist; anchors must be well-formed."""

    def test_vanished_unpinned_source_path_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing ``source_paths`` entry WITHOUT a blob pin is reported.

        No pin means no way to recover the cited content — the claim's
        provenance genuinely vanished.
        """
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('- "src/fixture.py"', '- "src/deleted.py"')
        page = page.replace('  "src/fixture.py": "0123456789abcdef0123456789abcdef01234567"\n', "")
        page = page.replace("source_git_blobs:\n", "")
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "source_paths entry 'src/deleted.py' does not exist" in out

    def test_vanished_blob_pinned_source_is_a_retired_source(self, tmp_path: Path) -> None:
        """A missing entry WITH a blob pin passes — the pin IS the provenance.

        Reversed 2026-08-17 (the analysis_scripts retirement, board
        task f0c3a532): a one-shot measurement script may leave the
        working tree, and the page citing it stays verifiable through
        ``git cat-file blob <pinned hash>``. The old rule demanded the
        path exist forever, which would have forced either keeping dead
        ungated code or orphaning the measurement record.
        """
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('- "src/fixture.py"', '- "src/deleted.py"')
        page = page.replace('"src/fixture.py": ', '"src/deleted.py": ')
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_line_locator_suffix_is_stripped(self, tmp_path: Path) -> None:
        """``file.md:42`` checks the file, not the literal string."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('- "src/fixture.py"', '- "src/fixture.py:42"')
        page = page.replace('"src/fixture.py": ', '"src/fixture.py:42": ')
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_line_range_locator_suffix_is_stripped(self, tmp_path: Path) -> None:
        """``file.md:42-58`` also resolves to the file."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('- "src/fixture.py"', '- "src/fixture.py:42-58"')
        page = page.replace('"src/fixture.py": ', '"src/fixture.py:42-58": ')
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_bare_pin_key_resolves_an_anchored_source_path(self, tmp_path: Path) -> None:
        """A pin keyed by the FILE satisfies a ``source_paths`` line anchor.

        A git blob id addresses a file, so a page citing several lines of
        one file pins it once by the bare path. The rule strips the ``:NN``
        locator on both sides, matching wiki-check's ``git-blob-hash-pin``.

        Latent until 2026-09-04: no page in this wiki had combined an
        anchored ``source_paths`` entry with a pin, so the branch that
        compared them verbatim had never run, and the twenty tpclient.js
        pages were the first to reach it.
        """
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('- "src/fixture.py"', '- "src/fixture.py:16"')
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_url_source_is_skipped(self, tmp_path: Path) -> None:
        """An ``https://`` source is not checked against the filesystem."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('- "src/fixture.py"', '- "https://tankpit.com"')
        page = page.replace('  "src/fixture.py": "0123456789abcdef0123456789abcdef01234567"\n', "")
        page = page.replace("source_git_blobs:\n", "")
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        assert run_wiki_rules(tmp_path) == 0

    def test_anchor_outside_source_paths_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An anchored path must also be declared in ``source_paths``."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace('"src/fixture.py": ', '"src/other.py": ')
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "source_git_blobs key 'src/other.py' is not in source_paths" in out

    def test_malformed_anchor_hash_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A hash that is not 40 lowercase hex chars is rejected."""
        _green_tree(tmp_path)
        page = GREEN_FRONTMATTER.replace(
            "0123456789abcdef0123456789abcdef01234567", "not-a-real-hash"
        )
        (tmp_path / "wiki" / "pages" / "alpha.md").write_text(page, encoding="utf-8")
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "is not a 40-hex object id" in out


class TestNavigation:
    """Hub links must resolve and no page may be orphaned."""

    def test_hub_link_to_missing_page_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A hub linking a non-existent page is reported."""
        _build_wiki(
            tmp_path,
            pages={"alpha.md": GREEN_FRONTMATTER},
            hub_links={"things": ["alpha.md", "ghost.md"]},
            index_text="1 content pages\n\n[Things](hubs/things.md) -- x (2 pages)\n",
        )
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "things.md: links to missing page 'ghost.md'" in out

    def test_orphan_page_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A page no hub links to is reported as an orphan."""
        _build_wiki(
            tmp_path,
            pages={"alpha.md": GREEN_FRONTMATTER, "lonely.md": GREEN_FRONTMATTER},
            hub_links={"things": ["alpha.md"]},
            index_text="2 content pages\n\n[Things](hubs/things.md) -- x (1 pages)\n",
        )
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "lonely.md: orphan -- no hub links it" in out

    def test_page_linked_from_two_hubs_is_not_an_orphan(self, tmp_path: Path) -> None:
        """Polyhierarchy is legal — one page, two hubs, no violation."""
        _build_wiki(
            tmp_path,
            pages={"alpha.md": GREEN_FRONTMATTER},
            hub_links={"things": ["alpha.md"], "others": ["alpha.md"]},
            index_text=(
                "1 content pages\n\n"
                "[Things](hubs/things.md) -- x (1 pages)\n"
                "[Others](hubs/others.md) -- y (1 pages)\n"
            ),
        )
        assert run_wiki_rules(tmp_path) == 0


class TestCounts:
    """The index must not lie about how much it indexes."""

    def test_wrong_hub_count_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A hub count that disagrees with the hub's links is reported."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "index.md").write_text(
            "1 content pages\n\n[Things](hubs/things.md) -- x (7 pages)\n", encoding="utf-8"
        )
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "hub 'things' claims 7 pages, hub links 1" in out

    def test_wrong_total_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A total that disagrees with the page count is reported."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "index.md").write_text(
            "60 content pages\n\n[Things](hubs/things.md) -- x (1 pages)\n", encoding="utf-8"
        )
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "claims 60 content pages, 1 exist" in out

    def test_index_link_to_missing_hub_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An index row naming a hub file that is absent is reported."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "index.md").write_text(
            "1 content pages\n\n"
            "[Things](hubs/things.md) -- x (1 pages)\n"
            "[Gone](hubs/gone.md) -- y (3 pages)\n",
            encoding="utf-8",
        )
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "index.md: links to missing hub 'gone'" in out

    def test_absent_total_is_a_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An index with no ``N content pages`` phrase is reported."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "index.md").write_text(
            "Some wiki.\n\n[Things](hubs/things.md) -- x (1 pages)\n", encoding="utf-8"
        )
        count, out = _capture(tmp_path, capsys)
        assert count == 1
        assert "no 'N content pages' total found" in out

    def test_singular_page_wording_is_accepted(self, tmp_path: Path) -> None:
        """``(1 page)`` parses the same as ``(1 pages)``."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "index.md").write_text(
            "1 content pages\n\n[Things](hubs/things.md) -- x (1 page)\n", encoding="utf-8"
        )
        assert run_wiki_rules(tmp_path) == 0

    def test_non_hub_index_lines_are_ignored(self, tmp_path: Path) -> None:
        """Prose lines in the index are not parsed as hub rows."""
        _green_tree(tmp_path)
        (tmp_path / "wiki" / "index.md").write_text(
            "# Wiki\n\nRead this first. 1 content pages.\n\n"
            "[Things](hubs/things.md) -- x (1 pages)\n\n"
            "## How this works\n\nThree tiers.\n",
            encoding="utf-8",
        )
        assert run_wiki_rules(tmp_path) == 0


class TestRealRepository:
    """The rule must be green against the wiki it was written for."""

    def test_real_wiki_is_conformant(self) -> None:
        """The live ``wiki/`` tree passes every structural check."""
        project_root = Path(__file__).resolve().parents[2]
        assert run_wiki_rules(project_root) == 0
