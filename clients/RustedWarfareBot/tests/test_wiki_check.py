"""The wiki gate, driven over fixture trees that break every rule once.

The checker's own first run against the real tree caught the hand audit
overcounting every hub by the format-comment line, so the fixtures here
include exactly that shape: a hub comment that LOOKS like an inclusion
link but carries no real slug.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.wiki_check import EXIT_BAD_USAGE, EXIT_OK, EXIT_VIOLATIONS, main, run_checks


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clean_tree(root: Path) -> None:
    """A minimal wiki that satisfies every rule."""
    _write(root, "cited.txt", "one\ntwo\nthree\n")
    _write(
        root,
        "wiki/pages/alpha.md",
        '---\ntitle: "Alpha"\nsource_paths:\n  - "cited.txt:2"\n'
        '  - "https://example.invalid/doc"\nsource_git_blobs:\n'
        '  "cited.txt": "0000000000000000000000000000000000000000"\n---\n'
        "# Alpha\n\nLinks to [[beta]].[^1]\n\n[^1]: `cited.txt:2` -- the claim.\n",
    )
    _write(root, "wiki/pages/beta.md", '---\ntitle: "Beta"\n---\n# Beta\n\nPlain.\n')
    _write(
        root,
        "wiki/hubs/topic.md",
        "# Topic\n\n[Alpha](../pages/alpha.md) -- one\n[Beta](../pages/beta.md) -- two\n"
        "<!-- Format: [Title](../pages/<slug>.md) -- description -->\n",
    )
    _write(
        root,
        "wiki/index.md",
        "# Wiki\n\n2 content pages.\n\n[Topic](hubs/topic.md) -- things (2 pages)\n",
    )


def test_a_clean_tree_passes_with_a_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _clean_tree(tmp_path)
    assert main([], root=tmp_path) == EXIT_OK
    assert capsys.readouterr().out == "[sources] 0 violation(s) across 2 pages\n"


def test_every_rule_fires_once_on_the_broken_tree(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """One violation per rule, asserted verbatim: a missing source path, an
    out-of-bounds anchor, a missing blob path, a dangling wikilink, both
    footnote directions, an orphan page, and all three index defects. A
    range anchor and a URL are present and must NOT fire."""
    _clean_tree(tmp_path)
    _write(
        tmp_path,
        "wiki/pages/gamma.md",
        '---\ntitle: "Gamma"\nsource_paths:\n  - "missing.txt"\n  - "cited.txt:9"\n'
        '  - "cited.txt:1-3"\nsource_git_blobs:\n'
        '  "gone.py": "1111111111111111111111111111111111111111"\n---\n'
        "# Gamma\n\nSee [[nowhere]].[^1] Also plain.[^2]\n\n[^2]: `cited.txt:1` -- fine.\n"
        "[^3]: `cited.txt:1` -- unreferenced.\n",
    )
    _write(
        tmp_path,
        "wiki/index.md",
        "# Wiki\n\n9 content pages.\n\n[Topic](hubs/topic.md) -- things (5 pages)\n"
        "[Ghost](hubs/ghost.md) -- absent (1 page)\n",
    )
    assert main([], root=tmp_path) == EXIT_VIOLATIONS
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "gamma.md: source path does not resolve: missing.txt",
        "gamma.md: anchor cited.txt:9 is beyond the file's 3 lines",
        "gamma.md: source path does not resolve: gone.py",
        "gamma.md: footnote [^1] is used but never defined",
        "gamma.md: footnote [^3] is defined but never used",
        "gamma.md: wikilink [[nowhere]] resolves to no page or hub",
        "gamma.md: linked from no hub; readers navigating the index never reach it",
        "index.md: hub ghost states 1 pages, links 0",
        "index.md: hub topic states 5 pages, links 2",
        "index.md: total states 9 pages, 3 exist",
        "[sources] 10 violation(s) across 3 pages",
    ]


def test_an_index_without_a_total_is_a_violation(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "wiki/index.md", "# Wiki\n\n[Topic](hubs/topic.md) -- things (2 pages)\n")
    found = run_checks(tmp_path)
    assert found == ("index.md: no content-page total found",)


def test_a_page_without_frontmatter_is_checked_for_links_only(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "wiki/pages/beta.md", "# Beta\n\nNo frontmatter, links to [[alpha]].\n")
    assert run_checks(tmp_path) == ()


def test_a_bad_argument_count_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--verbose"]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: wiki_check")


def test_the_module_entry_point_exits_with_the_check_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.wiki_check")
    sys.argv = ["wiki_check", "extra"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.wiki_check", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.wiki_check"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: wiki_check")


def test_the_real_wiki_passes_the_gate() -> None:
    """The gate's own dogfood: the tree this repo ships must hold the
    contract, and this test failing alongside make sources is the point."""
    assert run_checks(Path(__file__).resolve().parents[1]) == ()
