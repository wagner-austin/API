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
from scripts.wiki_check import (
    ARTIFACT_ROOTS,
    EXIT_BAD_USAGE,
    EXIT_OK,
    EXIT_VIOLATIONS,
    Report,
    absent_artifact_roots,
    main,
    run_checks,
)

NOT_APPLICABLE = (
    "artifact tier not applicable: runs/, .game/, .decompiled/ absent on this machine, "
    "{n} artifact citation(s) not checked"
)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _hold_the_store(root: Path) -> None:
    """Make every artifact root a directory, the way only the workstation has them."""
    for name in ARTIFACT_ROOTS:
        (root / name).mkdir(exist_ok=True)


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
    assert capsys.readouterr().out == (
        f"[sources] 0 violation(s) across 2 pages; {NOT_APPLICABLE.format(n=0)}\n"
    )


def test_a_missing_artifact_path_is_the_artifact_tiers_finding_alone(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The two-tier split (SCHEMA.md, 2026-09-07): a `runs/` citation that
    does not resolve is invisible to the repo tier -- a fresh clone holds no
    measurement record, honestly, and the summary counts the citation it did
    not check -- and fatal to the artifact tier, which runs on the machine
    that holds the store (every artifact root a directory, 2026-09-20)."""
    _clean_tree(tmp_path)
    _write(
        tmp_path,
        "wiki/pages/beta.md",
        '---\ntitle: "Beta"\nsource_paths:\n  - "runs/sweeps/demo/absent.txt"\n---\n'
        "# Beta\n\nPlain.\n",
    )
    assert run_checks(tmp_path) == Report(
        violations=(), pages=2, artifact_citations=1, absent_roots=ARTIFACT_ROOTS
    )
    assert main([], root=tmp_path) == EXIT_OK
    assert capsys.readouterr().out == (
        f"[sources] 0 violation(s) across 2 pages; {NOT_APPLICABLE.format(n=1)}\n"
    )

    _hold_the_store(tmp_path)

    assert run_checks(tmp_path) == Report(
        violations=("beta.md: source path does not resolve: runs/sweeps/demo/absent.txt",),
        pages=2,
        artifact_citations=1,
        absent_roots=(),
    )
    assert main([], root=tmp_path) == EXIT_VIOLATIONS
    assert capsys.readouterr().out == (
        "beta.md: source path does not resolve: runs/sweeps/demo/absent.txt\n"
        "[sources] 1 violation(s) across 2 pages; artifact tier: 1 citation(s) checked "
        "under runs/, .game/, .decompiled/\n"
    )


def test_a_partial_store_is_not_the_store(tmp_path: Path) -> None:
    """`runs/` alone is what the test suite leaves behind on any machine and
    `.game/` alone is what CI links in, so neither turns the artifact tier on;
    the summary names exactly which roots are missing."""
    _clean_tree(tmp_path)
    _write(
        tmp_path,
        "wiki/pages/beta.md",
        '---\ntitle: "Beta"\nsource_paths:\n  - ".game/fallback64.bat"\n---\n# Beta\n\nPlain.\n',
    )
    (tmp_path / "runs").mkdir()
    (tmp_path / ".game").mkdir()

    assert absent_artifact_roots(tmp_path) == (".decompiled/",)
    report = run_checks(tmp_path)
    assert report == Report(
        violations=(), pages=2, artifact_citations=1, absent_roots=(".decompiled/",)
    )
    assert report.summary() == (
        "[sources] 0 violation(s) across 2 pages; artifact tier not applicable: "
        ".decompiled/ absent on this machine, 1 artifact citation(s) not checked"
    )


def test_an_artifact_path_that_resolves_passes_both_tiers(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "runs/sweeps/demo/present.txt", "a scorecard\n")
    _write(
        tmp_path,
        "wiki/pages/beta.md",
        '---\ntitle: "Beta"\nsource_paths:\n  - "runs/sweeps/demo/present.txt"\n---\n'
        "# Beta\n\nPlain.\n",
    )
    assert run_checks(tmp_path).violations == ()
    _hold_the_store(tmp_path)
    assert run_checks(tmp_path) == Report(
        violations=(), pages=2, artifact_citations=1, absent_roots=()
    )
    assert main([], root=tmp_path) == EXIT_OK
    assert capsys.readouterr().out == (
        "[sources] 0 violation(s) across 2 pages; artifact tier: 1 citation(s) checked "
        "under runs/, .game/, .decompiled/\n"
    )


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
        f"[sources] 10 violation(s) across 3 pages; {NOT_APPLICABLE.format(n=0)}",
    ]


def test_an_index_without_a_total_is_a_violation(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "wiki/index.md", "# Wiki\n\n[Topic](hubs/topic.md) -- things (2 pages)\n")
    assert run_checks(tmp_path).violations == ("index.md: no content-page total found",)


def test_a_page_without_frontmatter_is_unpinnable(tmp_path: Path) -> None:
    """The healing page shipped bare and nothing noticed for two weeks: a
    page with no frontmatter has no title, no sources, and no build pin,
    which the schema's own terms call unverifiable -- so it is now a
    violation rather than a silent pass."""
    _clean_tree(tmp_path)
    _write(tmp_path, "wiki/pages/beta.md", "# Beta\n\nNo frontmatter, links to [[alpha]].\n")
    assert run_checks(tmp_path).violations == (
        "beta.md: no frontmatter title; the page is unpinnable",
    )


def test_a_bad_argument_count_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """Any argument is refused, including the `--artifacts` flag the Makefile
    passed until 2026-09-20: the machine decides the tier now, and a flag
    that could ask for the tier on a machine without the store would put the
    gate back to failing everywhere but the workstation."""
    assert main(["--artifacts"]) == EXIT_BAD_USAGE
    assert capsys.readouterr().out == "usage: wiki_check\n"


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
    client = Path(__file__).resolve().parents[1]
    report = run_checks(client)
    assert report.violations == ()
    assert report.pages == len(list((client / "wiki" / "pages").glob("*.md")))
