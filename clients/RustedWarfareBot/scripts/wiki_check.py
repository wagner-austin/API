"""Gate the wiki's mechanical contract: sources resolve, links land, counts tell the truth.

The schema (``wiki/SCHEMA.md``) demands provenance and navigation the
guard never enforced here: the sibling client's guard carries wiki rules,
this one was lifted without them, and the gap was paid for twice by hand
-- the 2026-08-14 sweep found five citation defects and an orphan that
had accumulated silently, and the same script had to be re-run from a
transcript the next day. A discipline that lives in a session is not a
discipline ([[policy-loop]] says the same about run procedure).

Seven checks, all mechanical, all fatal: every page opens with
frontmatter carrying a title (the healing page shipped without any and
nothing noticed for two weeks -- an unpinnable page is unverifiable by
the schema's own terms), every ``source_paths`` entry
resolves (line anchors in bounds, URLs skipped), every ``source_git_blobs``
path exists, every ``[[wikilink]]`` lands on a page or hub, every footnote
is both used and defined, every page is linked from at least one hub, and
the index's per-hub and total page counts match reality. Blob DRIFT --
a pinned hash no longer matching the file -- is deliberately not gated:
that means "nobody re-read this page since that blob", and reddening the
build on it rewards bumping pins without re-reading.

Run through the target that owns it::

    make sources
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_BAD_USAGE = 2

_WIKILINK = re.compile(r"\[\[([a-z0-9-]+)\]\]")
_FOOTNOTE_USED = re.compile(r"\[\^(\d+)\](?!:)")
_FOOTNOTE_DEFINED = re.compile(r"^\[\^(\d+)\]:", re.MULTILINE)
_QUOTED_ENTRY = re.compile(r'-\s*"([^"]+)"')
_BLOB_PATH = re.compile(r'^\s+"([^"]+)":\s*"[0-9a-f]{40}"', re.MULTILINE)
_HUB_LINK = re.compile(r"\]\(\.\./pages/([a-z0-9-]+)\.md\)")
_INDEX_HUB = re.compile(r"\]\(hubs/([a-z0-9-]+)\.md\).*\((\d+) pages?\)")
_INDEX_TOTAL = re.compile(r"(\d+) content pages")


def _frontmatter(text: str) -> str:
    """Return the page's YAML frontmatter block, empty when absent."""
    matter = re.match(r"---\n(.*?)\n---", text, re.DOTALL)
    return "" if matter is None else matter.group(1)


def _source_entries(matter: str) -> tuple[str, ...]:
    """Return the quoted entries of the ``source_paths`` block."""
    block = re.search(r"^source_paths:\n((?:\s+-\s+.*\n)+)", matter + "\n", re.MULTILINE)
    if block is None:
        return ()
    return tuple(m.group(1) for m in _QUOTED_ENTRY.finditer(block.group(1)))


def _blob_paths(matter: str) -> tuple[str, ...]:
    """Return the paths pinned in the ``source_git_blobs`` block."""
    return tuple(m.group(1) for m in _BLOB_PATH.finditer(matter))


def _first_group_all(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    """Return every match's first group, in match order."""
    return tuple(m.group(1) for m in pattern.finditer(text))


def _check_sources(page: str, matter: str, root: Path) -> list[str]:
    """Verify every cited path resolves and every line anchor is in bounds."""
    found: list[str] = []
    for entry in [*_source_entries(matter), *_blob_paths(matter)]:
        if entry.startswith("http"):
            continue
        path_part, _, line_part = entry.partition(":")
        cited = root / path_part
        if not cited.exists():
            found.append(f"{page}: source path does not resolve: {path_part}")
            continue
        if line_part and line_part.isdigit():
            length = len(cited.read_text(encoding="utf-8", errors="ignore").splitlines())
            if int(line_part) > length:
                found.append(f"{page}: anchor {entry} is beyond the file's {length} lines")
    return found


def _check_links(page: str, text: str, slugs: frozenset[str]) -> list[str]:
    """Verify every wikilink lands and every footnote is used and defined."""
    found = [
        f"{page}: wikilink [[{slug}]] resolves to no page or hub"
        for slug in _first_group_all(_WIKILINK, text)
        if slug not in slugs
    ]
    used = set(_first_group_all(_FOOTNOTE_USED, text))
    defined = set(_first_group_all(_FOOTNOTE_DEFINED, text))
    found.extend(f"{page}: footnote [^{ref}] is used but never defined" for ref in used - defined)
    found.extend(f"{page}: footnote [^{ref}] is defined but never used" for ref in defined - used)
    return sorted(found)


def _check_navigation(root: Path, pages: list[Path], hubs: list[Path]) -> list[str]:
    """Verify hub reachability and the index's counts."""
    found: list[str] = []
    linked: dict[str, int] = {}
    reachable: set[str] = set()
    for hub in hubs:
        slugs = _first_group_all(_HUB_LINK, hub.read_text(encoding="utf-8"))
        linked[hub.stem] = len(slugs)
        reachable.update(slugs)
    found.extend(
        f"{page.name}: linked from no hub; readers navigating the index never reach it"
        for page in pages
        if page.stem not in reachable
    )
    index = (root / "wiki" / "index.md").read_text(encoding="utf-8")
    for hub_slug, stated_text in _index_hub_counts(index):
        stated = int(stated_text)
        actual = linked.get(hub_slug, 0)
        if stated != actual:
            found.append(f"index.md: hub {hub_slug} states {stated} pages, links {actual}")
    total = _index_total(index)
    if total is None:
        found.append("index.md: no content-page total found")
    elif int(total) != len(pages):
        found.append(f"index.md: total states {total} pages, {len(pages)} exist")
    return sorted(found)


def _index_hub_counts(index: str) -> tuple[tuple[str, str], ...]:
    """Return each index hub line's slug and stated page count."""
    return tuple((m.group(1), m.group(2)) for m in _INDEX_HUB.finditer(index))


def _index_total(index: str) -> str | None:
    """Return the index's stated content-page total, None when absent."""
    total = _INDEX_TOTAL.search(index)
    return None if total is None else total.group(1)


def run_checks(root: Path) -> tuple[str, ...]:
    """Run every check over the wiki tree.

    Args:
        root: The client directory holding ``wiki/``.

    Returns:
        One line per violation, empty when the contract holds.

    Raises:
        OSError: When a wiki file cannot be read.
    """
    pages = sorted((root / "wiki" / "pages").glob("*.md"))
    hubs = sorted((root / "wiki" / "hubs").glob("*.md"))
    slugs = frozenset(p.stem for p in [*pages, *hubs])
    found: list[str] = []
    for page in pages:
        text = page.read_text(encoding="utf-8")
        matter = _frontmatter(text)
        if "title:" not in matter:
            found.append(f"{page.name}: no frontmatter title; the page is unpinnable")
        found.extend(_check_sources(page.name, matter, root))
        found.extend(_check_links(page.name, text, slugs))
    found.extend(_check_navigation(root, pages, hubs))
    return tuple(found)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    """Check the wiki and report.

    Args:
        argv: No arguments are accepted; present for the entry-point shape.
            ``None`` reads ``sys.argv[1:]``.
        root: The client directory, injectable for tests. ``None`` uses
            the working directory.

    Returns:
        ``EXIT_OK`` when the contract holds, ``EXIT_VIOLATIONS`` with one
        line per violation when it does not, ``EXIT_BAD_USAGE`` on any
        argument.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if args:
        sys.stdout.write("usage: wiki_check\n")
        return EXIT_BAD_USAGE
    base = root if root is not None else Path()
    found = run_checks(base)
    for line in found:
        sys.stdout.write(f"{line}\n")
    pages = len(list((base / "wiki" / "pages").glob("*.md")))
    sys.stdout.write(f"[sources] {len(found)} violation(s) across {pages} pages\n")
    return EXIT_OK if not found else EXIT_VIOLATIONS


if __name__ == "__main__":
    raise SystemExit(main(None))
