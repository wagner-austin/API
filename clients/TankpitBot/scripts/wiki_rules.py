"""Guard rule: the wiki's structure must match what it claims.

The wiki is the project's source of truth, but nothing in the gate
checked its own bookkeeping until this rule. Two 2026-07-31 audits
found index counts off by seven, a hub advertising twelve pages while
listing thirteen, frontmatter naming deleted source files, and a
hub-linked page carrying no frontmatter at all -- every one of them
invisible to ``make check``.

This rule closes the mechanically-checkable half of that gap:

* **Frontmatter** -- every page opens with a parseable block carrying
  the SCHEMA's required keys, a real ``YYYY-MM-DD`` ``fact_checked``
  date, and a known ``confidence`` level.
* **Provenance** -- every ``source_paths`` entry still exists on disk
  (a trailing ``:line`` or ``:start-end`` locator is stripped first,
  and ``http(s)://`` sources are skipped), and every
  ``source_git_blobs`` key is one of them with a well-formed 40-hex
  object id (catching typo'd or invented anchors).
* **Navigation** -- every hub inclusion link resolves, and every page
  is reachable from at least one hub (SCHEMA's orphan ban).
* **Counts** -- ``index.md``'s per-hub page counts equal each hub's
  actual link count, and its total equals the number of content pages.

Deliberately NOT checked: whether a ``source_git_blobs`` hash equals
the tree's current hash. A lagging anchor is not a defect -- it is the
marker for "this page has not been audited since that tree", and the
honest fix is an audit, not a bump. Gating on hash equality would turn
the gate red on every ``src/`` commit and would reward bumping the
anchor without re-reading the page, which is the exact failure this
wiki's log calls out. Drift is a report, not a gate.

A tree without a ``wiki/pages`` directory is out of scope (mirrors
``contract_rules`` skipping absent packages), so guard runs against
synthetic test trees stay green.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from platform_core.logging import get_logger

REQUIRED_FRONTMATTER_KEYS = ("title", "tags", "related", "fact_checked", "confidence")
"""Keys every content page must carry, per ``wiki/SCHEMA.md``."""

CONFIDENCE_VALUES = ("high", "medium", "low")
"""The only ``confidence`` levels ``wiki/SCHEMA.md`` defines."""

FRONTMATTER_FENCE = "---"
FACT_CHECKED_FORMAT = "%Y-%m-%d"
URL_PREFIXES = ("http://", "https://")

_BLOB_HASH = re.compile(r"^[0-9a-f]{40}$")
_LINE_LOCATOR = re.compile(r":\d+(-\d+)?$")
_HUB_LINK = re.compile(r"^\[[^\]]+\]\(\.\./pages/([a-z0-9-]+\.md)\)")
_INDEX_HUB = re.compile(r"^\[[^\]]+\]\(hubs/([a-z0-9-]+)\.md\).*?\((\d+) pages?")
_INDEX_TOTAL = re.compile(r"(\d+) content pages")
_SCALAR_LINE = re.compile(r"^([a-z_]+):\s*(.*)$")
_MAP_ENTRY_LINE = re.compile(r'^"([^"]+)":\s*"([^"]*)"$')

_LOGGER = get_logger(__name__)


class ParsedFrontmatter(TypedDict):
    """One page's frontmatter, split by value shape.

    The wiki's frontmatter uses three YAML shapes and no others, so a
    line-oriented parser covers it without adding a YAML dependency
    (the same reasoning that made physics claim blocks fenced JSON).

    Attributes:
        scalars: ``key: value`` pairs, quotes stripped.
        lists: ``key:`` followed by ``- item`` lines, or an inline
            ``[a, b]`` flow sequence.
        maps: ``key:`` followed by indented ``"name": "value"`` lines.
    """

    scalars: dict[str, str]
    lists: dict[str, list[str]]
    maps: dict[str, dict[str, str]]


def _strip_quotes(value: str) -> str:
    """Remove one layer of surrounding quotes from a scalar.

    Args:
        value: Raw scalar text.

    Returns:
        The value without a matching pair of surrounding quotes.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _parse_flow_sequence(value: str) -> list[str]:
    """Parse an inline ``[a, b, c]`` YAML flow sequence.

    Args:
        value: Text beginning with ``[`` and ending with ``]``.

    Returns:
        The sequence items, quote-stripped, empties dropped.
    """
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [_strip_quotes(item.strip()) for item in inner.split(",") if item.strip()]


def _parse_frontmatter(text: str, page: str) -> tuple[ParsedFrontmatter | None, list[str]]:
    """Parse one page's frontmatter block.

    Args:
        text: Full markdown text of the page.
        page: Page name for violation messages.

    Returns:
        Pair of (parsed frontmatter or None, violations). ``None``
        means the block was absent or unterminated and no per-key
        checks can run.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return None, [f"{page}: no frontmatter block (SCHEMA requires one on every page)"]
    scalars: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    maps: dict[str, dict[str, str]] = {}
    current_key = ""
    for line in lines[1:]:
        if line.strip() == FRONTMATTER_FENCE:
            return ParsedFrontmatter(scalars=scalars, lists=lists, maps=maps), []
        current_key = _absorb_line(line, current_key, scalars, lists, maps)
    return None, [f"{page}: unclosed frontmatter block"]


def _absorb_line(
    line: str,
    current_key: str,
    scalars: dict[str, str],
    lists: dict[str, list[str]],
    maps: dict[str, dict[str, str]],
) -> str:
    """Fold one frontmatter line into the accumulating containers.

    Args:
        line: Raw frontmatter line.
        current_key: Key most recently opened by a bare ``key:``.
        scalars: Scalar accumulator, mutated in place.
        lists: Sequence accumulator, mutated in place.
        maps: Mapping accumulator, mutated in place.

    Returns:
        The key that remains open after this line.
    """
    stripped = line.strip()
    if not stripped:
        return current_key
    if stripped.startswith("- ") and current_key:
        lists.setdefault(current_key, []).append(_strip_quotes(stripped[2:].strip()))
        return current_key
    map_entry = _MAP_ENTRY_LINE.match(stripped)
    if map_entry is not None and current_key:
        entry_key: str = map_entry.group(1)
        entry_value: str = map_entry.group(2)
        maps.setdefault(current_key, {})[entry_key] = entry_value
        return current_key
    scalar = _SCALAR_LINE.match(stripped)
    if scalar is None:
        return current_key
    key: str = scalar.group(1)
    value: str = scalar.group(2)
    raw_value = value.strip()
    if not raw_value:
        return key
    if raw_value.startswith("[") and raw_value.endswith("]"):
        lists[key] = _parse_flow_sequence(raw_value)
        return ""
    scalars[key] = _strip_quotes(raw_value)
    return ""


def parse_page_frontmatter(page_path: Path) -> ParsedFrontmatter | None:
    """Read and parse one wiki page's frontmatter.

    The public entry point onto this module's parser, so tools that need
    frontmatter (notably the anchor-drift report in
    ``scripts/wiki_anchors.py``) reuse it instead of restating the
    grammar.

    Args:
        page_path: Path to the wiki content page.

    Returns:
        Parsed frontmatter, or None when the page has no parseable
        block (which the guard rule reports separately).
    """
    matter, _violations = _parse_frontmatter(page_path.read_text(encoding="utf-8"), page_path.name)
    return matter


def _frontmatter_violations(matter: ParsedFrontmatter, page: str) -> list[str]:
    """Check required keys, the date format, and the confidence level.

    Args:
        matter: Parsed frontmatter for the page.
        page: Page name for violation messages.

    Returns:
        Violations found on this page's frontmatter.
    """
    violations: list[str] = []
    present = set(matter["scalars"]) | set(matter["lists"]) | set(matter["maps"])
    for key in REQUIRED_FRONTMATTER_KEYS:
        if key not in present:
            violations.append(f"{page}: frontmatter missing required key '{key}'")
    fact_checked = matter["scalars"].get("fact_checked")
    if fact_checked is not None:
        try:
            datetime.strptime(fact_checked, FACT_CHECKED_FORMAT)
        except ValueError as error:
            _LOGGER.warning(
                "wiki_fact_checked_unparseable page=%s value=%s error=%s",
                page,
                fact_checked,
                error,
            )
            violations.append(f"{page}: fact_checked '{fact_checked}' is not YYYY-MM-DD")
    confidence = matter["scalars"].get("confidence")
    if confidence is not None and confidence not in CONFIDENCE_VALUES:
        violations.append(f"{page}: confidence '{confidence}' not one of {list(CONFIDENCE_VALUES)}")
    return violations


def _provenance_violations(
    matter: ParsedFrontmatter,
    page: str,
    project_root: Path,
) -> list[str]:
    """Check that declared sources exist and anchors are well-formed.

    Args:
        matter: Parsed frontmatter for the page.
        page: Page name for violation messages.
        project_root: Project root the source paths are relative to.

    Returns:
        Violations for vanished source paths and malformed anchors.
    """
    violations: list[str] = []
    source_paths = matter["lists"].get("source_paths", [])
    blob_pins = matter["maps"].get("source_git_blobs", {})
    for source in source_paths:
        if source.startswith(URL_PREFIXES):
            continue
        target = _LINE_LOCATOR.sub("", source)
        if (project_root / target).exists():
            continue
        if source in blob_pins:
            # A RETIRED source: the file left the working tree but its
            # exact content stays addressable by the pinned blob id
            # (``git cat-file blob <hash>``), which is what the pin was
            # recorded for. Deleting a one-shot measurement script must
            # not orphan the page that cites it — the pin IS the
            # provenance. First case: the analysis_scripts retirement,
            # 2026-08-17 (board task f0c3a532).
            continue
        violations.append(f"{page}: source_paths entry '{source}' does not exist")
    for anchored, blob in blob_pins.items():
        if anchored not in source_paths:
            violations.append(f"{page}: source_git_blobs key '{anchored}' is not in source_paths")
        if _BLOB_HASH.match(blob) is None:
            violations.append(f"{page}: source_git_blobs['{anchored}'] is not a 40-hex object id")
    return violations


def _hub_links(hub_path: Path) -> list[str]:
    """Extract the page filenames one hub links to, in order.

    Args:
        hub_path: Path to the hub markdown file.

    Returns:
        Page filenames (e.g. ``module-map.md``) this hub links to.
    """
    targets: list[str] = []
    for line in hub_path.read_text(encoding="utf-8").splitlines():
        match = _HUB_LINK.match(line.strip())
        if match is not None:
            target: str = match.group(1)
            targets.append(target)
    return targets


def _navigation_violations(
    hubs_dir: Path,
    pages_dir: Path,
) -> tuple[dict[str, int], list[str]]:
    """Check hub links resolve and no content page is orphaned.

    Args:
        hubs_dir: Directory holding the hub pages.
        pages_dir: Directory holding the content pages.

    Returns:
        Pair of (hub stem -> link count, violations).
    """
    violations: list[str] = []
    counts: dict[str, int] = {}
    linked: set[str] = set()
    for hub_path in sorted(hubs_dir.glob("*.md")):
        targets = _hub_links(hub_path)
        counts[hub_path.stem] = len(targets)
        for target in targets:
            if not (pages_dir / target).is_file():
                violations.append(f"{hub_path.name}: links to missing page '{target}'")
            else:
                linked.add(target)
    for page_path in sorted(pages_dir.glob("*.md")):
        if page_path.name not in linked:
            violations.append(
                f"{page_path.name}: orphan -- no hub links it (SCHEMA hub-link discipline)"
            )
    return counts, violations


def _count_violations(
    index_path: Path,
    hub_counts: dict[str, int],
    page_total: int,
) -> list[str]:
    """Check the index's advertised counts against reality.

    Args:
        index_path: Path to ``wiki/index.md``.
        hub_counts: Hub stem -> actual number of links in that hub.
        page_total: Actual number of content pages on disk.

    Returns:
        Violations for every count the index states incorrectly.
    """
    violations: list[str] = []
    text = index_path.read_text(encoding="utf-8")
    seen_total = False
    for line in text.splitlines():
        stripped = line.strip()
        hub_match = _INDEX_HUB.match(stripped)
        if hub_match is None:
            continue
        stem: str = hub_match.group(1)
        claimed_text: str = hub_match.group(2)
        claimed = int(claimed_text)
        actual = hub_counts.get(stem)
        if actual is None:
            violations.append(f"index.md: links to missing hub '{stem}'")
        elif actual != claimed:
            violations.append(f"index.md: hub '{stem}' claims {claimed} pages, hub links {actual}")
    total_match = _INDEX_TOTAL.search(text)
    if total_match is not None:
        seen_total = True
        total_text: str = total_match.group(1)
        claimed_total = int(total_text)
        if claimed_total != page_total:
            violations.append(f"index.md: claims {claimed_total} content pages, {page_total} exist")
    if not seen_total:
        violations.append("index.md: no 'N content pages' total found")
    return violations


def run_wiki_rules(project_root: Path) -> int:
    """Run the wiki structure rule over a project tree.

    Args:
        project_root: Project root containing ``wiki/pages``.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    wiki_dir = project_root / "wiki"
    pages_dir = wiki_dir / "pages"
    hubs_dir = wiki_dir / "hubs"
    if not pages_dir.is_dir():
        return 0
    violations: list[str] = []
    page_paths = sorted(pages_dir.glob("*.md"))
    for page_path in page_paths:
        matter, parse_violations = _parse_frontmatter(
            page_path.read_text(encoding="utf-8"), page_path.name
        )
        violations.extend(parse_violations)
        if matter is not None:
            violations.extend(_frontmatter_violations(matter, page_path.name))
            violations.extend(_provenance_violations(matter, page_path.name, project_root))
    if hubs_dir.is_dir():
        hub_counts, nav_violations = _navigation_violations(hubs_dir, pages_dir)
        violations.extend(nav_violations)
        index_path = wiki_dir / "index.md"
        if index_path.is_file():
            violations.extend(_count_violations(index_path, hub_counts, len(page_paths)))
    for violation in violations:
        sys.stdout.write(f"wiki_structure_violation {violation}\n")
    return len(violations)


__all__ = [
    "CONFIDENCE_VALUES",
    "REQUIRED_FRONTMATTER_KEYS",
    "ParsedFrontmatter",
    "parse_page_frontmatter",
    "run_wiki_rules",
]
