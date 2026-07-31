"""Report which wiki pages have drifted from the trees they were audited against.

``source_git_blobs`` records the git tree or blob a page was last
audited against. A lagging anchor is NOT a defect -- it means "nobody
has re-read this page since that tree" -- so the ``wiki-structure``
guard rule deliberately does not gate on it (see
``wiki/SCHEMA.md`` and ``scripts/wiki_rules.py``). Gating would redden
the build on every source commit and would reward bumping anchors
without re-reading, which is the exact laundering the wiki's audit log
refuses.

This is the other half of that decision: a REPORT, not a gate. It
answers "which pages are owed an audit, and how big is the gap?" so
the triage that used to be an ad-hoc shell loop is reproducible.

Exit code is 0 whether or not drift exists -- ``make check`` must never
depend on this. Use ``--exit-code`` to opt into a nonzero exit for a
scripted audit sweep.

Usage::

    tankpit-wiki-anchors                 # table of every stale anchor
    tankpit-wiki-anchors --all           # include current anchors too
    tankpit-wiki-anchors --exit-code     # exit 1 when any anchor is stale
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from scripts import _test_hooks as script_hooks
from scripts.wiki_rules import parse_page_frontmatter
from tankpit_bot import _test_hooks as core_hooks

STATUS_STALE = "STALE"
STATUS_CURRENT = "CURRENT"
STATUS_UNRESOLVED = "UNRESOLVED"

_USAGE = (
    "usage: tankpit-wiki-anchors [--all] [--exit-code]\n"
    "\n"
    "  --all         list current anchors as well as stale ones\n"
    "  --exit-code   exit 1 when any anchor is stale (default: always 0)\n"
)


class AnchorStatus(TypedDict):
    """One page's anchor against the tree it names.

    Attributes:
        page: Wiki page filename.
        path: Repo-relative path the anchor covers.
        recorded: Object id recorded in the page's frontmatter.
        current: Object id HEAD resolves for that path, or "" when git
            cannot resolve it.
        status: ``STALE``, ``CURRENT``, or ``UNRESOLVED``.
        fact_checked: The page's ``fact_checked`` date, for triage order.
    """

    page: str
    path: str
    recorded: str
    current: str
    status: str
    fact_checked: str


def _classify(recorded: str, current: str | None) -> str:
    """Decide one anchor's status.

    Args:
        recorded: Object id from the page's frontmatter.
        current: Object id HEAD resolves, or None when unresolvable.

    Returns:
        One of :data:`STATUS_STALE`, :data:`STATUS_CURRENT`, or
        :data:`STATUS_UNRESOLVED`.
    """
    if current is None:
        return STATUS_UNRESOLVED
    if current == recorded:
        return STATUS_CURRENT
    return STATUS_STALE


def collect_anchor_statuses(project_root: Path) -> list[AnchorStatus]:
    """Resolve every wiki anchor against the current HEAD.

    Args:
        project_root: Project root containing ``wiki/pages``.

    Returns:
        One entry per ``source_git_blobs`` entry across all pages,
        sorted by page name then path. Empty when there is no wiki.
    """
    pages_dir = project_root / "wiki" / "pages"
    if not pages_dir.is_dir():
        return []
    statuses: list[AnchorStatus] = []
    for page_path in sorted(pages_dir.glob("*.md")):
        matter = parse_page_frontmatter(page_path)
        if matter is None:
            continue
        fact_checked = matter["scalars"].get("fact_checked", "")
        anchors = matter["maps"].get("source_git_blobs", {})
        for anchored_path in sorted(anchors):
            recorded = anchors[anchored_path]
            current = script_hooks.resolve_tree_hash(project_root, anchored_path)
            statuses.append(
                AnchorStatus(
                    page=page_path.name,
                    path=anchored_path,
                    recorded=recorded,
                    current=current if current is not None else "",
                    status=_classify(recorded, current),
                    fact_checked=fact_checked,
                )
            )
    return statuses


def format_report(statuses: list[AnchorStatus], *, show_all: bool) -> list[str]:
    """Render the anchor statuses as printable lines.

    Args:
        statuses: Anchor statuses to render.
        show_all: Include ``CURRENT`` rows as well as drifted ones.

    Returns:
        Lines to print, including a trailing summary line.
    """
    shown = [s for s in statuses if show_all or s["status"] != STATUS_CURRENT]
    lines: list[str] = []
    for status in shown:
        lines.append(
            f"{status['status']:<10} {status['page']:<34} "
            f"{status['path']:<34} audited {status['fact_checked'] or '(undated)'}"
        )
    stale = sum(1 for s in statuses if s["status"] == STATUS_STALE)
    unresolved = sum(1 for s in statuses if s["status"] == STATUS_UNRESOLVED)
    current = sum(1 for s in statuses if s["status"] == STATUS_CURRENT)
    lines.append(
        f"{len(statuses)} anchors: {stale} stale, {unresolved} unresolved, {current} current"
    )
    if stale > 0:
        lines.append(
            "A stale anchor means the page is owed an AUDIT, not a bump: "
            "re-read the page against the tree, correct what drifted, then move the anchor."
        )
    return lines


def run_report(project_root: Path, *, show_all: bool) -> int:
    """Print the anchor-drift report.

    Args:
        project_root: Project root containing ``wiki/pages``.
        show_all: Include current anchors in the listing.

    Returns:
        Number of stale anchors found.
    """
    statuses = collect_anchor_statuses(project_root)
    for line in format_report(statuses, show_all=show_all):
        sys.stdout.write(f"{line}\n")
    return sum(1 for s in statuses if s["status"] == STATUS_STALE)


def main() -> None:
    """Entry point for the ``tankpit-wiki-anchors`` command.

    Raises:
        SystemExit: On ``--help``, unrecognized arguments, or when
            ``--exit-code`` is set and any anchor is stale.
    """
    # ``get_argv`` returns the full ``sys.argv``; drop the program name
    # the same way ``scripts/analyze_session_timing.py`` does.
    full_argv = list(core_hooks.get_argv())
    show_all = False
    use_exit_code = False
    for token in full_argv[1:]:
        if token == "--all":
            show_all = True
        elif token == "--exit-code":
            use_exit_code = True
        else:
            sys.stdout.write(_USAGE)
            raise SystemExit(0 if token in ("--help", "-h") else 2)
    stale = run_report(Path.cwd(), show_all=show_all)
    if use_exit_code and stale > 0:
        raise SystemExit(1)


__all__ = [
    "STATUS_CURRENT",
    "STATUS_STALE",
    "STATUS_UNRESOLVED",
    "AnchorStatus",
    "collect_anchor_statuses",
    "format_report",
    "main",
    "run_report",
]
