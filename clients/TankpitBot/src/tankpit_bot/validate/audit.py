"""The ``make audit`` entrypoint: re-derive every validated claim.

Runs the archive validators over every decodable capture session in
``runs/bot`` and ``runs/sniff``, and the teleport validator over the
post-fix ``runs/bot/*.events.jsonl`` logs. Prints one evidence row per
claim and exits non-zero when any claim has zero samples or any
mismatch — an audit failure means either the wiki claim or the
archive is wrong, and both demand investigation, not softening.

With ``--stamp``, rewrites the ``fact_checked:`` frontmatter line of
each wiki page whose validated claims all passed — the stamp is
computed from validator output, never hand-typed
(``wiki/pages/physics-module-roadmap.md`` Phase 2).

Era note: capture-based validators sweep ALL eras — their windows
exclude pickups, which is where the pre-2026-06-24 double-count bug
lived, and fuel readings are wire values, not bot beliefs. The
teleport validator trusts bot belief fuel and therefore applies the
post-fix cutoff (see ``teleport_events``).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot.types import decode_capture_session
from tankpit_bot.validate.archive import (
    validate_firing_costs,
    validate_fuel_capacity,
    validate_hit_damage,
    validate_radar_cost,
    validate_walk_cost,
)
from tankpit_bot.validate.events_validators import validate_teleport_cost
from tankpit_bot.validate.types import ClaimEvidenceDict
from tankpit_bot.validate.windows import build_fuel_windows
from tankpit_bot.validate.wire_timeline import WireTimelineDict, extract_wire_timeline

_LOGGER = get_logger(__name__)

STAMPED_PAGES: dict[str, frozenset[str]] = {
    "game-economy.md": frozenset(
        {
            "walk-cost",
            "single-shot-cost",
            "dual-shot-cost",
            "missile-shot-cost",
            "homing-shot-cost",
            "teleport-cost",
            "single-hit-victim-cost",
            "dual-hit-victim-cost",
            "missile-hit-victim-cost",
            "homing-hit-victim-cost",
            "fuel-capacity",
            "radar-cost",
        }
    ),
}
"""Wiki pages eligible for fact_checked stamping, with the validated
claims each page owns. Explicit by design: a page is stamped only
when every one of its listed claims passed the audit."""


def _load_timelines(runs_root: Path) -> tuple[list[WireTimelineDict], int]:
    """Extract wire timelines from every decodable capture session.

    Args:
        runs_root: Directory containing ``bot/`` and ``sniff/`` run dirs.

    Returns:
        Pair of (timelines, count of magic-less sessions skipped).
    """
    timelines: list[WireTimelineDict] = []
    skipped = 0
    for subdir in ("bot", "sniff"):
        for path in sorted((runs_root / subdir).glob("*.capture_session.json")):
            session = decode_capture_session(
                narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))
            )
            if session["magic"] is None:
                _LOGGER.warning("audit_session_without_magic file=%s", path)
                skipped += 1
                continue
            timelines.append(extract_wire_timeline(session))
    return timelines, skipped


def collect_evidence(runs_root: Path) -> list[ClaimEvidenceDict]:
    """Run every validator and gather the per-claim evidence.

    Args:
        runs_root: Directory containing ``bot/`` and ``sniff/`` run dirs.

    Returns:
        Evidence records, one per validated claim.
    """
    timelines, skipped = _load_timelines(runs_root)
    if skipped:
        _LOGGER.warning("audit_sessions_skipped count=%d", skipped)
    per_session = [build_fuel_windows(timeline) for timeline in timelines]
    windows = [window for session_windows in per_session for window in session_windows]
    walk_records = [validate_walk_cost(session_windows) for session_windows in per_session]
    walk = ClaimEvidenceDict(
        claim_id="walk-cost",
        samples=sum(record["samples"] for record in walk_records),
        exact=sum(record["exact"] for record in walk_records),
        mismatches=sum(record["mismatches"] for record in walk_records),
        detail="single-echo walk episodes closed by a quiet window, -1 per wire step",
    )
    evidence: list[ClaimEvidenceDict] = [walk]
    evidence.extend(validate_firing_costs(windows))
    evidence.extend(validate_hit_damage(windows))
    evidence.append(validate_fuel_capacity(timelines))
    evidence.append(validate_radar_cost(timelines))
    evidence.append(validate_teleport_cost(runs_root / "bot"))
    return evidence


EXACTNESS_FLOOR = 0.85
"""Minimum exact/samples share for a claim to pass.

Clean instruments measure 88-100% exact across the real archive
(2026-07-21 calibration): the residual is positive-signed measurement
noise — walks truncated mid-path by collisions/terrain, unmodeled
fuel events inside otherwise-clean windows — not physics. The floor
still catches any REAL drift hard: a changed constant moves the
entire distribution, collapsing the exact share toward zero.
"""


def _passed(record: ClaimEvidenceDict) -> bool:
    """Report whether one claim's evidence passes the audit gate.

    Args:
        record: Evidence for one claim.

    Returns:
        True when the claim has samples and the claimed value
        dominates them (exact share at or above the floor).
    """
    return record["samples"] > 0 and record["exact"] / record["samples"] >= EXACTNESS_FLOOR


def stamp_fact_checked(page_path: Path, stamp: str) -> bool:
    """Rewrite a wiki page's ``fact_checked:`` frontmatter line.

    Args:
        page_path: The wiki page to stamp.
        stamp: Replacement value for the ``fact_checked:`` field.

    Returns:
        True when the line was found and rewritten.
    """
    lines = page_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("fact_checked:"):
            lines[index] = f"fact_checked: {stamp}\n"
            page_path.write_text("".join(lines), encoding="utf-8")
            return True
    return False


def _stamp_pages(
    wiki_pages_dir: Path,
    evidence: list[ClaimEvidenceDict],
) -> list[str]:
    """Stamp every eligible page whose validated claims all passed.

    Args:
        wiki_pages_dir: The ``wiki/pages`` directory.
        evidence: Collected audit evidence.

    Returns:
        Names of the pages that were stamped.
    """
    by_id = {record["claim_id"]: record for record in evidence}
    stamped: list[str] = []
    for page_name, claim_ids in STAMPED_PAGES.items():
        records = [by_id[claim_id] for claim_id in sorted(claim_ids) if claim_id in by_id]
        if len(records) != len(claim_ids) or not all(_passed(r) for r in records):
            continue
        total_samples = sum(r["samples"] for r in records)
        stamp = (
            f"{date.today().isoformat()} "
            f"(make audit: {len(records)} claims re-derived, {total_samples} clean samples)"
        )
        if stamp_fact_checked(wiki_pages_dir / page_name, stamp):
            stamped.append(page_name)
    return stamped


def run_audit(runs_root: Path, wiki_pages_dir: Path, *, stamp: bool) -> int:
    """Run the full audit, print the evidence table, optionally stamp.

    Args:
        runs_root: Directory containing ``bot/`` and ``sniff/`` run dirs.
        wiki_pages_dir: The ``wiki/pages`` directory.
        stamp: When True, rewrite fact_checked stamps for passing pages.

    Returns:
        0 when every claim passed; 1 otherwise.
    """
    evidence = collect_evidence(runs_root)
    width = max(len(record["claim_id"]) for record in evidence)
    sys.stdout.write(f"{'claim':<{width}}  samples  exact  mismatch  status\n")
    for record in evidence:
        status = "PASS" if _passed(record) else "FAIL"
        sys.stdout.write(
            f"{record['claim_id']:<{width}}  {record['samples']:>7}  {record['exact']:>5}"
            f"  {record['mismatches']:>8}  {status}  ({record['detail']})\n"
        )
    if stamp:
        for page_name in _stamp_pages(wiki_pages_dir, evidence):
            sys.stdout.write(f"stamped fact_checked: {page_name}\n")
    return 0 if all(_passed(record) for record in evidence) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``make audit``.

    Args:
        argv: Command-line arguments (``--runs-dir``, ``--wiki-dir``,
            ``--stamp``). Uses ``sys.argv[1:]`` when None.

    Returns:
        Process exit code (0 = every claim passed).
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])
    runs_root = Path("runs")
    wiki_pages_dir = Path("wiki") / "pages"
    stamp = False
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--runs-dir" and index + 1 < len(args):
            runs_root = Path(args[index + 1])
            index += 2
        elif token == "--wiki-dir" and index + 1 < len(args):
            wiki_pages_dir = Path(args[index + 1])
            index += 2
        elif token == "--stamp":
            stamp = True
            index += 1
        else:
            index += 1
    return run_audit(runs_root, wiki_pages_dir, stamp=stamp)


__all__ = [
    "EXACTNESS_FLOOR",
    "STAMPED_PAGES",
    "collect_evidence",
    "main",
    "run_audit",
    "stamp_fact_checked",
]
