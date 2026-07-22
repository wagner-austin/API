"""The ``make shadow`` entrypoint: price the sim's laws on the archive.

Loads every decodable capture session in ``runs/bot`` and
``runs/sniff``, extracts the shadow-law event timelines, and judges
each sim law (predictors imported from the sim source) against what
the real server actually did. Prints one evidence row per law and
exits non-zero when any law has zero samples or falls below the
audit's exactness floor — a shadow failure means the sim and the real
game disagree, which is either a wiki gap or a sim bug, and both
demand investigation, not softening.

This is the standing instrument the one-off mining sweeps graduated
into: every future live run lands in ``runs/`` and is automatically
judged against the sim on the next ``make shadow``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot.types import decode_capture_session
from tankpit_bot.validate.audit import EXACTNESS_FLOOR
from tankpit_bot.validate.shadow_laws import (
    shadow_corpse_window,
    shadow_grant_invariants,
    shadow_mercy_bundle,
    shadow_sync_cadence,
)
from tankpit_bot.validate.shadow_timeline import ShadowTimelineDict, extract_shadow_timeline
from tankpit_bot.validate.types import ClaimEvidenceDict

_LOGGER = get_logger(__name__)


def _load_shadow_timelines(runs_root: Path) -> list[ShadowTimelineDict]:
    """Extract shadow timelines from every decodable capture session.

    Args:
        runs_root: Directory containing ``bot/`` and ``sniff/`` run dirs.

    Returns:
        One timeline per session with a magic key; magic-less sessions
        are skipped with a warning (cannot XOR-decode).
    """
    timelines: list[ShadowTimelineDict] = []
    for subdir in ("bot", "sniff"):
        for path in sorted((runs_root / subdir).glob("*.capture_session.json")):
            session = decode_capture_session(
                narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))
            )
            if session["magic"] is None:
                _LOGGER.warning("shadow_session_without_magic file=%s", path)
                continue
            timelines.append(extract_shadow_timeline(session))
    return timelines


def collect_shadow_evidence(runs_root: Path) -> list[ClaimEvidenceDict]:
    """Run every shadow-law validator and gather the evidence.

    Args:
        runs_root: Directory containing ``bot/`` and ``sniff/`` run dirs.

    Returns:
        Evidence records, one per shadowed law.
    """
    timelines = _load_shadow_timelines(runs_root)
    return [
        shadow_sync_cadence(timelines),
        shadow_grant_invariants(timelines),
        shadow_mercy_bundle(timelines),
        shadow_corpse_window(timelines),
    ]


def _passed(record: ClaimEvidenceDict) -> bool:
    """Report whether one law's evidence passes the shadow gate.

    Args:
        record: Evidence for one law.

    Returns:
        True when the law has samples and the sim's prediction
        dominates them (exact share at or above the audit floor).
    """
    return record["samples"] > 0 and record["exact"] / record["samples"] >= EXACTNESS_FLOOR


def run_shadow(runs_root: Path) -> int:
    """Run the full shadow comparison and print the evidence table.

    Args:
        runs_root: Directory containing ``bot/`` and ``sniff/`` run dirs.

    Returns:
        0 when every law passed; 1 otherwise.
    """
    evidence = collect_shadow_evidence(runs_root)
    width = max(len(record["claim_id"]) for record in evidence)
    sys.stdout.write(f"{'law':<{width}}  samples  exact  mismatch  status\n")
    for record in evidence:
        status = "PASS" if _passed(record) else "FAIL"
        sys.stdout.write(
            f"{record['claim_id']:<{width}}  {record['samples']:>7}  {record['exact']:>5}"
            f"  {record['mismatches']:>8}  {status}  ({record['detail']})\n"
        )
    return 0 if all(_passed(record) for record in evidence) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``make shadow``.

    Args:
        argv: Command-line arguments (``--runs-dir``). Uses
            ``sys.argv[1:]`` when None.

    Returns:
        Process exit code (0 = every law passed).
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])
    runs_root = Path("runs")
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--runs-dir" and index + 1 < len(args):
            runs_root = Path(args[index + 1])
            index += 2
        else:
            index += 1
    return run_shadow(runs_root)


__all__ = [
    "collect_shadow_evidence",
    "main",
    "run_shadow",
]
