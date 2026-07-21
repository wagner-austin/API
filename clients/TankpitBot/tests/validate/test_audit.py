"""End-to-end tests for the ``make audit`` orchestrator."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot.types import CaptureSession, encode_capture_session
from tankpit_bot.validate.audit import (
    STAMPED_PAGES,
    collect_evidence,
    main,
    run_audit,
    stamp_fact_checked,
)
from tankpit_bot.validate.types import (
    ClaimEvidenceDict,
    decode_claim_evidence,
    encode_claim_evidence,
    encode_evidence_list,
)
from tests.validate.builders import (
    ENEMY_ID,
    SELF_ID,
    fuel_gain_message,
    identity_message,
    make_session,
    move_message,
    shot_message,
    sync_message,
)

_ECONOMY_PAGE = "---\ntitle: Game Economy\nfact_checked: 2026-07-06\n---\n\n# Economy\n"


def _write_capture(path: Path, session: CaptureSession) -> None:
    """Serialize one capture session to disk.

    Args:
        path: Target file path.
        session: Session to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json_str(encode_capture_session(session)), encoding="utf-8")


def _green_capture() -> CaptureSession:
    """Build a capture producing one clean window for every archive claim."""
    messages = [
        identity_message(1000, SELF_ID),
        sync_message(2000, SELF_ID, 1, 1000),
        move_message(2500, SELF_ID, "ee"),
        fuel_gain_message(3000, 998),
        fuel_gain_message(3200, 998),
        shot_message(3500, SELF_ID, 0),
        fuel_gain_message(4000, 992),
        shot_message(4500, SELF_ID, 1),
        fuel_gain_message(5000, 982),
        shot_message(5500, SELF_ID, 2),
        fuel_gain_message(6000, 972),
        shot_message(6500, SELF_ID, 3),
        fuel_gain_message(7000, 962),
        shot_message(7500, ENEMY_ID, 0),
        fuel_gain_message(8000, 917),
        shot_message(8500, ENEMY_ID, 1),
        fuel_gain_message(9000, 827),
    ]
    return make_session(messages)


def _green_events(runs_root: Path) -> None:
    """Write one eligible events log with a clean exact teleport pair."""
    events = (
        '{"diagnostic_kind": "self_alignment_sample",'
        ' "belief_x": 0, "belief_y": 0, "belief_fuel": 500}\n'
        '{"diagnostic_kind": "action_outcome", "action_kind": "teleport",'
        ' "outcome": "landed_exact", "landed_x": 3, "landed_y": 4}\n'
        '{"diagnostic_kind": "self_alignment_sample",'
        ' "belief_x": 3, "belief_y": 4, "belief_fuel": 470}\n'
    )
    (runs_root / "bot").mkdir(parents=True, exist_ok=True)
    (runs_root / "bot" / "bot-20260701-000000.events.jsonl").write_text(events, encoding="utf-8")


def _green_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a runs tree + wiki dir where every claim passes.

    Args:
        tmp_path: Test temp root.

    Returns:
        Pair of (runs_root, wiki_pages_dir).
    """
    runs_root = tmp_path / "runs"
    _write_capture(runs_root / "bot" / "a.capture_session.json", _green_capture())
    _write_capture(runs_root / "sniff" / "b.capture_session.json", make_session([]))
    _green_events(runs_root)
    wiki_dir = tmp_path / "wiki" / "pages"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "game-economy.md").write_text(_ECONOMY_PAGE, encoding="utf-8")
    return runs_root, wiki_dir


def test_green_tree_passes_and_stamps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Every claim re-derives, the audit passes, the page is stamped."""
    runs_root, wiki_dir = _green_tree(tmp_path)
    rc = run_audit(runs_root, wiki_dir, stamp=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "teleport-cost" in out
    assert "FAIL" not in out
    assert "stamped fact_checked: game-economy.md" in out
    page_text = (wiki_dir / "game-economy.md").read_text(encoding="utf-8")
    expected = (
        f"fact_checked: {date.today().isoformat()} "
        "(make audit: 9 claims re-derived, 17 clean samples)"
    )
    assert expected in page_text


def test_empty_archive_fails_without_stamping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Zero samples for every claim is an audit failure, not a pass."""
    runs_root = tmp_path / "runs"
    wiki_dir = tmp_path / "wiki" / "pages"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "game-economy.md").write_text(_ECONOMY_PAGE, encoding="utf-8")
    rc = run_audit(runs_root, wiki_dir, stamp=True)
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "stamped" not in out
    assert "fact_checked: 2026-07-06" in (wiki_dir / "game-economy.md").read_text(encoding="utf-8")


def test_magicless_session_is_skipped(tmp_path: Path) -> None:
    """A capture without a magic key contributes nothing but does not crash."""
    runs_root = tmp_path / "runs"
    _write_capture(runs_root / "bot" / "nomagic.capture_session.json", make_session([], magic=None))
    evidence = collect_evidence(runs_root)
    assert all(record["samples"] == 0 for record in evidence)


def test_stamp_fact_checked_reports_missing_line(tmp_path: Path) -> None:
    """A page without a fact_checked line cannot be stamped."""
    page = tmp_path / "page.md"
    page.write_text("---\ntitle: X\n---\n", encoding="utf-8")
    assert stamp_fact_checked(page, "2026-07-21") is False


def test_partial_evidence_never_stamps(tmp_path: Path) -> None:
    """A page is stamped only when every one of its claims was validated."""
    from tankpit_bot.validate.audit import _stamp_pages

    wiki_dir = tmp_path
    (wiki_dir / "game-economy.md").write_text(_ECONOMY_PAGE, encoding="utf-8")
    partial = [
        ClaimEvidenceDict(claim_id="walk-cost", samples=5, exact=5, mismatches=0, detail="x")
    ]
    assert _stamp_pages(wiki_dir, partial) == []
    assert sorted(STAMPED_PAGES) == ["game-economy.md"]


def test_low_exact_share_fails_the_floor(tmp_path: Path) -> None:
    """A claim whose exact share sits below the floor never stamps."""
    from tankpit_bot.validate.audit import EXACTNESS_FLOOR, _stamp_pages

    (tmp_path / "game-economy.md").write_text(_ECONOMY_PAGE, encoding="utf-8")
    noisy = [
        ClaimEvidenceDict(claim_id=claim_id, samples=10, exact=8, mismatches=2, detail="x")
        for claim_id in sorted(STAMPED_PAGES["game-economy.md"])
    ]
    assert EXACTNESS_FLOOR == 0.85
    assert _stamp_pages(tmp_path, noisy) == []


def test_passing_page_without_fact_checked_line_is_not_stamped(tmp_path: Path) -> None:
    """Full green evidence cannot stamp a page lacking the frontmatter line."""
    from tankpit_bot.validate.audit import _stamp_pages

    (tmp_path / "game-economy.md").write_text("---\ntitle: X\n---\n", encoding="utf-8")
    green = [
        ClaimEvidenceDict(claim_id=claim_id, samples=1, exact=1, mismatches=0, detail="x")
        for claim_id in sorted(STAMPED_PAGES["game-economy.md"])
    ]
    assert _stamp_pages(tmp_path, green) == []


def test_main_parses_args_and_runs(tmp_path: Path) -> None:
    """The CLI wires --runs-dir/--wiki-dir/--stamp and unknown args."""
    runs_root, wiki_dir = _green_tree(tmp_path)
    rc = main(
        [
            "--runs-dir",
            str(runs_root),
            "--wiki-dir",
            str(wiki_dir),
            "--stamp",
            "--unknown-flag",
        ]
    )
    assert rc == 0


def test_main_reads_sys_argv_when_argv_is_none(tmp_path: Path) -> None:
    """main(None) falls back to sys.argv (the console-script path)."""
    runs_root, wiki_dir = _green_tree(tmp_path)
    original_argv = sys.argv
    sys.argv = ["tankpit-audit", "--runs-dir", str(runs_root), "--wiki-dir", str(wiki_dir)]
    rc = main(None)
    sys.argv = original_argv
    assert rc == 0


def test_evidence_codec_round_trip() -> None:
    """Evidence encode/decode is lossless and list encoding preserves order."""
    record = ClaimEvidenceDict(claim_id="walk-cost", samples=3, exact=2, mismatches=1, detail="d")
    assert decode_claim_evidence(encode_claim_evidence(record)) == record
    assert encode_evidence_list([record]) == [encode_claim_evidence(record)]
