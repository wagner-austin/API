"""End-to-end tests for the ``make roundtrip`` encoder validator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot.types import CaptureSession, encode_capture_session
from tankpit_bot.validate.roundtrip import (
    collect_roundtrip_evidence,
    main,
    run_roundtrip,
)
from tests.validate.builders import (
    SELF_ID,
    frame_message,
    fuel_gain_message,
    identity_message,
    make_session,
    move_message,
    pickup_message,
    sent_command_message,
    short_sync_message,
    sync_message,
    tank_remove_message,
    xor_encode_body,
)


def _write_capture(path: Path, session: CaptureSession) -> None:
    """Serialize one capture session to disk.

    Args:
        path: Target file path.
        session: Session to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json_str(encode_capture_session(session)), encoding="utf-8")


def _green_session() -> CaptureSession:
    """Build a session whose every binary message round-trips exactly.

    Includes skipped traffic (a sent command, a lobby text frame, an
    unknown outer type), the two plaintext toggle acks (un-XORed raw
    echoes, round-tripped against the raw-frame encoder), and three
    invalid-but-counted frames: an undecodable top-level 0x45, a 0x43
    cache body with a torn record, and a 0x53 shorter than its minimum.
    """
    messages = [
        identity_message(1000, SELF_ID),
        move_message(1500, SELF_ID, "sse"),
        sync_message(2000, SELF_ID, 1, 1000),
        short_sync_message(2500, SELF_ID, 1),
        fuel_gain_message(3000, 998),
        pickup_message(3500),
        tank_remove_message(4000, 9),
        sent_command_message(4500, 112, 5, 6),
        frame_message(5000, b"=1|Jan. 08, 2013|Artax|1|9", "received"),
        frame_message(5200, b"A1", "received"),
        frame_message(5400, b"C0", "received"),
        frame_message(5500, bytes([0x99, 1, 2, 3]), "received"),
        frame_message(6000, xor_encode_body(0x45, bytes([40, 41])), "received"),
        frame_message(6500, xor_encode_body(0x43, bytes([1, 2, 3])), "received"),
        frame_message(7000, xor_encode_body(0x53, bytes([1, 2])), "received"),
    ]
    return make_session(messages)


def _mismatch_session() -> CaptureSession:
    """Build a session with one known encode/decode asymmetry.

    An 11-byte tunneled MovementResponse decodes with a defaulted
    carrying byte; the encoder always emits the 12-byte wire form, so
    the round trip must report a mismatch.
    """
    inner = bytes([1, 9, 0, 5, 6, 0, 0, 1, 0, 0, 7])
    body = xor_encode_body(0x2E, bytes([0x3D]) + inner)
    return make_session(
        [
            frame_message(1000, body, "received"),
            frame_message(2000, body, "received"),
        ]
    )


def test_green_tree_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Every message round-trips; invalid frames are counted, not judged."""
    runs_root = tmp_path / "runs"
    _write_capture(runs_root / "sniff" / "a.capture_session.json", _green_session())
    _write_capture(runs_root / "bot" / "nomagic.capture_session.json", make_session([], magic=None))
    _write_capture(runs_root / "bot" / "latest.capture_session.json", _mismatch_session())
    rc = run_roundtrip(runs_root)
    out = capsys.readouterr().out
    assert rc == 0
    assert "FAIL" not in out
    assert "roundtrip-33" in out
    assert "roundtrip-autoscroll_ack" in out
    assert "roundtrip-chat_ack" in out
    assert "invalid-frames  skipped=3" in out
    assert "0 mismatches" in out


def test_mismatch_fails_with_first_diff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A decode/encode asymmetry is reported with its first byte diff."""
    runs_root = tmp_path / "runs"
    _write_capture(runs_root / "bot" / "bad.capture_session.json", _mismatch_session())
    rc = run_roundtrip(runs_root)
    out = capsys.readouterr().out
    assert rc == 1
    assert "roundtrip-61" in out
    assert "FAIL" in out
    assert "first diff: want=" in out


def test_empty_archive_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An archive with no binary messages is a failure, not a pass."""
    runs_root = tmp_path / "runs"
    (runs_root / "bot").mkdir(parents=True)
    rc = run_roundtrip(runs_root)
    out = capsys.readouterr().out
    assert rc == 1
    assert "no binary messages" in out


def test_collect_evidence_families(tmp_path: Path) -> None:
    """Evidence is one record per family plus the invalid-frames tail."""
    runs_root = tmp_path / "runs"
    _write_capture(runs_root / "sniff" / "a.capture_session.json", _green_session())
    evidence = collect_roundtrip_evidence(runs_root)
    by_id = {record["claim_id"]: record for record in evidence}
    assert by_id["roundtrip-46"]["exact"] == 2
    assert by_id["roundtrip-46"]["mismatches"] == 0
    assert by_id["roundtrip-46"]["detail"] == "byte-identical"
    assert by_id["invalid-frames"]["samples"] == 3
    assert evidence[-1]["claim_id"] == "invalid-frames"


def test_main_parses_runs_dir(tmp_path: Path) -> None:
    """The CLI accepts --runs-dir and unknown flags."""
    runs_root = tmp_path / "runs"
    _write_capture(runs_root / "sniff" / "a.capture_session.json", _green_session())
    assert main(["--runs-dir", str(runs_root), "--unknown-flag"]) == 0


def test_main_reads_sys_argv_when_argv_is_none(tmp_path: Path) -> None:
    """main(None) falls back to sys.argv (the console-script path)."""
    runs_root = tmp_path / "runs"
    _write_capture(runs_root / "sniff" / "a.capture_session.json", _green_session())
    original_argv = sys.argv
    sys.argv = ["tankpit-roundtrip", "--runs-dir", str(runs_root)]
    rc = main(None)
    sys.argv = original_argv
    assert rc == 0
