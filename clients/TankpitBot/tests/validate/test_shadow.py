"""The shadow runner and CLI over a real on-disk runs tree."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot.sim.server import CORPSE_WINDOW_TICKS, TICK_MS
from tankpit_bot.types import CaptureSession, encode_capture_session
from tankpit_bot.validate.shadow import collect_shadow_evidence, main, run_shadow
from tests.validate.builders import (
    ENEMY_ID,
    SELF_ID,
    deactivation_message,
    equipment_gain_message,
    identity_message,
    inventory_message,
    make_session,
    sync_message,
    tank_remove_message,
)

VICTIM_ID = 12
CORPSE_MS = CORPSE_WINDOW_TICKS * TICK_MS


def _write_capture(path: Path, session: CaptureSession) -> None:
    path.write_text(dump_json_str(encode_capture_session(session)), encoding="utf-8")


def _lawful_session() -> CaptureSession:
    messages = [identity_message(0, SELF_ID)]
    messages.extend(sync_message(index * TICK_MS, ENEMY_ID, 3, 400) for index in range(8))
    messages.append(equipment_gain_message(20_000, [0, 7, 0, 0, 0], True))
    messages.append(inventory_message(20_100, [5, 12, 5, 5, 3]))
    messages.append(inventory_message(30_000, [5, 12, 5, 5, 3]))
    messages.append(deactivation_message(40_000, VICTIM_ID, SELF_ID))
    messages.append(tank_remove_message(40_000 + CORPSE_MS, VICTIM_ID))
    return make_session(messages)


def _lawless_session() -> CaptureSession:
    messages = [identity_message(0, SELF_ID)]
    messages.extend(sync_message(index * 5000, ENEMY_ID, 3, 400) for index in range(8))
    messages.append(equipment_gain_message(20_000, [0, 2, 0, 1, 0], True))
    messages.append(inventory_message(20_100, [5, 7, 5, 6, 3]))
    messages.append(inventory_message(30_000, [5, 7, 5, 6, 0]))
    messages.append(deactivation_message(40_000, VICTIM_ID, SELF_ID))
    messages.append(tank_remove_message(48_000, VICTIM_ID))
    return make_session(messages)


def _make_runs(tmp_path: Path, session: CaptureSession) -> Path:
    runs_root = tmp_path / "runs"
    (runs_root / "bot").mkdir(parents=True)
    (runs_root / "sniff").mkdir(parents=True)
    _write_capture(runs_root / "bot" / "a.capture_session.json", session)
    return runs_root


def test_lawful_archive_passes_every_law(tmp_path: Path) -> None:
    runs_root = _make_runs(tmp_path, _lawful_session())
    evidence = collect_shadow_evidence(runs_root)
    by_id = {record["claim_id"]: record for record in evidence}
    assert by_id["sync-cadence"]["exact"] == 1
    assert by_id["grant-invariants"]["exact"] == 1
    assert by_id["kill-mercy-bundle"]["exact"] == 1
    assert by_id["corpse-window"]["exact"] == 1
    assert all(record["mismatches"] == 0 for record in evidence)
    assert run_shadow(runs_root) == 0


def test_lawless_archive_fails_every_law(tmp_path: Path) -> None:
    runs_root = _make_runs(tmp_path, _lawless_session())
    evidence = collect_shadow_evidence(runs_root)
    for record in evidence:
        assert record["mismatches"] == 1, record["claim_id"]
    assert run_shadow(runs_root) == 1


def test_magicless_session_is_skipped(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    (runs_root / "bot").mkdir(parents=True)
    (runs_root / "sniff").mkdir(parents=True)
    _write_capture(runs_root / "bot" / "nomagic.capture_session.json", make_session([], magic=None))
    evidence = collect_shadow_evidence(runs_root)
    assert all(record["samples"] == 0 for record in evidence)
    assert run_shadow(runs_root) == 1


def test_run_shadow_prints_the_evidence_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_root = _make_runs(tmp_path, _lawful_session())
    assert run_shadow(runs_root) == 0
    out = capsys.readouterr().out
    assert "sync-cadence" in out
    assert "kill-mercy-bundle" in out
    assert "PASS" in out


def test_main_parses_runs_dir(tmp_path: Path) -> None:
    runs_root = _make_runs(tmp_path, _lawful_session())
    assert main(["--runs-dir", str(runs_root)]) == 0


def test_main_skips_unknown_tokens(tmp_path: Path) -> None:
    runs_root = _make_runs(tmp_path, _lawful_session())
    assert main(["--verbose", "--runs-dir", str(runs_root)]) == 0


def test_main_reads_sys_argv_when_argv_is_none(tmp_path: Path) -> None:
    """main(None) falls back to sys.argv (the console-script path)."""
    runs_root = _make_runs(tmp_path, _lawful_session())
    original_argv = sys.argv
    sys.argv = ["tankpit-shadow", "--runs-dir", str(runs_root)]
    rc = main(None)
    sys.argv = original_argv
    assert rc == 0
