"""Tests for the per-run digest builder, renderer, and CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.run_digest import (
    build_run_digest,
    main,
    render_run_digest,
)


def _event(timestamp: str, channel: str, message: str, **fields: str | int | bool) -> str:
    """Encode one runtime event JSONL line.

    Args:
        timestamp: Event timestamp.
        channel: Event channel.
        message: Event message.
        **fields: Structured fields spread at the top level.

    Returns:
        One JSON line.
    """
    return dump_json_str(
        {
            "timestamp": timestamp,
            "level": "INFO",
            "logger": "tankpit_bot.runtime.events",
            "mode": "bot",
            "channel": channel,
            "message": message,
            **fields,
        }
    )


def _write_session(path: Path, lines: list[str]) -> Path:
    """Write a synthetic events artifact.

    Args:
        path: Target file path.
        lines: JSONL lines.

    Returns:
        The written path.
    """
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _full_session_lines() -> list[str]:
    """Build a session exercising every digest lane.

    The clearance shot at 00:00:20 converts (pickup 4 s later); the one
    at 00:06:00 does not (no pickup inside the 10 s window). The second
    displacement repeats the first's tile so the histogram aggregates.
    """
    return [
        _event(
            "2026-08-05T00:00:00",
            "DIAGNOSTIC",
            "diagnostic_kind=session_room_joined",
            diagnostic_kind="session_room_joined",
            room_id="1",
        ),
        _event(
            "2026-08-05T00:00:01",
            "DIAGNOSTIC",
            "diagnostic_kind=tank_identity",
            diagnostic_kind="tank_identity",
            tank_id=1301,
        ),
        _event(
            "2026-08-05T00:00:02",
            "DIAGNOSTIC",
            "diagnostic_kind=session_account_stats",
            diagnostic_kind="session_account_stats",
            rank_name="private",
            rank_number=26,
            promotion_points=267291,
        ),
        _event(
            "2026-08-05T00:00:03",
            "DIAGNOSTIC",
            "diagnostic_kind=inventory_sample",
            diagnostic_kind="inventory_sample",
            armor=25,
            dual=20,
            missile=25,
            homing=23,
            radar=22,
        ),
        _event("2026-08-05T00:00:05", "WIRE", "teleport(59,95)"),
        _event(
            "2026-08-05T00:00:06",
            "DIAGNOSTIC",
            "diagnostic_kind=teleport_displacement",
            diagnostic_kind="teleport_displacement",
            requested_x=59,
            requested_y=95,
            landed_x=60,
            landed_y=96,
        ),
        _event(
            "2026-08-05T00:00:08",
            "DIAGNOSTIC",
            "diagnostic_kind=teleport_displacement",
            diagnostic_kind="teleport_displacement",
            requested_x=59,
            requested_y=95,
            landed_x=60,
            landed_y=94,
        ),
        _event(
            "2026-08-05T00:00:10",
            "DIAGNOSTIC",
            "diagnostic_kind=plan_released",
            diagnostic_kind="plan_released",
            reason="unservable",
        ),
        _event(
            "2026-08-05T00:00:20",
            "DIAGNOSTIC",
            "COLLECT score=925 cmd=shoot",
            behavior_reason="mine_clearance_shot",
            combat_target_x=58,
            combat_target_y=94,
        ),
        _event("2026-08-05T00:00:24", "WIRE", "pickup_equipment"),
        _event("2026-08-05T00:00:30", "WIRE", "shoot(100,100,id=501)"),
        _event("2026-08-05T00:00:31", "AI", "kill registered (tank_id=501)"),
        # Our own death arrives ONLY as the self_deactivated
        # diagnostic (0x41 origin here; the Normal-field fuel-wrap
        # emits the same kind). The old free-text DEACTIVATED regex
        # matched neither producer and read 0 through arterial's two
        # 2026-08-26 deaths.
        _event(
            "2026-08-05T00:00:40",
            "DIAGNOSTIC",
            "diagnostic_kind=self_deactivated",
            diagnostic_kind="self_deactivated",
            origin="protocol_0x41",
            killer_id=501,
        ),
        _event("2026-08-05T00:00:41", "WORLD", "DEACTIVATED: tank=502 killed by 1301"),
        _event(
            "2026-08-05T00:06:00",
            "DIAGNOSTIC",
            "COLLECT score=925 cmd=shoot",
            behavior_reason="mine_clearance_shot",
            combat_target_x=10,
            combat_target_y=10,
        ),
        _event("2026-08-05T00:06:30", "WIRE", "pickup_fuel"),
        _event("2026-08-05T00:06:31", "WIRE", "map_open"),
        _event(
            "2026-08-05T00:06:40",
            "DIAGNOSTIC",
            "diagnostic_kind=inventory_sample",
            diagnostic_kind="inventory_sample",
            armor=25,
            dual=25,
            missile=25,
            homing=25,
            radar=20,
        ),
        _event(
            "2026-08-05T00:07:00",
            "DIAGNOSTIC",
            "diagnostic_kind=session_scorecard",
            diagnostic_kind="session_scorecard",
            exit_reason="session_complete",
        ),
    ]


def test_full_session_digest(tmp_path: Path) -> None:
    """Every lane lands in the digest with the right counts."""
    source = _write_session(tmp_path / "run.events.jsonl", _full_session_lines())

    digest = build_run_digest(source)

    assert digest["room_id"] == "1"
    assert digest["self_tank_id"] == 1301
    assert digest["duration_s"] == 420
    assert digest["clean_exit"] is True
    assert digest["exit_reason"] == "session_complete"
    assert digest["kills"] == 1
    assert digest["deaths"] == 1
    assert digest["shots"] == 1
    assert digest["teleports"] == 1
    assert digest["pickups"] == 2
    assert digest["displacements"] == 2
    assert digest["displacement_top"] == [{"requested_x": 59, "requested_y": 95, "count": 2}]
    assert digest["releases_by_reason"] == {"unservable": 1}
    assert digest["rank_name"] == "private"
    assert digest["rank_number"] == 26
    assert digest["promotion_points"] == 267291
    assert digest["inventory_first"] == [25, 20, 25, 23, 22]
    assert digest["inventory_last"] == [25, 25, 25, 25, 20]
    assert [s["pickup_followed"] for s in digest["clearance_shots"]] == [True, False]
    assert [b["minute"] for b in digest["timeline"]] == [0, 5]
    assert digest["timeline"][0]["kills"] == 1
    assert digest["timeline"][0]["shots"] == 1
    assert digest["timeline"][0]["teleports"] == 1
    assert digest["timeline"][0]["pickups"] == 1
    assert digest["timeline"][1]["pickups"] == 1


def test_crashed_session_has_no_clean_exit(tmp_path: Path) -> None:
    """No scorecard, no account scrape, no inventory: honest gaps."""
    source = _write_session(
        tmp_path / "crashed.events.jsonl",
        [
            _event("2026-08-05T00:00:00", "WIRE", "shoot(1,1,id=5)"),
            _event("2026-08-05T00:00:01", "AI", "kill registered (tank_id=5)"),
        ],
    )

    digest = build_run_digest(source)

    assert digest["clean_exit"] is False
    assert digest["exit_reason"] == ""
    assert digest["rank_number"] == -1
    assert digest["inventory_first"] == []
    assert digest["kills"] == 1

    rendered = render_run_digest(digest)
    assert "CRASHED (no teardown scorecard)" in rendered
    assert "account" not in rendered
    assert "inventory" not in rendered


def test_pre_rename_archives_read_rank_points(tmp_path: Path) -> None:
    """Artifacts written before the rank_number rename still digest."""
    source = _write_session(
        tmp_path / "old.events.jsonl",
        [
            _event(
                "2026-08-05T00:00:00",
                "DIAGNOSTIC",
                "diagnostic_kind=session_account_stats",
                diagnostic_kind="session_account_stats",
                rank_name="private",
                rank_points=27,
                promotion_points=241167,
            ),
        ],
    )

    assert build_run_digest(source)["rank_number"] == 27


def test_other_tank_deactivation_is_not_our_death(tmp_path: Path) -> None:
    """A victim's deactivation line never books a death for us."""
    source = _write_session(
        tmp_path / "kills.events.jsonl",
        [
            _event(
                "2026-08-05T00:00:00",
                "DIAGNOSTIC",
                "diagnostic_kind=tank_identity",
                diagnostic_kind="tank_identity",
                tank_id=1301,
            ),
            _event("2026-08-05T00:00:01", "WORLD", "DEACTIVATED: tank=502 killed by 1301"),
        ],
    )

    assert build_run_digest(source)["deaths"] == 0


def test_second_identity_does_not_overwrite_self_id(tmp_path: Path) -> None:
    """Only the first tank_identity names the self id."""
    source = _write_session(
        tmp_path / "ids.events.jsonl",
        [
            _event(
                "2026-08-05T00:00:00",
                "DIAGNOSTIC",
                "diagnostic_kind=tank_identity",
                diagnostic_kind="tank_identity",
                tank_id=1301,
            ),
            _event(
                "2026-08-05T00:00:01",
                "DIAGNOSTIC",
                "diagnostic_kind=tank_identity",
                diagnostic_kind="tank_identity",
                tank_id=999,
            ),
        ],
    )

    assert build_run_digest(source)["self_tank_id"] == 1301


def test_empty_artifact_raises(tmp_path: Path) -> None:
    """A no-events artifact is an error, not a zero digest."""
    source = tmp_path / "empty.events.jsonl"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no events"):
        build_run_digest(source)


def test_bad_field_type_raises(tmp_path: Path) -> None:
    """A non-int structured field surfaces as a loud error."""
    source = _write_session(
        tmp_path / "bad.events.jsonl",
        [
            _event(
                "2026-08-05T00:00:00",
                "DIAGNOSTIC",
                "diagnostic_kind=teleport_displacement",
                diagnostic_kind="teleport_displacement",
                requested_x="not-an-int",
                requested_y=95,
            ),
        ],
    )

    with pytest.raises(ValueError, match="requested_x"):
        build_run_digest(source)


def test_cli_writes_digest_json(tmp_path: Path) -> None:
    """The CLI persists the machine-readable digest beside the source."""
    source = _write_session(tmp_path / "run.events.jsonl", _full_session_lines())
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: ["tankpit-run-digest", str(source)]
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original_get_argv

    persisted = narrow_json_to_dict(
        load_json_str((tmp_path / "run.digest.json").read_text(encoding="utf-8"))
    )
    assert persisted["kills"] == 1
    assert persisted["exit_reason"] == "session_complete"
    assert persisted["displacement_top"] == [{"requested_x": 59, "requested_y": 95, "count": 2}]


def test_combat_and_radar_yield_counters(tmp_path: Path) -> None:
    """Hits, misses, zero-yield radars, and damage totals all count.

    Two radar dispatches with no pickup between them make the first
    zero-yield; the second is still open at session end and counts
    too, while the pickup-closed third stays productive. Damage
    totals come from the teardown ledger emission's numeric fields
    (2026-08-06 extension).
    """
    lines = [
        _event(
            "2026-08-06T21:00:00",
            "DIAGNOSTIC",
            "diagnostic_kind=radar_dispatch",
            diagnostic_kind="radar_dispatch",
        ),
        _event(
            "2026-08-06T21:00:02",
            "DIAGNOSTIC",
            "diagnostic_kind=radar_dispatch",
            diagnostic_kind="radar_dispatch",
        ),
        _event(
            "2026-08-06T21:00:04",
            "DIAGNOSTIC",
            "diagnostic_kind=container_pickup_dispatched",
            diagnostic_kind="container_pickup_dispatched",
        ),
        _event(
            "2026-08-06T21:00:06",
            "DIAGNOSTIC",
            "diagnostic_kind=radar_dispatch",
            diagnostic_kind="radar_dispatch",
        ),
        _event(
            "2026-08-06T21:00:08",
            "DIAGNOSTIC",
            "diagnostic_kind=action_outcome",
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="hit",
        ),
        _event(
            "2026-08-06T21:00:10",
            "DIAGNOSTIC",
            "diagnostic_kind=action_outcome",
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="miss",
        ),
        _event(
            "2026-08-06T21:00:11",
            "DIAGNOSTIC",
            "diagnostic_kind=action_outcome",
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="reject",
        ),
        _event(
            "2026-08-06T21:00:12",
            "DIAGNOSTIC",
            "diagnostic_kind=damage_ledger",
            diagnostic_kind="damage_ledger",
            dealt="orange-5: dual x3 135",
            taken="orange-1: single x2 50",
            dealt_fuel=135,
            taken_fuel=50,
        ),
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    digest = build_run_digest(path)

    assert digest["hits"] == 1
    assert digest["misses"] == 1
    # First radar superseded unproductive, third still open at end.
    assert digest["zero_yield_radars"] == 2
    assert digest["damage_dealt"] == 135
    assert digest["damage_taken"] == 50


def test_pre_extension_damage_ledger_reads_zero(tmp_path: Path) -> None:
    """Archives without the numeric ledger fields stay at zero."""
    lines = [
        _event(
            "2026-08-06T21:00:00",
            "DIAGNOSTIC",
            "diagnostic_kind=damage_ledger",
            diagnostic_kind="damage_ledger",
            dealt="none",
            taken="none",
        ),
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    digest = build_run_digest(path)

    assert digest["damage_dealt"] == 0
    assert digest["damage_taken"] == 0


def test_a_shot_outcome_does_not_also_write_the_damage_ledger(tmp_path: Path) -> None:
    """The combat router claims an outcome record and stops reading it.

    ``_apply_combat_diagnostic`` handles two kinds in one cascade: the
    ``action_outcome`` arm counts the hit, and the FALL-THROUGH reads
    ``dealt_fuel`` / ``taken_fuel`` as the authoritative end-of-run
    damage totals. No archived outcome record carries those fields --
    42,082 of them, none -- so removing the arm's return changes no
    published number today.

    The shape below is constructed, and the reason to hold the line is
    what the fall-through means: those totals are the ledger's
    fuel-confirmed figures, emitted ONCE at teardown. A per-shot record
    that reached them would overwrite the run's damage accounting with
    whatever a single shot happened to carry, and the digest is what a
    post-mortem trusts for how much damage the session actually did.
    """
    lines = [
        _event("2026-08-06T21:00:00", "STATE", "start"),
        _event(
            "2026-08-06T21:00:02",
            "DIAGNOSTIC",
            "diagnostic_kind=action_outcome",
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="hit",
            dealt_fuel=999,
            taken_fuel=888,
        ),
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    digest = build_run_digest(path)

    assert digest["hits"] == 1
    assert (digest["damage_dealt"], digest["damage_taken"]) == (0, 0)
