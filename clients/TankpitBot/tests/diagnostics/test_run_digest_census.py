"""Tests for the digest's wasted-tick census and kill attribution.

Split from ``tests/diagnostics/test_run_digest.py`` (2026-08-28, at
the file-size bar) when the census landed: stalls, superseded splits,
wire-gap tracking, and the 0x41 killer-attribution rule that replaced
the free-text "kill registered" count.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str

from tankpit_bot.diagnostics.run_digest import build_run_digest
from tankpit_bot.diagnostics.run_digest_render import render_run_digest


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


def _identity(timestamp: str, tank_id: int) -> str:
    """Build a ``tank_identity`` DIAGNOSTIC line.

    Args:
        timestamp: Event timestamp.
        tank_id: This session's own wire id.

    Returns:
        One JSON line.
    """
    return _event(
        timestamp,
        "DIAGNOSTIC",
        "diagnostic_kind=tank_identity",
        diagnostic_kind="tank_identity",
        tank_id=tank_id,
    )


def _deactivation(timestamp: str, victim_id: int, killer_id: int) -> str:
    """Build a ``tank_deactivated`` DIAGNOSTIC line.

    Args:
        timestamp: Event timestamp.
        victim_id: The killed tank's wire id.
        killer_id: The killer's wire id.

    Returns:
        One JSON line.
    """
    return _event(
        timestamp,
        "DIAGNOSTIC",
        "diagnostic_kind=tank_deactivated",
        diagnostic_kind="tank_deactivated",
        origin="protocol_0x41",
        victim_id=victim_id,
        killer_id=killer_id,
    )


def test_a_siblings_kill_never_counts_as_ours(tmp_path: Path) -> None:
    """A fleet sibling's 0x41 in our stream books no kill for us.

    The pre-fleet raw count broke the day two bots shared a room
    (2026-08-14: arterial banked artax's two kills on zero shots); the
    digest applies the same killer-must-be-us rule as the scorecard.
    """
    source = _write_session(
        tmp_path / "sibling.events.jsonl",
        [
            _identity("2026-08-05T00:00:00", 1301),
            _deactivation("2026-08-05T00:00:01", 502, 777),
        ],
    )

    assert build_run_digest(source)["kills"] == 0


def test_an_unidentified_session_books_no_kills(tmp_path: Path) -> None:
    """Without a ``tank_identity``, no 0x41 is attributable to us.

    The ``-1`` unidentified sentinel must never match a ``-1``
    killer_id from a pre-fleet artifact that lacked the field.
    """
    source = _write_session(
        tmp_path / "unidentified.events.jsonl",
        [_deactivation("2026-08-05T00:00:00", 502, -1)],
    )

    digest = build_run_digest(source)

    assert digest["self_tank_id"] == -1
    assert digest["kills"] == 0


def test_the_census_counts_stalls_and_superseded_splits(tmp_path: Path) -> None:
    """Stalls and both superseded flavors land in the digest and render.

    The livelock detector fired unread through the 2026-08-21 gatherer
    livelock era because nothing consumed ``liveness_stall``; the
    superseded split separates planner churn (undispatched) from
    re-aims on real output (dispatched).
    """
    source = _write_session(
        tmp_path / "census.events.jsonl",
        [
            _event(
                "2026-08-05T00:00:00",
                "DIAGNOSTIC",
                "diagnostic_kind=liveness_stall",
                diagnostic_kind="liveness_stall",
                action_kind="teleport",
                streak=25,
            ),
            _event(
                "2026-08-05T00:00:02",
                "DIAGNOSTIC",
                "diagnostic_kind=action_outcome",
                diagnostic_kind="action_outcome",
                action_kind="teleport",
                outcome="superseded",
                dispatched=False,
            ),
            _event(
                "2026-08-05T00:00:04",
                "DIAGNOSTIC",
                "diagnostic_kind=action_outcome",
                diagnostic_kind="action_outcome",
                action_kind="shoot",
                outcome="superseded",
                dispatched=True,
            ),
            # A pre-detector archive row without the dispatched field
            # reads as churn -- nothing proved it reached the wire.
            _event(
                "2026-08-05T00:00:06",
                "DIAGNOSTIC",
                "diagnostic_kind=action_outcome",
                diagnostic_kind="action_outcome",
                action_kind="shoot",
                outcome="superseded",
            ),
        ],
    )

    digest = build_run_digest(source)

    assert digest["liveness_stalls"] == 1
    assert digest["superseded_undispatched"] == 2
    assert digest["superseded_dispatched"] == 1
    rendered = render_run_digest(digest)
    assert "activity   stalls=1 superseded_plans=2 re_aims=1" in rendered


def test_wire_gap_census_tracks_the_longest_silence(tmp_path: Path) -> None:
    """Inter-dispatch gaps over the stall bar count; short ones do not.

    The 2026-08-20 arterial run idled 193 s inside a 261 s session and
    the digest showed nothing -- five-minute timeline buckets smooth a
    dead loop right over.
    """
    source = _write_session(
        tmp_path / "gaps.events.jsonl",
        [
            _event("2026-08-05T00:00:00", "WIRE", "shoot(1,1,id=5)"),
            _event("2026-08-05T00:00:02", "WIRE", "shoot(1,1,id=5)"),
            _event("2026-08-05T00:03:15", "WIRE", "teleport(59,95)"),
            _event("2026-08-05T00:03:50", "WIRE", "pickup_fuel"),
        ],
    )

    digest = build_run_digest(source)

    assert digest["max_wire_gap_s"] == 193
    assert digest["wire_gaps_over_30s"] == 2
    assert "max_wire_gap=193s gaps_over_30s=2" in render_run_digest(digest)
