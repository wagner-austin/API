"""Tests for the corpus audit: analyzers diffed against wire receipts."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.corpus_audit import (
    RunAuditDict,
    _collect_flags,
    audit_corpus,
    audit_events_artifact,
    main,
    render_corpus_audit,
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


def _kill(timestamp: str, victim_id: int, killer_id: int) -> str:
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


def _inventory(timestamp: str, radar: int) -> str:
    """Build an ``inventory_sample`` DIAGNOSTIC line.

    Args:
        timestamp: Event timestamp.
        radar: Extra-radar count.

    Returns:
        One JSON line.
    """
    return _event(
        timestamp,
        "DIAGNOSTIC",
        "diagnostic_kind=inventory_sample",
        diagnostic_kind="inventory_sample",
        armor=0,
        dual=0,
        missile=0,
        homing=0,
        radar=radar,
        radar_enabled=True,
    )


def _displacement(timestamp: str, x: int, y: int) -> str:
    """Build a ``teleport_displacement`` DIAGNOSTIC line.

    Args:
        timestamp: Event timestamp.
        x: Requested landing X.
        y: Requested landing Y.

    Returns:
        One JSON line.
    """
    return _event(
        timestamp,
        "DIAGNOSTIC",
        "diagnostic_kind=teleport_displacement",
        diagnostic_kind="teleport_displacement",
        requested_x=x,
        requested_y=y,
        landed_x=x + 1,
        landed_y=y,
        displacement=1,
    )


def _write_run(path: Path, lines: list[str]) -> Path:
    """Write a synthetic events artifact.

    Args:
        path: Target file path.
        lines: JSONL lines.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _clean_run_lines() -> list[str]:
    """Build a run every analyzer agrees on: 1 kill, 1 death, no flags."""
    return [
        _identity("2026-08-05T00:00:00", 601),
        # A second identity must not re-bind the self id.
        _identity("2026-08-05T00:00:01", 999),
        _event("2026-08-05T00:00:02", "WIRE", "shoot(1,1,id=529)"),
        _kill("2026-08-05T00:00:04", 529, 601),
        # A sibling's kill counts for nobody, in every counter.
        _kill("2026-08-05T00:00:06", 530, 777),
        _event(
            "2026-08-05T00:00:08",
            "DIAGNOSTIC",
            "diagnostic_kind=self_deactivated",
            diagnostic_kind="self_deactivated",
            origin="protocol_0x41",
            killer_id=777,
        ),
        # Kinds outside both tally families ride along untallied,
        # and WIRE lines other than shoots skip the beat census.
        _event(
            "2026-08-05T00:00:10",
            "DIAGNOSTIC",
            "diagnostic_kind=plan_released",
            diagnostic_kind="plan_released",
            reason="unservable",
        ),
        _event("2026-08-05T00:00:12", "WIRE", "teleport(59,95)"),
    ]


def _audit(
    *,
    digest_kills: int = 2,
    digest_deaths: int = 1,
    scorecard_kills: int = 2,
    radar_drift: int = 0,
) -> RunAuditDict:
    """Build a clean audit row with per-case overrides.

    Args:
        digest_kills: Digest kill count (wire recount stays 2).
        digest_deaths: Digest death count (wire recount stays 1).
        scorecard_kills: Scorecard kill count.
        radar_drift: Radar book prediction error.

    Returns:
        The audit row.
    """
    return RunAuditDict(
        source="run.events.jsonl",
        wire_kills=2,
        wire_deaths=1,
        digest_kills=digest_kills,
        digest_deaths=digest_deaths,
        scorecard_kills=scorecard_kills,
        fast_shots=0,
        reaims_within_30s=0,
        radar_drift=radar_drift,
        physics_divergences=0,
        kind_counts={},
        flags=[],
    )


class TestAuditEventsArtifact:
    """Tests for the single-artifact audit."""

    def test_agreeing_run_carries_no_flags(self, tmp_path: Path) -> None:
        """Wire recount, digest, and scorecard agree on a clean run."""
        source = _write_run(tmp_path / "run.events.jsonl", _clean_run_lines())

        audit = audit_events_artifact(source)

        assert audit["wire_kills"] == 1
        assert audit["digest_kills"] == 1
        assert audit["scorecard_kills"] == 1
        assert audit["wire_deaths"] == 1
        assert audit["digest_deaths"] == 1
        assert audit["kind_counts"]["tank_deactivated"] == 2
        assert audit["flags"] == []

    def test_fast_shots_violate_the_serve_beat(self, tmp_path: Path) -> None:
        """Two shoots inside one beat flag the dispatcher."""
        source = _write_run(
            tmp_path / "fast.events.jsonl",
            [
                _event("2026-08-05T00:00:00", "WIRE", "shoot(1,1,id=5)"),
                _event("2026-08-05T00:00:01", "WIRE", "shoot(1,1,id=5)"),
                _event("2026-08-05T00:00:03", "WIRE", "shoot(1,1,id=5)"),
            ],
        )

        audit = audit_events_artifact(source)

        assert audit["fast_shots"] == 1
        assert any("serve" in flag for flag in audit["flags"])

    def test_reaim_window_splits_orbits_from_fresh_aims(self, tmp_path: Path) -> None:
        """Same tile inside 30 s counts; outside it, or another tile, does not."""
        source = _write_run(
            tmp_path / "reaim.events.jsonl",
            [
                _displacement("2026-08-05T00:00:00", 18, 123),
                _displacement("2026-08-05T00:00:10", 18, 123),
                _displacement("2026-08-05T00:00:15", 40, 50),
                _displacement("2026-08-05T00:01:30", 18, 123),
            ],
        )

        audit = audit_events_artifact(source)

        assert audit["reaims_within_30s"] == 1
        assert any("tombstone" in flag for flag in audit["flags"])

    def test_radar_book_balances_with_the_uses_extra_split(self, tmp_path: Path) -> None:
        """first + gains - paid presses = last, free presses excluded."""
        source = _write_run(
            tmp_path / "radar.events.jsonl",
            [
                _inventory("2026-08-05T00:00:00", 5),
                _event(
                    "2026-08-05T00:00:02",
                    "DIAGNOSTIC",
                    "diagnostic_kind=radar_dispatch",
                    diagnostic_kind="radar_dispatch",
                    uses_extra=True,
                ),
                _event(
                    "2026-08-05T00:00:04",
                    "DIAGNOSTIC",
                    "diagnostic_kind=radar_dispatch",
                    diagnostic_kind="radar_dispatch",
                    uses_extra=False,
                ),
                _event(
                    "2026-08-05T00:00:06",
                    "DIAGNOSTIC",
                    "diagnostic_kind=equipment_gain",
                    diagnostic_kind="equipment_gain",
                    armor=0,
                    dual=0,
                    missile=0,
                    homing=0,
                    radar=2,
                ),
                _inventory("2026-08-05T00:00:08", 6),
            ],
        )

        audit = audit_events_artifact(source)

        assert audit["radar_drift"] == 0
        assert audit["flags"] == []

    def test_a_run_without_inventory_samples_has_zero_drift(self, tmp_path: Path) -> None:
        """No samples means no book to disagree with."""
        source = _write_run(
            tmp_path / "bare.events.jsonl",
            [_event("2026-08-05T00:00:00", "STATE", "INITIALIZING")],
        )

        assert audit_events_artifact(source)["radar_drift"] == 0

    def test_an_era_shape_the_analyzers_reject_raises(self, tmp_path: Path) -> None:
        """A record missing a strictly-required field raises, not lies.

        The first real-corpus sweep hit an archive
        ``teleport_displacement`` with no requested tile; the digest
        rejects that shape too, so the single-artifact audit raises and
        the SWEEP books a named skip (covered in the sweep tests).
        """
        source = _write_run(
            tmp_path / "old-displacement.events.jsonl",
            [
                _event(
                    "2026-08-05T00:00:00",
                    "DIAGNOSTIC",
                    "diagnostic_kind=teleport_displacement",
                    diagnostic_kind="teleport_displacement",
                ),
            ],
        )

        with pytest.raises(KeyError, match="requested_x"):
            audit_events_artifact(source)

    def test_physics_divergences_are_tallied(self, tmp_path: Path) -> None:
        """The fuel book's residual detector count rides along unflagged."""
        source = _write_run(
            tmp_path / "physics.events.jsonl",
            [
                _event(
                    "2026-08-05T00:00:00",
                    "DIAGNOSTIC",
                    "diagnostic_kind=physics_divergence",
                    diagnostic_kind="physics_divergence",
                    residual=-20,
                ),
            ],
        )

        audit = audit_events_artifact(source)

        assert audit["physics_divergences"] == 1
        assert audit["flags"] == []

    def test_empty_artifact_raises(self, tmp_path: Path) -> None:
        """A no-events artifact is an error, not a zero audit."""
        source = tmp_path / "empty.events.jsonl"
        source.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="no events"):
            audit_events_artifact(source)


class TestCollectFlags:
    """Each disagreement produces its own named flag."""

    def test_a_clean_audit_has_no_flags(self) -> None:
        """Agreement everywhere flags nothing."""
        assert _collect_flags(_audit()) == []

    def test_digest_kill_disagreement_is_flagged(self) -> None:
        """The exact bug class this instrument exists for."""
        flags = _collect_flags(_audit(digest_kills=1))

        assert flags == ["digest kills=1 disagrees with wire recount 2"]

    def test_scorecard_kill_disagreement_is_flagged(self) -> None:
        """The scorecard's copy of the kill law is diffed too."""
        flags = _collect_flags(_audit(scorecard_kills=3))

        assert flags == ["scorecard kills=3 disagrees with wire recount 2"]

    def test_death_disagreement_is_flagged(self) -> None:
        """deaths=0 through real deaths was the founding bug."""
        flags = _collect_flags(_audit(digest_deaths=0))

        assert flags == ["digest deaths=0 disagrees with wire recount 1"]

    def test_radar_drift_beyond_tolerance_is_flagged(self) -> None:
        """One press of drift is a sampling boundary; more is a defect."""
        assert _collect_flags(_audit(radar_drift=1)) == []
        assert _collect_flags(_audit(radar_drift=-2)) == [
            "radar book drift -2 -- inventory tracking disagrees with first+gains-spends"
        ]


class TestCorpusSweep:
    """Tests for the directory sweep and its roll-up."""

    def test_sweep_counts_clean_flagged_and_skipped(self, tmp_path: Path) -> None:
        """Every artifact lands in exactly one bucket, none silently."""
        _write_run(tmp_path / "a" / "bot-1.events.jsonl", _clean_run_lines())
        _write_run(
            tmp_path / "a" / "bot-2.events.jsonl",
            [
                _event("2026-08-05T00:00:00", "WIRE", "shoot(1,1,id=5)"),
                _event("2026-08-05T00:00:01", "WIRE", "shoot(1,1,id=5)"),
            ],
        )
        (tmp_path / "a" / "empty.events.jsonl").write_text("", encoding="utf-8")
        _write_run(tmp_path / "a" / "latest.events.jsonl", _clean_run_lines())
        # An archive era missing a field a strict reader requires is a
        # NAMED skip, not a crash (the first real-corpus sweep died on
        # this before the KeyError arm existed).
        _write_run(
            tmp_path / "a" / "old-era.events.jsonl",
            [
                _event(
                    "2026-08-05T00:00:00",
                    "DIAGNOSTIC",
                    "diagnostic_kind=inventory_sample",
                    diagnostic_kind="inventory_sample",
                    radar=5,
                ),
            ],
        )

        corpus = audit_corpus(tmp_path)

        assert corpus["runs_audited"] == 2
        assert corpus["runs_flagged"] == 1
        assert corpus["runs_skipped"] == 2
        assert "empty.events.jsonl" in corpus["skipped"][0]
        assert "old-era.events.jsonl" in corpus["skipped"][1]
        assert corpus["kind_counts"]["tank_identity"] == 2
        assert len(corpus["audits"]) == 1

    def test_render_names_every_bucket(self, tmp_path: Path) -> None:
        """Flags, skips, and kind tallies all reach the table."""
        _write_run(
            tmp_path / "bot-1.events.jsonl",
            [
                _event("2026-08-05T00:00:00", "WIRE", "shoot(1,1,id=5)"),
                _event("2026-08-05T00:00:01", "WIRE", "shoot(1,1,id=5)"),
                _identity("2026-08-05T00:00:02", 601),
            ],
        )
        (tmp_path / "empty.events.jsonl").write_text("", encoding="utf-8")

        rendered = render_corpus_audit(audit_corpus(tmp_path))

        assert "1 audited, 1 flagged, 1 skipped" in rendered
        assert "SKIPPED" in rendered
        assert "FLAG" in rendered
        assert "tank_identity x1" in rendered

    def test_cli_exits_one_on_flags_and_zero_when_clean(self, tmp_path: Path) -> None:
        """The sweep can gate: flagged corpus fails, clean corpus passes."""
        clean_root = tmp_path / "clean"
        _write_run(clean_root / "bot-1.events.jsonl", _clean_run_lines())
        flagged_root = tmp_path / "flagged"
        _write_run(
            flagged_root / "bot-1.events.jsonl",
            [
                _event("2026-08-05T00:00:00", "WIRE", "shoot(1,1,id=5)"),
                _event("2026-08-05T00:00:01", "WIRE", "shoot(1,1,id=5)"),
            ],
        )
        original_get_argv = _test_hooks.get_argv
        try:
            _test_hooks.get_argv = lambda: ["tankpit-corpus-audit", str(clean_root)]
            assert main() == 0
            _test_hooks.get_argv = lambda: ["tankpit-corpus-audit", str(flagged_root)]
            assert main() == 1
        finally:
            _test_hooks.get_argv = original_get_argv
