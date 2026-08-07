"""Tests for the deterministic ledger checks.

Records are built as literal :class:`RuntimeEventRecordDict` rows so
each check's timing windows and thresholds can be driven exactly.
"""

from __future__ import annotations

from tankpit_bot.diagnostics.ledger_audit import audit_ledger
from tankpit_bot.diagnostics.run_audit_types import FindingDict, make_finding
from tankpit_bot.runtime_records import RuntimeEventRecordDict


def _record(
    timestamp: str,
    channel: str = "DIAGNOSTIC",
    **fields: str | int | float | bool,
) -> RuntimeEventRecordDict:
    """Build one event record with the given fields."""
    return RuntimeEventRecordDict(
        timestamp=timestamp,
        level="INFO",
        logger="tankpit_bot.runtime.events",
        mode="bot",
        channel=channel,
        message="",
        fields=dict(fields),
    )


def _scorecard(timestamp: str = "2026-07-19T00:51:26") -> RuntimeEventRecordDict:
    """Build a session_scorecard record so exit checks stay quiet."""
    return _record(
        timestamp,
        diagnostic_kind="session_scorecard",
        exit_reason="completed",
        ticks=144,
        kills=4,
    )


def _by_check(findings: list[FindingDict], check: str) -> list[FindingDict]:
    """Return the findings produced by one check."""
    return [f for f in findings if f["check"] == check]


def test_empty_artifact_is_a_critical_finding() -> None:
    """No records means the session died before producing anything."""
    assert audit_ledger([]) == [
        make_finding(
            "empty_run",
            "critical",
            "the events artifact contains no records -- the session "
            "died before the game loop produced anything",
        )
    ]


def test_kill_double_registration_inside_window() -> None:
    """The same victim registered twice within 30s is flagged."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T00:50:37",
                diagnostic_kind="tank_deactivated",
                victim_id=511,
                killer_id=1301,
            ),
            _record(
                "2026-07-19T00:50:39",
                diagnostic_kind="tank_deactivated",
                victim_id=511,
                killer_id=-1,
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "kill_double_registration") == [
        make_finding(
            "kill_double_registration",
            "critical",
            "victim 511 registered twice within 30s -- two channels counted one death",
            victim_id=511,
            first="2026-07-19T00:50:37",
            second="2026-07-19T00:50:39",
        )
    ]


def test_kill_re_registration_outside_window_is_a_respawn_re_kill() -> None:
    """The same victim killed again after the window is legitimate."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T00:40:00",
                diagnostic_kind="tank_deactivated",
                victim_id=507,
                killer_id=1301,
            ),
            _record(
                "2026-07-19T00:45:00",
                diagnostic_kind="tank_deactivated",
                victim_id=507,
                killer_id=1301,
            ),
            _record(
                "2026-07-19T00:45:01",
                diagnostic_kind="tank_deactivated",
                killer_id=1301,
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "kill_double_registration") == []


def test_unresolved_decisions_surface_per_action_kind() -> None:
    """The shutdown sweep's pending decisions become one warning each."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T00:51:25",
                diagnostic_kind="session_unresolved_decisions",
                shoot=235,
                teleport=240,
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "unresolved_decision") == [
        make_finding(
            "unresolved_decision",
            "warning",
            "teleport decision 240 never got an outcome before shutdown",
            action_kind="teleport",
            decision_event_id=240,
        ),
        make_finding(
            "unresolved_decision",
            "warning",
            "shoot decision 235 never got an outcome before shutdown",
            action_kind="shoot",
            decision_event_id=235,
        ),
    ]


def test_stall_timeout_is_critical() -> None:
    """Every stall_timeout outcome is its own critical finding."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T00:48:00",
                diagnostic_kind="action_outcome",
                action_kind="scan",
                outcome="stall_timeout",
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "stall_timeout") == [
        make_finding(
            "stall_timeout",
            "critical",
            "scan hit the stall timeout -- the wire never answered and "
            "the bot burned the full wait",
            action_kind="scan",
            timestamp="2026-07-19T00:48:00",
        )
    ]


def test_single_command_rejection_is_info() -> None:
    """One rejection is informational; a missing error code renders -1."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T00:48:31",
                diagnostic_kind="action_outcome",
                action_kind="collect",
                outcome="command_rejected",
                target_x=195,
                target_y=136,
                error_code=4,
            ),
            _record(
                "2026-07-19T00:49:00",
                diagnostic_kind="action_outcome",
                action_kind="move",
                outcome="command_rejected",
                target_x=10,
                target_y=11,
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "command_rejection") == [
        make_finding(
            "command_rejection",
            "info",
            "server rejected a collect with error code 4",
            action_kind="collect",
            error_code=4,
            timestamp="2026-07-19T00:48:31",
        ),
        make_finding(
            "command_rejection",
            "info",
            "server rejected a move with error code -1",
            action_kind="move",
            error_code=-1,
            timestamp="2026-07-19T00:49:00",
        ),
    ]
    assert _by_check(findings, "rejection_retry_loop") == []


def test_superseded_churn_fires_only_above_threshold() -> None:
    """Five superseded outcomes stay quiet; six become churn."""
    quiet = audit_ledger(
        [
            *(
                _record(
                    f"2026-07-19T00:48:0{index}",
                    diagnostic_kind="action_outcome",
                    action_kind="shoot",
                    outcome="superseded",
                )
                for index in range(5)
            ),
            _scorecard(),
        ]
    )
    assert _by_check(quiet, "superseded_churn") == []
    noisy = audit_ledger(
        [
            *(
                _record(
                    f"2026-07-19T00:48:0{index}",
                    diagnostic_kind="action_outcome",
                    action_kind="shoot",
                    outcome="superseded",
                )
                for index in range(6)
            ),
            _scorecard(),
        ]
    )
    assert _by_check(noisy, "superseded_churn") == [
        make_finding(
            "superseded_churn",
            "warning",
            "6 shoot decisions were superseded mid-action -- heavy re-dispatch churn",
            action_kind="shoot",
            count=6,
        )
    ]


def test_tick_cadence_gap_above_threshold_is_flagged() -> None:
    """A 10s tick-to-tick gap is flagged; boolean tick fields are ignored."""
    findings = audit_ledger(
        [
            _record("2026-07-19T00:48:00", channel="AI", tick_n=10),
            _record("2026-07-19T00:48:02", channel="AI", tick_n=11),
            _record("2026-07-19T00:48:12", channel="AI", tick_n=12),
            _record("2026-07-19T00:48:13", channel="AI", tick_n=True),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "tick_cadence_gap") == [
        make_finding(
            "tick_cadence_gap",
            "warning",
            "10s of wall clock between ticks 11 and 12 -- something "
            "waited longer than any healthy cause explains",
            prev_tick=11,
            next_tick=12,
            gap_s=10,
            at="2026-07-19T00:48:12",
        )
    ]


def test_session_exit_reads_the_scorecard() -> None:
    """The scorecard's exit reason becomes the run's exit finding."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T00:51:26",
                diagnostic_kind="session_scorecard",
                exit_reason="no_viable_targets",
                ticks=144,
                kills=4,
            )
        ]
    )
    assert _by_check(findings, "session_exit") == [
        make_finding(
            "session_exit",
            "info",
            "session ended: no_viable_targets",
            exit_reason="no_viable_targets",
            ticks=144,
            kills=4,
        )
    ]


def test_missing_scorecard_is_a_warning() -> None:
    """A run with records but no scorecard died before shutdown ran."""
    findings = audit_ledger([_record("2026-07-19T00:46:23", channel="STATE")])
    assert _by_check(findings, "session_exit") == [
        make_finding(
            "session_exit",
            "warning",
            "no session scorecard in the artifact -- the run died before the shutdown path ran",
        )
    ]


def test_scorecard_with_missing_fields_uses_sentinels() -> None:
    """A malformed scorecard still yields an exit finding with sentinels."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T00:51:26",
                diagnostic_kind="session_scorecard",
                exit_reason=7,
            )
        ]
    )
    assert _by_check(findings, "session_exit") == [
        make_finding(
            "session_exit",
            "info",
            "session ended: unknown",
            exit_reason="unknown",
            ticks=-1,
            kills=-1,
        )
    ]


def test_successful_and_malformed_outcomes_produce_no_findings() -> None:
    """Success outcomes and rows missing typed fields are not failures."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T00:48:00",
                diagnostic_kind="action_outcome",
                action_kind="teleport",
                outcome="landed_exact",
                target_x=100,
                target_y=100,
            ),
            _record(
                "2026-07-19T00:48:02",
                diagnostic_kind="action_outcome",
                action_kind=7,
                outcome="landed_exact",
            ),
            _record(
                "2026-07-19T00:48:04",
                diagnostic_kind="session_unresolved_decisions",
                shoot="not-an-id",
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "command_rejection") == []
    assert _by_check(findings, "rejection_retry_loop") == []
    assert _by_check(findings, "executor_discards") == []
    assert _by_check(findings, "unresolved_decision") == []


def test_repeated_records_for_one_tick_keep_the_first_timestamp() -> None:
    """Only each tick's first timestamp feeds the cadence check."""
    findings = audit_ledger(
        [
            _record("2026-07-19T00:48:00", channel="AI", tick_n=10),
            _record("2026-07-19T00:48:01", channel="WORLD", tick_n=10),
            _record("2026-07-19T00:48:02", channel="AI", tick_n=11),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "tick_cadence_gap") == []


def test_typed_collect_resolutions_classify_distinctly() -> None:
    """pickup_empty and inventory_full get info verdicts; clamped_transfer none.

    A clamped transfer is a SUCCESS (the fuel arrived; the server's
    code 5 is the completion signal) -- it must produce no finding and
    never feed the retry-loop detector. The empty pickup and the
    inventory-full refusal are surfaced as info with their own
    explanations.
    """
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T17:38:44",
                diagnostic_kind="action_outcome",
                action_kind="collect",
                outcome="clamped_transfer",
                target_x=36,
                target_y=113,
            ),
            _record(
                "2026-07-19T17:39:00",
                diagnostic_kind="action_outcome",
                action_kind="collect",
                outcome="pickup_empty",
                target_x=200,
                target_y=128,
            ),
            _record(
                "2026-07-19T17:39:10",
                diagnostic_kind="action_outcome",
                action_kind="collect",
                outcome="inventory_full",
                target_x=210,
                target_y=130,
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "command_rejection") == [
        make_finding(
            "command_rejection",
            "info",
            "pickup found the container drained -- consumed by someone "
            "else between scan and pickup",
            action_kind="collect",
            timestamp="2026-07-19T17:39:00",
        ),
        make_finding(
            "command_rejection",
            "info",
            "equipment pickup refused: all inventory slots full "
            "(beliefs reconciled) -- the fullness gate should have "
            "prevented this dispatch",
            action_kind="collect",
            timestamp="2026-07-19T17:39:10",
        ),
    ]
    assert _by_check(findings, "rejection_retry_loop") == []


def test_repeated_clamped_transfers_are_not_a_retry_loop() -> None:
    """Two clamped transfers on one target are two successes, not churn."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T17:38:44",
                diagnostic_kind="action_outcome",
                action_kind="collect",
                outcome="clamped_transfer",
                target_x=36,
                target_y=113,
            ),
            _record(
                "2026-07-19T17:39:44",
                diagnostic_kind="action_outcome",
                action_kind="collect",
                outcome="clamped_transfer",
                target_x=36,
                target_y=113,
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "rejection_retry_loop") == []
    assert _by_check(findings, "command_rejection") == []


def test_repeated_empty_pickups_on_one_target_are_a_retry_loop() -> None:
    """Two pickup_empty on the same target mean the belief is not learning."""
    findings = audit_ledger(
        [
            _record(
                "2026-07-19T17:39:00",
                diagnostic_kind="action_outcome",
                action_kind="collect",
                outcome="pickup_empty",
                target_x=200,
                target_y=128,
            ),
            _record(
                "2026-07-19T17:39:30",
                diagnostic_kind="action_outcome",
                action_kind="collect",
                outcome="pickup_empty",
                target_x=200,
                target_y=128,
            ),
            _scorecard(),
        ]
    )
    assert _by_check(findings, "rejection_retry_loop") == [
        make_finding(
            "rejection_retry_loop",
            "critical",
            "collect at (200,128) failed 2 times -- replanning is not learning from the failure",
            action_kind="collect",
            target_x=200,
            target_y=128,
            failures="pickup_empty,pickup_empty",
        )
    ]
