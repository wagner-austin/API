"""Tests for streaming worker helper functions.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from covenant_radar_api.streaming.worker import (
    _count_breaches,
    _current_iso_timestamp,
    _determine_alert_severity,
    _determine_alert_type,
    _determine_evaluation_status,
    _generate_alert_message,
    _generate_event_id,
    _make_buffer_key,
    _scale_metrics,
    make_default_worker_config,
)

from ._test_worker_fixtures import make_covenant_result, make_measurement_event


class TestMakeDefaultWorkerConfig:
    """Tests for make_default_worker_config."""

    def test_returns_worker_config(self) -> None:
        """Returns a WorkerConfig with all required fields."""
        config = make_default_worker_config()

        assert config["model_version"] == "v1.0.0"
        assert config["batch_size"] == 100
        assert config["poll_timeout_seconds"] == 1.0
        assert config["alert_threshold"] == 0.80
        assert config["commit_interval"] == 10
        assert config["buffer_timeout_seconds"] == 5.0
        assert config["min_metrics_per_period"] == 3
        assert config["tolerance_ratio_scaled"] == 100_000


class TestMakeBufferKey:
    """Tests for _make_buffer_key."""

    def test_creates_tuple_key(self) -> None:
        """Creates tuple from event fields."""
        event = make_measurement_event(
            deal_id="deal-123",
            period_start="2024-01-01",
            period_end="2024-03-31",
        )
        key = _make_buffer_key(event)
        assert key == ("deal-123", "2024-01-01", "2024-03-31")


class TestDetermineEvaluationStatus:
    """Tests for _determine_evaluation_status."""

    def test_returns_ok_for_empty(self) -> None:
        """Returns OK for empty results."""
        status = _determine_evaluation_status(())
        assert status == "OK"

    def test_returns_ok_for_all_ok(self) -> None:
        """Returns OK when all results are OK."""
        results = (
            make_covenant_result(status="OK"),
            make_covenant_result(covenant_id="cov-002", status="OK"),
        )
        status = _determine_evaluation_status(results)
        assert status == "OK"

    def test_returns_breach_for_any_breach(self) -> None:
        """Returns BREACH if any result is BREACH."""
        results = (
            make_covenant_result(status="OK"),
            make_covenant_result(covenant_id="cov-002", status="BREACH"),
            make_covenant_result(covenant_id="cov-003", status="NEAR_BREACH"),
        )
        status = _determine_evaluation_status(results)
        assert status == "BREACH"

    def test_returns_warning_for_near_breach(self) -> None:
        """Returns WARNING for NEAR_BREACH without BREACH."""
        results = (
            make_covenant_result(status="OK"),
            make_covenant_result(covenant_id="cov-002", status="NEAR_BREACH"),
        )
        status = _determine_evaluation_status(results)
        assert status == "WARNING"


class TestCountBreaches:
    """Tests for _count_breaches."""

    def test_counts_zero_for_empty(self) -> None:
        """Returns 0 for empty results."""
        count = _count_breaches(())
        assert count == 0

    def test_counts_breaches(self) -> None:
        """Counts BREACH status results."""
        results = (
            make_covenant_result(status="OK"),
            make_covenant_result(covenant_id="cov-002", status="BREACH"),
            make_covenant_result(covenant_id="cov-003", status="BREACH"),
            make_covenant_result(covenant_id="cov-004", status="NEAR_BREACH"),
        )
        count = _count_breaches(results)
        assert count == 2


class TestDetermineAlertSeverity:
    """Tests for _determine_alert_severity."""

    def test_critical_at_90_percent(self) -> None:
        """Returns critical at 90% or above."""
        assert _determine_alert_severity(0.90) == "critical"
        assert _determine_alert_severity(0.95) == "critical"
        assert _determine_alert_severity(1.0) == "critical"

    def test_warning_below_90_percent(self) -> None:
        """Returns warning below 90%."""
        assert _determine_alert_severity(0.89) == "warning"
        assert _determine_alert_severity(0.80) == "warning"
        assert _determine_alert_severity(0.50) == "warning"


class TestDetermineAlertType:
    """Tests for _determine_alert_type."""

    def test_breach_for_breach_status(self) -> None:
        """Returns breach for BREACH status."""
        assert _determine_alert_type("BREACH") == "breach"

    def test_high_risk_for_other_status(self) -> None:
        """Returns high_risk for non-BREACH status."""
        assert _determine_alert_type("OK") == "high_risk"
        assert _determine_alert_type("WARNING") == "high_risk"


class TestGenerateAlertMessage:
    """Tests for _generate_alert_message."""

    def test_breach_message(self) -> None:
        """Generates breach message."""
        msg = _generate_alert_message(
            deal_id="deal-001",
            deal_name="Test Deal",
            risk_probability=0.85,
            evaluation_status="BREACH",
            breaches_count=2,
        )
        assert "Test Deal" in msg
        assert "deal-001" in msg
        assert "2 covenant breach(es)" in msg
        assert "85.0%" in msg
        assert "Immediate review required" in msg

    def test_high_risk_message(self) -> None:
        """Generates high risk message without breach."""
        msg = _generate_alert_message(
            deal_id="deal-001",
            deal_name="Test Deal",
            risk_probability=0.82,
            evaluation_status="OK",
            breaches_count=0,
        )
        assert "Test Deal" in msg
        assert "elevated risk" in msg
        assert "82.0%" in msg
        assert "No covenant breaches detected" in msg


class TestScaleMetrics:
    """Tests for _scale_metrics."""

    def test_scales_to_million(self) -> None:
        """Scales float metrics to 1M integers."""
        metrics = {"debt_to_equity": 1.5, "current_ratio": 2.25}
        scaled = _scale_metrics(metrics)
        assert scaled == {"debt_to_equity": 1_500_000, "current_ratio": 2_250_000}


class TestGenerateEventId:
    """Tests for _generate_event_id."""

    def test_generates_uuid(self) -> None:
        """Generates a UUID string."""
        event_id = _generate_event_id()
        assert len(event_id) == 36  # UUID format
        assert event_id.count("-") == 4


class TestCurrentIsoTimestamp:
    """Tests for _current_iso_timestamp."""

    def test_generates_iso_timestamp(self) -> None:
        """Generates ISO format timestamp."""
        ts = _current_iso_timestamp()
        assert ts.endswith("Z")
        assert "T" in ts
        assert len(ts) == 20
