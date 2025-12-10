"""Tests for row conversion functions in postgres module."""

from __future__ import annotations

import pytest

from covenant_persistence.postgres import (
    _require_frequency,
    _require_status,
    _require_threshold_direction,
    _row_to_covenant,
    _row_to_covenant_result,
    _row_to_deal,
    _row_to_measurement,
)


class TestRowToDeal:
    """Tests for _row_to_deal function."""

    def test_valid_row_returns_deal(self) -> None:
        """Valid row tuple converts to Deal TypedDict."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "Test Deal",
            "Acme Corp",
            "Technology",
            "North America",
            100_000_000,
            "USD",
            "2025-12-31",
        )
        deal = _row_to_deal(row)
        assert deal["id"]["value"] == "deal-uuid-123"
        assert deal["name"] == "Test Deal"
        assert deal["borrower"] == "Acme Corp"
        assert deal["sector"] == "Technology"
        assert deal["region"] == "North America"
        assert deal["commitment_amount_cents"] == 100_000_000
        assert deal["currency"] == "USD"
        assert deal["maturity_date_iso"] == "2025-12-31"

    def test_invalid_id_type_raises(self) -> None:
        """Non-string id raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            123,  # Should be str
            "Test Deal",
            "Acme Corp",
            "Technology",
            "North America",
            100_000_000,
            "USD",
            "2025-12-31",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_deal(row)
        assert "Expected str for id" in str(exc_info.value)

    def test_invalid_name_type_raises(self) -> None:
        """Non-string name raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            123,  # Should be str
            "Acme Corp",
            "Technology",
            "North America",
            100_000_000,
            "USD",
            "2025-12-31",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_deal(row)
        assert "Expected str for name" in str(exc_info.value)

    def test_invalid_borrower_type_raises(self) -> None:
        """Non-string borrower raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "Test Deal",
            123,  # Should be str
            "Technology",
            "North America",
            100_000_000,
            "USD",
            "2025-12-31",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_deal(row)
        assert "Expected str for borrower" in str(exc_info.value)

    def test_invalid_sector_type_raises(self) -> None:
        """Non-string sector raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "Test Deal",
            "Acme Corp",
            123,  # Should be str
            "North America",
            100_000_000,
            "USD",
            "2025-12-31",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_deal(row)
        assert "Expected str for sector" in str(exc_info.value)

    def test_invalid_region_type_raises(self) -> None:
        """Non-string region raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "Test Deal",
            "Acme Corp",
            "Technology",
            123,  # Should be str
            100_000_000,
            "USD",
            "2025-12-31",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_deal(row)
        assert "Expected str for region" in str(exc_info.value)

    def test_invalid_commitment_type_raises(self) -> None:
        """Non-int commitment raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "Test Deal",
            "Acme Corp",
            "Technology",
            "North America",
            "100000000",  # Should be int
            "USD",
            "2025-12-31",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_deal(row)
        assert "Expected int for commitment" in str(exc_info.value)

    def test_invalid_currency_type_raises(self) -> None:
        """Non-string currency raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "Test Deal",
            "Acme Corp",
            "Technology",
            "North America",
            100_000_000,
            123,  # Should be str
            "2025-12-31",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_deal(row)
        assert "Expected str for currency" in str(exc_info.value)

    def test_invalid_maturity_type_raises(self) -> None:
        """Non-string maturity raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "Test Deal",
            "Acme Corp",
            "Technology",
            "North America",
            100_000_000,
            "USD",
            123,  # Should be str
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_deal(row)
        assert "Expected str for maturity" in str(exc_info.value)


class TestRowToCovenant:
    """Tests for _row_to_covenant function."""

    def test_valid_row_returns_covenant(self) -> None:
        """Valid row tuple converts to Covenant TypedDict."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "deal-uuid-456",
            "Debt to EBITDA",
            "total_debt / ebitda",
            3_500_000,
            "<=",
            "QUARTERLY",
        )
        covenant = _row_to_covenant(row)
        assert covenant["id"]["value"] == "cov-uuid-123"
        assert covenant["deal_id"]["value"] == "deal-uuid-456"
        assert covenant["name"] == "Debt to EBITDA"
        assert covenant["formula"] == "total_debt / ebitda"
        assert covenant["threshold_value_scaled"] == 3_500_000
        assert covenant["threshold_direction"] == "<="
        assert covenant["frequency"] == "QUARTERLY"

    def test_gte_threshold_direction(self) -> None:
        """Greater-than-or-equal direction is accepted."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "deal-uuid-456",
            "Interest Coverage",
            "ebitda / interest",
            2_000_000,
            ">=",
            "ANNUAL",
        )
        covenant = _row_to_covenant(row)
        assert covenant["threshold_direction"] == ">="
        assert covenant["frequency"] == "ANNUAL"

    def test_invalid_id_type_raises(self) -> None:
        """Non-string id raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            123,  # Should be str
            "deal-uuid-456",
            "Test",
            "a / b",
            3_500_000,
            "<=",
            "QUARTERLY",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant(row)
        assert "Expected str for id" in str(exc_info.value)

    def test_invalid_deal_id_type_raises(self) -> None:
        """Non-string deal_id raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            123,  # Should be str
            "Test",
            "a / b",
            3_500_000,
            "<=",
            "QUARTERLY",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant(row)
        assert "Expected str for deal_id" in str(exc_info.value)

    def test_invalid_name_type_raises(self) -> None:
        """Non-string name raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "deal-uuid-456",
            123,  # Should be str
            "a / b",
            3_500_000,
            "<=",
            "QUARTERLY",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant(row)
        assert "Expected str for name" in str(exc_info.value)

    def test_invalid_formula_type_raises(self) -> None:
        """Non-string formula raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "deal-uuid-456",
            "Test",
            123,  # Should be str
            3_500_000,
            "<=",
            "QUARTERLY",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant(row)
        assert "Expected str for formula" in str(exc_info.value)

    def test_invalid_threshold_type_raises(self) -> None:
        """Non-int threshold raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "deal-uuid-456",
            "Test",
            "a / b",
            "3500000",  # Should be int
            "<=",
            "QUARTERLY",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant(row)
        assert "Expected int for threshold" in str(exc_info.value)

    def test_invalid_direction_type_raises(self) -> None:
        """Non-string direction raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "deal-uuid-456",
            "Test",
            "a / b",
            3_500_000,
            123,  # Should be str
            "QUARTERLY",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant(row)
        assert "Expected str for direction" in str(exc_info.value)

    def test_invalid_frequency_type_raises(self) -> None:
        """Non-string frequency raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "deal-uuid-456",
            "Test",
            "a / b",
            3_500_000,
            "<=",
            123,  # Should be str
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant(row)
        assert "Expected str for frequency" in str(exc_info.value)


class TestRowToMeasurement:
    """Tests for _row_to_measurement function."""

    def test_valid_row_returns_measurement(self) -> None:
        """Valid row tuple converts to Measurement TypedDict."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "2024-01-01",
            "2024-03-31",
            "total_debt",
            100_000_000,
        )
        measurement = _row_to_measurement(row)
        assert measurement["deal_id"]["value"] == "deal-uuid-123"
        assert measurement["period_start_iso"] == "2024-01-01"
        assert measurement["period_end_iso"] == "2024-03-31"
        assert measurement["metric_name"] == "total_debt"
        assert measurement["metric_value_scaled"] == 100_000_000

    def test_invalid_deal_id_type_raises(self) -> None:
        """Non-string deal_id raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            123,  # Should be str
            "2024-01-01",
            "2024-03-31",
            "total_debt",
            100_000_000,
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_measurement(row)
        assert "Expected str for deal_id" in str(exc_info.value)

    def test_invalid_period_start_type_raises(self) -> None:
        """Non-string period_start raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            123,  # Should be str
            "2024-03-31",
            "total_debt",
            100_000_000,
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_measurement(row)
        assert "Expected str for period_start" in str(exc_info.value)

    def test_invalid_period_end_type_raises(self) -> None:
        """Non-string period_end raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "2024-01-01",
            123,  # Should be str
            "total_debt",
            100_000_000,
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_measurement(row)
        assert "Expected str for period_end" in str(exc_info.value)

    def test_invalid_metric_name_type_raises(self) -> None:
        """Non-string metric_name raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "2024-01-01",
            "2024-03-31",
            123,  # Should be str
            100_000_000,
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_measurement(row)
        assert "Expected str for metric_name" in str(exc_info.value)

    def test_invalid_metric_value_type_raises(self) -> None:
        """Non-int metric_value raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "deal-uuid-123",
            "2024-01-01",
            "2024-03-31",
            "total_debt",
            "100000000",  # Should be int
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_measurement(row)
        assert "Expected int for metric_value" in str(exc_info.value)


class TestRowToCovenantResult:
    """Tests for _row_to_covenant_result function."""

    def test_valid_row_returns_result(self) -> None:
        """Valid row tuple converts to CovenantResult TypedDict."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "2024-01-01",
            "2024-03-31",
            2_500_000,
            "OK",
        )
        result = _row_to_covenant_result(row)
        assert result["covenant_id"]["value"] == "cov-uuid-123"
        assert result["period_start_iso"] == "2024-01-01"
        assert result["period_end_iso"] == "2024-03-31"
        assert result["calculated_value_scaled"] == 2_500_000
        assert result["status"] == "OK"

    def test_near_breach_status(self) -> None:
        """NEAR_BREACH status is accepted."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "2024-01-01",
            "2024-03-31",
            3_400_000,
            "NEAR_BREACH",
        )
        result = _row_to_covenant_result(row)
        assert result["status"] == "NEAR_BREACH"

    def test_breach_status(self) -> None:
        """BREACH status is accepted."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "2024-01-01",
            "2024-03-31",
            4_000_000,
            "BREACH",
        )
        result = _row_to_covenant_result(row)
        assert result["status"] == "BREACH"

    def test_invalid_covenant_id_type_raises(self) -> None:
        """Non-string covenant_id raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            123,  # Should be str
            "2024-01-01",
            "2024-03-31",
            2_500_000,
            "OK",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant_result(row)
        assert "Expected str for covenant_id" in str(exc_info.value)

    def test_invalid_period_start_type_raises(self) -> None:
        """Non-string period_start raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            123,  # Should be str
            "2024-03-31",
            2_500_000,
            "OK",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant_result(row)
        assert "Expected str for period_start" in str(exc_info.value)

    def test_invalid_period_end_type_raises(self) -> None:
        """Non-string period_end raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "2024-01-01",
            123,  # Should be str
            2_500_000,
            "OK",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant_result(row)
        assert "Expected str for period_end" in str(exc_info.value)

    def test_invalid_calculated_type_raises(self) -> None:
        """Non-int calculated raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "2024-01-01",
            "2024-03-31",
            "2500000",  # Should be int
            "OK",
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant_result(row)
        assert "Expected int for calculated" in str(exc_info.value)

    def test_invalid_status_type_raises(self) -> None:
        """Non-string status raises TypeError."""
        row: tuple[str | int | bool | None, ...] = (
            "cov-uuid-123",
            "2024-01-01",
            "2024-03-31",
            2_500_000,
            123,  # Should be str
        )
        with pytest.raises(TypeError) as exc_info:
            _row_to_covenant_result(row)
        assert "Expected str for status" in str(exc_info.value)


class TestRequireThresholdDirection:
    """Tests for _require_threshold_direction function."""

    def test_lte_returns_lte(self) -> None:
        """Less-than-or-equal string returns typed literal."""
        result = _require_threshold_direction("<=")
        assert result == "<="

    def test_gte_returns_gte(self) -> None:
        """Greater-than-or-equal string returns typed literal."""
        result = _require_threshold_direction(">=")
        assert result == ">="

    def test_invalid_raises(self) -> None:
        """Invalid direction raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _require_threshold_direction("<")
        assert "Invalid threshold direction" in str(exc_info.value)


class TestRequireFrequency:
    """Tests for _require_frequency function."""

    def test_quarterly_returns_quarterly(self) -> None:
        """QUARTERLY string returns typed literal."""
        result = _require_frequency("QUARTERLY")
        assert result == "QUARTERLY"

    def test_annual_returns_annual(self) -> None:
        """ANNUAL string returns typed literal."""
        result = _require_frequency("ANNUAL")
        assert result == "ANNUAL"

    def test_invalid_raises(self) -> None:
        """Invalid frequency raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _require_frequency("MONTHLY")
        assert "Invalid frequency" in str(exc_info.value)


class TestRequireStatus:
    """Tests for _require_status function."""

    def test_ok_returns_ok(self) -> None:
        """OK string returns typed literal."""
        result = _require_status("OK")
        assert result == "OK"

    def test_near_breach_returns_near_breach(self) -> None:
        """NEAR_BREACH string returns typed literal."""
        result = _require_status("NEAR_BREACH")
        assert result == "NEAR_BREACH"

    def test_breach_returns_breach(self) -> None:
        """BREACH string returns typed literal."""
        result = _require_status("BREACH")
        assert result == "BREACH"

    def test_invalid_raises(self) -> None:
        """Invalid status raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _require_status("INVALID")
        assert "Invalid status" in str(exc_info.value)
