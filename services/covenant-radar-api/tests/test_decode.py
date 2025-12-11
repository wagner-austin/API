"""Tests for HTTP request body parsing."""

from __future__ import annotations

import pytest
from covenant_domain import DealId
from platform_core.json_utils import InvalidJsonError, JSONTypeError

from covenant_radar_api.api.decode import (
    parse_covenant_id_request,
    parse_covenant_request,
    parse_deal_id_request,
    parse_deal_request,
    parse_evaluate_request,
    parse_measurements_request,
    parse_predict_request,
    parse_train_request,
    parse_update_deal_request,
)


class TestParseDealRequest:
    """Tests for parse_deal_request."""

    def test_valid_deal_request(self) -> None:
        """Test parsing a valid deal request."""
        body = b"""{
            "id": {"value": "deal-123"},
            "name": "Test Deal",
            "borrower": "Acme Corp",
            "sector": "Technology",
            "region": "North America",
            "commitment_amount_cents": 1000000,
            "currency": "USD",
            "maturity_date_iso": "2025-12-31"
        }"""
        result = parse_deal_request(body)

        assert result["id"]["value"] == "deal-123"
        assert result["name"] == "Test Deal"
        assert result["borrower"] == "Acme Corp"
        assert result["sector"] == "Technology"
        assert result["region"] == "North America"
        assert result["commitment_amount_cents"] == 1000000
        assert result["currency"] == "USD"
        assert result["maturity_date_iso"] == "2025-12-31"

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing required field raises JSONTypeError."""
        body = b"""{"id": {"value": "deal-123"}}"""
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_deal_request(body)

    def test_invalid_json_raises(self) -> None:
        """Test that invalid JSON raises InvalidJsonError."""
        body = b"not valid json"
        with pytest.raises(InvalidJsonError):
            parse_deal_request(body)

    def test_non_object_raises_type_error(self) -> None:
        """Test that non-object JSON raises TypeError."""
        body = b"[]"
        with pytest.raises(TypeError, match="Request body must be a JSON object"):
            parse_deal_request(body)


class TestParseUpdateDealRequest:
    """Tests for parse_update_deal_request."""

    def test_valid_update_request(self) -> None:
        """Test parsing a valid update deal request."""
        body = b"""{
            "name": "Updated Deal",
            "borrower": "New Corp",
            "sector": "Finance",
            "region": "Europe",
            "commitment_amount_cents": 2000000,
            "currency": "EUR",
            "maturity_date_iso": "2026-06-30"
        }"""
        deal_id = DealId(value="existing-deal-id")
        result = parse_update_deal_request(body, deal_id)

        assert result["id"]["value"] == "existing-deal-id"
        assert result["name"] == "Updated Deal"
        assert result["borrower"] == "New Corp"
        assert result["commitment_amount_cents"] == 2000000

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing required field raises JSONTypeError."""
        body = b"""{"name": "Test"}"""
        deal_id = DealId(value="test-id")
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_update_deal_request(body, deal_id)

    def test_wrong_type_raises_json_type_error(self) -> None:
        """Test that wrong field type raises JSONTypeError."""
        body = b"""{
            "name": 123,
            "borrower": "Corp",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": 1000,
            "currency": "USD",
            "maturity_date_iso": "2025-01-01"
        }"""
        deal_id = DealId(value="test-id")
        with pytest.raises(JSONTypeError, match="Field 'name' must be a string"):
            parse_update_deal_request(body, deal_id)

    def test_wrong_int_type_raises_json_type_error(self) -> None:
        """Test that wrong int field type raises JSONTypeError."""
        body = b"""{
            "name": "Deal",
            "borrower": "Corp",
            "sector": "Tech",
            "region": "NA",
            "commitment_amount_cents": "not an int",
            "currency": "USD",
            "maturity_date_iso": "2025-01-01"
        }"""
        deal_id = DealId(value="test-id")
        match_msg = "Field 'commitment_amount_cents' must be an integer"
        with pytest.raises(JSONTypeError, match=match_msg):
            parse_update_deal_request(body, deal_id)


class TestParseDealIdRequest:
    """Tests for parse_deal_id_request."""

    def test_valid_deal_id_request(self) -> None:
        """Test parsing a valid deal ID request."""
        body = b"""{"value": "deal-123"}"""
        result = parse_deal_id_request(body)

        assert result["value"] == "deal-123"

    def test_missing_value_raises_json_type_error(self) -> None:
        """Test that missing value field raises JSONTypeError."""
        body = b"""{}"""
        with pytest.raises(JSONTypeError, match="Missing required field 'value'"):
            parse_deal_id_request(body)


class TestParseCovenantIdRequest:
    """Tests for parse_covenant_id_request."""

    def test_valid_covenant_id_request(self) -> None:
        """Test parsing a valid covenant ID request."""
        body = b"""{"value": "cov-456"}"""
        result = parse_covenant_id_request(body)

        assert result["value"] == "cov-456"

    def test_missing_value_raises_json_type_error(self) -> None:
        """Test that missing value field raises JSONTypeError."""
        body = b"""{}"""
        with pytest.raises(JSONTypeError, match="Missing required field 'value'"):
            parse_covenant_id_request(body)


class TestParseCovenantRequest:
    """Tests for parse_covenant_request."""

    def test_valid_covenant_request(self) -> None:
        """Test parsing a valid covenant request."""
        body = b"""{
            "id": {"value": "cov-123"},
            "deal_id": {"value": "deal-456"},
            "name": "Debt to EBITDA",
            "formula": "total_debt / ebitda",
            "threshold_value_scaled": 3500000,
            "threshold_direction": "<=",
            "frequency": "QUARTERLY"
        }"""
        result = parse_covenant_request(body)

        assert result["id"]["value"] == "cov-123"
        assert result["deal_id"]["value"] == "deal-456"
        assert result["name"] == "Debt to EBITDA"
        assert result["formula"] == "total_debt / ebitda"
        assert result["threshold_value_scaled"] == 3500000
        assert result["threshold_direction"] == "<="
        assert result["frequency"] == "QUARTERLY"

    def test_invalid_direction_raises_json_type_error(self) -> None:
        """Test that invalid threshold direction raises JSONTypeError."""
        body = b"""{
            "id": {"value": "cov-123"},
            "deal_id": {"value": "deal-456"},
            "name": "Test",
            "formula": "a / b",
            "threshold_value_scaled": 1000000,
            "threshold_direction": "==",
            "frequency": "QUARTERLY"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid ThresholdDirection"):
            parse_covenant_request(body)

    def test_invalid_frequency_raises_json_type_error(self) -> None:
        """Test that invalid frequency raises JSONTypeError."""
        body = b"""{
            "id": {"value": "cov-123"},
            "deal_id": {"value": "deal-456"},
            "name": "Test",
            "formula": "a / b",
            "threshold_value_scaled": 1000000,
            "threshold_direction": "<=",
            "frequency": "MONTHLY"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid CovenantFrequency"):
            parse_covenant_request(body)


class TestParseMeasurementsRequest:
    """Tests for parse_measurements_request."""

    def test_valid_measurements_request(self) -> None:
        """Test parsing a valid measurements request."""
        body = b"""{
            "measurements": [
                {
                    "deal_id": {"value": "deal-123"},
                    "period_start_iso": "2024-01-01",
                    "period_end_iso": "2024-03-31",
                    "metric_name": "total_debt",
                    "metric_value_scaled": 5000000000
                },
                {
                    "deal_id": {"value": "deal-123"},
                    "period_start_iso": "2024-01-01",
                    "period_end_iso": "2024-03-31",
                    "metric_name": "ebitda",
                    "metric_value_scaled": 1500000000
                }
            ]
        }"""
        result = parse_measurements_request(body)

        assert len(result) == 2
        assert result[0]["metric_name"] == "total_debt"
        assert result[1]["metric_name"] == "ebitda"

    def test_empty_measurements_list(self) -> None:
        """Test parsing empty measurements list."""
        body = b"""{"measurements": []}"""
        result = parse_measurements_request(body)

        assert len(result) == 0

    def test_missing_measurements_key_raises(self) -> None:
        """Test that missing measurements key raises JSONTypeError."""
        body = b"""{}"""
        with pytest.raises(JSONTypeError, match="Missing required field 'measurements'"):
            parse_measurements_request(body)

    def test_non_list_measurements_raises_json_type_error(self) -> None:
        """Test that non-list measurements raises JSONTypeError."""
        body = b"""{"measurements": "not a list"}"""
        with pytest.raises(JSONTypeError, match="Field 'measurements' must be an array"):
            parse_measurements_request(body)

    def test_non_object_measurement_raises_type_error(self) -> None:
        """Test that non-object measurement item raises TypeError."""
        body = b"""{"measurements": ["not an object"]}"""
        with pytest.raises(TypeError, match="Each measurement must be a JSON object"):
            parse_measurements_request(body)

    def test_invalid_measurement_field_raises(self) -> None:
        """Test that invalid measurement field raises appropriate error."""
        body = b"""{
            "measurements": [
                {
                    "deal_id": {"value": "deal-123"},
                    "period_start_iso": 12345,
                    "period_end_iso": "2024-03-31",
                    "metric_name": "test",
                    "metric_value_scaled": 1000
                }
            ]
        }"""
        with pytest.raises(JSONTypeError, match="Field 'period_start_iso' must be a string"):
            parse_measurements_request(body)


class TestParseEvaluateRequest:
    """Tests for parse_evaluate_request."""

    def test_valid_evaluate_request(self) -> None:
        """Test parsing a valid evaluate request."""
        body = b"""{
            "deal_id": "deal-123",
            "period_start_iso": "2024-01-01",
            "period_end_iso": "2024-03-31",
            "tolerance_ratio_scaled": 100000
        }"""
        result = parse_evaluate_request(body)

        assert result["deal_id"] == "deal-123"
        assert result["period_start_iso"] == "2024-01-01"
        assert result["period_end_iso"] == "2024-03-31"
        assert result["tolerance_ratio_scaled"] == 100000

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing field raises JSONTypeError."""
        body = b"""{"deal_id": "deal-123"}"""
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_evaluate_request(body)


class TestParsePredictRequest:
    """Tests for parse_predict_request."""

    def test_valid_predict_request(self) -> None:
        """Test parsing a valid predict request."""
        body = b"""{"deal_id": "deal-456"}"""
        result = parse_predict_request(body)

        assert result["deal_id"] == "deal-456"

    def test_missing_deal_id_raises_json_type_error(self) -> None:
        """Test that missing deal_id raises JSONTypeError."""
        body = b"""{}"""
        with pytest.raises(JSONTypeError, match="Missing required field 'deal_id'"):
            parse_predict_request(body)


class TestParseTrainRequest:
    """Tests for parse_train_request."""

    def test_valid_train_request(self) -> None:
        """Test parsing a valid train request."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "early_stopping_rounds": 10
        }"""
        result = parse_train_request(body)

        assert result["learning_rate"] == 0.1
        assert result["max_depth"] == 6
        assert result["n_estimators"] == 100
        assert result["subsample"] == 0.8
        assert result["colsample_bytree"] == 0.8
        assert result["random_state"] == 42
        assert result["train_ratio"] == 0.7
        assert result["val_ratio"] == 0.15
        assert result["test_ratio"] == 0.15
        assert result["early_stopping_rounds"] == 10
        # reg_alpha/reg_lambda default when not provided
        assert result["reg_alpha"] == 0.0
        assert result["reg_lambda"] == 1.0
        assert result["device"] == "auto"

    def test_request_with_defaults(self) -> None:
        """Test parsing with optional fields defaulted."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }"""
        result = parse_train_request(body)

        assert result["learning_rate"] == 0.1
        # Default values for optional fields
        assert result["train_ratio"] == 0.7
        assert result["val_ratio"] == 0.15
        assert result["test_ratio"] == 0.15
        assert result["early_stopping_rounds"] == 10
        assert result["reg_alpha"] == 0.0
        assert result["reg_lambda"] == 1.0
        assert result["device"] == "auto"

    def test_train_request_with_regularization_and_scale(self) -> None:
        """Test parsing reg params, device, and scale_pos_weight."""
        body = b"""{
            "learning_rate": 0.2,
            "max_depth": 4,
            "n_estimators": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "random_state": 7,
            "device": "cuda",
            "reg_alpha": 2.5,
            "reg_lambda": 3.5,
            "scale_pos_weight": 1.2
        }"""
        result = parse_train_request(body)

        assert result["device"] == "cuda"
        assert result["reg_alpha"] == 2.5
        assert result["reg_lambda"] == 3.5
        assert result["scale_pos_weight"] == 1.2
        assert result["n_estimators"] == 50

    def test_train_request_invalid_scale_pos_weight(self) -> None:
        """Test parsing rejects invalid scale_pos_weight type."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "scale_pos_weight": "heavy"
        }"""
        with pytest.raises(JSONTypeError, match="scale_pos_weight must be a number"):
            parse_train_request(body)

    def test_train_request_invalid_ratio_type(self) -> None:
        """Test parsing rejects non-numeric ratio values."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": "big"
        }"""
        with pytest.raises(JSONTypeError, match="Field 'train_ratio' must be a number"):
            parse_train_request(body)

    def test_train_request_invalid_device(self) -> None:
        """Test parsing rejects unsupported device value."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "tpu"
        }"""
        with pytest.raises(JSONTypeError, match="device must be one of: cpu, cuda, auto"):
            parse_train_request(body)

    def test_train_request_device_cpu(self) -> None:
        """Test parsing accepts explicit CPU device."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "cpu"
        }"""
        result = parse_train_request(body)
        assert result["device"] == "cpu"

    def test_train_request_device_auto_string(self) -> None:
        """Test parsing accepts explicit auto device string."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": "auto"
        }"""
        result = parse_train_request(body)
        assert result["device"] == "auto"

    def test_train_request_non_string_device(self) -> None:
        """Test parsing rejects non-string device types."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device": 123
        }"""
        with pytest.raises(JSONTypeError, match="device must be a string"):
            parse_train_request(body)

    def test_early_stopping_as_float(self) -> None:
        """Test parsing early_stopping_rounds as float (converts to int)."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 15.0
        }"""
        result = parse_train_request(body)

        assert result["early_stopping_rounds"] == 15

    def test_early_stopping_invalid_type(self) -> None:
        """Test parsing rejects non-numeric early_stopping_rounds."""
        body = b"""{
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": "fast"
        }"""
        with pytest.raises(JSONTypeError, match="Field 'early_stopping_rounds' must be a number"):
            parse_train_request(body)

    def test_missing_field_raises_json_type_error(self) -> None:
        """Test that missing field raises JSONTypeError."""
        body = b"""{"learning_rate": 0.1}"""
        with pytest.raises(JSONTypeError, match="Missing required field"):
            parse_train_request(body)
