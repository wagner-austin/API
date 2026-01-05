"""Tests for helper functions in train_external_job."""

from __future__ import annotations

import pytest
from covenant_ml.types import EvalMetrics
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.worker.train_external_job import (
    _get_meta_filename,
    _metrics_to_json,
    _optional_float,
    _optional_int,
    _parse_device,
)


class TestParseDevice:
    """Tests for _parse_device function."""

    def test_defaults_to_auto(self) -> None:
        """None input returns 'auto'."""
        assert _parse_device(None) == "auto"

    def test_accepts_cpu(self) -> None:
        """'cpu' is accepted."""
        assert _parse_device("cpu") == "cpu"

    def test_accepts_cuda(self) -> None:
        """'cuda' is accepted."""
        assert _parse_device("cuda") == "cuda"

    def test_accepts_auto(self) -> None:
        """'auto' is accepted."""
        assert _parse_device("auto") == "auto"

    def test_rejects_invalid_string(self) -> None:
        """Invalid device string raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            _parse_device("tpu")

    def test_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="device must be a string"):
            _parse_device(123)


class TestOptionalHelpers:
    """Tests for _optional_float and _optional_int helpers."""

    def test_optional_float_returns_default_on_missing(self) -> None:
        """_optional_float returns default when key is missing."""
        assert _optional_float({}, "missing", 0.5) == 0.5

    def test_optional_float_returns_value_when_present(self) -> None:
        """_optional_float returns value when present."""
        assert _optional_float({"val": 0.8}, "val", 0.5) == 0.8

    def test_optional_float_converts_int_to_float(self) -> None:
        """_optional_float converts int to float."""
        assert _optional_float({"val": 5}, "val", 0.0) == 5.0

    def test_optional_float_raises_on_invalid_type(self) -> None:
        """_optional_float raises JSONTypeError on invalid type."""
        with pytest.raises(JSONTypeError, match="must be a number"):
            _optional_float({"val": "string"}, "val", 0.0)

    def test_optional_int_returns_default_on_missing(self) -> None:
        """_optional_int returns default when key is missing."""
        assert _optional_int({}, "missing", 10) == 10

    def test_optional_int_returns_value_when_present(self) -> None:
        """_optional_int returns value when present."""
        assert _optional_int({"val": 20}, "val", 10) == 20

    def test_optional_int_converts_float_to_int(self) -> None:
        """_optional_int converts float to int."""
        assert _optional_int({"val": 15.5}, "val", 0) == 15

    def test_optional_int_raises_on_invalid_type(self) -> None:
        """_optional_int raises JSONTypeError on invalid type."""
        with pytest.raises(JSONTypeError, match="must be a number"):
            _optional_int({"val": "string"}, "val", 0)


class TestGetMetaFilename:
    """Tests for _get_meta_filename function."""

    def test_mlp(self) -> None:
        """Returns correct filename for MLP."""
        assert _get_meta_filename("mlp") == "active_mlp_meta.json"

    def test_lstm(self) -> None:
        """Returns correct filename for LSTM."""
        assert _get_meta_filename("lstm") == "active_lstm_meta.json"

    def test_lightgbm(self) -> None:
        """Returns correct filename for LightGBM."""
        assert _get_meta_filename("lightgbm") == "active_lgbm_meta.json"

    def test_xgboost(self) -> None:
        """Returns empty string for XGBoost (self-describing format)."""
        assert _get_meta_filename("xgboost") == ""


class TestMetricsToJson:
    """Tests for _metrics_to_json function."""

    def test_conversion(self) -> None:
        """_metrics_to_json correctly converts EvalMetrics."""
        metrics: EvalMetrics = {
            "loss": 0.25,
            "ppl": 1.284,
            "auc": 0.85,
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
        }
        result = _metrics_to_json(metrics)

        assert result["loss"] == 0.25
        assert result["auc"] == 0.85
        assert result["accuracy"] == 0.80
        assert result["precision"] == 0.75
        assert result["recall"] == 0.70
        assert result["f1_score"] == 0.72
