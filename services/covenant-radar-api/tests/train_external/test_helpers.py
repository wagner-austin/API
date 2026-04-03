"""Tests for helper functions in train_external_job."""

from __future__ import annotations

import pytest
from covenant_ml.types import EvalMetrics
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.worker._train_external_parsers import (
    _optional_float,
    _optional_int,
    _parse_device,
    _parse_monotonic_constraints,
    _parse_optional_bool,
)
from covenant_radar_api.worker.train_external_job import (
    _get_active_filename,
    _get_meta_filename,
    _metrics_to_json,
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


class TestParseOptionalBool:
    """Tests for _parse_optional_bool function."""

    def test_returns_default_on_missing(self) -> None:
        """Returns default when key is missing."""
        assert _parse_optional_bool({}, "flag", False) is False
        assert _parse_optional_bool({}, "flag", True) is True

    def test_returns_value_when_true(self) -> None:
        """Returns True when value is True."""
        assert _parse_optional_bool({"flag": True}, "flag", False) is True

    def test_returns_value_when_false(self) -> None:
        """Returns False when value is False."""
        assert _parse_optional_bool({"flag": False}, "flag", True) is False

    def test_raises_on_non_boolean(self) -> None:
        """Raises JSONTypeError for non-boolean value."""
        with pytest.raises(JSONTypeError, match="must be a boolean"):
            _parse_optional_bool({"flag": "yes"}, "flag", False)

    def test_raises_on_int(self) -> None:
        """Raises JSONTypeError for int value (not bool)."""
        with pytest.raises(JSONTypeError, match="must be a boolean"):
            _parse_optional_bool({"flag": 1}, "flag", False)


class TestParseMonotonicConstraints:
    """Tests for _parse_monotonic_constraints function."""

    def test_returns_none_when_missing(self) -> None:
        """Returns None when key is not in dict."""
        assert _parse_monotonic_constraints({}) is None

    def test_returns_none_when_null(self) -> None:
        """Returns None when value is null."""
        assert _parse_monotonic_constraints({"monotonic_constraints": None}) is None

    def test_returns_dict_when_valid(self) -> None:
        """Returns dict when value is valid dict[str, int]."""
        result = _parse_monotonic_constraints({"monotonic_constraints": {"a": 1, "b": -1}})
        assert result == {"a": 1, "b": -1}

    def test_raises_on_non_dict(self) -> None:
        """Raises JSONTypeError for non-dict value."""
        with pytest.raises(JSONTypeError, match="monotonic_constraints must be a dict"):
            _parse_monotonic_constraints({"monotonic_constraints": "invalid"})

    def test_raises_on_non_int_values(self) -> None:
        """Raises JSONTypeError for non-int values."""
        with pytest.raises(JSONTypeError, match="monotonic_constraints values must be ints"):
            _parse_monotonic_constraints({"monotonic_constraints": {"a": "not_int"}})


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

    def test_logreg(self) -> None:
        """Returns correct filename for LogReg."""
        assert _get_meta_filename("logreg") == "active_logreg_meta.json"

    def test_random_forest(self) -> None:
        """Returns correct filename for RandomForest."""
        assert _get_meta_filename("random_forest") == "active_rf_meta.json"

    def test_xgboost(self) -> None:
        """Returns empty string for XGBoost (self-describing format)."""
        assert _get_meta_filename("xgboost") == ""

    def test_cleargbm(self) -> None:
        """Returns empty string for ClearGBM (self-describing format)."""
        assert _get_meta_filename("cleargbm") == ""


class TestGetActiveFilename:
    """Tests for _get_active_filename function."""

    def test_xgboost(self) -> None:
        """Returns correct filename for XGBoost."""
        assert _get_active_filename("xgboost") == "active_xgb.ubj"

    def test_mlp(self) -> None:
        """Returns correct filename for MLP."""
        assert _get_active_filename("mlp") == "active_mlp.pt"

    def test_lstm(self) -> None:
        """Returns correct filename for LSTM."""
        assert _get_active_filename("lstm") == "active_lstm.pt"

    def test_lightgbm(self) -> None:
        """Returns correct filename for LightGBM."""
        assert _get_active_filename("lightgbm") == "active_lgbm.txt"

    def test_cleargbm(self) -> None:
        """Returns correct filename for ClearGBM."""
        assert _get_active_filename("cleargbm") == "active_cgbm.json"

    def test_logreg(self) -> None:
        """Returns correct filename for LogReg."""
        assert _get_active_filename("logreg") == "active_logreg.joblib"

    def test_random_forest(self) -> None:
        """Returns correct filename for RandomForest."""
        assert _get_active_filename("random_forest") == "active_rf.joblib"

    def test_unknown_raises_value_error(self) -> None:
        """Unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            _get_active_filename("unknown")


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
