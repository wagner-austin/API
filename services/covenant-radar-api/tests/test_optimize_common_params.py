"""Tests for worker/_optimize_common.py shared utilities and time-series dataset support.

Tests use dependency injection via worker/_test_hooks to verify actual code paths.
All code paths are tested with strong assertions on actual behavior.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from covenant_radar_api.worker._optimize_common import (
    build_optimization_config,
    optional_int,
    parse_backend_name,
    parse_device,
    parse_feature_preset,
)


class TestParseDevice:
    """Tests for parse_device function."""

    def test_parse_device_defaults_to_auto(self) -> None:
        """None input returns 'auto'."""
        assert parse_device(None) == "auto"

    def test_parse_device_accepts_cpu(self) -> None:
        """'cpu' is accepted."""
        assert parse_device("cpu") == "cpu"

    def test_parse_device_accepts_cuda(self) -> None:
        """'cuda' is accepted."""
        assert parse_device("cuda") == "cuda"

    def test_parse_device_accepts_auto(self) -> None:
        """'auto' is accepted."""
        assert parse_device("auto") == "auto"

    def test_parse_device_rejects_invalid_string(self) -> None:
        """Invalid device string raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            parse_device("tpu")

    def test_parse_device_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="device must be a string"):
            parse_device(123)


class TestParseFeaturePreset:
    """Tests for parse_feature_preset function."""

    def test_parse_feature_preset_defaults_to_none(self) -> None:
        """None input returns 'none'."""
        assert parse_feature_preset(None) == "none"

    def test_parse_feature_preset_accepts_none(self) -> None:
        """'none' is accepted."""
        assert parse_feature_preset("none") == "none"

    def test_parse_feature_preset_accepts_log_only(self) -> None:
        """'log_only' is accepted."""
        assert parse_feature_preset("log_only") == "log_only"

    def test_parse_feature_preset_accepts_ratios_only(self) -> None:
        """'ratios_only' is accepted."""
        assert parse_feature_preset("ratios_only") == "ratios_only"

    def test_parse_feature_preset_accepts_full(self) -> None:
        """'full' is accepted."""
        assert parse_feature_preset("full") == "full"

    def test_parse_feature_preset_rejects_invalid_string(self) -> None:
        """Invalid feature_preset string raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_preset must be one of"):
            parse_feature_preset("invalid")

    def test_parse_feature_preset_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_preset must be a string"):
            parse_feature_preset(123)


class TestParseBackendName:
    """Tests for parse_backend_name function."""

    def test_parse_backend_name_defaults_to_xgboost(self) -> None:
        """None input returns 'xgboost'."""
        assert parse_backend_name(None) == "xgboost"

    def test_parse_backend_name_accepts_xgboost(self) -> None:
        """'xgboost' is accepted."""
        assert parse_backend_name("xgboost") == "xgboost"

    def test_parse_backend_name_accepts_mlp(self) -> None:
        """'mlp' is accepted."""
        assert parse_backend_name("mlp") == "mlp"

    def test_parse_backend_name_accepts_lstm(self) -> None:
        """'lstm' is accepted."""
        assert parse_backend_name("lstm") == "lstm"

    def test_parse_backend_name_accepts_lightgbm(self) -> None:
        """'lightgbm' is accepted."""
        assert parse_backend_name("lightgbm") == "lightgbm"

    def test_parse_backend_name_accepts_cleargbm(self) -> None:
        """'cleargbm' is accepted."""
        assert parse_backend_name("cleargbm") == "cleargbm"

    def test_parse_backend_name_accepts_logreg(self) -> None:
        """'logreg' is accepted."""
        assert parse_backend_name("logreg") == "logreg"

    def test_parse_backend_name_accepts_random_forest(self) -> None:
        """'random_forest' is accepted."""
        assert parse_backend_name("random_forest") == "random_forest"

    def test_parse_backend_name_rejects_invalid_string(self) -> None:
        """Invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="backend must be one of"):
            parse_backend_name("invalid")

    def test_parse_backend_name_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="backend must be a string"):
            parse_backend_name(123)


class TestOptionalInt:
    """Tests for optional_int function."""

    def test_optional_int_returns_default_on_missing(self) -> None:
        """optional_int returns default when key is missing."""
        assert optional_int({}, "missing", 10) == 10

    def test_optional_int_returns_value_when_present(self) -> None:
        """optional_int returns value when present."""
        assert optional_int({"val": 20}, "val", 10) == 20

    def test_optional_int_converts_float_to_int(self) -> None:
        """optional_int converts float to int."""
        assert optional_int({"val": 15.5}, "val", 0) == 15

    def test_optional_int_raises_on_invalid_type(self) -> None:
        """optional_int raises JSONTypeError on invalid type."""
        with pytest.raises(JSONTypeError, match="must be a number"):
            optional_int({"val": "string"}, "val", 0)


class TestBuildOptimizationConfig:
    """Tests for build_optimization_config function."""

    def test_build_config_with_timeout(self) -> None:
        """build_optimization_config creates config with timeout."""
        config = build_optimization_config(
            n_trials=50,
            timeout_seconds=3600,
            random_state=42,
        )

        assert config["n_trials"] == 50
        assert config["timeout_seconds"] == 3600
        assert config["random_state"] == 42

    def test_build_config_without_timeout(self) -> None:
        """build_optimization_config creates config without timeout."""
        config = build_optimization_config(
            n_trials=25,
            timeout_seconds=None,
            random_state=123,
        )

        assert config["n_trials"] == 25
        assert config["timeout_seconds"] is None
        assert config["random_state"] == 123
