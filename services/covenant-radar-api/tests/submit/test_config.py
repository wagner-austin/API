"""Tests for submit config builders.

Tests the dataset config and backend-specific training config builders.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

from scripts.submit.pipeline import (
    SubmitConfig,
    _build_lightgbm_config,
    _build_lstm_config,
    _build_mlp_config,
    _build_xgboost_config,
    _get_train_config,
    build_dataset_config,
)


class TestBuildDatasetConfig:
    """Tests for build_dataset_config function."""

    def test_build_with_last_aggregation(self, tmp_path: Path) -> None:
        """Test building config with last aggregation."""
        config = build_dataset_config(
            data_dir=tmp_path,
            aggregation="last",
            include_rank_features=True,
            include_diff_features=True,
        )
        assert config["name"] == "submit_data"
        assert config["time_series"]["aggregation"] == "last"
        assert config["time_series"]["include_rank_features"] is True
        assert config["time_series"]["include_diff_features"] is True

    def test_build_with_statistics_aggregation(self, tmp_path: Path) -> None:
        """Test building config with statistics aggregation."""
        config = build_dataset_config(
            data_dir=tmp_path,
            aggregation="statistics",
            include_rank_features=False,
            include_diff_features=False,
        )
        assert config["time_series"]["aggregation"] == "statistics"
        assert config["time_series"]["include_rank_features"] is False
        assert config["time_series"]["include_diff_features"] is False

    def test_build_with_mean_aggregation(self, tmp_path: Path) -> None:
        """Test building config with mean aggregation."""
        config = build_dataset_config(
            data_dir=tmp_path,
            aggregation="mean",
            include_rank_features=True,
            include_diff_features=False,
        )
        assert config["time_series"]["aggregation"] == "mean"

    def test_build_with_first_aggregation(self, tmp_path: Path) -> None:
        """Test building config with first aggregation."""
        config = build_dataset_config(
            data_dir=tmp_path,
            aggregation="first",
            include_rank_features=False,
            include_diff_features=True,
        )
        assert config["time_series"]["aggregation"] == "first"


class TestBuildLightGBMConfig:
    """Tests for _build_lightgbm_config function."""

    def test_build_lightgbm_config(self) -> None:
        """Test building LightGBM config."""
        submit_config = SubmitConfig(
            backend="lightgbm",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            aggregation="last",
            include_rank_features=True,
            include_diff_features=True,
        )
        config = _build_lightgbm_config(submit_config)
        assert config["learning_rate"] == 0.1
        assert config["n_estimators"] == 100
        assert config["num_leaves"] == 31
        assert config["device"] == "cpu"
        assert config["train_ratio"] == 0.7


class TestBuildXGBoostConfig:
    """Tests for _build_xgboost_config function."""

    def test_build_xgboost_config(self) -> None:
        """Test building XGBoost config."""
        submit_config = SubmitConfig(
            backend="xgboost",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=6,
            aggregation="last",
            include_rank_features=True,
            include_diff_features=True,
        )
        config = _build_xgboost_config(submit_config)
        assert config["learning_rate"] == 0.1
        assert config["max_depth"] == 6
        assert config["n_estimators"] == 100


class TestBuildMLPConfig:
    """Tests for _build_mlp_config function."""

    def test_build_mlp_config(self) -> None:
        """Test building MLP config."""
        submit_config = SubmitConfig(
            backend="mlp",
            n_estimators=50,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=-1,
            aggregation="last",
            include_rank_features=True,
            include_diff_features=True,
        )
        config = _build_mlp_config(submit_config)
        assert config["learning_rate"] == 0.01
        assert config["n_epochs"] == 50
        assert config["hidden_sizes"] == (256, 128, 64)


class TestBuildLSTMConfig:
    """Tests for _build_lstm_config function."""

    def test_build_lstm_config(self) -> None:
        """Test building LSTM config."""
        submit_config = SubmitConfig(
            backend="lstm",
            n_estimators=30,
            learning_rate=0.001,
            num_leaves=31,
            max_depth=-1,
            aggregation="last",
            include_rank_features=True,
            include_diff_features=True,
        )
        config = _build_lstm_config(submit_config)
        assert config["learning_rate"] == 0.001
        assert config["n_epochs"] == 30
        assert config["hidden_size"] == 128


class TestGetTrainConfig:
    """Tests for _get_train_config dispatcher function."""

    def test_dispatcher_lightgbm(self) -> None:
        """Test dispatcher returns LightGBM config for lightgbm backend."""
        submit_config = SubmitConfig(
            backend="lightgbm",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            aggregation="last",
            include_rank_features=True,
            include_diff_features=True,
        )
        config = _get_train_config(submit_config)
        expected = _build_lightgbm_config(submit_config)
        assert config == expected

    def test_dispatcher_xgboost(self) -> None:
        """Test dispatcher returns XGBoost config for xgboost backend."""
        submit_config = SubmitConfig(
            backend="xgboost",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            aggregation="mean",
            include_rank_features=False,
            include_diff_features=True,
        )
        config = _get_train_config(submit_config)
        expected = _build_xgboost_config(submit_config)
        assert config == expected

    def test_dispatcher_mlp(self) -> None:
        """Test dispatcher returns MLP config for mlp backend."""
        submit_config = SubmitConfig(
            backend="mlp",
            n_estimators=50,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=-1,
            aggregation="statistics",
            include_rank_features=True,
            include_diff_features=False,
        )
        config = _get_train_config(submit_config)
        expected = _build_mlp_config(submit_config)
        assert config == expected

    def test_dispatcher_lstm(self) -> None:
        """Test dispatcher returns LSTM config for lstm backend."""
        submit_config = SubmitConfig(
            backend="lstm",
            n_estimators=30,
            learning_rate=0.001,
            num_leaves=31,
            max_depth=-1,
            aggregation="first",
            include_rank_features=False,
            include_diff_features=False,
        )
        config = _get_train_config(submit_config)
        expected = _build_lstm_config(submit_config)
        assert config == expected
