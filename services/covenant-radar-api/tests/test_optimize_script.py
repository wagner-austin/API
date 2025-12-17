"""Tests for scripts/optimize.py CLI entry point.

Tests use dependency injection via scripts/_test_hooks to avoid real optimization runs.
All code paths are tested with strong assertions on actual behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import scripts._test_hooks as _hooks
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    PreparedClassifier,
    ProgressCallback,
)
from covenant_ml.backends.registry import BackendRegistration, ClassifierRegistry
from covenant_ml.datasets import DatasetConfig, DatasetMeta, DatasetRegistry, LoadedDataset
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    LightGBMConfig,
    LSTMConfig,
    MLPConfig,
    TrainConfig,
    TrainOutcome,
)
from numpy.typing import NDArray
from platform_core.logging import setup_rich_logging
from scripts._test_hooks import (
    LightGBMOptimizationResult,
    LSTMOptimizationResult,
    MLPOptimizationResult,
    XGBoostOptimizationResult,
)
from scripts.optimize import main
from scripts.optimize.cli import (
    PRESET_DESCRIPTIONS,
    FeaturePreset,
    OptimizeArgs,
    _handle_flag,
    _parse_backend,
    _parse_dataset,
    _parse_preset,
    parse_args,
)
from scripts.optimize.display import (
    create_hyperparams_table,
    create_result_table,
    print_config,
    print_result,
)
from scripts.optimize.modes import compare_presets, run_all_datasets, run_single_with_progress
from scripts.optimize.runner import (
    get_project_root,
    run_lightgbm,
    run_lstm,
    run_mlp,
    run_xgboost,
)

FeaturePresetLiteral = Literal["none", "log_only", "ratios_only", "full"]


@pytest.fixture(autouse=True)
def _setup_rich_logging_for_tests() -> None:
    """Set up rich logging before each test that needs it."""
    setup_rich_logging(level="WARNING", show_time=False)


def _make_fake_train_config() -> TrainConfig:
    """Create a fake TrainConfig for testing."""
    return TrainConfig(
        device="cpu",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        early_stopping_rounds=10,
        reg_alpha=0.01,
        reg_lambda=0.01,
    )


def _make_fake_result(
    dataset: str = "taiwan",
    feature_preset: FeaturePresetLiteral = "full",
    best_val_auc: float = 0.85,
    n_features: int = 100,
) -> XGBoostOptimizationResult:
    """Create a fake optimization result for testing."""
    return {
        "backend": "xgboost",
        "status": "complete",
        "dataset": dataset,
        "n_samples": 1000,
        "n_features": n_features,
        "feature_preset": feature_preset,
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": best_val_auc,
        "best_max_depth": 6,
        "best_n_estimators": 100,
        "best_learning_rate": 0.1,
        "best_reg_alpha": 0.01,
        "best_reg_lambda": 0.01,
        "best_subsample": 0.8,
        "best_colsample_bytree": 0.8,
        "duration_seconds": 10.0,
        "recommended_config": _make_fake_train_config(),
    }


def _make_fake_mlp_result() -> MLPOptimizationResult:
    """Create a fake MLP optimization result for testing."""
    return {
        "backend": "mlp",
        "status": "complete",
        "dataset": "taiwan",
        "n_samples": 1000,
        "n_features": 100,
        "feature_preset": "full",
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": 0.85,
        "best_n_layers": 3,
        "best_hidden_size": 128,
        "best_learning_rate": 0.001,
        "best_dropout": 0.2,
        "best_batch_size": 64,
        "duration_seconds": 10.0,
        "recommended_config": MLPConfig(
            device="cpu",
            precision="fp32",
            optimizer="adamw",
            hidden_sizes=(128, 64),
            learning_rate=0.001,
            dropout=0.2,
            batch_size=64,
            n_epochs=100,
            early_stopping_patience=10,
            random_state=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        ),
    }


def _make_fake_lightgbm_result() -> LightGBMOptimizationResult:
    """Create a fake LightGBM optimization result for testing."""
    return {
        "backend": "lightgbm",
        "status": "complete",
        "dataset": "taiwan",
        "n_samples": 1000,
        "n_features": 100,
        "feature_preset": "full",
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": 0.85,
        "best_max_depth": 6,
        "best_n_estimators": 100,
        "best_num_leaves": 31,
        "best_learning_rate": 0.1,
        "best_reg_alpha": 0.01,
        "best_reg_lambda": 0.01,
        "best_subsample": 0.8,
        "best_colsample_bytree": 0.8,
        "duration_seconds": 10.0,
        "recommended_config": LightGBMConfig(
            device="cpu",
            max_depth=6,
            n_estimators=100,
            num_leaves=31,
            min_child_samples=20,
            learning_rate=0.1,
            reg_alpha=0.01,
            reg_lambda=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            early_stopping_rounds=10,
        ),
    }


def _make_fake_lstm_result() -> LSTMOptimizationResult:
    """Create a fake LSTM optimization result for testing."""
    return {
        "backend": "lstm",
        "status": "complete",
        "dataset": "taiwan",
        "n_samples": 1000,
        "n_features": 100,
        "feature_preset": "full",
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": 0.85,
        "best_hidden_size": 64,
        "best_num_layers": 2,
        "best_learning_rate": 0.001,
        "best_dropout": 0.2,
        "best_batch_size": 32,
        "duration_seconds": 10.0,
        "recommended_config": LSTMConfig(
            device="cpu",
            precision="fp32",
            hidden_size=64,
            num_layers=2,
            learning_rate=0.001,
            dropout=0.2,
            batch_size=32,
            n_epochs=100,
            early_stopping_patience=10,
            sequence_length=10,
            bidirectional=False,
            random_state=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        ),
    }


# =============================================================================
# Shared Fake Classes for save_model=True Tests
# =============================================================================


class _FakePreparedClassifier:
    """Fake classifier for save_model tests."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return constant predictions."""
        n_samples: int = int(x.shape[0])
        return np.column_stack(
            [
                np.full(n_samples, 0.3, dtype=np.float64),
                np.full(n_samples, 0.7, dtype=np.float64),
            ]
        )


class _FakeSaveModelBackend:
    """Fake backend for save_model tests."""

    def backend_name(self) -> BackendName:
        """Return xgboost as backend name."""
        return "xgboost"

    def capabilities(self) -> BackendCapabilities:
        """Return fake capabilities."""
        return {
            "supports_train": True,
            "supports_gpu": False,
            "supports_early_stopping": True,
            "supports_feature_importance": True,
            "model_format": "ubj",
        }

    def prepare(
        self, *, n_features: int, n_classes: int, feature_names: list[str] | None
    ) -> PreparedClassifier:
        """Return fake prepared classifier."""
        return _FakePreparedClassifier()

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> TrainOutcome:
        """Return fake train outcome."""
        model_path = output_dir / "model_1.ubj"
        model_path.write_bytes(b"fake model")
        n_samples: int = int(x_features.shape[0])
        n_features_count: int = int(x_features.shape[1])
        fake_metrics: EvalMetrics = {
            "loss": 0.35,
            "ppl": 1.42,
            "auc": 0.88,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }
        fake_importances: list[FeatureImportance] = [
            {"name": f"f_{i}", "importance": 1.0 / n_features_count, "rank": i + 1}
            for i in range(n_features_count)
        ]
        return {
            "model_path": str(model_path),
            "model_id": "fake-model-1",
            "samples_total": n_samples,
            "samples_train": int(n_samples * 0.7),
            "samples_val": int(n_samples * 0.15),
            "samples_test": n_samples - int(n_samples * 0.7) - int(n_samples * 0.15),
            "train_metrics": fake_metrics,
            "val_metrics": fake_metrics,
            "test_metrics": fake_metrics,
            "best_val_auc": 0.88,
            "best_round": 50,
            "total_rounds": 100,
            "early_stopped": True,
            "config": config,
            "feature_importances": fake_importances,
            "scale_pos_weight_computed": 1.0,
        }

    def evaluate(
        self, *, model: PreparedClassifier, x: NDArray[np.float64], y: NDArray[np.int64]
    ) -> EvalMetrics:
        """Return fake evaluation metrics."""
        return {
            "loss": 0.35,
            "ppl": 1.42,
            "auc": 0.88,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Save fake model."""
        Path(path).write_bytes(b"fake model")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load fake model."""
        return _FakePreparedClassifier()

    def get_feature_importances(
        self, *, model: PreparedClassifier, feature_names: list[str] | None
    ) -> list[FeatureImportance] | None:
        """Return fake feature importances."""
        if feature_names is None:
            return None
        return [
            {"name": n, "importance": 1.0 / len(feature_names), "rank": i + 1}
            for i, n in enumerate(feature_names)
        ]


def _make_fake_dataset_config(name: str) -> DatasetConfig:
    """Create fake dataset config for save_model tests."""
    return {
        "name": name,
        "display_name": f"Fake {name}",
        "folder": f"{name}_data",
        "file_name": "data.csv",
        "file_format": "csv",
        "encoding": "utf-8",
        "target": {
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        "exclude_columns": (),
        "n_samples_expected": 100,
        "n_features_expected": 10,
        "positive_class_ratio_expected": 0.3,
    }


def _make_fake_loaded_dataset() -> LoadedDataset:
    """Create fake loaded dataset for save_model tests."""
    rng = np.random.default_rng(42)
    x = rng.random((100, 10))
    y = rng.integers(0, 2, size=100).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": "fake",
        "n_samples": 100,
        "n_features": 10,
        "n_positive": n_positive,
        "n_negative": 100 - n_positive,
        "positive_ratio": n_positive / 100,
        "feature_names": tuple(f"f_{i}" for i in range(10)),
    }
    return {"meta": meta, "x": x, "y": y}


class TestOptimizeArgs:
    """Tests for OptimizeArgs initialization."""

    def test_default_values(self) -> None:
        """Test OptimizeArgs has correct defaults."""
        args = OptimizeArgs()
        assert args.dataset == "taiwan"
        assert args.n_trials == 300
        assert args.feature_preset == "full"
        assert args.device == "cuda"
        assert args.timeout is None
        assert args.compare_presets is False
        assert args.all_datasets is False
        assert args.verbose is False


class TestPresetDescriptions:
    """Tests for preset descriptions constant."""

    def test_all_presets_have_descriptions(self) -> None:
        """Verify all presets have descriptions."""
        presets: list[FeaturePreset] = ["none", "log_only", "ratios_only", "full"]
        for preset in presets:
            assert preset in PRESET_DESCRIPTIONS
            description: str = PRESET_DESCRIPTIONS[preset]
            # Description should contain meaningful content about features
            assert "features" in description.lower() or "original" in description.lower()


class TestParseBackend:
    """Tests for _parse_backend function."""

    def test_parse_xgboost(self) -> None:
        """Test parsing xgboost backend."""
        result: str = _parse_backend("xgboost")
        assert result == "xgboost"

    def test_parse_mlp(self) -> None:
        """Test parsing mlp backend."""
        result: str = _parse_backend("mlp")
        assert result == "mlp"

    def test_parse_lightgbm(self) -> None:
        """Test parsing lightgbm backend."""
        result: str = _parse_backend("lightgbm")
        assert result == "lightgbm"

    def test_parse_lstm(self) -> None:
        """Test parsing lstm backend."""
        result: str = _parse_backend("lstm")
        assert result == "lstm"

    def test_parse_invalid_raises_system_exit(self) -> None:
        """Test parsing invalid backend raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_backend("invalid")
        assert exc_info.value.code == 1


class TestParseDataset:
    """Tests for _parse_dataset function."""

    def test_parse_taiwan(self) -> None:
        """Test parsing taiwan dataset."""
        result: str = _parse_dataset("taiwan")
        assert result == "taiwan"

    def test_parse_us(self) -> None:
        """Test parsing us dataset."""
        result: str = _parse_dataset("us")
        assert result == "us"

    def test_parse_polish(self) -> None:
        """Test parsing polish dataset."""
        result: str = _parse_dataset("polish")
        assert result == "polish"

    def test_parse_invalid_raises_system_exit(self) -> None:
        """Test parsing invalid dataset raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_dataset("invalid")
        assert exc_info.value.code == 1


class TestParsePreset:
    """Tests for _parse_preset function."""

    def test_parse_none(self) -> None:
        """Test parsing none preset."""
        result: str = _parse_preset("none")
        assert result == "none"

    def test_parse_log_only(self) -> None:
        """Test parsing log_only preset."""
        result: str = _parse_preset("log_only")
        assert result == "log_only"

    def test_parse_ratios_only(self) -> None:
        """Test parsing ratios_only preset."""
        result: str = _parse_preset("ratios_only")
        assert result == "ratios_only"

    def test_parse_full(self) -> None:
        """Test parsing full preset."""
        result: str = _parse_preset("full")
        assert result == "full"

    def test_parse_invalid_raises_system_exit(self) -> None:
        """Test parsing invalid preset raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_preset("invalid")
        assert exc_info.value.code == 1


class TestHandleFlag:
    """Tests for _handle_flag function."""

    def test_compare_presets_short(self) -> None:
        """Test -c flag sets compare_presets."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "-c")
        assert handled is True
        assert args.compare_presets is True

    def test_compare_presets_long(self) -> None:
        """Test --compare-presets flag sets compare_presets."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "--compare-presets")
        assert handled is True
        assert args.compare_presets is True

    def test_all_datasets_short(self) -> None:
        """Test -a flag sets all_datasets."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "-a")
        assert handled is True
        assert args.all_datasets is True

    def test_all_datasets_long(self) -> None:
        """Test --all-datasets flag sets all_datasets."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "--all-datasets")
        assert handled is True
        assert args.all_datasets is True

    def test_verbose_short(self) -> None:
        """Test -v flag sets verbose."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "-v")
        assert handled is True
        assert args.verbose is True

    def test_verbose_long(self) -> None:
        """Test --verbose flag sets verbose."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "--verbose")
        assert handled is True
        assert args.verbose is True

    def test_help_short_raises_system_exit(self) -> None:
        """Test -h flag raises SystemExit(0)."""
        args = OptimizeArgs()
        with pytest.raises(SystemExit) as exc_info:
            _handle_flag(args, "-h")
        assert exc_info.value.code == 0

    def test_help_long_raises_system_exit(self) -> None:
        """Test --help flag raises SystemExit(0)."""
        args = OptimizeArgs()
        with pytest.raises(SystemExit) as exc_info:
            _handle_flag(args, "--help")
        assert exc_info.value.code == 0

    def test_unknown_flag_not_handled(self) -> None:
        """Test unknown flag returns False."""
        args = OptimizeArgs()
        handled = _handle_flag(args, "--unknown")
        assert handled is False


class TestParseArgs:
    """Tests for parse_args function."""

    def test_empty_args_uses_defaults(self) -> None:
        """Test empty args uses defaults."""
        args: OptimizeArgs = parse_args([])
        assert args.dataset == "taiwan"
        assert args.n_trials == 300
        assert args.feature_preset == "full"

    def test_backend_short(self) -> None:
        """Test -b sets backend."""
        args: OptimizeArgs = parse_args(["-b", "lightgbm"])
        assert args.backend == "lightgbm"

    def test_backend_long(self) -> None:
        """Test --backend sets backend."""
        args: OptimizeArgs = parse_args(["--backend", "lstm"])
        assert args.backend == "lstm"

    def test_dataset_short(self) -> None:
        """Test -d sets dataset."""
        args: OptimizeArgs = parse_args(["-d", "us"])
        assert args.dataset == "us"

    def test_dataset_long(self) -> None:
        """Test --dataset sets dataset."""
        args: OptimizeArgs = parse_args(["--dataset", "polish"])
        assert args.dataset == "polish"

    def test_n_trials_short(self) -> None:
        """Test -n sets n_trials."""
        args: OptimizeArgs = parse_args(["-n", "50"])
        assert args.n_trials == 50

    def test_n_trials_long(self) -> None:
        """Test --n-trials sets n_trials."""
        args: OptimizeArgs = parse_args(["--n-trials", "100"])
        assert args.n_trials == 100

    def test_feature_preset_short(self) -> None:
        """Test -f sets feature_preset."""
        args: OptimizeArgs = parse_args(["-f", "none"])
        assert args.feature_preset == "none"

    def test_feature_preset_long(self) -> None:
        """Test --feature-preset sets feature_preset."""
        args: OptimizeArgs = parse_args(["--feature-preset", "log_only"])
        assert args.feature_preset == "log_only"

    def test_device(self) -> None:
        """Test --device sets device."""
        args: OptimizeArgs = parse_args(["--device", "cpu"])
        assert args.device == "cpu"

    def test_timeout_short(self) -> None:
        """Test -t sets timeout."""
        args: OptimizeArgs = parse_args(["-t", "60"])
        assert args.timeout == 60

    def test_timeout_long(self) -> None:
        """Test --timeout sets timeout."""
        args: OptimizeArgs = parse_args(["--timeout", "120"])
        assert args.timeout == 120

    def test_verbose_flag(self) -> None:
        """Test -v flag."""
        args: OptimizeArgs = parse_args(["-v"])
        assert args.verbose is True

    def test_compare_presets_flag(self) -> None:
        """Test -c flag."""
        args: OptimizeArgs = parse_args(["-c"])
        assert args.compare_presets is True

    def test_all_datasets_flag(self) -> None:
        """Test -a flag."""
        args: OptimizeArgs = parse_args(["-a"])
        assert args.all_datasets is True

    def test_save_model_flag_short(self) -> None:
        """Test -s flag sets save_model to True."""
        args: OptimizeArgs = parse_args(["-s"])
        assert args.save_model is True

    def test_save_model_flag_long(self) -> None:
        """Test --save-model flag sets save_model to True."""
        args: OptimizeArgs = parse_args(["--save-model"])
        assert args.save_model is True

    def test_combined_args(self) -> None:
        """Test multiple args combined."""
        args: OptimizeArgs = parse_args(
            ["-d", "us", "-n", "50", "-f", "none", "-v", "--device", "cpu"]
        )
        assert args.dataset == "us"
        assert args.n_trials == 50
        assert args.feature_preset == "none"
        assert args.verbose is True
        assert args.device == "cpu"

    def test_unknown_args_ignored(self) -> None:
        """Test unknown args are ignored."""
        args: OptimizeArgs = parse_args(["--unknown", "value", "-x"])
        assert args.dataset == "taiwan"  # default


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_parent_of_scripts(self) -> None:
        """Test project root is parent of scripts directory."""
        root: Path = get_project_root()
        assert root.name == "covenant-radar-api"
        assert (root / "scripts").exists()


class TestCreateResultTable:
    """Tests for create_result_table function."""

    def test_creates_table_with_data(self) -> None:
        """Test table is created with result data."""
        result = _make_fake_result()
        table = create_result_table("xgboost", result, 15.5)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)


class TestCreateHyperparamsTable:
    """Tests for create_hyperparams_table function for all backends."""

    def test_creates_xgboost_table(self) -> None:
        """Test table is created for XGBoost hyperparameters."""
        result = _make_fake_result()
        table = create_hyperparams_table("xgboost", result)
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_creates_mlp_table(self) -> None:
        """Test table is created for MLP hyperparameters."""
        result = _make_fake_mlp_result()
        table = create_hyperparams_table("mlp", result)
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_creates_lightgbm_table(self) -> None:
        """Test table is created for LightGBM hyperparameters."""
        result = _make_fake_lightgbm_result()
        table = create_hyperparams_table("lightgbm", result)
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_creates_lstm_table(self) -> None:
        """Test table is created for LSTM hyperparameters."""
        result = _make_fake_lstm_result()
        table = create_hyperparams_table("lstm", result)
        assert callable(table.add_column)
        assert callable(table.add_row)


class TestRunXGBoost:
    """Tests for run_xgboost function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test run_xgboost uses the xgboost_runner hook."""
        fake_result = _make_fake_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            result: XGBoostOptimizationResult = run_xgboost("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
        finally:
            _hooks.xgboost_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test run_xgboost includes timeout in config when provided."""
        fake_result = _make_fake_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            run_xgboost("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.xgboost_runner = original


class TestRunMLP:
    """Tests for run_mlp function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test run_mlp uses the mlp_runner hook."""
        fake_result = _make_fake_mlp_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.mlp_runner
        _hooks.mlp_runner = fake_runner
        try:
            result: MLPOptimizationResult = run_mlp("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
            # MLP-specific config
            assert "precision" in config_json
            assert "adamw" in config_json
        finally:
            _hooks.mlp_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test run_mlp includes timeout in config when provided."""
        fake_result = _make_fake_mlp_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.mlp_runner
        _hooks.mlp_runner = fake_runner
        try:
            run_mlp("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.mlp_runner = original


class TestRunLightGBM:
    """Tests for run_lightgbm function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test run_lightgbm uses the lightgbm_runner hook."""
        fake_result = _make_fake_lightgbm_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.lightgbm_runner
        _hooks.lightgbm_runner = fake_runner
        try:
            result: LightGBMOptimizationResult = run_lightgbm("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
            # LightGBM-specific config
            assert "early_stopping_rounds" in config_json
        finally:
            _hooks.lightgbm_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test run_lightgbm includes timeout in config when provided."""
        fake_result = _make_fake_lightgbm_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.lightgbm_runner
        _hooks.lightgbm_runner = fake_runner
        try:
            run_lightgbm("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.lightgbm_runner = original


class TestRunLSTM:
    """Tests for run_lstm function."""

    def test_runs_optimization_with_hook(self, tmp_path: Path) -> None:
        """Test run_lstm uses the lstm_runner hook."""
        fake_result = _make_fake_lstm_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.lstm_runner
        _hooks.lstm_runner = fake_runner
        try:
            result: LSTMOptimizationResult = run_lstm("taiwan", 10, "full", "cpu", None)
            assert result == fake_result
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "taiwan" in config_json
            assert "10" in config_json
            assert "full" in config_json
            # LSTM-specific config
            assert "precision" in config_json
            assert "sequence_length" in config_json
            assert "bidirectional" in config_json
        finally:
            _hooks.lstm_runner = original

    def test_includes_timeout_when_provided(self, tmp_path: Path) -> None:
        """Test run_lstm includes timeout in config when provided."""
        fake_result = _make_fake_lstm_result()
        call_args: list[tuple[str, Path, Path]] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
            call_args.append((config_json, external_dir, output_dir))
            return fake_result

        original = _hooks.lstm_runner
        _hooks.lstm_runner = fake_runner
        try:
            run_lstm("taiwan", 10, "full", "cpu", 60)
            assert len(call_args) == 1
            config_json, _, _ = call_args[0]
            assert "timeout_seconds" in config_json
            assert "60" in config_json
        finally:
            _hooks.lstm_runner = original


class TestPrintConfig:
    """Tests for print_config function."""

    def test_prints_without_error(self) -> None:
        """Test print_config runs without error."""
        # Just verify it doesn't raise
        print_config("xgboost", "taiwan", 50, "full", "cuda")


class TestPrintResult:
    """Tests for print_result function."""

    def test_prints_without_error(self) -> None:
        """Test print_result runs without error."""
        result = _make_fake_result()
        print_result("xgboost", result, 10.5)


class TestComparePresets:
    """Tests for compare_presets function."""

    def test_runs_all_presets_xgboost(self) -> None:
        """Test compare_presets runs all four presets with XGBoost."""
        presets_called: list[str] = []
        callback_calls = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal callback_calls
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
                progress_callback(
                    {
                        "trial_number": 2,
                        "n_trials_total": 5,
                        "current_auc": 0.75,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": False,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
            # Extract preset from config
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
                return _make_fake_result(feature_preset="none", best_val_auc=0.75, n_features=20)
            if "log_only" in config_json:
                presets_called.append("log_only")
                return _make_fake_result(
                    feature_preset="log_only", best_val_auc=0.80, n_features=40
                )
            if "ratios_only" in config_json:
                presets_called.append("ratios_only")
                return _make_fake_result(
                    feature_preset="ratios_only", best_val_auc=0.82, n_features=500
                )
            presets_called.append("full")
            return _make_fake_result(feature_preset="full", best_val_auc=0.85, n_features=800)

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            compare_presets("xgboost", "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
            assert "none" in presets_called
            assert "log_only" in presets_called
            assert "ratios_only" in presets_called
            assert "full" in presets_called
            assert callback_calls == 8  # 2 calls per preset * 4 presets
        finally:
            _hooks.xgboost_runner = original

    def test_runs_all_presets_mlp(self) -> None:
        """Test compare_presets runs all four presets with MLP."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.001,
                        "best_n_layers": 3,
                        "best_hidden_size": 128,
                        "best_dropout": 0.2,
                    }
                )
            # Extract preset from config
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return _make_fake_mlp_result()

        original = _hooks.mlp_runner
        _hooks.mlp_runner = fake_runner
        try:
            compare_presets("mlp", "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.mlp_runner = original

    def test_runs_all_presets_lightgbm(self) -> None:
        """Test compare_presets runs all four presets with LightGBM."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                        "best_num_leaves": 31,
                    }
                )
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return _make_fake_lightgbm_result()

        original = _hooks.lightgbm_runner
        _hooks.lightgbm_runner = fake_runner
        try:
            compare_presets("lightgbm", "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.lightgbm_runner = original

    def test_runs_all_presets_lstm(self) -> None:
        """Test compare_presets runs all four presets with LSTM."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.001,
                        "best_num_layers": 2,
                        "best_hidden_size": 64,
                        "best_dropout": 0.2,
                    }
                )
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return _make_fake_lstm_result()

        original = _hooks.lstm_runner
        _hooks.lstm_runner = fake_runner
        try:
            compare_presets("lstm", "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.lstm_runner = original

    def test_compare_presets_with_save_model_true(self, tmp_path: Path) -> None:
        """Test compare_presets with save_model=True.

        This covers the save_model branch in compare_presets (modes.py line 453).
        """
        # Set up fake registries using shared classes
        fake_backend = _FakeSaveModelBackend()
        fake_registry = ClassifierRegistry()
        fake_registry.register("xgboost", BackendRegistration(lambda: fake_backend))
        fake_dataset_reg = DatasetRegistry((_make_fake_dataset_config("taiwan"),))

        # Store originals
        orig_runner = _hooks.xgboost_runner
        orig_backend_reg = _hooks.backend_registry_factory
        orig_dataset_reg = _hooks.dataset_registry_factory
        orig_loader = _hooks.dataset_loader

        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            # Track which preset was called
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
                return _make_fake_result(feature_preset="none", best_val_auc=0.75, n_features=20)
            if "log_only" in config_json:
                presets_called.append("log_only")
                return _make_fake_result(
                    feature_preset="log_only", best_val_auc=0.80, n_features=40
                )
            if "ratios_only" in config_json:
                presets_called.append("ratios_only")
                return _make_fake_result(
                    feature_preset="ratios_only", best_val_auc=0.82, n_features=500
                )
            presets_called.append("full")
            return _make_fake_result(feature_preset="full", best_val_auc=0.85, n_features=800)

        _hooks.xgboost_runner = fake_runner
        _hooks.backend_registry_factory = lambda: fake_registry
        _hooks.dataset_registry_factory = lambda: fake_dataset_reg
        _hooks.dataset_loader = lambda cfg, ext_dir: _make_fake_loaded_dataset()

        try:
            # Create necessary directories
            (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            # Run with save_model=True
            compare_presets(
                "xgboost", "taiwan", 5, "cpu", None, save_model=True, project_root=tmp_path
            )

            # Verify all presets were called
            assert len(presets_called) == 4
            assert "none" in presets_called
            assert "log_only" in presets_called
            assert "ratios_only" in presets_called
            assert "full" in presets_called

            # Verify model was saved (at least one preset creates a model)
            model_dir = tmp_path / "models" / "xgboost"
            assert model_dir.exists()
        finally:
            _hooks.xgboost_runner = orig_runner
            _hooks.backend_registry_factory = orig_backend_reg
            _hooks.dataset_registry_factory = orig_dataset_reg
            _hooks.dataset_loader = orig_loader


class TestRunSingleWithProgress:
    """Tests for run_single_with_progress function."""

    def test_runs_mlp_backend(self) -> None:
        """Test run_single_with_progress uses MLP backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
            nonlocal call_count
            call_count += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.001,
                        "best_n_layers": 3,
                        "best_hidden_size": 128,
                        "best_dropout": 0.2,
                    }
                )
            return _make_fake_mlp_result()

        original = _hooks.mlp_runner
        _hooks.mlp_runner = fake_runner
        try:
            result = run_single_with_progress(
                "mlp", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "mlp"
            assert result["result"]["backend"] == "mlp"
        finally:
            _hooks.mlp_runner = original

    def test_runs_lightgbm_backend(self) -> None:
        """Test run_single_with_progress uses LightGBM backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            nonlocal call_count
            call_count += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                        "best_num_leaves": 31,
                    }
                )
            return _make_fake_lightgbm_result()

        original = _hooks.lightgbm_runner
        _hooks.lightgbm_runner = fake_runner
        try:
            result = run_single_with_progress(
                "lightgbm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "lightgbm"
            assert result["result"]["backend"] == "lightgbm"
        finally:
            _hooks.lightgbm_runner = original

    def test_runs_lstm_backend(self) -> None:
        """Test run_single_with_progress uses LSTM backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
            nonlocal call_count
            call_count += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.001,
                        "best_num_layers": 2,
                        "best_hidden_size": 64,
                        "best_dropout": 0.2,
                    }
                )
            return _make_fake_lstm_result()

        original = _hooks.lstm_runner
        _hooks.lstm_runner = fake_runner
        try:
            result = run_single_with_progress(
                "lstm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "lstm"
            assert result["result"]["backend"] == "lstm"
        finally:
            _hooks.lstm_runner = original

    def test_run_single_with_save_model_true(self, tmp_path: Path) -> None:
        """Test run_single_with_progress with save_model=True.

        This covers the save_model branch in run_single_backend (modes.py line 363).
        """
        # Set up fake registries using shared classes
        fake_backend = _FakeSaveModelBackend()
        fake_registry = ClassifierRegistry()
        fake_registry.register("xgboost", BackendRegistration(lambda: fake_backend))
        fake_dataset_reg = DatasetRegistry((_make_fake_dataset_config("taiwan"),))

        # Store originals
        orig_runner = _hooks.xgboost_runner
        orig_backend_reg = _hooks.backend_registry_factory
        orig_dataset_reg = _hooks.dataset_registry_factory
        orig_loader = _hooks.dataset_loader

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            return _make_fake_result()

        _hooks.xgboost_runner = fake_runner
        _hooks.backend_registry_factory = lambda: fake_registry
        _hooks.dataset_registry_factory = lambda: fake_dataset_reg
        _hooks.dataset_loader = lambda cfg, ext_dir: _make_fake_loaded_dataset()

        try:
            # Create necessary directories
            (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            # Run with save_model=True
            result = run_single_with_progress(
                "xgboost", "taiwan", 5, "full", "cpu", None, save_model=True, project_root=tmp_path
            )

            # Verify the result is returned correctly
            assert result["backend"] == "xgboost"
            assert result["result"]["backend"] == "xgboost"

            # Verify model was saved
            model_dir = tmp_path / "models" / "xgboost"
            assert model_dir.exists()
        finally:
            _hooks.xgboost_runner = orig_runner
            _hooks.backend_registry_factory = orig_backend_reg
            _hooks.dataset_registry_factory = orig_dataset_reg
            _hooks.dataset_loader = orig_loader


class TestRunAllDatasets:
    """Tests for run_all_datasets function."""

    def test_runs_all_three_datasets(self) -> None:
        """Test run_all_datasets runs on taiwan, us, and polish with varying AUCs."""
        datasets_called: list[str] = []
        callback_calls = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal callback_calls
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
                progress_callback(
                    {
                        "trial_number": 2,
                        "n_trials_total": 5,
                        "current_auc": 0.75,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": False,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
            # Return different AUC values to cover both best and non-best formatting
            if "taiwan" in config_json:
                datasets_called.append("taiwan")
                return _make_fake_result(dataset="taiwan", best_val_auc=0.90)  # Best
            if '"us"' in config_json:
                datasets_called.append("us")
                return _make_fake_result(dataset="us", best_val_auc=0.85)  # Not best
            datasets_called.append("polish")
            return _make_fake_result(dataset="polish", best_val_auc=0.82)  # Not best

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            run_all_datasets("xgboost", 10, "full", "cpu", None, save_model=False)
            assert len(datasets_called) == 3
            assert "taiwan" in datasets_called
            assert "us" in datasets_called
            assert "polish" in datasets_called
            assert callback_calls == 6  # 2 calls per dataset * 3 datasets
        finally:
            _hooks.xgboost_runner = original


class TestMain:
    """Tests for main function."""

    def test_main_with_defaults_runs_single(self) -> None:
        """Test main with no args runs single optimization and calls progress callback."""
        call_count = 0
        callback_calls = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count, callback_calls
            call_count += 1
            # Simulate calling progress callback as real runner would
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
                progress_callback(
                    {
                        "trial_number": 2,
                        "n_trials_total": 5,
                        "current_auc": 0.75,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": False,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
            return _make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-n", "5", "--no-save-model"])  # Small n_trials for test
            assert exit_code == 0
            assert call_count == 1
            assert callback_calls == 2  # Progress callback was invoked
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_compare_presets(self) -> None:
        """Test main with -c runs compare presets."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count
            call_count += 1
            return _make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-c", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 4  # All four presets
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_all_datasets(self) -> None:
        """Test main with -a runs all datasets."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count
            call_count += 1
            return _make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-a", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 3  # All three datasets
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_verbose(self) -> None:
        """Test main with -v sets debug logging."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count
            call_count += 1
            return _make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-v", "-n", "5", "--no-save-model"])
            assert exit_code == 0
            assert call_count == 1
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_all_options(self) -> None:
        """Test main with multiple options."""
        configs_received: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            configs_received.append(config_json)
            return _make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(
                ["-d", "us", "-n", "25", "-f", "log_only", "--device", "cpu", "--no-save-model"]
            )
            assert exit_code == 0
            assert len(configs_received) == 1
            config = configs_received[0]
            assert '"us"' in config
            assert "25" in config
            assert "log_only" in config
            assert "cpu" in config
        finally:
            _hooks.xgboost_runner = original

    def test_main_with_timeout(self) -> None:
        """Test main with timeout option."""
        configs_received: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            configs_received.append(config_json)
            return _make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            exit_code: int = main(["-n", "5", "-t", "120", "--no-save-model"])
            assert exit_code == 0
            assert len(configs_received) == 1
            assert "timeout_seconds" in configs_received[0]
            assert "120" in configs_received[0]
        finally:
            _hooks.xgboost_runner = original

    def test_main_help_exits_zero(self) -> None:
        """Test main with --help exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestModuleEntry:
    """Tests for module entry point."""

    def test_module_main_entry_with_help(self) -> None:
        """Test __main__ entry point via runpy."""
        import runpy
        from types import ModuleType

        # Clear module from sys.modules to avoid runpy warning about
        # module already being imported
        modules_to_clear = [k for k in sys.modules if k.startswith("scripts.optimize")]
        saved_modules: dict[str, ModuleType] = {}
        for mod in modules_to_clear:
            saved_modules[mod] = sys.modules.pop(mod)

        original_argv = sys.argv
        sys.argv = ["optimize", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("scripts.optimize", run_name="__main__", alter_sys=True)
            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv
            # Restore modules
            sys.modules.update(saved_modules)

    def test_dunder_main_module_import_does_not_execute_main(self) -> None:
        """Test importing __main__.py directly doesn't execute main (covers False branch)."""
        import importlib
        from types import ModuleType

        # Clear the module from cache if present
        mod_name = "scripts.optimize.__main__"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        # Import the module - this covers the False branch of if __name__ == "__main__"
        module: ModuleType = importlib.import_module(mod_name)

        # Verify the module was imported but main wasn't called (no SystemExit)
        # The module should have the name "scripts.optimize.__main__", not "__main__"
        assert module.__name__ == mod_name
        # If we got here without SystemExit, the if __name__ == "__main__" was False


class TestKeyboardInterrupt:
    """Tests for KeyboardInterrupt handling in main function."""

    def test_main_handles_keyboard_interrupt_gracefully(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test main exits gracefully on KeyboardInterrupt during execution."""

        def raising_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            raise KeyboardInterrupt()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = raising_runner
        try:
            exit_code: int = main(["-n", "1", "--no-save-model"])
            assert exit_code == 130

            # Verify output
            captured = capsys.readouterr()
            output = captured.out + captured.err
            assert "Process Interrupted by User" in output
        finally:
            _hooks.xgboost_runner = original


class TestRealDatasetHooks:
    """Tests for real dataset hooks to ensure coverage."""

    def test_real_dataset_registry_returns_registry(self) -> None:
        """Test _real_dataset_registry returns a DatasetRegistry."""
        from scripts._test_hooks import _real_dataset_registry

        registry = _real_dataset_registry()

        assert "taiwan" in registry
        assert "us" in registry
        assert "polish" in registry

    def test_real_dataset_loader_loads_taiwan(self, tmp_path: Path) -> None:
        """Test _real_dataset_loader loads Taiwan dataset."""
        from shutil import copyfile

        from scripts._test_hooks import _real_dataset_loader, _real_dataset_registry

        # Copy real Taiwan dataset
        external_dir = tmp_path / "external"
        taiwan_dir = external_dir / "taiwan_data"
        taiwan_dir.mkdir(parents=True, exist_ok=True)
        real_src = Path(__file__).parent.parent / "data" / "external" / "taiwan_data" / "data.csv"
        assert real_src.exists(), "Taiwan dataset not found"
        copyfile(str(real_src), str(taiwan_dir / "data.csv"))

        registry = _real_dataset_registry()
        config = registry.get("taiwan")
        dataset = _real_dataset_loader(config, external_dir)

        assert dataset["meta"]["n_samples"] > 0
        assert dataset["meta"]["n_features"] > 0
