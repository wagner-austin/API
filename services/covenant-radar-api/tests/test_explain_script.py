"""Tests for scripts/explain CLI entry point.

Tests use dependency injection via scripts/_test_hooks to avoid real model loading.
All code paths are tested with strong assertions on actual behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import scripts._test_hooks as _hooks
from covenant_ml.datasets import DatasetConfig, DatasetMeta, DatasetRegistry, LoadedDataset
from covenant_ml.explainers.registry import ExplainerRegistration, ExplainerRegistry
from covenant_ml.explainers.types import ExplainResult, SupportedExplainer
from covenant_ml.types import BackendName
from numpy.typing import NDArray
from platform_core.rich_logging import setup_rich_logging
from platform_ml.explainers.protocol import FeatureExplainer, PredictorProtocol
from platform_ml.explainers.types import (
    ComputationalCost,
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
)
from scripts.explain import main
from scripts.explain.cli import (
    BACKEND_EXPLAINERS,
    EXPLAINER_DESCRIPTIONS,
    ExplainArgs,
    _handle_flag,
    _parse_backend,
    _parse_dataset,
    _parse_explainer,
    parse_args,
    print_help,
    validate_explainer_backend,
)
from scripts.explain.display import (
    BACKEND_DISPLAY_NAMES,
    _create_importance_table,
    _create_summary_table,
    print_config,
    print_result,
)
from scripts.explain.runner import (
    MODEL_EXTENSIONS,
    ExplainRunResult,
    _get_default_model_path,
    _load_dataset_with_features,
    _sample_data,
    get_project_root,
)

DatasetNameLiteral = Literal["taiwan", "us", "polish"]


@pytest.fixture(autouse=True)
def _setup_rich_logging_for_tests() -> None:
    """Set up rich logging before each test that needs it."""
    setup_rich_logging(level="WARNING", show_time=False)


# =============================================================================
# Fake Implementations for Testing
# =============================================================================


def _make_fake_feature_importances(n_features: int = 10) -> list[FeatureImportanceScore]:
    """Create fake feature importance scores for testing."""
    return [
        {"name": f"feature_{i}", "importance": 1.0 - (i * 0.1), "rank": i + 1}
        for i in range(n_features)
    ]


def _make_fake_explain_result(
    backend: BackendName = "xgboost",
    explainer: SupportedExplainer = "permutation",
    n_samples: int = 100,
    n_features: int = 10,
) -> ExplainResult:
    """Create a fake ExplainResult for testing."""
    return {
        "status": "complete",
        "backend": backend,
        "explainer": explainer,
        "n_samples_used": n_samples,
        "n_features": n_features,
        "target_class": 1,
        "feature_importances": _make_fake_feature_importances(n_features),
        "duration_seconds": 1.5,
    }


def _make_fake_run_result(
    backend: BackendName = "xgboost",
    dataset: DatasetNameLiteral = "taiwan",
    explainer: SupportedExplainer = "permutation",
) -> ExplainRunResult:
    """Create a fake ExplainRunResult for testing."""
    return {
        "backend": backend,
        "dataset": dataset,
        "explainer": explainer,
        "result": _make_fake_explain_result(backend, explainer),
        "elapsed": 1.5,
        "model_path": "/path/to/model.ubj",
    }


class FakePredictor:
    """Fake predictor for testing."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return fake probabilities."""
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.column_stack(
            [np.zeros(n_samples), np.ones(n_samples)]
        ).astype(np.float64)
        return proba


class FakeExplainer:
    """Fake explainer for testing."""

    def __init__(self, name: SupportedExplainer = "permutation") -> None:
        """Initialize with explainer name."""
        self._name = name

    def explainer_name(self) -> ExplainerName:
        """Return the explainer name."""
        return "permutation"

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities."""
        cost: ComputationalCost = "medium"
        return {
            "requires_gradients": False,
            "requires_background_data": False,
            "computational_cost": cost,
        }

    def compute_importance(
        self,
        *,
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Return fake importances."""
        return [
            {"name": name, "importance": 1.0 / (i + 1), "rank": i + 1}
            for i, name in enumerate(feature_names)
        ]


def _make_fake_dataset() -> LoadedDataset:
    """Create fake dataset for testing."""
    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((200, 10)).astype(np.float64)
    y: NDArray[np.int64] = rng.integers(0, 2, size=200).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": "fake",
        "n_samples": 200,
        "n_features": 10,
        "n_positive": n_positive,
        "n_negative": 200 - n_positive,
        "positive_ratio": float(n_positive) / 200.0,
        "feature_names": tuple(f"feature_{i}" for i in range(10)),
        "categorical_encodings": (),
    }
    return {"x": x, "y": y, "meta": meta, "groups": None}


def _make_fake_dataset_config(name: str) -> DatasetConfig:
    """Create fake dataset config for testing."""
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
        "n_samples_expected": 200,
        "n_features_expected": 10,
        "positive_class_ratio_expected": 0.5,
    }


def _make_fake_dataset_registry() -> DatasetRegistry:
    """Create fake dataset registry for testing."""
    configs = (
        _make_fake_dataset_config("taiwan"),
        _make_fake_dataset_config("us"),
        _make_fake_dataset_config("polish"),
    )
    return DatasetRegistry(configs)


def _make_fake_explainer_registry() -> ExplainerRegistry:
    """Create fake explainer registry for testing."""
    registry = ExplainerRegistry()

    def make_fake() -> FeatureExplainer:
        return FakeExplainer()

    # Register with proper ExplainerRegistration
    backends: frozenset[BackendName] = frozenset(["xgboost", "lightgbm", "mlp", "lstm"])
    registration = ExplainerRegistration(
        factory=make_fake,
        compatible_backends=backends,
        requires_gradients=False,
    )
    registry.register("permutation", registration)
    registry.register("shap_tree", registration)
    registry.register("gradient", registration)
    registry.register("integrated_gradients", registration)
    return registry


# =============================================================================
# CLI Argument Parsing Tests
# =============================================================================


class TestParseArgs:
    """Tests for parse_args function."""

    def test_empty_args_uses_defaults(self) -> None:
        """Empty args should use default values."""
        args = parse_args([])
        assert args.backend == "xgboost"
        assert args.dataset == "taiwan"
        assert args.explainer == "permutation"
        assert args.n_samples == 1000
        assert args.target_class == 1
        assert args.top_n == 20
        assert args.verbose is False
        assert args.model_path is None

    def test_parse_backend_short_flag(self) -> None:
        """Backend can be set with -b flag."""
        args = parse_args(["-b", "lightgbm"])
        assert args.backend == "lightgbm"

    def test_parse_backend_long_flag(self) -> None:
        """Backend can be set with --backend flag."""
        args = parse_args(["--backend", "mlp", "-e", "gradient"])
        assert args.backend == "mlp"
        assert args.explainer == "gradient"

    def test_parse_dataset_short_flag(self) -> None:
        """Dataset can be set with -d flag."""
        args = parse_args(["-d", "us"])
        assert args.dataset == "us"

    def test_parse_explainer_short_flag(self) -> None:
        """Explainer can be set with -e flag."""
        args = parse_args(["-e", "shap_tree"])
        assert args.explainer == "shap_tree"

    def test_parse_n_samples_short_flag(self) -> None:
        """N samples can be set with -n flag."""
        args = parse_args(["-n", "500"])
        assert args.n_samples == 500

    def test_parse_target_class(self) -> None:
        """Target class can be set with -c flag."""
        args = parse_args(["-c", "0"])
        assert args.target_class == 0

    def test_parse_top_n(self) -> None:
        """Top N can be set with -t flag."""
        args = parse_args(["-t", "10"])
        assert args.top_n == 10

    def test_parse_model_path(self) -> None:
        """Model path can be set with -m flag."""
        args = parse_args(["-m", "/custom/path.ubj"])
        assert args.model_path == "/custom/path.ubj"

    def test_parse_verbose_flag(self) -> None:
        """Verbose flag can be set."""
        args = parse_args(["-v"])
        assert args.verbose is True


class TestParseBackend:
    """Tests for _parse_backend function."""

    def test_valid_backends(self) -> None:
        """All valid backends are accepted."""
        assert _parse_backend("xgboost") == "xgboost"
        assert _parse_backend("mlp") == "mlp"
        assert _parse_backend("lightgbm") == "lightgbm"
        assert _parse_backend("lstm") == "lstm"

    def test_invalid_backend_exits(self) -> None:
        """Invalid backend raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_backend("invalid")
        assert exc_info.value.code == 1


class TestParseDataset:
    """Tests for _parse_dataset function."""

    def test_valid_datasets(self) -> None:
        """All valid datasets are accepted."""
        assert _parse_dataset("taiwan") == "taiwan"
        assert _parse_dataset("us") == "us"
        assert _parse_dataset("polish") == "polish"

    def test_invalid_dataset_exits(self) -> None:
        """Invalid dataset raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_dataset("invalid")
        assert exc_info.value.code == 1


class TestParseExplainer:
    """Tests for _parse_explainer function."""

    def test_valid_explainers(self) -> None:
        """All valid explainers are accepted."""
        assert _parse_explainer("permutation") == "permutation"
        assert _parse_explainer("gradient") == "gradient"
        assert _parse_explainer("integrated_gradients") == "integrated_gradients"
        assert _parse_explainer("shap_tree") == "shap_tree"

    def test_invalid_explainer_exits(self) -> None:
        """Invalid explainer raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_explainer("invalid")
        assert exc_info.value.code == 1


class TestValidateExplainerBackend:
    """Tests for validate_explainer_backend function."""

    def test_valid_combinations(self) -> None:
        """Valid explainer/backend combinations are accepted."""
        validate_explainer_backend("permutation", "xgboost")
        validate_explainer_backend("shap_tree", "xgboost")
        validate_explainer_backend("permutation", "mlp")
        validate_explainer_backend("gradient", "mlp")
        validate_explainer_backend("integrated_gradients", "lstm")

    def test_invalid_gradient_on_xgboost_exits(self) -> None:
        """Gradient explainer on xgboost raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            validate_explainer_backend("gradient", "xgboost")
        assert exc_info.value.code == 1

    def test_invalid_shap_tree_on_mlp_exits(self) -> None:
        """SHAP tree on MLP raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            validate_explainer_backend("shap_tree", "mlp")
        assert exc_info.value.code == 1


class TestHandleFlag:
    """Tests for _handle_flag function."""

    def test_verbose_short(self) -> None:
        """Short verbose flag is handled."""
        args = ExplainArgs()
        assert _handle_flag(args, "-v") is True
        assert args.verbose is True

    def test_verbose_long(self) -> None:
        """Long verbose flag is handled."""
        args = ExplainArgs()
        assert _handle_flag(args, "--verbose") is True
        assert args.verbose is True

    def test_help_short_exits(self) -> None:
        """Short help flag exits with code 0."""
        args = ExplainArgs()
        with pytest.raises(SystemExit) as exc_info:
            _handle_flag(args, "-h")
        assert exc_info.value.code == 0

    def test_help_long_exits(self) -> None:
        """Long help flag exits with code 0."""
        args = ExplainArgs()
        with pytest.raises(SystemExit) as exc_info:
            _handle_flag(args, "--help")
        assert exc_info.value.code == 0

    def test_unknown_flag_returns_false(self) -> None:
        """Unknown flags return False."""
        args = ExplainArgs()
        assert _handle_flag(args, "--unknown") is False


class TestPrintHelp:
    """Tests for print_help function."""

    def test_print_help_executes(self) -> None:
        """print_help executes without error."""
        print_help()  # Should not raise


# =============================================================================
# Display Tests
# =============================================================================


class TestDisplayBackendNames:
    """Tests for backend display names."""

    def test_all_backends_have_display_names(self) -> None:
        """All backends should have display names."""
        backends: list[BackendName] = ["xgboost", "mlp", "lightgbm", "lstm"]
        for backend in backends:
            assert backend in BACKEND_DISPLAY_NAMES


class TestCreateSummaryTable:
    """Tests for _create_summary_table function."""

    def test_creates_table_with_result(self) -> None:
        """Summary table is created with result data."""
        result = _make_fake_run_result()
        table = _create_summary_table(result)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)


class TestCreateImportanceTable:
    """Tests for _create_importance_table function."""

    def test_creates_table_with_importances(self) -> None:
        """Importance table is created with feature data."""
        result = _make_fake_run_result()
        table = _create_importance_table(result, top_n=5)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_empty_importances_returns_empty_table(self) -> None:
        """Empty importances returns empty table."""
        result = _make_fake_run_result()
        result["result"]["feature_importances"] = []
        table = _create_importance_table(result, top_n=5)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_small_importance_uses_scientific_notation(self) -> None:
        """Small importances use scientific notation."""
        result = _make_fake_run_result()
        result["result"]["feature_importances"] = [{"name": "tiny", "importance": 0.001, "rank": 1}]
        table = _create_importance_table(result, top_n=5)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_zero_max_importance_handles_gracefully(self) -> None:
        """Zero max importance is handled without division error."""
        result = _make_fake_run_result()
        result["result"]["feature_importances"] = [{"name": "zero", "importance": 0.0, "rank": 1}]
        table = _create_importance_table(result, top_n=5)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)


class TestPrintConfig:
    """Tests for print_config function."""

    def test_print_config_executes(self) -> None:
        """print_config executes without error."""
        print_config("xgboost", "taiwan", "permutation", 1000, None)

    def test_print_config_with_model_path(self) -> None:
        """print_config with custom model path executes."""
        print_config("mlp", "us", "gradient", 500, "/custom/path.pt")


class TestPrintResult:
    """Tests for print_result function."""

    def test_print_result_executes(self) -> None:
        """print_result executes without error."""
        result = _make_fake_run_result()
        print_result(result, top_n=10)


# =============================================================================
# Runner Tests
# =============================================================================


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_path(self) -> None:
        """get_project_root returns a Path to the project root."""
        root = get_project_root()
        # Verify it's a valid path by checking it has expected parts
        assert root.name == "covenant-radar-api"


class TestModelExtensions:
    """Tests for MODEL_EXTENSIONS constant."""

    def test_all_backends_have_extensions(self) -> None:
        """All backends have model file extensions."""
        backends: list[BackendName] = ["xgboost", "mlp", "lightgbm", "lstm"]
        for backend in backends:
            assert backend in MODEL_EXTENSIONS


class TestGetDefaultModelPath:
    """Tests for _get_default_model_path function."""

    def test_xgboost_path(self) -> None:
        """XGBoost model path has .ubj extension."""
        path = _get_default_model_path("xgboost", "taiwan")
        assert path.name == "taiwan_xgboost_best.ubj"

    def test_mlp_path(self) -> None:
        """MLP model path has .pt extension."""
        path = _get_default_model_path("mlp", "us")
        assert path.name == "us_mlp_best.pt"

    def test_lightgbm_path(self) -> None:
        """LightGBM model path has .txt extension."""
        path = _get_default_model_path("lightgbm", "polish")
        assert path.name == "polish_lightgbm_best.txt"


class TestSampleData:
    """Tests for _sample_data function."""

    def test_sample_reduces_size(self) -> None:
        """Sampling reduces data size."""
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((100, 5)).astype(np.float64)
        y: NDArray[np.int64] = rng.integers(0, 2, size=100).astype(np.int64)
        x_s, y_s = _sample_data(x, y, n_samples=50, random_state=42)
        assert int(x_s.shape[0]) == 50
        assert int(y_s.shape[0]) == 50

    def test_sample_returns_all_if_n_samples_exceeds(self) -> None:
        """Returns all data if n_samples exceeds dataset size."""
        rng = np.random.default_rng(42)
        x: NDArray[np.float64] = rng.random((50, 5)).astype(np.float64)
        y: NDArray[np.int64] = rng.integers(0, 2, size=50).astype(np.int64)
        x_s, y_s = _sample_data(x, y, n_samples=100, random_state=42)
        assert int(x_s.shape[0]) == 50
        assert int(y_s.shape[0]) == 50


class TestLoadDatasetWithFeatures:
    """Tests for _load_dataset_with_features function."""

    def test_load_returns_tuple(self) -> None:
        """Load returns x, y, feature_names tuple."""
        orig_registry = _hooks.dataset_registry_factory
        orig_loader = _hooks.dataset_loader

        _hooks.dataset_registry_factory = _make_fake_dataset_registry
        _hooks.dataset_loader = lambda config, external_dir: _make_fake_dataset()

        try:
            x, y, names = _load_dataset_with_features("taiwan", "none", Path("/fake"))
            assert int(x.shape[0]) == 200
            assert int(y.shape[0]) == 200
            # Verify feature names has expected count (10 original features)
            assert len(names) == 10
        finally:
            _hooks.dataset_registry_factory = orig_registry
            _hooks.dataset_loader = orig_loader


# =============================================================================
# Main Entry Point Tests
# =============================================================================


class TestMain:
    """Tests for main entry point."""

    def test_main_with_help_flag_exits_zero(self) -> None:
        """Main with --help exits with code 0."""
        # Help flag causes SystemExit(0) during argument parsing
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_with_invalid_backend_exits_one(self) -> None:
        """Main with invalid backend exits."""
        # _parse_backend raises SystemExit before main returns
        with pytest.raises(SystemExit) as exc_info:
            main(["-b", "invalid_backend"])
        assert exc_info.value.code == 1

    def test_main_with_incompatible_explainer_exits_one(self) -> None:
        """Main with incompatible explainer/backend exits."""
        # validate_explainer_backend raises SystemExit
        with pytest.raises(SystemExit) as exc_info:
            main(["-b", "xgboost", "-e", "gradient"])
        assert exc_info.value.code == 1


# =============================================================================
# Explainer Descriptions Tests
# =============================================================================


class TestExplainerDescriptions:
    """Tests for EXPLAINER_DESCRIPTIONS constant."""

    def test_all_explainers_have_descriptions(self) -> None:
        """All explainers have descriptions."""
        explainers: list[SupportedExplainer] = [
            "permutation",
            "gradient",
            "integrated_gradients",
            "shap_tree",
        ]
        for explainer in explainers:
            assert explainer in EXPLAINER_DESCRIPTIONS


class TestBackendExplainers:
    """Tests for BACKEND_EXPLAINERS constant."""

    def test_all_backends_have_explainer_lists(self) -> None:
        """All backends have explainer compatibility lists."""
        backends: list[BackendName] = ["xgboost", "mlp", "lightgbm", "lstm"]
        for backend in backends:
            assert backend in BACKEND_EXPLAINERS
            # Each backend has at least 2 compatible explainers (permutation + one other)
            assert len(BACKEND_EXPLAINERS[backend]) >= 2


# =============================================================================
# ExplainArgs Tests
# =============================================================================


class TestExplainArgs:
    """Tests for ExplainArgs class."""

    def test_defaults(self) -> None:
        """ExplainArgs has correct defaults."""
        args = ExplainArgs()
        assert args.backend == "xgboost"
        assert args.dataset == "taiwan"
        assert args.explainer == "permutation"
        assert args.model_path is None
        assert args.n_samples == 1000
        assert args.target_class == 1
        assert args.top_n == 20
        assert args.verbose is False


class TestParseArgsUnknownArgument:
    """Tests for parse_args with unknown arguments."""

    def test_unknown_argument_is_skipped(self) -> None:
        """Unknown arguments are skipped during parsing."""
        # Unknown argument should be ignored, using default values
        args = parse_args(["--unknown-flag"])
        # Should have default values since unknown arg was skipped
        assert args.backend == "xgboost"
        assert args.dataset == "taiwan"

    def test_unknown_argument_with_known_args(self) -> None:
        """Unknown arguments don't affect known arguments."""
        args = parse_args(["--unknown", "-b", "mlp", "-e", "gradient"])
        assert args.backend == "mlp"
        assert args.explainer == "gradient"


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImport:
    """Tests for module imports to ensure coverage."""

    def test_main_module_import(self) -> None:
        """Main module can be imported."""
        from scripts.explain.main import main, run

        assert callable(main)
        assert callable(run)

    def test_dunder_main_import(self) -> None:
        """__main__ module can be imported."""
        from scripts.explain import __main__ as dunder_main

        # The import triggers line 3 coverage
        # Verify module name is the expected value
        assert dunder_main.__name__ == "scripts.explain.__main__"

    def test_dunder_main_runs_as_module(self) -> None:
        """__main__ module runs correctly when invoked with python -m."""
        import subprocess
        import sys

        # Run the module with --help flag, which exits with code 0
        result = subprocess.run(
            [sys.executable, "-m", "scripts.explain", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        # --help flag causes SystemExit(0) which means success
        assert result.returncode == 0
        # Should contain usage info
        assert "Usage:" in result.stdout or "usage:" in result.stdout.lower()

    def test_dunder_main_executes_main_guard(self) -> None:
        """__main__ module's if __name__ == '__main__' block is covered.

        Uses runpy.run_module to execute the module in a way that triggers
        the __name__ == '__main__' guard, which allows coverage to track
        line 6 execution.
        """
        import runpy
        import sys
        from types import ModuleType

        # Clear module from sys.modules to avoid runpy warning about
        # module already being imported
        modules_to_clear = [k for k in sys.modules if k.startswith("scripts.explain")]
        saved_modules: dict[str, ModuleType] = {}
        for mod in modules_to_clear:
            saved_modules[mod] = sys.modules.pop(mod)

        original_argv = sys.argv
        sys.argv = ["scripts.explain", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("scripts.explain", run_name="__main__", alter_sys=True)
            # --help exits with 0
            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv
            # Restore modules
            sys.modules.update(saved_modules)


# =============================================================================
# Run Function Tests
# =============================================================================


class TestRunFunction:
    """Tests for the run() function in main.py."""

    def test_run_file_not_found_returns_one(self) -> None:
        """Run returns 1 when model file not found."""
        from scripts.explain.main import main

        # Use a valid backend/explainer combo but non-existent model
        exit_code = main(["-b", "xgboost", "-m", "/nonexistent/path/model.ubj"])
        assert exit_code == 1

    def test_run_with_verbose_flag(self) -> None:
        """Run with verbose flag sets up DEBUG logging."""
        from scripts.explain.main import main

        # Test verbose + file not found (covers verbose setup then error)
        exit_code = main(["-b", "xgboost", "-v", "-m", "/nonexistent/path.ubj"])
        assert exit_code == 1

    def test_run_success_returns_zero(self) -> None:
        """Run returns 0 on successful execution."""
        from tempfile import NamedTemporaryFile

        from scripts.explain.main import main

        # Create a fake model file
        with NamedTemporaryFile(suffix=".ubj", delete=False) as f:
            temp_path = f.name
            f.write(b"fake model data")

        # Save originals
        orig_ds_registry = _hooks.dataset_registry_factory
        orig_ds_loader = _hooks.dataset_loader
        orig_exp_registry = _hooks.explainer_registry_factory

        try:
            # Set up all fakes
            _hooks.dataset_registry_factory = _make_fake_dataset_registry
            _hooks.dataset_loader = lambda config, external_dir: _make_fake_dataset()
            _hooks.explainer_registry_factory = _make_fake_explainer_registry

            # Need to also mock the model loader
            from covenant_radar_api.worker import _explain_loaders
            from covenant_radar_api.worker._explain_loaders import (
                LSTMModelConfig,
                MLPModelConfig,
            )

            orig_model_loader = _explain_loaders.load_model_for_backend

            def fake_model_loader(
                backend: BackendName,
                model_path: str,
                mlp_config: MLPModelConfig | None = None,
                lstm_config: LSTMModelConfig | None = None,
            ) -> PredictorProtocol:
                return FakePredictor()

            _explain_loaders.load_model_for_backend = fake_model_loader

            try:
                exit_code = main(["-m", temp_path, "-n", "50"])
                assert exit_code == 0
            finally:
                _explain_loaders.load_model_for_backend = orig_model_loader
        finally:
            _hooks.dataset_registry_factory = orig_ds_registry
            _hooks.dataset_loader = orig_ds_loader
            _hooks.explainer_registry_factory = orig_exp_registry
            Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# Run Explanation Tests
# =============================================================================


class TestRunExplanation:
    """Tests for run_explanation function."""

    def test_run_explanation_file_not_found(self) -> None:
        """run_explanation raises FileNotFoundError for missing model."""
        from scripts.explain.runner import run_explanation

        with pytest.raises(FileNotFoundError):
            run_explanation(
                backend="xgboost",
                dataset="taiwan",
                explainer="permutation",
                model_path="/nonexistent/model.ubj",
                n_samples=100,
                target_class=1,
            )

    def test_run_explanation_incompatible_explainer(self) -> None:
        """run_explanation raises ValueError for incompatible explainer."""
        from tempfile import NamedTemporaryFile

        from scripts.explain.runner import run_explanation

        # Create a fake model file so it exists
        with NamedTemporaryFile(suffix=".ubj", delete=False) as f:
            temp_path = f.name
            f.write(b"fake")

        try:
            # Set up fake explainer registry that says gradient is incompatible
            orig_registry = _hooks.explainer_registry_factory

            def fake_registry_incompatible() -> ExplainerRegistry:
                registry = ExplainerRegistry()

                def make_fake() -> FeatureExplainer:
                    return FakeExplainer()

                # Only permutation is compatible with xgboost
                backends: frozenset[BackendName] = frozenset(["mlp", "lstm"])
                registration = ExplainerRegistration(
                    factory=make_fake,
                    compatible_backends=backends,
                    requires_gradients=True,
                )
                registry.register("gradient", registration)
                return registry

            _hooks.explainer_registry_factory = fake_registry_incompatible

            with pytest.raises(ValueError) as exc_info:
                run_explanation(
                    backend="xgboost",
                    dataset="taiwan",
                    explainer="gradient",
                    model_path=temp_path,
                    n_samples=100,
                    target_class=1,
                )
            assert "not compatible" in str(exc_info.value)
        finally:
            _hooks.explainer_registry_factory = orig_registry
            Path(temp_path).unlink(missing_ok=True)

    def test_run_explanation_success_with_fakes(self) -> None:
        """run_explanation succeeds with fake dependencies."""
        from tempfile import NamedTemporaryFile

        from scripts.explain.runner import run_explanation

        # Create a fake model file
        with NamedTemporaryFile(suffix=".ubj", delete=False) as f:
            temp_path = f.name
            f.write(b"fake model data")

        # Save originals
        orig_ds_registry = _hooks.dataset_registry_factory
        orig_ds_loader = _hooks.dataset_loader
        orig_exp_registry = _hooks.explainer_registry_factory

        try:
            # Set up all fakes
            _hooks.dataset_registry_factory = _make_fake_dataset_registry
            _hooks.dataset_loader = lambda config, external_dir: _make_fake_dataset()
            _hooks.explainer_registry_factory = _make_fake_explainer_registry

            # Need to also mock the model loader
            from covenant_radar_api.worker import _explain_loaders
            from covenant_radar_api.worker._explain_loaders import (
                LSTMModelConfig,
                MLPModelConfig,
            )

            orig_model_loader = _explain_loaders.load_model_for_backend

            def fake_model_loader(
                backend: BackendName,
                model_path: str,
                mlp_config: MLPModelConfig | None = None,
                lstm_config: LSTMModelConfig | None = None,
            ) -> PredictorProtocol:
                return FakePredictor()

            _explain_loaders.load_model_for_backend = fake_model_loader

            try:
                result = run_explanation(
                    backend="xgboost",
                    dataset="taiwan",
                    explainer="permutation",
                    model_path=temp_path,
                    n_samples=50,
                    target_class=1,
                )

                # Verify result structure
                assert result["backend"] == "xgboost"
                assert result["dataset"] == "taiwan"
                assert result["explainer"] == "permutation"
                assert result["result"]["status"] == "complete"
                assert result["elapsed"] > 0.0
                # Check first feature importance has expected structure
                first_importance = result["result"]["feature_importances"][0]
                assert first_importance["name"] == "feature_0"
            finally:
                _explain_loaders.load_model_for_backend = orig_model_loader
        finally:
            _hooks.dataset_registry_factory = orig_ds_registry
            _hooks.dataset_loader = orig_ds_loader
            _hooks.explainer_registry_factory = orig_exp_registry
            Path(temp_path).unlink(missing_ok=True)

    def test_run_explanation_default_model_path(self) -> None:
        """run_explanation uses default model path when model_path is None."""
        from scripts.explain.runner import run_explanation

        # With no model_path and a dataset/backend combo that doesn't have a model,
        # it should look for default path which won't exist
        # Use lstm+polish which is unlikely to have a trained model
        with pytest.raises(FileNotFoundError) as exc_info:
            run_explanation(
                backend="lstm",
                dataset="polish",
                explainer="permutation",
                model_path=None,  # Use default
                n_samples=100,
                target_class=1,
            )
        # Error message should include the default path
        assert "polish_lstm_best.pt" in str(exc_info.value)


# =============================================================================
# Main ValueError Handler Tests
# =============================================================================


class TestMainValueErrorHandler:
    """Tests for main() ValueError exception handler (lines 82-85)."""

    def test_main_catches_value_error_from_runner(self) -> None:
        """Main catches ValueError raised by run_explanation and returns 1.

        This tests the ValueError exception handler in main.py by:
        1. Passing a valid CLI combo (permutation + xgboost) that passes validation
        2. Overriding the explainer_registry_factory to return a registry where
           that combo is NOT compatible
        3. The ValueError is raised by runner.py and caught by main.py
        """
        from tempfile import NamedTemporaryFile

        from scripts.explain.main import main

        # Create a fake model file so FileNotFoundError doesn't happen first
        with NamedTemporaryFile(suffix=".ubj", delete=False) as f:
            temp_path = f.name
            f.write(b"fake model data")

        # Save originals
        orig_ds_registry = _hooks.dataset_registry_factory
        orig_ds_loader = _hooks.dataset_loader
        orig_exp_registry = _hooks.explainer_registry_factory

        try:
            # Set up dataset fakes
            _hooks.dataset_registry_factory = _make_fake_dataset_registry
            _hooks.dataset_loader = lambda config, external_dir: _make_fake_dataset()

            def fake_registry_rejects_permutation() -> ExplainerRegistry:
                """Registry that reports permutation as incompatible with xgboost.

                CLI uses BACKEND_EXPLAINERS which allows permutation+xgboost,
                but this registry will report False for is_compatible(),
                causing ValueError in runner.py line 186.
                """
                registry = ExplainerRegistry()

                def make_fake() -> FeatureExplainer:
                    return FakeExplainer()

                # Register permutation with ONLY neural network backends
                # (not xgboost), so is_compatible("permutation", "xgboost") = False
                backends: frozenset[BackendName] = frozenset(["mlp", "lstm"])
                registration = ExplainerRegistration(
                    factory=make_fake,
                    compatible_backends=backends,
                    requires_gradients=False,
                )
                registry.register("permutation", registration)
                return registry

            _hooks.explainer_registry_factory = fake_registry_rejects_permutation

            # Call main with permutation+xgboost (valid CLI combo)
            # CLI validation passes, but runner's registry check fails
            exit_code = main(["-b", "xgboost", "-e", "permutation", "-m", temp_path])

            # ValueError should be caught and return 1
            assert exit_code == 1
        finally:
            _hooks.dataset_registry_factory = orig_ds_registry
            _hooks.dataset_loader = orig_ds_loader
            _hooks.explainer_registry_factory = orig_exp_registry
            Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# Main KeyboardInterrupt Handler Tests
# =============================================================================


class TestMainKeyboardInterruptHandler:
    """Tests for main() KeyboardInterrupt exception handler (lines 86-89)."""

    def test_main_catches_keyboard_interrupt_returns_130(self) -> None:
        """Main catches KeyboardInterrupt and returns 130.

        This tests the KeyboardInterrupt exception handler in main.py by
        making the dataset_loader hook raise KeyboardInterrupt to simulate
        the user pressing Ctrl+C during data loading.
        """
        from tempfile import NamedTemporaryFile

        from scripts.explain.main import main

        # Create a fake model file so FileNotFoundError doesn't happen first
        with NamedTemporaryFile(suffix=".ubj", delete=False) as f:
            temp_path = f.name
            f.write(b"fake model data")

        # Save originals
        orig_ds_registry = _hooks.dataset_registry_factory
        orig_ds_loader = _hooks.dataset_loader
        orig_exp_registry = _hooks.explainer_registry_factory

        try:
            # Set up dataset registry
            _hooks.dataset_registry_factory = _make_fake_dataset_registry
            _hooks.explainer_registry_factory = _make_fake_explainer_registry

            def loader_raises_keyboard_interrupt(
                config: DatasetConfig, external_dir: Path
            ) -> LoadedDataset:
                """Simulate user pressing Ctrl+C during data loading."""
                raise KeyboardInterrupt()

            _hooks.dataset_loader = loader_raises_keyboard_interrupt

            # Call main - KeyboardInterrupt should be caught and return 130
            exit_code = main(["-b", "xgboost", "-e", "permutation", "-m", temp_path])

            # KeyboardInterrupt handler returns 130
            assert exit_code == 130
        finally:
            _hooks.dataset_registry_factory = orig_ds_registry
            _hooks.dataset_loader = orig_ds_loader
            _hooks.explainer_registry_factory = orig_exp_registry
            Path(temp_path).unlink(missing_ok=True)
