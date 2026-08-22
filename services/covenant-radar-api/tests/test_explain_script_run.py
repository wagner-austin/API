"""Tests for scripts/explain CLI entry point.

Tests use dependency injection via scripts/_test_hooks to avoid real model loading.
All code paths are tested with strong assertions on actual behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import scripts._test_hooks as _hooks
from covenant_ml.datasets import DatasetConfig, LoadedDataset
from covenant_ml.explainers.registry import ExplainerRegistration, ExplainerRegistry
from covenant_ml.types import BackendName
from platform_core.rich_logging import setup_rich_logging
from platform_ml.explainers.protocol import FeatureExplainer, PredictorProtocol
from scripts.explain import main

from tests._explain_script_fixtures import (
    FakeExplainer,
    FakePredictor,
    _make_fake_dataset,
    _make_fake_dataset_registry,
    _make_fake_explainer_registry,
)


@pytest.fixture(autouse=True)
def _setup_rich_logging_for_tests() -> None:
    """Set up rich logging before each test that needs it."""
    setup_rich_logging(level="WARNING", show_time=False)


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
