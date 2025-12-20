"""Integration tests for submit pipeline.

Tests train_model, run_pipeline, and main with fake backend via hooks.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray
from scripts.submit import _hooks as submit_hooks
from scripts.submit.__main__ import main
from scripts.submit.pipeline import (
    SubmitConfig,
    run_pipeline,
    train_model,
)

from .conftest import (
    create_fake_registry,
    get_captured_console,
    set_fake_backend_path,
)

# =============================================================================
# Test Array Factories
# =============================================================================


def make_train_features() -> NDArray[np.float64]:
    """Create training feature array with 10 samples, 3 features."""
    arr: NDArray[np.float64] = np.zeros((10, 3), dtype=np.float64)
    for i in range(10):
        arr[i, 0] = float(i)
        arr[i, 1] = float(i * 2)
        arr[i, 2] = float(i * 3)
    return arr


def make_train_labels() -> NDArray[np.int64]:
    """Create training label array with 10 samples."""
    arr: NDArray[np.int64] = np.zeros(10, dtype=np.int64)
    for i in range(5):
        arr[i] = 0
    for i in range(5, 10):
        arr[i] = 1
    return arr


# =============================================================================
# Tests for train_model
# =============================================================================


class TestTrainModel:
    """Tests for train_model function."""

    def test_train_model_with_fake_backend(self, tmp_path: Path) -> None:
        """Test train_model with fake backend via hook."""
        # Set up fake registry hook
        set_fake_backend_path(tmp_path)
        submit_hooks.registry_hook = create_fake_registry

        x_train = make_train_features()
        y_train = make_train_labels()
        feature_names: tuple[str, ...] = ("f1", "f2", "f3")

        config = SubmitConfig(
            backend="lightgbm",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            aggregation="last",
            include_rank_features=True,
            include_diff_features=True,
        )

        output_dir = tmp_path / "models"
        output_dir.mkdir()

        model, result = train_model(
            x_train=x_train,
            y_train=y_train,
            feature_names=feature_names,
            config=config,
            output_dir=output_dir,
        )

        # Verify result
        assert result["n_samples"] == 10
        assert result["n_features"] == 3
        assert result["feature_names"] == ("f1", "f2", "f3")
        assert abs(result["val_auc"] - 0.82) < 0.001

        # Verify model can predict
        predictions = model.predict_proba(x_train)
        n_preds: int = int(predictions.shape[0])
        assert n_preds == 10

        # Verify console output
        captured = get_captured_console()
        assert len(captured.messages) >= 2


# =============================================================================
# Tests for run_pipeline
# =============================================================================


class TestRunPipeline:
    """Tests for run_pipeline function."""

    def test_run_pipeline_end_to_end(self, timeseries_fixture_dir: Path, tmp_path: Path) -> None:
        """Test full pipeline with fake backend."""
        # Set up fake registry hook
        set_fake_backend_path(tmp_path)
        submit_hooks.registry_hook = create_fake_registry

        # Create a copy for test data
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        shutil.copy(timeseries_fixture_dir / "data.csv", test_dir / "data.csv")
        shutil.copy(timeseries_fixture_dir / "labels.csv", test_dir / "labels.csv")

        output_path = tmp_path / "submissions" / "submission.csv"
        model_output_dir = tmp_path / "models"
        model_output_dir.mkdir()

        config = SubmitConfig(
            backend="lightgbm",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            aggregation="last",
            include_rank_features=False,
            include_diff_features=False,
        )

        result = run_pipeline(
            train_dir=timeseries_fixture_dir,
            test_dir=test_dir,
            output_path=output_path,
            config=config,
            model_output_dir=model_output_dir,
        )

        # Verify result
        assert result["n_samples"] == 3  # 3 entities in fixture
        assert len(result["entity_ids"]) == 3
        assert len(result["predictions"]) == 3

        # Verify output file was created
        assert output_path.exists()
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 4  # Header + 3 rows

    def test_run_pipeline_uses_default_model_output_dir(
        self, timeseries_fixture_dir: Path, tmp_path: Path
    ) -> None:
        """Test that run_pipeline uses output_path.parent as default model dir."""
        # Set up fake registry hook
        set_fake_backend_path(tmp_path)
        submit_hooks.registry_hook = create_fake_registry

        # Create a copy for test data
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        shutil.copy(timeseries_fixture_dir / "data.csv", test_dir / "data.csv")
        shutil.copy(timeseries_fixture_dir / "labels.csv", test_dir / "labels.csv")

        output_path = tmp_path / "output" / "submission.csv"

        config = SubmitConfig(
            backend="lightgbm",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            aggregation="last",
            include_rank_features=False,
            include_diff_features=False,
        )

        # Call without model_output_dir
        result = run_pipeline(
            train_dir=timeseries_fixture_dir,
            test_dir=test_dir,
            output_path=output_path,
            config=config,
        )

        assert result["n_samples"] == 3
        assert output_path.exists()


# =============================================================================
# Tests for main function
# =============================================================================


class TestMainFunction:
    """Tests for main entry point."""

    def test_main_with_fake_registry(self, timeseries_fixture_dir: Path, tmp_path: Path) -> None:
        """Test main function with fake backend via hook."""
        # Set up fake registry hook
        set_fake_backend_path(tmp_path)
        submit_hooks.registry_hook = create_fake_registry

        # Create test data
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        shutil.copy(timeseries_fixture_dir / "data.csv", test_dir / "data.csv")
        shutil.copy(timeseries_fixture_dir / "labels.csv", test_dir / "labels.csv")

        output_path = tmp_path / "submissions" / "submission.csv"

        exit_code = main(
            [
                "--train-dir",
                str(timeseries_fixture_dir),
                "--test-dir",
                str(test_dir),
                "--output",
                str(output_path),
                "-n",
                "10",
                "-l",
                "0.1",
                "--no-rank-features",
                "--no-diff-features",
            ]
        )

        assert exit_code == 0
        assert output_path.exists()

        # Verify console output includes final message
        captured = get_captured_console()
        found_generated = False
        for msg in captured.messages:
            if "Generated" in msg and "predictions" in msg:
                found_generated = True
                break
        assert found_generated


# =============================================================================
# Tests for module entry point
# =============================================================================


class TestModuleEntryPoint:
    """Tests for module entry point."""

    def test_module_main_entry_with_help(self) -> None:
        """Test __main__ entry point via runpy."""
        import runpy

        # Save original state
        original_argv = sys.argv.copy()
        saved_modules: dict[str, ModuleType] = {}

        # Clear any cached modules
        modules_to_clear: list[str] = [k for k in sys.modules if k.startswith("scripts.submit")]
        for mod in modules_to_clear:
            saved_modules[mod] = sys.modules.pop(mod)

        # Run with --help flag which causes SystemExit(0)
        sys.argv = ["submit", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("scripts.submit", run_name="__main__", alter_sys=True)
            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv
            # Restore saved modules
            sys.modules.update(saved_modules)

    def test_dunder_main_module_import_does_not_execute_main(self) -> None:
        """Test importing __main__.py directly doesn't execute main."""
        import importlib

        # Clear the module from cache if present
        mod_name = "scripts.submit.__main__"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        # Import the module - this covers the False branch of if __name__ == "__main__"
        module: ModuleType = importlib.import_module(mod_name)

        # Verify the module was imported but main wasn't called (no SystemExit)
        assert module.__name__ == mod_name
