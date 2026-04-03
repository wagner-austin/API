"""Shared fixtures for submit pipeline tests.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import shutil
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
    ProgressCallback,
)
from covenant_ml.backends.registry import BackendRegistration, ClassifierRegistry
from covenant_ml.optimizer import SearchSpace
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    LightGBMConfig,
    TrainOutcome,
)
from numpy.typing import NDArray
from scripts.submit import _hooks as submit_hooks
from scripts.submit._hooks import ConsoleProtocol

# =============================================================================
# Fixtures Directory
# =============================================================================

# Path: conftest.py -> submit -> tests -> covenant-radar-api -> services -> API
_API_ROOT = Path(__file__).parent.parent.parent.parent.parent
FIXTURES_DIR = _API_ROOT / "libs" / "covenant_ml" / "tests" / "datasets" / "fixtures"


# =============================================================================
# Test Console
# =============================================================================


class TestConsole:
    """Test console that captures output."""

    def __init__(self) -> None:
        """Initialize empty output list."""
        self.messages: list[str] = []

    def write(self, message: str) -> None:
        """Capture message.

        Args:
            message: Message to capture.
        """
        self.messages.append(message)


_test_console: TestConsole | None = None


def _get_test_console() -> ConsoleProtocol:
    """Get test console for dependency injection."""
    global _test_console
    if _test_console is None:
        _test_console = TestConsole()
    return _test_console


def get_captured_console() -> TestConsole:
    """Get global test console with guaranteed non-None.

    Returns:
        The current test console.

    Raises:
        RuntimeError: If console not initialized.
    """
    global _test_console
    console = _test_console
    if console is None:
        msg = "_test_console not initialized"
        raise RuntimeError(msg)
    return console


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def setup_test_hooks() -> Generator[None, None, None]:
    """Set up test hooks before each test."""
    global _test_console
    _test_console = TestConsole()

    # Save original hooks
    original_console_hook = submit_hooks.console_hook
    original_project_root_hook = submit_hooks.project_root_hook
    original_registry_hook = submit_hooks.registry_hook

    # Install test hooks
    submit_hooks.console_hook = _get_test_console

    yield

    # Restore original hooks
    submit_hooks.console_hook = original_console_hook
    submit_hooks.project_root_hook = original_project_root_hook
    submit_hooks.registry_hook = original_registry_hook
    _test_console = None


# =============================================================================
# Fake Backend for Integration Tests
# =============================================================================


class FakeTrainedClassifier:
    """Fake trained classifier for testing."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return fake probabilities.

        Args:
            x: Input features.

        Returns:
            2D array with shape (n_samples, 2).
        """
        n_samples: int = int(x.shape[0])
        probs: NDArray[np.float64] = np.column_stack(
            [
                np.full(n_samples, 0.3, dtype=np.float64),
                np.full(n_samples, 0.7, dtype=np.float64),
            ]
        )
        return probs


# Global for storing fake backend path
_fake_backend_path: Path | None = None


class FakeBackend:
    """Fake classifier backend for testing."""

    def __init__(self) -> None:
        """Initialize fake backend."""
        pass

    def backend_name(self) -> BackendName:
        """Return backend name."""
        return "lightgbm"

    def capabilities(self) -> BackendCapabilities:
        """Return backend capabilities."""
        return BackendCapabilities(
            supports_train=True,
            supports_gpu=False,
            supports_early_stopping=True,
            supports_feature_importance=True,
            model_format="txt",
        )

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Prepare a classifier."""
        return FakeTrainedClassifier()

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
        """Train a classifier."""
        global _fake_backend_path
        n_samples: int = int(x_features.shape[0])

        if _fake_backend_path is None:
            model_path = output_dir / "model.txt"
        else:
            model_path = _fake_backend_path / "model.txt"

        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("fake model")

        train_metrics: EvalMetrics = EvalMetrics(
            loss=0.3,
            ppl=1.35,
            auc=0.85,
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1_score=0.72,
        )
        val_metrics: EvalMetrics = EvalMetrics(
            loss=0.35,
            ppl=1.42,
            auc=0.82,
            accuracy=0.78,
            precision=0.72,
            recall=0.68,
            f1_score=0.70,
        )
        test_metrics: EvalMetrics = EvalMetrics(
            loss=0.38,
            ppl=1.46,
            auc=0.80,
            accuracy=0.76,
            precision=0.70,
            recall=0.66,
            f1_score=0.68,
        )

        fake_config: LightGBMConfig = LightGBMConfig(
            device="cpu",
            learning_rate=0.1,
            max_depth=-1,
            n_estimators=100,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
            early_stopping_rounds=10,
        )

        feature_importances: list[FeatureImportance] = []

        return TrainOutcome(
            model_path=str(model_path),
            model_id="fake-model-001",
            samples_total=n_samples,
            samples_train=int(n_samples * 0.7),
            samples_val=int(n_samples * 0.15),
            samples_test=int(n_samples * 0.15),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_val_auc=0.82,
            best_round=50,
            total_rounds=100,
            early_stopped=True,
            config=fake_config,
            feature_importances=feature_importances,
            scale_pos_weight_computed=1.0,
        )

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Evaluate a model."""
        return EvalMetrics(
            loss=0.35,
            ppl=1.42,
            auc=0.82,
            accuracy=0.78,
            precision=0.72,
            recall=0.68,
            f1_score=0.70,
        )

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Save model."""
        Path(path).write_text("fake saved model")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load model."""
        return FakeTrainedClassifier()

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Get feature importances."""
        return None

    def get_default_search_space(self) -> SearchSpace:
        """Not used in submit tests."""
        raise NotImplementedError

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Not used in submit tests."""
        raise NotImplementedError


def create_fake_backend() -> ClassifierBackend:
    """Factory function to create a fake backend."""
    return FakeBackend()


def create_fake_registry() -> ClassifierRegistry:
    """Create a ClassifierRegistry with fake backend registered."""
    registry = ClassifierRegistry()
    registration = BackendRegistration(create_fake_backend)
    registry.register("lightgbm", registration)
    return registry


def set_fake_backend_path(path: Path) -> None:
    """Set the path for fake backend to save models.

    Args:
        path: Path for model artifacts.
    """
    global _fake_backend_path
    _fake_backend_path = path


@pytest.fixture()
def timeseries_fixture_dir(tmp_path: Path) -> Path:
    """Create a copy of the timeseries_simple fixture for testing.

    Args:
        tmp_path: Pytest temporary path.

    Returns:
        Path to the fixture directory with data.csv and labels.csv.
    """
    source_dir = FIXTURES_DIR / "timeseries_simple"
    dest_dir = tmp_path / "timeseries_simple"
    dest_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(source_dir / "data.csv", dest_dir / "data.csv")
    shutil.copy(source_dir / "labels.csv", dest_dir / "labels.csv")

    return dest_dir
