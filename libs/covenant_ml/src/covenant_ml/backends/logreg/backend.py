"""Logistic Regression backend for tabular binary classification.

Implements ClassifierBackend protocol using sklearn LogisticRegression with:
- Stratified train/val/test splits
- L1/L2/ElasticNet regularization support
- Class weight balancing for imbalanced data
- Feature coefficient extraction as importances
- Strict typing (no Any, no casts, no type: ignore)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    LogRegConfig,
    TrainOutcome,
    TrainProgress,
)
from covenant_ml.types_model_meta import LogRegModelMeta

from ...metrics import compute_all_metrics
from ...optimizer.search_spaces import make_logreg_default_space, make_logreg_focused_space
from ...optimizer.types import SampledFloatParams, SampledIntParams, SearchSpace
from ...trainer import stratified_split
from ..protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
)

_log = get_logger(__name__)


def _is_logreg_config(cfg: ClassifierTrainConfig) -> TypeGuard[LogRegConfig]:
    """Check if config is LogRegConfig by looking for LogReg-specific keys.

    Args:
        cfg: Classifier training configuration to check.

    Returns:
        True if config contains LogReg-specific keys (solver, penalty, C).
    """
    return (
        isinstance(cfg, dict)
        and "solver" in cfg
        and "penalty" in cfg
        and "C" in cfg
        and "max_iter" in cfg
    )


class _LogRegModelProtocol(Protocol):
    """Protocol for sklearn LogisticRegression classifier."""

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Coefficient weights for features."""
        ...

    @property
    def intercept_(self) -> NDArray[np.float64]:
        """Intercept (bias) term."""
        ...

    @property
    def classes_(self) -> NDArray[np.int64]:
        """Class labels known to the classifier."""
        ...

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
    ) -> _LogRegModelProtocol:
        """Fit the model to training data."""
        ...

    def predict_proba(self, x_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        ...


class _JoblibDumpProtocol(Protocol):
    """Protocol for joblib.dump function."""

    def __call__(self, value: _LogRegModelProtocol, filename: str) -> list[str]:
        """Dump object to file."""
        ...


class _JoblibLoadProtocol(Protocol):
    """Protocol for joblib.load function."""

    def __call__(self, filename: str) -> _LogRegModelProtocol:
        """Load object from file."""
        ...


class _LogRegPrepared:
    """Prepared LogisticRegression model for inference.

    Wraps a fitted sklearn LogisticRegression model to satisfy
    the PreparedClassifier protocol.
    """

    def __init__(self, model: _LogRegModelProtocol) -> None:
        """Initialize with fitted model.

        Args:
            model: Fitted LogisticRegression model.
        """
        self._model = model

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        raw_proba: NDArray[np.float64] = self._model.predict_proba(x)
        proba: NDArray[np.float64] = np.asarray(raw_proba, dtype=np.float64)
        return proba


LOGREG_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": False,
    "supports_early_stopping": False,
    "supports_feature_importance": True,
    "model_format": "joblib",
}


def _get_joblib_imports() -> tuple[_JoblibDumpProtocol, _JoblibLoadProtocol]:
    """Get joblib dump/load via dynamic import.

    Returns:
        Tuple of (joblib.dump, joblib.load).
    """
    joblib_module = __import__("joblib", fromlist=["dump", "load"])
    dump_fn: _JoblibDumpProtocol = joblib_module.dump
    load_fn: _JoblibLoadProtocol = joblib_module.load
    return dump_fn, load_fn


def _create_logreg_model(
    *,
    penalty: str | None,
    inverse_reg_strength: float,
    solver: str,
    max_iter: int,
    tol: float,
    random_state: int,
    class_weight: str | None,
    l1_ratio: float | None,
    n_jobs: int,
) -> _LogRegModelProtocol:
    """Create sklearn LogisticRegression with given parameters.

    Maps inverse_reg_strength to sklearn's uppercase C parameter.

    Args:
        penalty: Regularization type ("l1", "l2", "elasticnet", or None).
        inverse_reg_strength: Inverse of regularization strength (sklearn C).
        solver: Optimization algorithm.
        max_iter: Maximum iterations for convergence.
        tol: Tolerance for stopping criteria.
        random_state: Random seed.
        class_weight: Class weight strategy ("balanced" or None).
        l1_ratio: ElasticNet mixing (only for elasticnet penalty).
        n_jobs: Number of parallel jobs.

    Returns:
        Fitted LogisticRegression model satisfying _LogRegModelProtocol.
    """
    sklearn_module = __import__(
        "sklearn.linear_model",
        fromlist=["LogisticRegression"],
    )
    model: _LogRegModelProtocol = sklearn_module.LogisticRegression(
        penalty=penalty,
        C=inverse_reg_strength,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        n_jobs=n_jobs,
    )
    return model


def _compute_class_weight(y_train: NDArray[np.int64]) -> float:
    """Compute scale_pos_weight from training labels.

    Args:
        y_train: Binary training labels.

    Returns:
        Ratio of negative to positive samples.

    Raises:
        ValueError: If no positive samples exist.
    """
    pos_mask: NDArray[np.bool_] = y_train == 1
    neg_mask: NDArray[np.bool_] = y_train == 0
    n_positive = int(np.count_nonzero(pos_mask))
    n_negative = int(np.count_nonzero(neg_mask))
    if n_positive == 0:
        raise ValueError("Training set has no positive samples")
    computed = float(n_negative) / float(n_positive)
    _log.info(
        "Computed class weight ratio for LogReg",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed


def _extract_feature_importances(
    model: _LogRegModelProtocol,
    feature_names: list[str],
) -> list[FeatureImportance]:
    """Extract feature importances from logistic regression coefficients.

    Uses absolute coefficient values as importance measure.
    Higher absolute coefficient = more important feature.

    Args:
        model: Fitted LogisticRegression model.
        feature_names: List of feature names.

    Returns:
        List of FeatureImportance sorted by importance (descending).
    """
    coef_raw: NDArray[np.float64] = model.coef_
    coef_flat: NDArray[np.float64] = np.asarray(coef_raw, dtype=np.float64).ravel()
    abs_coef: NDArray[np.float64] = np.abs(coef_flat)

    unsorted: list[tuple[str, float]] = []
    for i in range(len(feature_names)):
        # Use slice indexing to get typed float
        val_slice = np.asarray(abs_coef[i : i + 1], dtype=np.float64).flat
        importance_val: float = float(val_slice[0])
        unsorted.append((feature_names[i], importance_val))

    def get_importance(pair: tuple[str, float]) -> float:
        return pair[1]

    sorted_by_importance = sorted(unsorted, key=get_importance, reverse=True)

    result: list[FeatureImportance] = []
    for rank, (name, importance) in enumerate(sorted_by_importance, start=1):
        result.append(
            {
                "name": name,
                "importance": importance,
                "rank": rank,
            }
        )

    return result


class LogRegBackend(ClassifierBackend):
    """Logistic Regression backend for tabular binary classification."""

    def backend_name(self) -> BackendName:
        """Return the backend name identifier.

        Returns:
            Literal "logreg" backend name.
        """
        return "logreg"

    def capabilities(self) -> BackendCapabilities:
        """Return backend capabilities.

        Returns:
            BackendCapabilities dict with supported features.
        """
        return LOGREG_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Prepare is not supported for LogReg.

        LogReg uses on-demand training via train(). Use train() to create
        a fitted model, then load() to get a PreparedClassifier for inference.

        Args:
            n_features: Number of input features.
            n_classes: Number of output classes.
            feature_names: Optional feature names.

        Raises:
            RuntimeError: Always, as prepare is not supported.
        """
        raise RuntimeError(
            "LogRegBackend.prepare not supported; use train() then load() for inference."
        )

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: Callable[[TrainProgress], None] | None,
        groups: NDArray[np.int64] | None = None,
    ) -> TrainOutcome:
        """Train Logistic Regression model on tabular data.

        Args:
            x_features: Feature matrix of shape (n_samples, n_features).
            y_labels: Binary labels of shape (n_samples,).
            feature_names: Optional list of feature names.
            config: LogRegConfig with hyperparameters.
            output_dir: Directory to save model artifacts.
            progress: Optional progress callback.

        Returns:
            TrainOutcome with training results and metrics.

        Raises:
            RuntimeError: If config is not LogRegConfig.
            ValueError: If training set has no positive samples.
        """
        if not _is_logreg_config(config):
            raise RuntimeError("LogRegBackend requires LogRegConfig")

        cfg: LogRegConfig = config
        dump_fn, _ = _get_joblib_imports()

        splits = stratified_split(
            x_features,
            y_labels,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            random_state=cfg["random_state"],
            groups=groups,
        )

        scale_pos_weight = _compute_class_weight(splits.y_train)

        n_feats = int(x_features.shape[1])
        resolved_names: list[str]
        if feature_names is None:
            resolved_names = [f"f{i}" for i in range(n_feats)]
        else:
            resolved_names = feature_names

        class_weight_arg: str | None = "balanced" if cfg["class_weight_balanced"] else None
        penalty_arg: str | None = None if cfg["penalty"] == "none" else cfg["penalty"]
        l1_ratio_arg: float | None = cfg["l1_ratio"] if cfg["penalty"] == "elasticnet" else None

        model = _create_logreg_model(
            penalty=penalty_arg,
            inverse_reg_strength=cfg["C"],
            solver=cfg["solver"],
            max_iter=cfg["max_iter"],
            tol=cfg["tol"],
            random_state=cfg["random_state"],
            class_weight=class_weight_arg,
            l1_ratio=l1_ratio_arg,
            n_jobs=-1,
        )

        _log.info(
            "Training LogisticRegression model",
            extra={
                "n_samples": splits.n_train,
                "n_features": n_feats,
                "solver": cfg["solver"],
                "penalty": cfg["penalty"],
                "C": cfg["C"],
            },
        )
        model.fit(splits.x_train, splits.y_train)

        train_proba_raw: NDArray[np.float64] = model.predict_proba(splits.x_train)
        val_proba_raw: NDArray[np.float64] = model.predict_proba(splits.x_val)
        test_proba_raw: NDArray[np.float64] = model.predict_proba(splits.x_test)

        train_proba: NDArray[np.float64] = np.asarray(train_proba_raw[:, 1], dtype=np.float64)
        val_proba: NDArray[np.float64] = np.asarray(val_proba_raw[:, 1], dtype=np.float64)
        test_proba: NDArray[np.float64] = np.asarray(test_proba_raw[:, 1], dtype=np.float64)

        train_metrics = compute_all_metrics(splits.y_train, train_proba)
        val_metrics = compute_all_metrics(splits.y_val, val_proba)
        test_metrics = compute_all_metrics(splits.y_test, test_proba)

        if progress is not None:
            prog: TrainProgress = {
                "round": 1,
                "total_rounds": 1,
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["auc"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
            }
            progress(prog)

        feature_importances = _extract_feature_importances(model, resolved_names)

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "logreg_model.joblib"
        dump_fn(model, str(model_path))

        meta: LogRegModelMeta = {
            "backend": "logreg",
            "n_features": n_feats,
            "penalty": cfg["penalty"],
            "solver": cfg["solver"],
        }
        meta_path = output_dir / "logreg_model.json"
        meta_json = dump_json_str(meta)
        meta_path.write_text(meta_json, encoding="utf-8")

        _log.info(
            "LogReg training complete",
            extra={
                "model_path": str(model_path),
                "val_auc": val_metrics["auc"],
                "test_auc": test_metrics["auc"],
            },
        )

        return TrainOutcome(
            model_path=str(model_path),
            model_id="logreg",
            samples_total=splits.n_total,
            samples_train=splits.n_train,
            samples_val=splits.n_val,
            samples_test=splits.n_test,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_val_auc=val_metrics["auc"],
            best_round=1,
            total_rounds=1,
            early_stopped=False,
            config=cfg,
            feature_importances=feature_importances,
            scale_pos_weight_computed=scale_pos_weight,
        )

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Evaluate model on given data.

        Args:
            model: Prepared classifier for inference.
            x: Feature matrix of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).

        Returns:
            EvalMetrics with loss, AUC, accuracy, precision, recall, F1.
        """
        proba = model.predict_proba(x)
        return compute_all_metrics(y, proba[:, 1])

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Save is not supported for LogRegBackend.

        Model saving is handled in train() via joblib.

        Args:
            model: Prepared classifier.
            path: Target file path.

        Raises:
            RuntimeError: Always, as save is not supported.
        """
        raise RuntimeError("LogRegBackend.save not supported; use TrainOutcome.model_path.")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load LogReg model from file.

        Args:
            path: Path to the saved model file (.joblib format).

        Returns:
            PreparedClassifier wrapping the loaded model.
        """
        _, load_fn = _get_joblib_imports()
        model = load_fn(path)
        return _LogRegPrepared(model)

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Get feature importances.

        Feature importances are provided via TrainOutcome, so this returns None.

        Args:
            model: Prepared classifier.
            feature_names: Optional feature names.

        Returns:
            None (importances provided via TrainOutcome).
        """
        return None

    def get_default_search_space(self) -> SearchSpace:
        """Return default LogReg search space.

        Returns:
            LogRegSearchSpace with sensible default ranges.
        """
        return make_logreg_default_space()

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return focused LogReg search space around prior best params.

        Args:
            best_int_params: Best integer params (unused for LogReg).
            best_float_params: Best float params (reads C, tol).

        Returns:
            LogRegSearchSpace with narrowed ranges.
        """
        return make_logreg_focused_space(
            best_c=best_float_params["C"],
            best_tol=best_float_params["tol"],
        )


def create_logreg_backend() -> LogRegBackend:
    """Create LogReg backend instance.

    Returns:
        New LogRegBackend instance.
    """
    return LogRegBackend()


__all__ = [
    "LOGREG_CAPABILITIES",
    "LogRegBackend",
    "create_logreg_backend",
]
