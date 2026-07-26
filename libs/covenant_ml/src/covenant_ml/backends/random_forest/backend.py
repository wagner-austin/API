"""Random Forest backend for tabular binary classification.

Implements ClassifierBackend protocol using sklearn RandomForestClassifier with:
- Stratified train/val/test splits
- Bagging with feature randomization
- Class weight balancing for imbalanced data
- Built-in feature importance extraction
- Out-of-bag score computation (optional)
- Strict typing (no Any, no casts, no type: ignore)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from ...metrics import compute_all_metrics
from ...optimizer.search_spaces import (
    make_random_forest_default_space,
    make_random_forest_focused_space,
)
from ...optimizer.types import SampledFloatParams, SampledIntParams, SearchSpace
from ...trainer import stratified_split
from ...types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    RandomForestConfig,
    RandomForestModelMeta,
    TrainOutcome,
    TrainProgress,
)
from ..protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
    ProgressCallback,
)

_log = get_logger(__name__)


def _is_random_forest_config(cfg: ClassifierTrainConfig) -> TypeGuard[RandomForestConfig]:
    """Check if config is RandomForestConfig by looking for RF-specific keys.

    Args:
        cfg: Classifier training configuration to check.

    Returns:
        True if config contains RandomForest-specific keys.
    """
    return (
        isinstance(cfg, dict)
        and "n_estimators" in cfg
        and "bootstrap" in cfg
        and "min_samples_split" in cfg
        and "min_samples_leaf" in cfg
        and "oob_score" in cfg
    )


class _RFModelProtocol(Protocol):
    """Protocol for sklearn RandomForestClassifier."""

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        """Feature importance scores (Gini importance)."""
        ...

    @property
    def n_estimators(self) -> int:
        """Number of trees in the forest."""
        ...

    @property
    def oob_score_(self) -> float:
        """Out-of-bag score (only available if oob_score=True)."""
        ...

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
    ) -> _RFModelProtocol:
        """Fit the model to training data."""
        ...

    def predict_proba(self, x_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        ...


class _RFClassifierCtor(Protocol):
    """Protocol for RandomForestClassifier constructor."""

    def __call__(
        self,
        *,
        n_estimators: int,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: Literal["sqrt", "log2"] | float | int | None,
        bootstrap: bool,
        class_weight: str | None,
        n_jobs: int,
        random_state: int,
        oob_score: bool,
    ) -> _RFModelProtocol:
        """Construct RandomForestClassifier with given parameters."""
        ...


class _JoblibDumpProtocol(Protocol):
    """Protocol for joblib.dump function."""

    def __call__(self, value: _RFModelProtocol, filename: str) -> list[str]:
        """Dump object to file."""
        ...


class _JoblibLoadProtocol(Protocol):
    """Protocol for joblib.load function."""

    def __call__(self, filename: str) -> _RFModelProtocol:
        """Load object from file."""
        ...


class _RandomForestPrepared:
    """Prepared RandomForest model for inference.

    Wraps a fitted sklearn RandomForestClassifier model to satisfy
    the PreparedClassifier protocol.
    """

    def __init__(self, model: _RFModelProtocol) -> None:
        """Initialize with fitted model.

        Args:
            model: Fitted RandomForestClassifier model.
        """
        self._model = model

    @property
    def raw_model(self) -> _RFModelProtocol:
        """Return the underlying sklearn RandomForestClassifier.

        SHAP TreeExplainer accepts sklearn ensembles directly and rejects
        wrappers with "Model type not yet supported by TreeExplainer", so the
        native handle has to be reachable for shap_tree to work here.

        Returns:
            The raw sklearn model.
        """
        return self._model

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


RANDOM_FOREST_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": False,
    "supports_early_stopping": False,
    "supports_feature_importance": True,
    "model_format": "joblib",
}


def _get_sklearn_imports() -> tuple[_RFClassifierCtor, _JoblibDumpProtocol, _JoblibLoadProtocol]:
    """Get sklearn RandomForestClassifier and joblib via dynamic import.

    Returns:
        Tuple of (RandomForestClassifier constructor, joblib.dump, joblib.load).
    """
    sklearn_module = __import__(
        "sklearn.ensemble",
        fromlist=["RandomForestClassifier"],
    )
    rf_ctor: _RFClassifierCtor = sklearn_module.RandomForestClassifier

    joblib_module = __import__("joblib", fromlist=["dump", "load"])
    dump_fn: _JoblibDumpProtocol = joblib_module.dump
    load_fn: _JoblibLoadProtocol = joblib_module.load

    return rf_ctor, dump_fn, load_fn


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
        "Computed class weight ratio for RandomForest",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed


def _extract_feature_importances(
    model: _RFModelProtocol,
    feature_names: list[str],
) -> list[FeatureImportance]:
    """Extract feature importances from Random Forest.

    Uses Gini importance (mean decrease in impurity).

    Args:
        model: Fitted RandomForestClassifier model.
        feature_names: List of feature names.

    Returns:
        List of FeatureImportance sorted by importance (descending).
    """
    raw_importances: NDArray[np.float64] = model.feature_importances_
    importances: NDArray[np.float64] = np.asarray(raw_importances, dtype=np.float64)

    unsorted: list[tuple[str, float]] = []
    for i in range(len(feature_names)):
        # Use slice indexing to get typed float
        val_slice = np.asarray(importances[i : i + 1], dtype=np.float64).flat
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


class RandomForestBackend(ClassifierBackend):
    """Random Forest backend for tabular binary classification."""

    def backend_name(self) -> BackendName:
        """Return the backend name identifier.

        Returns:
            Literal "random_forest" backend name.
        """
        return "random_forest"

    def capabilities(self) -> BackendCapabilities:
        """Return backend capabilities.

        Returns:
            BackendCapabilities dict with supported features.
        """
        return RANDOM_FOREST_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Prepare is not supported for RandomForest.

        RandomForest uses on-demand training via train(). Use train() to create
        a fitted model, then load() to get a PreparedClassifier for inference.

        Args:
            n_features: Number of input features.
            n_classes: Number of output classes.
            feature_names: Optional feature names.

        Raises:
            RuntimeError: Always, as prepare is not supported.
        """
        raise RuntimeError(
            "RandomForestBackend.prepare not supported; use train() then load() for inference."
        )

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
        """Train Random Forest model on tabular data.

        Args:
            x_features: Feature matrix of shape (n_samples, n_features).
            y_labels: Binary labels of shape (n_samples,).
            feature_names: Optional list of feature names.
            config: RandomForestConfig with hyperparameters.
            output_dir: Directory to save model artifacts.
            progress: Optional progress callback.

        Returns:
            TrainOutcome with training results and metrics.

        Raises:
            RuntimeError: If config is not RandomForestConfig.
            ValueError: If training set has no positive samples.
        """
        if not _is_random_forest_config(config):
            raise RuntimeError("RandomForestBackend requires RandomForestConfig")

        cfg: RandomForestConfig = config
        rf_ctor, dump_fn, _ = _get_sklearn_imports()

        splits = stratified_split(
            x_features,
            y_labels,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            random_state=cfg["random_state"],
        )

        scale_pos_weight = _compute_class_weight(splits.y_train)

        n_feats = int(x_features.shape[1])
        resolved_names: list[str]
        if feature_names is None:
            resolved_names = [f"f{i}" for i in range(n_feats)]
        else:
            resolved_names = feature_names

        class_weight_arg: str | None = "balanced" if cfg["class_weight_balanced"] else None

        model = rf_ctor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            bootstrap=cfg["bootstrap"],
            class_weight=class_weight_arg,
            n_jobs=cfg["n_jobs"],
            random_state=cfg["random_state"],
            oob_score=cfg["oob_score"],
        )

        _log.info(
            "Training RandomForest model",
            extra={
                "n_samples": splits.n_train,
                "n_features": n_feats,
                "n_estimators": cfg["n_estimators"],
                "max_depth": cfg["max_depth"],
                "bootstrap": cfg["bootstrap"],
            },
        )
        model.fit(splits.x_train, splits.y_train)

        if cfg["oob_score"] and cfg["bootstrap"]:
            oob_score_val: float = float(model.oob_score_)
            _log.info(
                "RandomForest OOB score",
                extra={"oob_score": oob_score_val},
            )

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
        model_path = output_dir / "random_forest_model.joblib"
        dump_fn(model, str(model_path))

        meta: RandomForestModelMeta = {
            "backend": "random_forest",
            "n_features": n_feats,
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
        }
        meta_path = output_dir / "random_forest_model.json"
        meta_json = dump_json_str(meta)
        meta_path.write_text(meta_json, encoding="utf-8")

        _log.info(
            "RandomForest training complete",
            extra={
                "model_path": str(model_path),
                "val_auc": val_metrics["auc"],
                "test_auc": test_metrics["auc"],
            },
        )

        return TrainOutcome(
            model_path=str(model_path),
            model_id="random_forest",
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
        """Save is not supported for RandomForestBackend.

        Model saving is handled in train() via joblib.

        Args:
            model: Prepared classifier.
            path: Target file path.

        Raises:
            RuntimeError: Always, as save is not supported.
        """
        raise RuntimeError("RandomForestBackend.save not supported; use TrainOutcome.model_path.")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load RandomForest model from file.

        Args:
            path: Path to the saved model file (.joblib format).

        Returns:
            PreparedClassifier wrapping the loaded model.
        """
        _, _, load_fn = _get_sklearn_imports()
        model = load_fn(path)
        return _RandomForestPrepared(model)

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
        """Return default Random Forest search space.

        Returns:
            RandomForestSearchSpace with sensible default ranges.
        """
        return make_random_forest_default_space()

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return focused Random Forest search space around prior best params.

        Args:
            best_int_params: Best integer params (reads max_depth, n_estimators).
            best_float_params: Best float params (unused for Random Forest).

        Returns:
            RandomForestSearchSpace with narrowed ranges.
        """
        return make_random_forest_focused_space(
            best_max_depth=best_int_params["max_depth"],
            best_n_estimators=best_int_params["n_estimators"],
        )


def create_random_forest_backend() -> RandomForestBackend:
    """Create RandomForest backend instance.

    Returns:
        New RandomForestBackend instance.
    """
    return RandomForestBackend()


__all__ = [
    "RANDOM_FOREST_CAPABILITIES",
    "RandomForestBackend",
    "create_random_forest_backend",
]
