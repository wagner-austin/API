"""Concrete trainers for the two learners under measurement.

Each adapter satisfies
:class:`~covenant_ml.benchmarking.protocols.TrainerProto` and is responsible
for one thing: translating the shared :class:`BenchmarkConfig` into its
library's own hyperparameters, then doing the whole fit inside ``fit`` so the
runner's timing brackets exactly the work being compared. Prediction and model
introspection happen on the returned handle, after timing has stopped.

LightGBM is reached through :func:`__import__` with the module's members
assigned directly to Protocol-typed names, which is how this package keeps a
third-party surface fully typed without stubs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from cleargbm.ensemble import (
    PyGbmModelProto,
    export_model_json,
    predict_proba,
    train_gradient_boosting,
)
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from .model_shape import mean_leaves_from_cleargbm_json, mean_leaves_from_lightgbm_dump
from .protocols import DataSplit, TrainedModelProto
from .types import BenchmarkConfig, BenchmarkModelName


class LGBMBoosterProto(Protocol):
    """Protocol for the LightGBM ``Booster`` members this package reads."""

    def dump_model(self) -> dict[str, JSONValue]:
        """Dump the fitted ensemble structure.

        Returns:
            Mapping containing ``tree_info``.
        """
        ...

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score rows, returning the positive-class probability directly.

        Args:
            data: Feature matrix, shape (n_samples, n_features).

        Returns:
            Positive-class probabilities, shape (n_samples,), for a model
            trained with the binary objective.
        """
        ...


class LGBMClassifierProto(Protocol):
    """Protocol for the LightGBM classifier members this package uses."""

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int64]) -> None:
        """Fit the classifier.

        Args:
            x: Feature matrix.
            y: Binary labels.
        """
        ...

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Feature matrix.

        Returns:
            Array of shape (n_samples, 2).
        """
        ...

    @property
    def booster_(self) -> LGBMBoosterProto:
        """Return the underlying fitted booster.

        Returns:
            The booster handle.
        """
        ...


class _LGBMClassifierCtor(Protocol):
    """Protocol for the LightGBM classifier constructor."""

    def __call__(
        self,
        *,
        objective: str,
        n_estimators: int,
        num_leaves: int,
        max_depth: int,
        learning_rate: float,
        max_bin: int,
        min_child_samples: int,
        reg_alpha: float,
        reg_lambda: float,
        n_jobs: int,
        random_state: int,
        verbose: int,
    ) -> LGBMClassifierProto: ...


def _load_lightgbm_ctor() -> _LGBMClassifierCtor:
    """Resolve LightGBM's classifier constructor as a Protocol-typed callable.

    Returns:
        The ``LGBMClassifier`` constructor.
    """
    module = __import__("lightgbm", fromlist=["LGBMClassifier"])
    constructor: _LGBMClassifierCtor = module.LGBMClassifier
    return constructor


class ClearGbmTrainedModel:
    """A fitted ClearGBM ensemble, exposed through the benchmark's Protocol."""

    def __init__(self, model: PyGbmModelProto) -> None:
        """Capture the fitted model handle.

        Args:
            model: Trained ``PyGbmModel`` handle.
        """
        self._model = model

    def predict_positive_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict positive-class probabilities.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Positive-class probabilities, shape (n_samples,).
        """
        pairs = predict_proba(self._model, x)
        positives = [pair[1] for pair in pairs]
        return np.asarray(positives, dtype=np.float64)

    def mean_leaves(self) -> float:
        """Return the mean leaves per tree of the fitted ensemble.

        Returns:
            Mean leaves per tree.
        """
        return mean_leaves_from_cleargbm_json(export_model_json(self._model))


class LightGbmTrainedModel:
    """A fitted LightGBM ensemble, exposed through the benchmark's Protocol."""

    def __init__(self, classifier: LGBMClassifierProto) -> None:
        """Capture the fitted classifier.

        Args:
            classifier: The fitted ``LGBMClassifier``.
        """
        self._classifier = classifier

    def predict_positive_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict positive-class probabilities.

        Scores through the underlying ``Booster`` rather than the scikit-learn
        wrapper. Fitting from a plain numpy array makes LightGBM synthesise
        ``feature_names_in_`` as ``Column_0, Column_1, ...``; scikit-learn's
        estimator validation then warns on every numpy prediction because the
        array carries no matching names. The names are LightGBM's own
        invention, so the warning is spurious. The booster performs no such
        validation and returns the positive-class probability directly, which
        also avoids stacking both classes only to discard one.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Positive-class probabilities, shape (n_samples,).
        """
        positive: NDArray[np.float64] = np.asarray(
            self._classifier.booster_.predict(x), dtype=np.float64
        )
        return positive

    def mean_leaves(self) -> float:
        """Return the mean leaves per tree of the fitted ensemble.

        Returns:
            Mean leaves per tree.
        """
        return mean_leaves_from_lightgbm_dump(self._classifier.booster_.dump_model())


class ClearGbmTrainer:
    """Trains ClearGBM under the shared benchmark configuration."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Bind the shared configuration.

        Args:
            config: Hyperparameters held identical across both learners.
        """
        self._config = config

    @property
    def model_name(self) -> BenchmarkModelName:
        """Name recorded for this trainer's results.

        Returns:
            The literal ``"cleargbm"``.
        """
        return "cleargbm"

    def fit(self, split: DataSplit, seed: int) -> TrainedModelProto:
        """Fit ClearGBM on the split's training partition.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        config: GradientBoostingConfig = {
            "n_estimators": self._config["n_estimators"],
            "max_depth": self._config["max_depth"],
            "learning_rate": self._config["learning_rate"],
            "min_samples_split": 2,
            "min_samples_leaf": self._config["min_data_in_leaf"],
            "max_features": None,
            "max_bins": self._config["max_bins"],
            "subsample": 1.0,
            "random_state": seed,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": self._config["reg_alpha"],
            "reg_lambda": self._config["reg_lambda"],
            "n_jobs": self._config["n_jobs"],
            "early_stopping_rounds": None,
        }
        n_features = int(split.x_train.shape[1])
        feature_names = tuple(f"f{index}" for index in range(n_features))
        model = train_gradient_boosting(
            x_train=split.x_train,
            y_train=split.y_train,
            x_val=split.x_val,
            y_val=split.y_val,
            config=config,
            feature_names=feature_names,
        )
        return ClearGbmTrainedModel(model)


class LightGbmTrainer:
    """Trains LightGBM under the shared benchmark configuration."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Bind the shared configuration.

        Args:
            config: Hyperparameters held identical across both learners.
        """
        self._config = config
        self._constructor = _load_lightgbm_ctor()

    @property
    def model_name(self) -> BenchmarkModelName:
        """Name recorded for this trainer's results.

        Returns:
            The literal ``"lightgbm"``.
        """
        return "lightgbm"

    def fit(self, split: DataSplit, seed: int) -> TrainedModelProto:
        """Fit LightGBM on the split's training partition.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        classifier = self._constructor(
            objective="binary",
            n_estimators=self._config["n_estimators"],
            num_leaves=self._config["num_leaves"],
            max_depth=self._config["max_depth"],
            learning_rate=self._config["learning_rate"],
            max_bin=self._config["max_bins"],
            min_child_samples=self._config["min_data_in_leaf"],
            reg_alpha=self._config["reg_alpha"],
            reg_lambda=self._config["reg_lambda"],
            n_jobs=self._config["n_jobs"],
            random_state=seed,
            verbose=-1,
        )
        classifier.fit(split.x_train, split.y_train)
        return LightGbmTrainedModel(classifier)


__all__ = [
    "ClearGbmTrainedModel",
    "ClearGbmTrainer",
    "LGBMBoosterProto",
    "LGBMClassifierProto",
    "LightGbmTrainedModel",
    "LightGbmTrainer",
]
