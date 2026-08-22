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
from cleargbm.types import GradientBoostingConfig, GrowthStrategy
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from .model_shape import (
    mean_leaves_from_cleargbm_json,
    mean_leaves_from_lightgbm_dump,
    mean_leaves_from_xgb_dump,
)
from .protocols import DataSplit, TrainedModelProto
from .types import BenchmarkConfig, BenchmarkModelName


class XgbBoosterProto(Protocol):
    """Protocol for the fitted XGBoost booster members this package reads."""

    def get_dump(self) -> list[str]:
        """Dump every tree in the ensemble to text.

        Returns:
            One text representation per boosted tree.
        """
        ...


class XgbClassifierProto(Protocol):
    """Protocol for the fitted XGBoost classifier members this package uses."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Class probabilities, shape (n_samples, 2), column 1 positive.
        """
        ...

    def get_booster(self) -> XgbBoosterProto:
        """Return the underlying booster.

        Returns:
            The fitted booster.
        """
        ...


class XgbFittableProto(Protocol):
    """Protocol for an unfitted XGBoost classifier."""

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int64]) -> XgbClassifierProto:
        """Fit the classifier.

        Args:
            x: Training features, shape (n_samples, n_features).
            y: Training labels (0 or 1), shape (n_samples,).

        Returns:
            The fitted classifier.
        """
        ...


class XgbClassifierCtor(Protocol):
    """Protocol for XGBoost's classifier constructor.

    ``max_depth`` and ``max_leaves`` are both always passed: XGBoost treats
    ``0`` as "no bound", so a depth-wise arm passes ``max_leaves=0`` and a
    leaf-wise arm passes ``max_depth=0``, and neither arm leaves a budget
    implicit.
    """

    def __call__(
        self,
        *,
        n_estimators: int,
        learning_rate: float,
        max_bin: int,
        min_child_weight: int,
        tree_method: str,
        grow_policy: str,
        max_depth: int,
        max_leaves: int,
        reg_alpha: float,
        reg_lambda: float,
        n_jobs: int,
        random_state: int,
        eval_metric: str,
    ) -> XgbFittableProto:
        """Construct an unfitted XGBoost classifier.

        Args:
            n_estimators: Boosting rounds.
            learning_rate: Shrinkage per tree.
            max_bin: Histogram bin count.
            min_child_weight: Minimum child hessian sum.
            tree_method: Tree construction algorithm.
            grow_policy: ``depthwise`` or ``lossguide``.
            max_depth: Depth bound, ``0`` for unbounded.
            max_leaves: Leaf bound, ``0`` for unbounded.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            n_jobs: Worker threads.
            random_state: Seed.
            eval_metric: Metric name reported during fitting.

        Returns:
            The unfitted classifier.
        """
        ...


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
    """Trains ClearGBM under the shared benchmark configuration.

    One class covers both growth policies rather than a subclass per arm: the
    arms differ only in two config values, and the whole point of the variant
    axis is that a variant is a configuration rather than a fork.
    """

    def __init__(self, config: BenchmarkConfig, growth_strategy: GrowthStrategy) -> None:
        """Bind the shared configuration and this arm's growth policy.

        Args:
            config: Hyperparameters held identical across every arm.
            growth_strategy: Tree growth policy for this arm. Required with no
                default, matching the config axis itself: an arm constructed
                without naming its policy would report a measurement for a
                policy nobody chose. Under ``"leaf_wise"`` the shared
                ``num_leaves`` becomes the leaf budget; under ``"depth_wise"``
                the budget is unset and ``max_depth`` bounds the tree, which
                is what every manifest written before the growth axis existed
                recorded.
        """
        self._config = config
        self._growth_strategy: GrowthStrategy = growth_strategy

    @property
    def model_name(self) -> BenchmarkModelName:
        """Name recorded for this trainer's results.

        Returns:
            ``"cleargbm"`` for the baseline arm, ``"cleargbm@leaf_wise"`` for
            the leaf-wise variant.
        """
        if self._growth_strategy == "leaf_wise":
            return "cleargbm@leaf_wise"
        return "cleargbm"

    def fit(self, split: DataSplit, seed: int) -> TrainedModelProto:
        """Fit ClearGBM on the split's training partition.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        leaf_wise = self._growth_strategy == "leaf_wise"
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
            "growth_strategy": self._growth_strategy,
            # The same shared `num_leaves` that binds LightGBM's leaf-wise
            # growth, so the two leaf-wise arms are held to one budget.
            "num_leaves": self._config["num_leaves"] if leaf_wise else None,
            # Unweighted, matching the LightGBM/XGBoost benchmark arms which
            # set no class weight — and keeping this arm bit-identical to
            # every manifest recorded before the weighting axis existed.
            "scale_pos_weight": 1.0,
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


class XgBoostTrainedModel:
    """A fitted XGBoost ensemble, exposed through the benchmark's Protocol."""

    def __init__(self, classifier: XgbClassifierProto) -> None:
        """Capture the fitted classifier.

        Args:
            classifier: The fitted XGBoost classifier.
        """
        self._classifier = classifier

    def predict_positive_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the positive-class probability for each row.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Positive-class probabilities, shape (n_samples,).
        """
        return self._classifier.predict_proba(x)[:, 1]

    def mean_leaves(self) -> float:
        """Return the mean leaves per tree of the fitted ensemble.

        Returns:
            Mean leaves per tree.
        """
        return mean_leaves_from_xgb_dump(self._classifier.get_booster().get_dump())


def _load_xgboost_ctor() -> XgbClassifierCtor:
    """Resolve XGBoost's classifier constructor as a Protocol-typed callable.

    Returns:
        The ``XGBClassifier`` constructor.
    """
    module = __import__("xgboost", fromlist=["XGBClassifier"])
    constructor: XgbClassifierCtor = module.XGBClassifier
    return constructor


class XgBoostTrainer:
    """Trains XGBoost under the shared benchmark configuration.

    Present as a third reference implementation rather than as a variant of
    either other arm: it grows depth-wise like ClearGBM's baseline while
    sharing LightGBM's histogram approach, so it separates "ClearGBM is slow"
    from "depth-wise is slow" without any further configuration.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Bind the shared configuration.

        Args:
            config: Hyperparameters held identical across every arm.
        """
        self._config = config
        self._constructor = _load_xgboost_ctor()

    @property
    def model_name(self) -> BenchmarkModelName:
        """Name recorded for this trainer's results.

        Returns:
            The literal ``"xgboost"``.
        """
        return "xgboost"

    def fit(self, split: DataSplit, seed: int) -> TrainedModelProto:
        """Fit XGBoost on the split's training partition.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        classifier = self._constructor(
            n_estimators=self._config["n_estimators"],
            learning_rate=self._config["learning_rate"],
            max_bin=self._config["max_bins"],
            min_child_weight=self._config["min_data_in_leaf"],
            tree_method="hist",
            # Depth-wise, bounded by max_depth, with the leaf budget explicitly
            # released: XGBoost reads 0 as "no bound", so this arm matches
            # ClearGBM's baseline shape rather than LightGBM's.
            grow_policy="depthwise",
            max_depth=self._config["max_depth"],
            max_leaves=0,
            reg_alpha=self._config["reg_alpha"],
            reg_lambda=self._config["reg_lambda"],
            n_jobs=self._config["n_jobs"],
            random_state=seed,
            eval_metric="logloss",
        )
        return XgBoostTrainedModel(classifier.fit(split.x_train, split.y_train))


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
    "XgBoostTrainedModel",
    "XgBoostTrainer",
    "XgbBoosterProto",
    "XgbClassifierCtor",
    "XgbClassifierProto",
    "XgbFittableProto",
]
