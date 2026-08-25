"""The measurable arms of the growth-policy experiment.

Three trainers, each satisfying
:class:`covenant_ml.growth_policy.protocols.ArmTrainerProto` structurally.
XGBoost is the instrument -- it implements both growth policies, so switching
``grow_policy`` moves the one variable under study while the code, the splits
and the constraint semantics stay fixed. LightGBM and ClearGBM are anchors, so
the instrument's arms can be read against production learners rather than only
against each other.

The fitted-model wrappers for LightGBM and ClearGBM are imported from
:mod:`covenant_ml.benchmarking.adapters`: a fitted ensemble is used here
exactly as the benchmark uses one, so only the trainers are new.
"""

from __future__ import annotations

import numpy as np
from cleargbm.ensemble import train_gradient_boosting
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray

from ..benchmarking.adapters import ClearGbmTrainedModel, LightGbmTrainedModel
from ..benchmarking.model_shape import mean_leaves_from_xgb_dump
from .protocols import ArmSpec, TrainedModelProto, TwoWaySplit
from .types import ExperimentConfig
from .vendors import LgbClassifierCtor, XgbClassifierCtor, XgbClassifierProto


class XgbTrainedModel:
    """A fitted XGBoost ensemble, exposed through the experiment's Protocol."""

    def __init__(self, classifier: XgbClassifierProto) -> None:
        """Capture the fitted classifier.

        Args:
            classifier: The fitted ``XGBClassifier``.
        """
        self._classifier = classifier

    def predict_positive_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict positive-class probabilities.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Positive-class probabilities, shape (n_samples,).
        """
        proba = np.asarray(self._classifier.predict_proba(x), dtype=np.float64)
        positive: NDArray[np.float64] = proba[:, 1]
        return positive

    def mean_leaves(self) -> float:
        """Report the mean leaves per tree of the fitted ensemble.

        Returns:
            Mean leaves per tree.
        """
        return mean_leaves_from_xgb_dump(self._classifier.get_booster().get_dump())


class XgbArmTrainer:
    """Trains one XGBoost growth-policy arm."""

    def __init__(
        self,
        spec: ArmSpec,
        config: ExperimentConfig,
        constructor: XgbClassifierCtor,
    ) -> None:
        """Bind the arm's growth policy, the shared configuration and the vendor.

        Args:
            spec: The growth policy and budgets distinguishing this arm.
            config: Hyperparameters shared with every other arm.
            constructor: XGBoost's classifier constructor, injected so this
                class names no vendor and can be driven in tests by a real
                constructor built elsewhere.
        """
        self._spec = spec
        self._config = config
        self._constructor = constructor

    @property
    def arm_name(self) -> str:
        """Name recorded for this arm's results.

        Returns:
            The arm's display name.
        """
        return self._spec.name

    def fit(self, split: TwoWaySplit, seed: int) -> TrainedModelProto:
        """Fit this arm on the split's training partition.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        estimator = self._constructor(
            n_estimators=self._config["n_estimators"],
            learning_rate=self._config["learning_rate"],
            max_bin=self._config["max_bins"],
            min_child_weight=self._config["min_leaf"],
            tree_method="hist",
            grow_policy=self._spec.grow_policy,
            max_depth=self._spec.max_depth,
            max_leaves=self._spec.max_leaves,
            reg_alpha=self._config["reg_alpha"],
            reg_lambda=self._config["reg_lambda"],
            n_jobs=self._config["n_jobs"],
            random_state=seed,
            eval_metric="logloss",
        )
        return XgbTrainedModel(estimator.fit(split.x_train, split.y_train))


class LgbAnchorTrainer:
    """Trains LightGBM as the leaf-wise anchor."""

    def __init__(
        self,
        num_leaves: int,
        max_depth: int,
        config: ExperimentConfig,
        constructor: LgbClassifierCtor,
    ) -> None:
        """Bind the anchor's shape, the shared configuration and the vendor.

        Args:
            num_leaves: Leaf cap binding LightGBM's leaf-wise growth.
            max_depth: Depth cap applied alongside the leaf cap.
            config: Hyperparameters shared with every other arm.
            constructor: LightGBM's classifier constructor, injected so this
                class names no vendor.
        """
        self._num_leaves = num_leaves
        self._max_depth = max_depth
        self._config = config
        self._constructor = constructor

    @property
    def arm_name(self) -> str:
        """Name recorded for this arm's results.

        Returns:
            The arm's display name.
        """
        return f"lgb leafwise L{self._num_leaves}"

    def fit(self, split: TwoWaySplit, seed: int) -> TrainedModelProto:
        """Fit LightGBM on the split's training partition.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        estimator = self._constructor(
            n_estimators=self._config["n_estimators"],
            max_depth=self._max_depth,
            learning_rate=self._config["learning_rate"],
            max_bin=self._config["max_bins"],
            min_child_samples=self._config["min_leaf"],
            num_leaves=self._num_leaves,
            reg_alpha=self._config["reg_alpha"],
            reg_lambda=self._config["reg_lambda"],
            n_jobs=self._config["n_jobs"],
            random_state=seed,
            verbose=-1,
        )
        estimator.fit(split.x_train, split.y_train)
        return LightGbmTrainedModel(estimator)


class ClearGbmAnchorTrainer:
    """Trains ClearGBM as the depth-wise anchor.

    Does not reuse :class:`covenant_ml.benchmarking.adapters.ClearGbmTrainer`,
    which forwards a validation fold to ``train_gradient_boosting``. This
    experiment passes none, matching the run the recorded figures came from:
    scoring a validation fold every round would change the fit time that is
    the whole reason this arm is measured.
    """

    def __init__(self, max_depth: int, config: ExperimentConfig) -> None:
        """Bind the anchor's depth and the shared configuration.

        Args:
            max_depth: Depth cap for the depth-wise ensemble.
            config: Hyperparameters shared with every other arm.
        """
        self._max_depth = max_depth
        self._config = config

    @property
    def arm_name(self) -> str:
        """Name recorded for this arm's results.

        Returns:
            The arm's display name.
        """
        return f"cleargbm depthwise d{self._max_depth}"

    def fit(self, split: TwoWaySplit, seed: int) -> TrainedModelProto:
        """Fit ClearGBM on the split's training partition.

        Args:
            split: The partition to train on.
            seed: Seed for the model's internal randomness.

        Returns:
            The fitted model.
        """
        config: GradientBoostingConfig = {
            "n_estimators": self._config["n_estimators"],
            "max_depth": self._max_depth,
            "learning_rate": self._config["learning_rate"],
            "min_samples_split": 2,
            "min_samples_leaf": self._config["min_leaf"],
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "lambdarank_truncation_level": None,
            "goss_top_rate": None,
            "goss_other_rate": None,
            "quantized_gradient_bins": None,
            "min_data_in_bin": None,
            "max_bins": self._config["max_bins"],
            "subsample": 1.0,
            "random_state": seed,
            "monotonic_constraints": None,
            "reg_alpha": self._config["reg_alpha"],
            "reg_lambda": self._config["reg_lambda"],
            "n_jobs": self._config["n_jobs"],
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        feature_count = int(split.x_train.shape[1])
        feature_names = tuple(f"X{index + 1}" for index in range(feature_count))
        model = train_gradient_boosting(
            x_train=split.x_train,
            y_train=split.y_train,
            x_val=None,
            y_val=None,
            config=config,
            feature_names=feature_names,
        )
        return ClearGbmTrainedModel(model)


__all__ = [
    "ClearGbmAnchorTrainer",
    "LgbAnchorTrainer",
    "XgbArmTrainer",
    "XgbTrainedModel",
]
