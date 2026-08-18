"""Drift tests: the declared vendor Protocols against the installed libraries.

A Protocol is a claim about somebody else's code. Left unchecked it is a claim
that was true when it was written, and the failure is silent -- the package
keeps type-checking against a signature the vendor has since changed, and the
first symptom is a wrong number in a table rather than a red build.

So every declared name is *called*, using the Protocol's own keyword parameter
names, and driven to a result only the real function could produce.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.growth_policy.vendors import (
    load_average_precision,
    load_lgb_ctor,
    load_log_loss,
    load_roc_auc,
    load_stratified_split,
    load_xgb_ctor,
)

from .factories import make_separable_dataset
from .numeric import floats, ints, mean_of, positive_rate


class TestXgbConstructor:
    """XGBoost still accepts every keyword the Protocol declares."""

    def test_accepts_the_declared_keywords_and_fits(self) -> None:
        """Constructing by keyword and fitting should produce a usable booster."""
        constructor = load_xgb_ctor()
        features, labels = make_separable_dataset(row_count=40, feature_count=3)

        estimator = constructor(
            n_estimators=2,
            learning_rate=0.3,
            max_bin=16,
            min_child_weight=1,
            tree_method="hist",
            grow_policy="depthwise",
            max_depth=2,
            max_leaves=0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=1,
            random_state=42,
            eval_metric="logloss",
        )
        fitted = estimator.fit(features, labels)

        assert fitted.predict_proba(features).shape == (40, 2)
        assert len(fitted.get_booster().get_dump()) == 2

    def test_lossguide_honours_the_leaf_budget(self) -> None:
        """The leaf-wise policy must actually bound leaves, or the arms are identical."""
        constructor = load_xgb_ctor()
        features, labels = make_separable_dataset(row_count=200, feature_count=4)

        estimator = constructor(
            n_estimators=1,
            learning_rate=0.3,
            max_bin=32,
            min_child_weight=1,
            tree_method="hist",
            grow_policy="lossguide",
            max_depth=0,
            max_leaves=4,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=1,
            random_state=42,
            eval_metric="logloss",
        )
        dump = estimator.fit(features, labels).get_booster().get_dump()

        assert dump[0].count("leaf=") <= 4


class TestLgbConstructor:
    """LightGBM still accepts every keyword the Protocol declares."""

    def test_accepts_the_declared_keywords_and_fits(self) -> None:
        """Constructing by keyword and fitting should expose a booster."""
        constructor = load_lgb_ctor()
        features, labels = make_separable_dataset(row_count=60, feature_count=3)

        estimator = constructor(
            n_estimators=2,
            max_depth=2,
            learning_rate=0.3,
            max_bin=16,
            min_child_samples=2,
            num_leaves=4,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=1,
            random_state=42,
            verbose=-1,
        )
        estimator.fit(features, labels)

        predictions: NDArray[np.float64] = np.asarray(
            estimator.booster_.predict(features), dtype=np.float64
        )
        assert predictions.shape == (60,)


class TestMetricCallables:
    """The three scikit-learn metrics still have the declared signatures."""

    def test_roc_auc_ranks_a_perfect_separation_at_one(self) -> None:
        """A perfectly ordered score should give an AUC of exactly one."""
        metric = load_roc_auc()
        labels = ints([0, 0, 1, 1])
        scores = floats([0.1, 0.2, 0.8, 0.9])

        assert metric(labels, scores) == 1.0

    def test_average_precision_ranks_a_perfect_separation_at_one(self) -> None:
        """A perfectly ordered score should give an average precision of one."""
        metric = load_average_precision()
        labels = ints([0, 0, 1, 1])
        scores = floats([0.1, 0.2, 0.8, 0.9])

        assert metric(labels, scores) == 1.0

    def test_log_loss_accepts_the_labels_keyword(self) -> None:
        """A single-class fold must still score, which is why labels is passed."""
        metric = load_log_loss()
        labels = ints([1, 1])
        scores = floats([0.9, 0.9])

        assert metric(labels, scores, labels=[0, 1]) > 0.0


class TestStratifiedSplit:
    """scikit-learn's splitter still returns four folds in the declared order."""

    def test_returns_features_then_labels(self) -> None:
        """The four folds should be x_train, x_test, y_train, y_test."""
        splitter = load_stratified_split()
        features, labels = make_separable_dataset(row_count=100, feature_count=3)

        folds = splitter(features, labels, test_size=0.30, random_state=42, stratify=labels)

        assert len(folds) == 4
        assert folds[0].shape == (70, 3)
        assert folds[1].shape == (30, 3)
        assert folds[2].shape == (70,)
        assert folds[3].shape == (30,)

    def test_preserves_class_proportions(self) -> None:
        """Stratification should keep both folds near the source positive rate."""
        splitter = load_stratified_split()
        features, labels = make_separable_dataset(row_count=100, feature_count=3)

        folds = splitter(features, labels, test_size=0.30, random_state=42, stratify=labels)

        source_rate = positive_rate(labels)
        test_fold: NDArray[np.float64] = np.asarray(folds[3], dtype=np.float64)
        test_rate = mean_of(test_fold)
        assert abs(test_rate - source_rate) < 0.05
