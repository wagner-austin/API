"""Multiclass quality benchmark: ClearGBM softmax against LightGBM.

Measures the ``multiclass_softmax`` objective on a deterministic synthetic
corpus — Gaussian clusters with deliberate overlap, so log loss is an
informative axis rather than a race to zero. Both arms train under matched
hyperparameters (trees per class, depth, learning rate, bins, leaf minimum,
no subsampling, single-threaded) and score the same held-out quarter.

The corpus is synthetic because the library carries no multiclass service
dataset yet; determinism (a seeded generator, no wall-clock anywhere) keeps
every rerun comparable byte-for-byte on the corpus itself.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from cleargbm.ensemble_multiclass import (
    predict_class,
    predict_proba_multiclass,
    train_gradient_boosting_multiclass,
)
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from ..metrics import compute_accuracy, compute_multiclass_log_loss


class MulticlassBenchConfig(TypedDict):
    """Shared hyperparameters for both arms of the multiclass benchmark.

    Args:
        n_samples: Corpus rows per seed.
        n_features: Corpus feature count.
        n_classes: Class count (>= 2).
        n_estimators: Boosting rounds; ClearGBM trains ``n_classes`` trees
            per round and LightGBM does the same internally.
        max_depth: Maximum tree depth for both arms.
        learning_rate: Shrinkage for both arms.
        max_bins: Histogram bin count for both arms.
        min_samples_leaf: Minimum rows per leaf for both arms.
    """

    n_samples: int
    n_features: int
    n_classes: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    max_bins: int
    min_samples_leaf: int


class MulticlassQuality(TypedDict):
    """Held-out quality for one arm at one seed.

    Args:
        log_loss: Multiclass cross-entropy on the held-out quarter.
        accuracy: Argmax accuracy on the held-out quarter.
    """

    log_loss: float
    accuracy: float


class MulticlassArmResult(TypedDict):
    """One arm's measurement at one seed.

    Args:
        model: Arm name (``"cleargbm"`` or ``"lightgbm"``).
        seed: Corpus seed.
        quality: Held-out quality record.
    """

    model: str
    seed: int
    quality: MulticlassQuality


class MulticlassManifest(TypedDict):
    """Complete multiclass benchmark manifest.

    Args:
        config: The shared hyperparameters.
        seeds: Every corpus seed measured.
        results: One record per arm per seed.
    """

    config: MulticlassBenchConfig
    seeds: list[int]
    results: list[MulticlassArmResult]


class _LGBMMulticlassCtor(Protocol):
    """Protocol for LightGBM's classifier constructor, multiclass shape."""

    def __call__(
        self,
        *,
        objective: str,
        num_class: int,
        n_estimators: int,
        max_depth: int,
        num_leaves: int,
        learning_rate: float,
        max_bin: int,
        min_child_samples: int,
        reg_alpha: float,
        reg_lambda: float,
        n_jobs: int,
        random_state: int,
        verbose: int,
    ) -> _LGBMMulticlassProto: ...


class _LGBMMulticlassProto(Protocol):
    """Protocol for the LightGBM classifier members this module uses."""

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int64]) -> None:
        """Fit the classifier.

        Args:
            x: Feature matrix.
            y: Class labels.
        """
        ...

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Feature matrix.

        Returns:
            Array of shape (n_samples, n_classes).
        """
        ...


def _load_lightgbm_multiclass_ctor() -> _LGBMMulticlassCtor:
    """Resolve LightGBM's classifier constructor as a Protocol-typed callable.

    Returns:
        The ``LGBMClassifier`` constructor.
    """
    module = __import__("lightgbm", fromlist=["LGBMClassifier"])
    constructor: _LGBMMulticlassCtor = module.LGBMClassifier
    return constructor


def make_synthetic_multiclass(
    config: MulticlassBenchConfig,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate a deterministic overlapping-cluster multiclass corpus.

    Each class is a noisy cluster: uniform noise of half-width 3.0 around a
    class center shifted 2.0 along a rotating feature axis, so neighbouring
    classes overlap and no model reaches zero loss.

    Args:
        config: Corpus shape (rows, features, classes).
        seed: Generator seed; rows are class-interleaved, so every class has
            either ``floor`` or ``ceil`` of its equal share.

    Returns:
        Tuple of ``(features, labels)``.
    """
    rng = np.random.default_rng(seed)
    n = config["n_samples"]
    d = config["n_features"]
    k = config["n_classes"]
    noise: NDArray[np.float64] = rng.random((n, d), dtype=np.float64)
    x_rows: list[list[float]] = []
    y_list: list[int] = []
    for i in range(n):
        label = i % k
        row: list[float] = []
        for j in range(d):
            value = (float(noise.flat[i * d + j].item()) - 0.5) * 6.0
            if j == label % d:
                value += 2.0 * float(label // d + 1)
            if j == (label + 1) % d:
                value -= 1.0
            row.append(value)
        x_rows.append(row)
        y_list.append(label)
    x: NDArray[np.float64] = np.asarray(x_rows, dtype=np.float64)
    y: NDArray[np.int64] = np.asarray(y_list, dtype=np.int64)
    return x, y


def _split(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Shuffle deterministically and hold out the final quarter.

    Args:
        x: Feature matrix.
        y: Labels.
        seed: Shuffle seed (offset from the corpus seed by the caller).

    Returns:
        ``(x_train, y_train, x_test, y_test)``.
    """
    rng = np.random.default_rng(seed + 1_000_003)
    order: NDArray[np.intp] = rng.permutation(len(y))
    x_shuffled: NDArray[np.float64] = x[order]
    y_shuffled: NDArray[np.int64] = y[order]
    n_test = len(y) // 4
    n_train = len(y) - n_test
    return (
        x_shuffled[:n_train],
        y_shuffled[:n_train],
        x_shuffled[n_train:],
        y_shuffled[n_train:],
    )


def _cleargbm_config(config: MulticlassBenchConfig, seed: int) -> GradientBoostingConfig:
    """Build the ClearGBM training config for one arm run.

    Args:
        config: Shared hyperparameters.
        seed: Random seed for the run.

    Returns:
        The full ClearGBM config with the multiclass pairing.
    """
    return GradientBoostingConfig(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        min_samples_split=2 * config["min_samples_leaf"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=None,
        colsample_bytree=None,
        categorical_features=None,
        n_classes=config["n_classes"],
        lambdarank_truncation_level=None,
        goss_top_rate=None,
        goss_other_rate=None,
        max_bins=config["max_bins"],
        subsample=1.0,
        random_state=seed,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=1,
        early_stopping_rounds=None,
        growth_strategy="depth_wise",
        num_leaves=None,
        objective="multiclass_softmax",
        scale_pos_weight=None,
    )


def _measure_cleargbm(
    config: MulticlassBenchConfig,
    seed: int,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
) -> MulticlassArmResult:
    """Train and score the ClearGBM arm.

    Args:
        config: Shared hyperparameters.
        seed: Run seed.
        x_train: Training features.
        y_train: Training labels.
        x_test: Held-out features.
        y_test: Held-out labels.

    Returns:
        The arm's result record.
    """
    names = tuple(f"f{i}" for i in range(config["n_features"]))
    model = train_gradient_boosting_multiclass(
        x_train, y_train, None, None, _cleargbm_config(config, seed), names
    )
    proba = predict_proba_multiclass(model, x_test)
    predicted = predict_class(model, x_test)
    return MulticlassArmResult(
        model="cleargbm",
        seed=seed,
        quality=MulticlassQuality(
            log_loss=compute_multiclass_log_loss(y_test, proba),
            accuracy=compute_accuracy(y_test, predicted),
        ),
    )


def _measure_lightgbm(
    config: MulticlassBenchConfig,
    seed: int,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
) -> MulticlassArmResult:
    """Train and score the LightGBM arm.

    Args:
        config: Shared hyperparameters.
        seed: Run seed.
        x_train: Training features.
        y_train: Training labels.
        x_test: Held-out features.
        y_test: Held-out labels.

    Returns:
        The arm's result record.
    """
    classifier = _load_lightgbm_multiclass_ctor()(
        objective="multiclass",
        num_class=config["n_classes"],
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        num_leaves=1 << config["max_depth"],
        learning_rate=config["learning_rate"],
        max_bin=config["max_bins"],
        min_child_samples=config["min_samples_leaf"],
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=1,
        random_state=seed,
        verbose=-1,
    )
    classifier.fit(x_train, y_train)
    proba: NDArray[np.float64] = np.asarray(classifier.predict_proba(x_test), dtype=np.float64)
    argmaxes: NDArray[np.intp] = np.argmax(proba, axis=1)
    predicted: NDArray[np.int64] = argmaxes.astype(np.int64)
    return MulticlassArmResult(
        model="lightgbm",
        seed=seed,
        quality=MulticlassQuality(
            log_loss=compute_multiclass_log_loss(y_test, proba),
            accuracy=compute_accuracy(y_test, predicted),
        ),
    )


def run_multiclass_benchmark(
    config: MulticlassBenchConfig,
    seeds: list[int],
) -> MulticlassManifest:
    """Run both arms across every seed.

    Args:
        config: Shared hyperparameters.
        seeds: Corpus seeds to measure.

    Returns:
        The complete manifest.
    """
    results: list[MulticlassArmResult] = []
    for seed in seeds:
        x, y = make_synthetic_multiclass(config, seed)
        x_train, y_train, x_test, y_test = _split(x, y, seed)
        results.append(_measure_cleargbm(config, seed, x_train, y_train, x_test, y_test))
        results.append(_measure_lightgbm(config, seed, x_train, y_train, x_test, y_test))
    return MulticlassManifest(config=config, seeds=list(seeds), results=results)


def encode_multiclass_manifest(manifest: MulticlassManifest) -> JSONValue:
    """Encode the manifest to a JSON-serializable value.

    Args:
        manifest: The manifest to encode.

    Returns:
        JSON-shaped dictionary.
    """
    cfg = manifest["config"]
    return {
        "config": {
            "n_samples": cfg["n_samples"],
            "n_features": cfg["n_features"],
            "n_classes": cfg["n_classes"],
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "max_bins": cfg["max_bins"],
            "min_samples_leaf": cfg["min_samples_leaf"],
        },
        "seeds": list(manifest["seeds"]),
        "results": [
            {
                "model": r["model"],
                "seed": r["seed"],
                "quality": {
                    "log_loss": r["quality"]["log_loss"],
                    "accuracy": r["quality"]["accuracy"],
                },
            }
            for r in manifest["results"]
        ],
    }


__all__ = [
    "MulticlassArmResult",
    "MulticlassBenchConfig",
    "MulticlassManifest",
    "MulticlassQuality",
    "encode_multiclass_manifest",
    "make_synthetic_multiclass",
    "run_multiclass_benchmark",
]
