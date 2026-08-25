"""GOSS quality benchmark: sampled versus full training, both libraries.

Measures what gradient-based one-side sampling costs on a deterministic
synthetic binary corpus: four arms per seed — ClearGBM and LightGBM, each
trained full and trained under GOSS with the SAME rates — scored on the
same held-out quarter by AUC and log loss. The interesting number is the
within-library gap (GOSS minus full), and whether ClearGBM's gap matches
LightGBM's.

The corpus is synthetic because GOSS's quality effect is what is being
measured, not a service dataset's idiosyncrasies; determinism (a seeded
generator, no wall-clock anywhere) keeps every rerun comparable.
"""

from __future__ import annotations

import math
from typing import Protocol, TypedDict

import numpy as np
from cleargbm.ensemble import predict_proba, train_gradient_boosting
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from ..metrics import compute_auc, compute_log_loss


class GossBenchConfig(TypedDict):
    """Shared hyperparameters for every arm of the GOSS benchmark.

    Args:
        n_samples: Corpus rows per seed.
        n_features: Corpus feature count.
        n_estimators: Boosting rounds for every arm.
        max_depth: Maximum tree depth for every arm.
        learning_rate: Shrinkage for every arm.
        max_bins: Histogram bin count for every arm.
        min_samples_leaf: Minimum rows per leaf for every arm.
        top_rate: GOSS top rate for the sampled arms.
        other_rate: GOSS other rate for the sampled arms.
    """

    n_samples: int
    n_features: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    max_bins: int
    min_samples_leaf: int
    top_rate: float
    other_rate: float


class GossQuality(TypedDict):
    """Held-out quality for one arm at one seed.

    Args:
        auc: ROC AUC on the held-out quarter.
        log_loss: Binary cross-entropy on the held-out quarter.
    """

    auc: float
    log_loss: float


class GossArmResult(TypedDict):
    """One arm's measurement at one seed.

    Args:
        model: Arm name (``"cleargbm"`` or ``"lightgbm"``).
        sampling: ``"full"`` or ``"goss"``.
        seed: Corpus seed.
        quality: Held-out quality record.
    """

    model: str
    sampling: str
    seed: int
    quality: GossQuality


class GossManifest(TypedDict):
    """Complete GOSS benchmark manifest.

    Args:
        config: The shared hyperparameters.
        seeds: Every corpus seed measured.
        results: One record per arm per seed.
    """

    config: GossBenchConfig
    seeds: list[int]
    results: list[GossArmResult]


class _LGBMGossProto(Protocol):
    """Protocol for the LightGBM classifier members this module uses."""

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


class _LGBMGossCtor(Protocol):
    """Protocol for the LightGBM classifier constructor, GOSS shape."""

    def __call__(
        self,
        *,
        objective: str,
        data_sample_strategy: str,
        top_rate: float,
        other_rate: float,
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
    ) -> _LGBMGossProto: ...


def _load_lightgbm_goss_ctor() -> _LGBMGossCtor:
    """Resolve LightGBM's classifier constructor as a Protocol-typed callable.

    Returns:
        The ``LGBMClassifier`` constructor.
    """
    module = __import__("lightgbm", fromlist=["LGBMClassifier"])
    constructor: _LGBMGossCtor = module.LGBMClassifier
    return constructor


def make_synthetic_binary(
    n_samples: int,
    n_features: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate a deterministic noisy-logistic binary corpus.

    Each row's log-odds is a linear signal over the first three features
    plus uniform noise; the label thresholds the resulting probability at
    a uniform draw, so labels are stochastic-but-deterministic and neither
    library can reach zero loss. Shared by the GOSS and quantized-training
    benchmarks, which is why the shape arrives as plain ints rather than
    either harness's config record.

    Args:
        n_samples: Corpus rows.
        n_features: Corpus feature count.
        seed: Generator seed.

    Returns:
        Tuple of ``(features, labels)``.
    """
    rng = np.random.default_rng(seed)
    n = n_samples
    d = n_features
    noise: NDArray[np.float64] = rng.random((n, d + 1), dtype=np.float64)
    x_rows: list[list[float]] = []
    labels: list[int] = []
    for i in range(n):
        row: list[float] = []
        for j in range(d):
            row.append(float(noise.flat[i * (d + 1) + j].item()))
        x_rows.append(row)
        signal = 4.0 * (row[0] - 0.5) + 2.0 * (row[1] - 0.5) + 1.0 * (row[2] - 0.5)
        prob = 1.0 / (1.0 + math.exp(-signal))
        draw = float(noise.flat[i * (d + 1) + d].item())
        labels.append(1 if draw < prob else 0)
    x: NDArray[np.float64] = np.asarray(x_rows, dtype=np.float64)
    y: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)
    return x, y


def _split(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Hold out the final quarter of rows (the corpus is i.i.d.).

    Args:
        x: Feature matrix.
        y: Labels.

    Returns:
        ``(x_train, y_train, x_test, y_test)``.
    """
    n_test = len(y) // 4
    n_train = len(y) - n_test
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def _cleargbm_config(
    config: GossBenchConfig,
    seed: int,
    goss: bool,
) -> GradientBoostingConfig:
    """Build the ClearGBM training config for one arm run.

    Args:
        config: Shared hyperparameters.
        seed: Random seed for the run.
        goss: Whether the arm trains under GOSS.

    Returns:
        The full ClearGBM config.
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
        n_classes=None,
        lambdarank_truncation_level=None,
        goss_top_rate=config["top_rate"] if goss else None,
        goss_other_rate=config["other_rate"] if goss else None,
        quantized_gradient_bins=None,
        min_data_in_bin=None,
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
        objective="binary_log_loss",
        scale_pos_weight=1.0,
    )


def _positive_probas(pairs: tuple[tuple[float, float], ...]) -> NDArray[np.float64]:
    """Extract the positive-class column from probability pairs.

    Args:
        pairs: Per-sample ``(p0, p1)`` tuples.

    Returns:
        The positive-class probabilities as an array.
    """
    positives: list[float] = [pair[1] for pair in pairs]
    return np.asarray(positives, dtype=np.float64)


def _quality(y_test: NDArray[np.int64], probas: NDArray[np.float64]) -> GossQuality:
    """Score one arm's held-out probabilities.

    Args:
        y_test: Held-out labels.
        probas: Positive-class probabilities.

    Returns:
        The arm's quality record.
    """
    return GossQuality(
        auc=compute_auc(y_test, probas),
        log_loss=compute_log_loss(y_test, probas),
    )


def run_goss_benchmark(
    config: GossBenchConfig,
    seeds: list[int],
) -> GossManifest:
    """Run all four arms across every seed.

    Args:
        config: Shared hyperparameters.
        seeds: Corpus seeds to measure.

    Returns:
        The complete manifest.
    """
    results: list[GossArmResult] = []
    names = tuple(f"f{i}" for i in range(config["n_features"]))
    for seed in seeds:
        x, y = make_synthetic_binary(config["n_samples"], config["n_features"], seed)
        x_train, y_train, x_test, y_test = _split(x, y)

        for goss in [False, True]:
            model = train_gradient_boosting(
                x_train,
                y_train,
                None,
                None,
                _cleargbm_config(config, seed, goss),
                names,
            )
            probas = _positive_probas(predict_proba(model, x_test))
            results.append(
                GossArmResult(
                    model="cleargbm",
                    sampling="goss" if goss else "full",
                    seed=seed,
                    quality=_quality(y_test, probas),
                )
            )

        for goss in [False, True]:
            classifier = _load_lightgbm_goss_ctor()(
                objective="binary",
                data_sample_strategy="goss" if goss else "bagging",
                top_rate=config["top_rate"],
                other_rate=config["other_rate"],
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
            raw: NDArray[np.float64] = np.asarray(
                classifier.predict_proba(x_test), dtype=np.float64
            )
            positives: NDArray[np.float64] = raw[:, 1]
            results.append(
                GossArmResult(
                    model="lightgbm",
                    sampling="goss" if goss else "full",
                    seed=seed,
                    quality=_quality(y_test, positives),
                )
            )
    return GossManifest(config=config, seeds=list(seeds), results=results)


def encode_goss_manifest(manifest: GossManifest) -> JSONValue:
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
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "max_bins": cfg["max_bins"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "top_rate": cfg["top_rate"],
            "other_rate": cfg["other_rate"],
        },
        "seeds": list(manifest["seeds"]),
        "results": [
            {
                "model": r["model"],
                "sampling": r["sampling"],
                "seed": r["seed"],
                "quality": {
                    "auc": r["quality"]["auc"],
                    "log_loss": r["quality"]["log_loss"],
                },
            }
            for r in manifest["results"]
        ],
    }


__all__ = [
    "GossArmResult",
    "GossBenchConfig",
    "GossManifest",
    "GossQuality",
    "encode_goss_manifest",
    "make_synthetic_binary",
    "run_goss_benchmark",
]
