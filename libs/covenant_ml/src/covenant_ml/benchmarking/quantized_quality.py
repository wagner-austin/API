"""Quantized-training benchmark: packed integer versus float histograms.

Measures what gradient quantization (Shi 2022, as both libraries ship it)
buys and costs on a deterministic synthetic binary corpus: four arms per
seed — ClearGBM and LightGBM, each trained with float histograms and with
quantized gradients at the SAME bin count — scored on the same held-out
quarter by AUC and log loss, with each arm's fit wall clock recorded.
The interesting numbers are the within-library quality gap (quantized
minus full) and the within-library speed ratio, because quantization is a
speed lever whose price is quality.

Quality values are deterministic per config; ``fit_seconds`` is a wall
clock and varies with the machine — the manifest records it as a
measurement of THIS run's environment, exactly like the timing columns of
the identity manifests.
"""

from __future__ import annotations

import time
from typing import Protocol, TypedDict

import numpy as np
from cleargbm.ensemble import predict_proba, train_gradient_boosting
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from ..metrics import compute_auc, compute_log_loss
from .goss_quality import make_synthetic_binary


class QuantizedBenchConfig(TypedDict):
    """Shared hyperparameters for every arm of the quantized benchmark.

    Args:
        n_samples: Corpus rows per seed.
        n_features: Corpus feature count.
        n_estimators: Boosting rounds for every arm.
        max_depth: Maximum tree depth for every arm.
        learning_rate: Shrinkage for every arm.
        max_bins: Histogram bin count for every arm.
        min_samples_leaf: Minimum rows per leaf for every arm.
        quant_bins: Gradient quantization bin count for the quantized
            arms (both libraries receive the same count).
    """

    n_samples: int
    n_features: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    max_bins: int
    min_samples_leaf: int
    quant_bins: int


class QuantizedQuality(TypedDict):
    """Held-out quality for one arm at one seed.

    Args:
        auc: ROC AUC on the held-out quarter.
        log_loss: Binary cross-entropy on the held-out quarter.
    """

    auc: float
    log_loss: float


class QuantizedArmResult(TypedDict):
    """One arm's measurement at one seed.

    Args:
        model: Arm name (``"cleargbm"`` or ``"lightgbm"``).
        histogram: ``"float"`` or ``"quantized"``.
        seed: Corpus seed.
        quality: Held-out quality record.
        fit_seconds: Wall-clock training time for this arm.
    """

    model: str
    histogram: str
    seed: int
    quality: QuantizedQuality
    fit_seconds: float


class QuantizedManifest(TypedDict):
    """Complete quantized benchmark manifest.

    Args:
        config: The shared hyperparameters.
        seeds: Every corpus seed measured.
        results: One record per arm per seed.
    """

    config: QuantizedBenchConfig
    seeds: list[int]
    results: list[QuantizedArmResult]


class _LGBMQuantProto(Protocol):
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


class _LGBMQuantCtor(Protocol):
    """Protocol for the LightGBM classifier constructor, quantized shape.

    ``quant_train_renew_leaf`` is passed True on the quantized arm so
    LightGBM recomputes leaf values from the original float gradients —
    matching ClearGBM, whose leaf values ALWAYS come from the floats.
    """

    def __call__(
        self,
        *,
        objective: str,
        use_quantized_grad: bool,
        num_grad_quant_bins: int,
        quant_train_renew_leaf: bool,
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
    ) -> _LGBMQuantProto: ...


def _load_lightgbm_quant_ctor() -> _LGBMQuantCtor:
    """Resolve LightGBM's classifier constructor as a Protocol-typed callable.

    Returns:
        The ``LGBMClassifier`` constructor.
    """
    module = __import__("lightgbm", fromlist=["LGBMClassifier"])
    constructor: _LGBMQuantCtor = module.LGBMClassifier
    return constructor


def _cleargbm_config(
    config: QuantizedBenchConfig,
    seed: int,
    quantized: bool,
) -> GradientBoostingConfig:
    """Build the ClearGBM training config for one arm run.

    Args:
        config: Shared hyperparameters.
        seed: Random seed for the run.
        quantized: Whether the arm trains on packed integer histograms.

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
        goss_top_rate=None,
        goss_other_rate=None,
        quantized_gradient_bins=config["quant_bins"] if quantized else None,
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


def _positive_probas(pairs: tuple[tuple[float, float], ...]) -> NDArray[np.float64]:
    """Extract the positive-class column from probability pairs.

    Args:
        pairs: Per-sample ``(p0, p1)`` tuples.

    Returns:
        The positive-class probabilities as an array.
    """
    positives: list[float] = [pair[1] for pair in pairs]
    return np.asarray(positives, dtype=np.float64)


def _quality(y_test: NDArray[np.int64], probas: NDArray[np.float64]) -> QuantizedQuality:
    """Score one arm's held-out probabilities.

    Args:
        y_test: Held-out labels.
        probas: Positive-class probabilities.

    Returns:
        The arm's quality record.
    """
    return QuantizedQuality(
        auc=compute_auc(y_test, probas),
        log_loss=compute_log_loss(y_test, probas),
    )


def run_quantized_benchmark(
    config: QuantizedBenchConfig,
    seeds: list[int],
) -> QuantizedManifest:
    """Run all four arms across every seed, timing each fit.

    Args:
        config: Shared hyperparameters.
        seeds: Corpus seeds to measure.

    Returns:
        The complete manifest.
    """
    results: list[QuantizedArmResult] = []
    names = tuple(f"f{i}" for i in range(config["n_features"]))
    for seed in seeds:
        x, y = make_synthetic_binary(config["n_samples"], config["n_features"], seed)
        x_train, y_train, x_test, y_test = _split(x, y)

        for quantized in [False, True]:
            started = time.perf_counter()
            model = train_gradient_boosting(
                x_train,
                y_train,
                None,
                None,
                _cleargbm_config(config, seed, quantized),
                names,
            )
            fit_seconds = time.perf_counter() - started
            probas = _positive_probas(predict_proba(model, x_test))
            results.append(
                QuantizedArmResult(
                    model="cleargbm",
                    histogram="quantized" if quantized else "float",
                    seed=seed,
                    quality=_quality(y_test, probas),
                    fit_seconds=fit_seconds,
                )
            )

        for quantized in [False, True]:
            classifier = _load_lightgbm_quant_ctor()(
                objective="binary",
                use_quantized_grad=quantized,
                num_grad_quant_bins=config["quant_bins"],
                quant_train_renew_leaf=quantized,
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
            started = time.perf_counter()
            classifier.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            raw: NDArray[np.float64] = np.asarray(
                classifier.predict_proba(x_test), dtype=np.float64
            )
            positives: NDArray[np.float64] = raw[:, 1]
            results.append(
                QuantizedArmResult(
                    model="lightgbm",
                    histogram="quantized" if quantized else "float",
                    seed=seed,
                    quality=_quality(y_test, positives),
                    fit_seconds=fit_seconds,
                )
            )
    return QuantizedManifest(config=config, seeds=list(seeds), results=results)


def encode_quantized_manifest(manifest: QuantizedManifest) -> JSONValue:
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
            "quant_bins": cfg["quant_bins"],
        },
        "seeds": list(manifest["seeds"]),
        "results": [
            {
                "model": r["model"],
                "histogram": r["histogram"],
                "seed": r["seed"],
                "quality": {
                    "auc": r["quality"]["auc"],
                    "log_loss": r["quality"]["log_loss"],
                },
                "fit_seconds": r["fit_seconds"],
            }
            for r in manifest["results"]
        ],
    }


__all__ = [
    "QuantizedArmResult",
    "QuantizedBenchConfig",
    "QuantizedManifest",
    "QuantizedQuality",
    "encode_quantized_manifest",
    "run_quantized_benchmark",
]
