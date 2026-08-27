"""Ranking quality benchmark: ClearGBM LambdaMART against LightGBM's ranker.

Measures the ``lambdarank`` objective on a deterministic synthetic
query corpus — graded relevance derived from a noisy linear signal, so
NDCG is an informative axis rather than a race to 1.0. Both arms train
under matched hyperparameters (rounds, depth, learning rate, bins, leaf
minimum, truncation level, no subsampling, single-threaded) and score the
same held-out quarter of QUERIES (queries never straddle the split).

The corpus is synthetic because the library carries no graded-relevance
service dataset yet; determinism (a seeded generator, no wall-clock
anywhere) keeps every rerun comparable byte-for-byte on the corpus itself.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from cleargbm.ensemble import predict_raw
from cleargbm.ensemble_ranking import train_gradient_boosting_ranking
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.comparability import RunFingerprint, encode_run_fingerprint
from platform_core.json_utils import JSONValue

from ..metrics import compute_ndcg_at_k


class RankingBenchConfig(TypedDict):
    """Shared hyperparameters for both arms of the ranking benchmark.

    Args:
        n_queries: Queries per seed.
        docs_per_query: Documents in every query.
        n_features: Corpus feature count.
        n_estimators: Boosting rounds for both arms.
        max_depth: Maximum tree depth for both arms.
        learning_rate: Shrinkage for both arms.
        max_bins: Histogram bin count for both arms.
        min_samples_leaf: Minimum rows per leaf for both arms.
        truncation_level: NDCG truncation for training and evaluation.
    """

    n_queries: int
    docs_per_query: int
    n_features: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    max_bins: int
    min_samples_leaf: int
    truncation_level: int


class RankingQuality(TypedDict):
    """Held-out quality for one arm at one seed.

    Args:
        mean_ndcg: Mean NDCG at the truncation level over held-out queries.
    """

    mean_ndcg: float


class RankingArmResult(TypedDict):
    """One arm's measurement at one seed.

    Args:
        model: Arm name (``"cleargbm"`` or ``"lightgbm"``).
        seed: Corpus seed.
        quality: Held-out quality record.
    """

    model: str
    seed: int
    quality: RankingQuality


class RankingManifest(TypedDict):
    """Complete ranking benchmark manifest.

    Args:
        config: The shared hyperparameters.
        seeds: Every corpus seed measured.
        results: One record per arm per seed.
        fingerprint: The configuration these numbers were produced under,
            from :func:`~covenant_ml.benchmarking.provenance.benchmark_fingerprint`.
            These arms use no card, so the axis that decides their numbers is
            the BLAS thread count and the machine it was pinned on; a
            manifest recording only hyperparameters cannot tell a 24-core
            cluster node from an 8-core workstation.

            Taken as an argument rather than captured here. Building it reads
            installed metadata, and this module must remain importable in a
            process that has not pinned its thread count yet.
    """

    config: RankingBenchConfig
    seeds: list[int]
    results: list[RankingArmResult]
    fingerprint: RunFingerprint


class _LGBMRankerProto(Protocol):
    """Protocol for the LightGBM ranker members this module uses."""

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
        *,
        group: NDArray[np.int64],
    ) -> None:
        """Fit the ranker.

        Args:
            x: Feature matrix.
            y: Relevance grades.
            group: Documents per query, in row order.
        """
        ...

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict ranking scores.

        Args:
            x: Feature matrix.

        Returns:
            Score vector, shape (n_samples,).
        """
        ...


class _LGBMRankerCtor(Protocol):
    """Protocol for the LightGBM ranker constructor."""

    def __call__(
        self,
        *,
        objective: str,
        n_estimators: int,
        max_depth: int,
        num_leaves: int,
        learning_rate: float,
        max_bin: int,
        min_child_samples: int,
        lambdarank_truncation_level: int,
        reg_alpha: float,
        reg_lambda: float,
        n_jobs: int,
        random_state: int,
        verbose: int,
    ) -> _LGBMRankerProto: ...


def _load_lightgbm_ranker_ctor() -> _LGBMRankerCtor:
    """Resolve LightGBM's ranker constructor as a Protocol-typed callable.

    Returns:
        The ``LGBMRanker`` constructor.
    """
    module = __import__("lightgbm", fromlist=["LGBMRanker"])
    constructor: _LGBMRankerCtor = module.LGBMRanker
    return constructor


def make_synthetic_ranking(
    config: RankingBenchConfig,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Generate a deterministic graded-relevance query corpus.

    Each document's true utility is a linear signal over the first two
    features plus uniform noise; within every query the documents are
    graded 0-3 by utility quartile, so the grades are perfectly learnable
    only up to the noise floor.

    Args:
        config: Corpus shape (queries, documents, features).
        seed: Generator seed.

    Returns:
        Tuple of ``(features, grades, group_sizes)``, rows in query order.
    """
    rng = np.random.default_rng(seed)
    n_queries = config["n_queries"]
    docs = config["docs_per_query"]
    d = config["n_features"]
    noise: NDArray[np.float64] = rng.random((n_queries * docs, d + 1), dtype=np.float64)
    x_rows: list[list[float]] = []
    grades: list[int] = []
    for query in range(n_queries):
        utilities: list[float] = []
        for doc in range(docs):
            row_idx = query * docs + doc
            row: list[float] = []
            for j in range(d):
                row.append(float(noise.flat[row_idx * (d + 1) + j].item()))
            x_rows.append(row)
            noise_term = float(noise.flat[row_idx * (d + 1) + d].item())
            utilities.append(row[0] + 0.5 * row[1] + 0.4 * noise_term)

        indexed: list[tuple[float, int]] = [(utilities[i], i) for i in range(docs)]
        indexed.sort()
        order: list[int] = [i for _, i in indexed]
        query_grades = [0] * docs
        for position, doc in enumerate(order):
            query_grades[doc] = (4 * position) // docs
        grades.extend(query_grades)
    x: NDArray[np.float64] = np.asarray(x_rows, dtype=np.float64)
    y: NDArray[np.int64] = np.asarray(grades, dtype=np.int64)
    group_list: list[int] = [docs for _ in range(n_queries)]
    group: NDArray[np.int64] = np.asarray(group_list, dtype=np.int64)
    return x, y, group


def _split_queries(
    config: RankingBenchConfig,
    x: NDArray[np.float64],
    y: NDArray[np.int64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Hold out the final quarter of queries.

    Queries are generated i.i.d., so the tail is an unbiased holdout and
    no query straddles the boundary.

    Args:
        config: Corpus shape.
        x: Features, rows in query order.
        y: Grades.

    Returns:
        ``(x_train, y_train, group_train, x_test, y_test, group_test)``.
    """
    docs = config["docs_per_query"]
    n_queries = config["n_queries"]
    n_test_queries = n_queries // 4
    n_train_rows = (n_queries - n_test_queries) * docs
    train_list: list[int] = [docs for _ in range(n_queries - n_test_queries)]
    test_list: list[int] = [docs for _ in range(n_test_queries)]
    group_train: NDArray[np.int64] = np.asarray(train_list, dtype=np.int64)
    group_test: NDArray[np.int64] = np.asarray(test_list, dtype=np.int64)
    return (
        x[:n_train_rows],
        y[:n_train_rows],
        group_train,
        x[n_train_rows:],
        y[n_train_rows:],
        group_test,
    )


def _mean_ndcg(
    config: RankingBenchConfig,
    y_test: NDArray[np.int64],
    scores: NDArray[np.float64],
) -> float:
    """Average NDCG at the truncation level over the held-out queries.

    Args:
        config: Corpus shape (fixed documents per query).
        y_test: Held-out grades, rows in query order.
        scores: Predicted ranking scores, same order.

    Returns:
        The unweighted mean of the per-query NDCG values.
    """
    docs = config["docs_per_query"]
    n_queries = len(y_test) // docs
    total = 0.0
    for query in range(n_queries):
        start = query * docs
        end = start + docs
        total += compute_ndcg_at_k(y_test[start:end], scores[start:end], config["truncation_level"])
    return total / float(n_queries)


def _cleargbm_config(config: RankingBenchConfig, seed: int) -> GradientBoostingConfig:
    """Build the ClearGBM training config for one arm run.

    Args:
        config: Shared hyperparameters.
        seed: Random seed for the run.

    Returns:
        The full ClearGBM config with the lambdarank pairing.
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
        lambdarank_truncation_level=config["truncation_level"],
        goss_top_rate=None,
        goss_other_rate=None,
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
        objective="lambdarank",
        scale_pos_weight=None,
    )


def run_ranking_benchmark(
    config: RankingBenchConfig,
    seeds: list[int],
    fingerprint: RunFingerprint,
) -> RankingManifest:
    """Run both arms across every seed.

    Args:
        config: Shared hyperparameters.
        seeds: Corpus seeds to measure.
        fingerprint: The configuration this measurement runs under, from
            :func:`~covenant_ml.benchmarking.provenance.benchmark_fingerprint`.
            Required rather than optional: a manifest that could omit it
            would omit it, which is how thirty-seven of this project's
            forty-one published manifests came to carry no environment at
            all.

    Returns:
        The complete manifest.
    """
    results: list[RankingArmResult] = []
    for seed in seeds:
        x, y, _ = make_synthetic_ranking(config, seed)
        x_train, y_train, group_train, x_test, y_test, _ = _split_queries(config, x, y)

        model = train_gradient_boosting_ranking(
            x_train,
            y_train,
            group_train,
            None,
            None,
            None,
            _cleargbm_config(config, seed),
            tuple(f"f{i}" for i in range(config["n_features"])),
        )
        cleargbm_scores = predict_raw(model, x_test)
        results.append(
            RankingArmResult(
                model="cleargbm",
                seed=seed,
                quality=RankingQuality(mean_ndcg=_mean_ndcg(config, y_test, cleargbm_scores)),
            )
        )

        ranker = _load_lightgbm_ranker_ctor()(
            objective="lambdarank",
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            num_leaves=1 << config["max_depth"],
            learning_rate=config["learning_rate"],
            max_bin=config["max_bins"],
            min_child_samples=config["min_samples_leaf"],
            lambdarank_truncation_level=config["truncation_level"],
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=1,
            random_state=seed,
            verbose=-1,
        )
        ranker.fit(x_train, y_train, group=group_train)
        lgbm_scores: NDArray[np.float64] = np.asarray(ranker.predict(x_test), dtype=np.float64)
        results.append(
            RankingArmResult(
                model="lightgbm",
                seed=seed,
                quality=RankingQuality(mean_ndcg=_mean_ndcg(config, y_test, lgbm_scores)),
            )
        )
    return RankingManifest(
        config=config, seeds=list(seeds), results=results, fingerprint=fingerprint
    )


def encode_ranking_manifest(manifest: RankingManifest) -> JSONValue:
    """Encode the manifest to a JSON-serializable value.

    Args:
        manifest: The manifest to encode.

    Returns:
        JSON-shaped dictionary.
    """
    cfg = manifest["config"]
    return {
        "config": {
            "n_queries": cfg["n_queries"],
            "docs_per_query": cfg["docs_per_query"],
            "n_features": cfg["n_features"],
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "max_bins": cfg["max_bins"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "truncation_level": cfg["truncation_level"],
        },
        "seeds": list(manifest["seeds"]),
        "results": [
            {
                "model": r["model"],
                "seed": r["seed"],
                "quality": {"mean_ndcg": r["quality"]["mean_ndcg"]},
            }
            for r in manifest["results"]
        ],
        "fingerprint": encode_run_fingerprint(manifest["fingerprint"]),
    }


__all__ = [
    "RankingArmResult",
    "RankingBenchConfig",
    "RankingManifest",
    "RankingQuality",
    "encode_ranking_manifest",
    "make_synthetic_ranking",
    "run_ranking_benchmark",
]
