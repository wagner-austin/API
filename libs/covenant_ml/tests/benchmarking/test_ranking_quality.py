"""Tests for the ranking quality benchmark.

Runs the real measurement path with both real learners on a small corpus,
so the module is exercised end to end rather than through stand-ins.
"""

from __future__ import annotations

import numpy as np
from platform_core.comparability import NO_VALUE
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import determinism_record
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    narrow_json_to_str,
)
from platform_core.testing import sample_run_fingerprint

from covenant_ml.benchmarking.ranking_quality import (
    RankingBenchConfig,
    encode_ranking_manifest,
    make_synthetic_ranking,
    run_ranking_benchmark,
)


def _small_config() -> RankingBenchConfig:
    """Return a corpus/model config small enough for a fast test run."""
    return RankingBenchConfig(
        n_queries=40,
        docs_per_query=8,
        n_features=4,
        n_estimators=20,
        max_depth=3,
        learning_rate=0.2,
        max_bins=16,
        min_samples_leaf=5,
        truncation_level=8,
    )


#: A stated configuration, so every manifest these tests build carries the
#: axes a published one must. Built through the canonical builder rather than
#: written out, so it cannot fall behind the type.
_FINGERPRINT = sample_run_fingerprint(
    image_digest="sha256:" + "ef" * 32,
    gpu_model=NO_VALUE,
    driver_version=NO_VALUE,
    determinism=determinism_record("cpu", {"OMP_NUM_THREADS": SINGLE_THREAD}),
)


class TestSyntheticRankingCorpus:
    """The corpus is deterministic, query-shaped, and graded 0-3."""

    def test_the_manifest_says_what_it_ran_on(self) -> None:
        # Until 2026-08-27 this entry point pinned nothing and recorded
        # nothing about its environment, so two runs of it on two machines
        # were indistinguishable in the file.
        manifest = run_ranking_benchmark(_small_config(), [42], _FINGERPRINT)

        assert manifest["fingerprint"] == _FINGERPRINT

    def test_the_encoded_manifest_carries_the_configuration(self) -> None:
        manifest = run_ranking_benchmark(_small_config(), [42], _FINGERPRINT)

        encoded = narrow_json_to_dict(encode_ranking_manifest(manifest))
        fingerprint = narrow_json_to_dict(encoded["fingerprint"])
        host = narrow_json_to_dict(fingerprint["host"])

        assert host["logical_cores"] == _FINGERPRINT["host"]["logical_cores"]

    def test_same_seed_reproduces_the_corpus_exactly(self) -> None:
        """Two generations under one seed are byte-identical."""
        config = _small_config()
        x1, y1, g1 = make_synthetic_ranking(config, 42)
        x2, y2, g2 = make_synthetic_ranking(config, 42)
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
        assert np.array_equal(g1, g2)

    def test_different_seeds_differ(self) -> None:
        """Distinct seeds produce distinct feature matrices."""
        config = _small_config()
        x1, _, _ = make_synthetic_ranking(config, 42)
        x2, _, _ = make_synthetic_ranking(config, 43)
        assert not np.array_equal(x1, x2)

    def test_every_query_holds_all_four_grades(self) -> None:
        """Grades are quartiles of the within-query utility ordering."""
        config = _small_config()
        _, y, group = make_synthetic_ranking(config, 42)
        assert y.shape == (320,)
        assert group.shape == (40,)
        docs = config["docs_per_query"]
        for query in range(config["n_queries"]):
            counts = [0, 0, 0, 0]
            for doc in range(docs):
                grade = int(y.flat[query * docs + doc].item())
                counts[grade] += 1
            assert counts == [2, 2, 2, 2], f"query {query}: {counts}"


class TestRunRankingBenchmark:
    """Both arms measure, learn the corpus, and encode to a manifest."""

    def test_both_arms_report_and_beat_a_random_ordering(self) -> None:
        """One record per arm per seed, each well above the random floor.

        A random permutation of an 8-document, four-grade query scores a
        mean NDCG@8 around 0.75; a learner that found the signal clears
        0.85 comfortably even on the small test corpus.
        """
        manifest = run_ranking_benchmark(_small_config(), [42], _FINGERPRINT)
        pairs = [(r["model"], r["seed"]) for r in manifest["results"]]
        assert pairs == [("cleargbm", 42), ("lightgbm", 42)]
        for result in manifest["results"]:
            ndcg = result["quality"]["mean_ndcg"]
            assert 0.85 < ndcg <= 1.0, f"{result['model']}: {ndcg}"

    def test_manifest_encodes_to_json(self) -> None:
        """The encoded manifest round-trips through the JSON codec."""
        manifest = run_ranking_benchmark(_small_config(), [42], _FINGERPRINT)
        encoded = encode_ranking_manifest(manifest)
        decoded = narrow_json_to_dict(load_json_str(dump_json_str(encoded)))
        config = narrow_json_to_dict(decoded["config"])
        assert config["truncation_level"] == 8
        results = narrow_json_to_list(decoded["results"])
        assert len(results) == 2
        models = [narrow_json_to_str(narrow_json_to_dict(r)["model"]) for r in results]
        assert models == ["cleargbm", "lightgbm"]
