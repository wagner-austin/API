"""Tests for the multiclass quality benchmark.

Runs the real measurement path with both real learners on a small corpus,
so the module is exercised end to end rather than through stand-ins.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
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

from covenant_ml.benchmarking.multiclass_quality import (
    MulticlassBenchConfig,
    encode_multiclass_manifest,
    make_synthetic_multiclass,
    run_multiclass_benchmark,
)


def _small_config() -> MulticlassBenchConfig:
    """Return a corpus/model config small enough for a fast test run."""
    return MulticlassBenchConfig(
        n_samples=400,
        n_features=4,
        n_classes=3,
        n_estimators=20,
        max_depth=3,
        learning_rate=0.2,
        max_bins=16,
        min_samples_leaf=5,
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


class TestSyntheticCorpus:
    """The corpus is deterministic, class-balanced, and overlapping."""

    def test_the_manifest_says_what_it_ran_on(self) -> None:
        # Until 2026-08-27 this entry point pinned nothing and recorded
        # nothing about its environment, so two runs of it on two machines
        # were indistinguishable in the file.
        manifest = run_multiclass_benchmark(_small_config(), [42], _FINGERPRINT)

        assert manifest["fingerprint"] == _FINGERPRINT

    def test_the_encoded_manifest_carries_the_configuration(self) -> None:
        manifest = run_multiclass_benchmark(_small_config(), [42], _FINGERPRINT)

        encoded = narrow_json_to_dict(encode_multiclass_manifest(manifest))
        fingerprint = narrow_json_to_dict(encoded["fingerprint"])
        host = narrow_json_to_dict(fingerprint["host"])

        assert host["logical_cores"] == _FINGERPRINT["host"]["logical_cores"]

    def test_same_seed_reproduces_the_corpus_exactly(self) -> None:
        """Two generations under one seed are byte-identical."""
        config = _small_config()
        x1, y1 = make_synthetic_multiclass(config, 42)
        x2, y2 = make_synthetic_multiclass(config, 42)
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)

    def test_different_seeds_differ(self) -> None:
        """Distinct seeds produce distinct feature matrices."""
        config = _small_config()
        x1, _ = make_synthetic_multiclass(config, 42)
        x2, _ = make_synthetic_multiclass(config, 43)
        assert not np.array_equal(x1, x2)

    def test_classes_are_interleaved_and_balanced(self) -> None:
        """Every class holds an equal share of a divisible row count."""
        config = _small_config()
        _, y = make_synthetic_multiclass(config, 42)
        assert y.shape == (400,)
        for label in range(3):
            matches: NDArray[np.bool_] = y == label
            count = int(np.sum(matches))
            assert count in (133, 134)


class TestRunMulticlassBenchmark:
    """Both arms measure, learn the corpus, and encode to a manifest."""

    def test_both_arms_report_for_every_seed(self) -> None:
        """One record per arm per seed, in seed order."""
        manifest = run_multiclass_benchmark(_small_config(), [42, 43], _FINGERPRINT)
        pairs = [(r["model"], r["seed"]) for r in manifest["results"]]
        assert pairs == [
            ("cleargbm", 42),
            ("lightgbm", 42),
            ("cleargbm", 43),
            ("lightgbm", 43),
        ]

    def test_both_arms_beat_the_uniform_baseline(self) -> None:
        """Each arm's held-out loss is under log(K) and accuracy over 1/K.

        The corpus overlaps by construction, so this asserts genuine
        learning without demanding a corpus-specific score.
        """
        import math

        manifest = run_multiclass_benchmark(_small_config(), [42], _FINGERPRINT)
        for result in manifest["results"]:
            quality = result["quality"]
            assert quality["log_loss"] < math.log(3.0), result["model"]
            assert quality["accuracy"] > 1.0 / 3.0, result["model"]

    def test_manifest_encodes_to_json(self) -> None:
        """The encoded manifest round-trips through the JSON codec."""
        manifest = run_multiclass_benchmark(_small_config(), [42], _FINGERPRINT)
        encoded = encode_multiclass_manifest(manifest)
        decoded = narrow_json_to_dict(load_json_str(dump_json_str(encoded)))
        config = narrow_json_to_dict(decoded["config"])
        assert config["n_classes"] == 3
        results = narrow_json_to_list(decoded["results"])
        assert len(results) == 2
        models = [narrow_json_to_str(narrow_json_to_dict(r)["model"]) for r in results]
        assert models == ["cleargbm", "lightgbm"]
