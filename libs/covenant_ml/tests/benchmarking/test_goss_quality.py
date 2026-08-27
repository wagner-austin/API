"""Tests for the GOSS quality benchmark.

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

from covenant_ml.benchmarking.goss_quality import (
    GossBenchConfig,
    encode_goss_manifest,
    make_synthetic_binary,
    run_goss_benchmark,
)


def _small_config() -> GossBenchConfig:
    """Return a corpus/model config small enough for a fast test run."""
    return GossBenchConfig(
        n_samples=800,
        n_features=4,
        n_estimators=30,
        max_depth=3,
        learning_rate=0.2,
        max_bins=16,
        min_samples_leaf=5,
        top_rate=0.2,
        other_rate=0.1,
    )


class TestSyntheticBinaryCorpus:
    """The corpus is deterministic and carries a learnable noisy signal."""

    def test_same_seed_reproduces_the_corpus_exactly(self) -> None:
        """Two generations under one seed are byte-identical."""
        config = _small_config()
        x1, y1 = make_synthetic_binary(config["n_samples"], config["n_features"], 42)
        x2, y2 = make_synthetic_binary(config["n_samples"], config["n_features"], 42)
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)

    def test_different_seeds_differ(self) -> None:
        """Distinct seeds produce distinct feature matrices."""
        config = _small_config()
        x1, _ = make_synthetic_binary(config["n_samples"], config["n_features"], 42)
        x2, _ = make_synthetic_binary(config["n_samples"], config["n_features"], 43)
        assert not np.array_equal(x1, x2)

    def test_labels_are_binary_and_mixed(self) -> None:
        """Both classes appear, in stochastic-but-deterministic mixture."""
        config = _small_config()
        _, y = make_synthetic_binary(config["n_samples"], config["n_features"], 42)
        positives = int(np.sum(y))
        assert 0 < positives < len(y)


#: A stated configuration, so every manifest these tests build carries the
#: axes a published one must. Built through the canonical builder rather than
#: written out, so it cannot fall behind the type.
_FINGERPRINT = sample_run_fingerprint(
    image_digest="sha256:" + "ef" * 32,
    gpu_model=NO_VALUE,
    driver_version=NO_VALUE,
    determinism=determinism_record("cpu", {"OMP_NUM_THREADS": SINGLE_THREAD}),
)


class TestRunGossBenchmark:
    """All four arms measure, learn, and encode to a manifest."""

    def test_all_four_arms_report_and_learn(self) -> None:
        """One record per arm per seed, each with a discriminative AUC.

        The corpus's noisy-logistic signal caps attainable AUC well below
        1; 0.7 asserts genuine learning for every arm, sampled or full.
        """
        manifest = run_goss_benchmark(_small_config(), [42], _FINGERPRINT)
        arms = [(r["model"], r["sampling"]) for r in manifest["results"]]
        assert arms == [
            ("cleargbm", "full"),
            ("cleargbm", "goss"),
            ("lightgbm", "full"),
            ("lightgbm", "goss"),
        ]
        for result in manifest["results"]:
            auc = result["quality"]["auc"]
            assert 0.7 < auc <= 1.0, f"{result['model']}/{result['sampling']}: {auc}"

    def test_goss_changes_the_cleargbm_arm(self) -> None:
        """The sampled arm's numbers differ from the full arm's."""
        manifest = run_goss_benchmark(_small_config(), [42], _FINGERPRINT)
        by_arm = {(r["model"], r["sampling"]): r["quality"] for r in manifest["results"]}
        assert by_arm[("cleargbm", "goss")]["log_loss"] != by_arm[("cleargbm", "full")]["log_loss"]

    def test_the_manifest_says_what_it_ran_on(self) -> None:
        # Until 2026-08-27 this entry point pinned nothing and recorded
        # nothing about its environment, so two runs of it on two machines
        # were indistinguishable in the file.
        manifest = run_goss_benchmark(_small_config(), [42], _FINGERPRINT)

        assert manifest["fingerprint"] == _FINGERPRINT

    def test_the_encoded_manifest_carries_the_configuration(self) -> None:
        manifest = run_goss_benchmark(_small_config(), [42], _FINGERPRINT)

        encoded = narrow_json_to_dict(encode_goss_manifest(manifest))
        fingerprint = narrow_json_to_dict(encoded["fingerprint"])
        host = narrow_json_to_dict(fingerprint["host"])

        assert host["logical_cores"] == _FINGERPRINT["host"]["logical_cores"]

    def test_manifest_encodes_to_json(self) -> None:
        """The encoded manifest round-trips through the JSON codec."""
        manifest = run_goss_benchmark(_small_config(), [42], _FINGERPRINT)
        encoded = encode_goss_manifest(manifest)
        decoded = narrow_json_to_dict(load_json_str(dump_json_str(encoded)))
        config = narrow_json_to_dict(decoded["config"])
        assert config["top_rate"] == 0.2
        results = narrow_json_to_list(decoded["results"])
        assert len(results) == 4
        samplings = [narrow_json_to_str(narrow_json_to_dict(r)["sampling"]) for r in results]
        assert samplings == ["full", "goss", "full", "goss"]
