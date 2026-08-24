"""Tests for the quantized-training benchmark.

Runs the real measurement path with both real learners on a small corpus,
so the module is exercised end to end rather than through stand-ins.
"""

from __future__ import annotations

from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_list,
    narrow_json_to_str,
)

from covenant_ml.benchmarking.quantized_quality import (
    QuantizedBenchConfig,
    encode_quantized_manifest,
    run_quantized_benchmark,
)


def _small_config() -> QuantizedBenchConfig:
    """Return a corpus/model config small enough for a fast test run."""
    return QuantizedBenchConfig(
        n_samples=800,
        n_features=4,
        n_estimators=30,
        max_depth=3,
        learning_rate=0.2,
        max_bins=16,
        min_samples_leaf=5,
        quant_bins=4,
    )


class TestRunQuantizedBenchmark:
    """All four arms measure, learn, time, and encode to a manifest."""

    def test_all_four_arms_report_and_learn(self) -> None:
        """One record per arm per seed, each with a discriminative AUC.

        The corpus's noisy-logistic signal caps attainable AUC well below
        1; 0.7 asserts genuine learning for every arm, quantized or float.
        """
        manifest = run_quantized_benchmark(_small_config(), [42])
        arms = [(r["model"], r["histogram"]) for r in manifest["results"]]
        assert arms == [
            ("cleargbm", "float"),
            ("cleargbm", "quantized"),
            ("lightgbm", "float"),
            ("lightgbm", "quantized"),
        ]
        for result in manifest["results"]:
            auc = result["quality"]["auc"]
            assert 0.7 < auc <= 1.0, f"{result['model']}/{result['histogram']}: {auc}"

    def test_every_arm_records_a_positive_fit_time(self) -> None:
        """The speed measurement exists and is a positive wall clock."""
        manifest = run_quantized_benchmark(_small_config(), [42])
        for result in manifest["results"]:
            assert result["fit_seconds"] > 0.0

    def test_quantization_changes_the_cleargbm_arm(self) -> None:
        """The quantized arm's numbers differ from the float arm's."""
        manifest = run_quantized_benchmark(_small_config(), [42])
        by_arm = {(r["model"], r["histogram"]): r["quality"] for r in manifest["results"]}
        full = by_arm[("cleargbm", "float")]["log_loss"]
        quant = by_arm[("cleargbm", "quantized")]["log_loss"]
        assert quant != full

    def test_manifest_encodes_to_json(self) -> None:
        """The encoded manifest round-trips through the JSON codec."""
        manifest = run_quantized_benchmark(_small_config(), [42])
        encoded = encode_quantized_manifest(manifest)
        decoded = narrow_json_to_dict(load_json_str(dump_json_str(encoded)))
        config = narrow_json_to_dict(decoded["config"])
        assert config["quant_bins"] == 4
        results = narrow_json_to_list(decoded["results"])
        assert len(results) == 4
        histograms = [narrow_json_to_str(narrow_json_to_dict(r)["histogram"]) for r in results]
        assert histograms == ["float", "quantized", "float", "quantized"]
        for entry in results:
            fit_seconds = narrow_json_to_float(narrow_json_to_dict(entry)["fit_seconds"])
            assert fit_seconds > 0.0
