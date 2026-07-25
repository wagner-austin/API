"""Tests for benchmarking record shapes and their codecs."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue

from covenant_ml.benchmarking.types import (
    ERR_INVALID_REPEATS,
    ERR_NOT_BOOL,
    ERR_NOT_FLOAT,
    ERR_NOT_INT,
    ERR_NOT_LIST,
    ERR_NOT_MAPPING,
    ERR_NOT_STR,
    ERR_SCHEMA_VERSION,
    ERR_UNKNOWN_ESTIMATOR,
    ERR_UNKNOWN_MODEL,
    MANIFEST_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkManifest,
    DatasetInfo,
    QualityMetrics,
    SeedResult,
    TimingSummary,
    _require_bool,
    _require_estimator,
    _require_float,
    _require_float_list,
    _require_int,
    _require_int_list,
    _require_list,
    _require_mapping,
    _require_model_name,
    _require_str,
    decode_benchmark_config,
    decode_benchmark_manifest,
    decode_dataset_info,
    decode_quality_metrics,
    decode_seed_result,
    decode_timing_summary,
    encode_benchmark_config,
    encode_benchmark_manifest,
    encode_dataset_info,
    encode_quality_metrics,
    encode_seed_result,
    encode_timing_summary,
)


def make_timing() -> TimingSummary:
    """Build a timing summary fixture.

    Returns:
        A populated timing summary.
    """
    return {
        "canonical_s": 1.5,
        "min_s": 1.0,
        "median_s": 1.5,
        "mean_s": 1.6,
        "max_s": 2.0,
        "samples_s": [1.0, 1.5, 2.0],
    }


def make_quality() -> QualityMetrics:
    """Build a quality metrics fixture.

    Returns:
        A populated quality record.
    """
    return {
        "auc_roc": 0.68,
        "auc_pr": 0.14,
        "log_loss": 0.23,
        "brier": 0.06,
        "mean_pred": 0.065,
        "positive_rate": 0.066,
    }


def make_config() -> BenchmarkConfig:
    """Build a benchmark config fixture.

    Returns:
        A populated configuration.
    """
    return {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "max_bins": 64,
        "min_data_in_leaf": 20,
        "num_leaves": 31,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": 1,
        "repeats": 5,
        "warmups": 2,
    }


def make_dataset_info() -> DatasetInfo:
    """Build a dataset identity fixture.

    Returns:
        A populated dataset identity.
    """
    return {"sha256": "a" * 64, "n_rows": 100, "n_features": 18}


def make_seed_result() -> SeedResult:
    """Build a per-seed record fixture.

    Returns:
        A populated seed result.
    """
    return {
        "model": "cleargbm",
        "seed": 42,
        "ran_first": True,
        "timing": make_timing(),
        "quality": make_quality(),
        "mean_leaves": 57.9,
    }


def make_manifest() -> BenchmarkManifest:
    """Build a full manifest fixture.

    Returns:
        A populated manifest.
    """
    second: SeedResult = {
        "model": "lightgbm",
        "seed": 42,
        "ran_first": False,
        "timing": make_timing(),
        "quality": make_quality(),
        "mean_leaves": 31.0,
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "estimator": "median",
        "config": make_config(),
        "dataset": make_dataset_info(),
        "seeds": [42],
        "results": [make_seed_result(), second],
    }


def test_require_mapping_accepts_object() -> None:
    value: JSONValue = {"a": 1}
    assert _require_mapping(value, "f") == {"a": 1}


def test_require_mapping_rejects_non_object() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_MAPPING):
        _require_mapping("nope", "f")


def test_require_str_accepts_string() -> None:
    assert _require_str("x", "f") == "x"


def test_require_str_rejects_non_string() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_STR):
        _require_str(1, "f")


def test_require_int_accepts_integer() -> None:
    assert _require_int(7, "f") == 7


def test_require_int_rejects_bool() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_INT):
        _require_int(True, "f")


def test_require_int_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_INT):
        _require_int("7", "f")


def test_require_float_accepts_int_and_float() -> None:
    assert _require_float(2, "f") == 2.0
    assert _require_float(2.5, "f") == 2.5


def test_require_float_rejects_bool() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_FLOAT):
        _require_float(False, "f")


def test_require_float_rejects_non_number() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_FLOAT):
        _require_float("2.5", "f")


def test_require_bool_accepts_bool() -> None:
    assert _require_bool(True, "f") is True


def test_require_bool_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_BOOL):
        _require_bool(1, "f")


def test_require_list_accepts_array() -> None:
    assert _require_list([1, 2], "f") == [1, 2]


def test_require_list_rejects_non_array() -> None:
    with pytest.raises(ValueError, match=ERR_NOT_LIST):
        _require_list({"a": 1}, "f")


def test_require_float_list_accepts_numbers() -> None:
    assert _require_float_list([1, 2.5], "f") == [1.0, 2.5]


def test_require_float_list_reports_element_index() -> None:
    with pytest.raises(ValueError, match=r"f\[1\]"):
        _require_float_list([1.0, "x"], "f")


def test_require_int_list_accepts_integers() -> None:
    assert _require_int_list([1, 2], "f") == [1, 2]


def test_require_int_list_reports_element_index() -> None:
    with pytest.raises(ValueError, match=r"f\[0\]"):
        _require_int_list(["x"], "f")


def test_require_model_name_accepts_both_models() -> None:
    assert _require_model_name("cleargbm", "f") == "cleargbm"
    assert _require_model_name("lightgbm", "f") == "lightgbm"


def test_require_model_name_rejects_unknown() -> None:
    with pytest.raises(ValueError, match=ERR_UNKNOWN_MODEL):
        _require_model_name("xgboost", "f")


def test_require_estimator_accepts_median() -> None:
    assert _require_estimator("median", "f") == "median"


def test_require_estimator_rejects_minimum() -> None:
    with pytest.raises(ValueError, match=ERR_UNKNOWN_ESTIMATOR):
        _require_estimator("min", "f")


def test_timing_summary_round_trips() -> None:
    original = make_timing()
    assert decode_timing_summary(encode_timing_summary(original), "t") == original


def test_quality_metrics_round_trips() -> None:
    original = make_quality()
    assert decode_quality_metrics(encode_quality_metrics(original), "q") == original


def test_benchmark_config_round_trips() -> None:
    original = make_config()
    assert decode_benchmark_config(encode_benchmark_config(original), "c") == original


def test_dataset_info_round_trips() -> None:
    original = make_dataset_info()
    assert decode_dataset_info(encode_dataset_info(original), "d") == original


def test_seed_result_round_trips() -> None:
    original = make_seed_result()
    assert decode_seed_result(encode_seed_result(original), "r") == original


def test_manifest_round_trips() -> None:
    original = make_manifest()
    assert decode_benchmark_manifest(encode_benchmark_manifest(original)) == original


def test_manifest_rejects_wrong_schema_version() -> None:
    document = encode_benchmark_manifest(make_manifest())
    document["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match=ERR_SCHEMA_VERSION):
        decode_benchmark_manifest(document)


def test_manifest_reports_result_index_on_bad_element() -> None:
    document = encode_benchmark_manifest(make_manifest())
    document["results"] = ["not-an-object"]
    with pytest.raises(ValueError, match=r"results\[0\]"):
        decode_benchmark_manifest(document)


def test_error_codes_are_distinct() -> None:
    codes = [
        ERR_NOT_MAPPING,
        ERR_NOT_STR,
        ERR_NOT_FLOAT,
        ERR_NOT_INT,
        ERR_NOT_BOOL,
        ERR_NOT_LIST,
        ERR_UNKNOWN_MODEL,
        ERR_UNKNOWN_ESTIMATOR,
        ERR_SCHEMA_VERSION,
        ERR_INVALID_REPEATS,
    ]
    assert len(set(codes)) == len(codes)
