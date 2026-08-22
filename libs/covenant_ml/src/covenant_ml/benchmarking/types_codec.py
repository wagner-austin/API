"""Encoders and strict decoders for the benchmark manifest document."""

from __future__ import annotations

from platform_core.json_utils import JSONValue

from covenant_ml.benchmarking.types import (
    BENCHMARK_MODEL_NAMES,
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
    BenchmarkModelName,
    DatasetInfo,
    QualityMetrics,
    SeedResult,
    TimingEstimator,
    TimingSummary,
)


def _require_mapping(value: JSONValue, field: str) -> dict[str, JSONValue]:
    """Validate that a decoded JSON value is an object.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The value as a string-keyed mapping.

    Raises:
        ValueError: If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise ValueError(
            f"[{ERR_NOT_MAPPING}] Field '{field}' must be a JSON object, got {type(value).__name__}"
        )
    return value


def _require_str(value: JSONValue, field: str) -> str:
    """Validate that a decoded JSON value is a string.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated string.

    Raises:
        ValueError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"[{ERR_NOT_STR}] Field '{field}' must be a string, got {type(value).__name__}"
        )
    return value


def _require_int(value: JSONValue, field: str) -> int:
    """Validate that a decoded JSON value is an integer.

    Booleans are rejected even though ``bool`` subclasses ``int``, because a
    boolean in an integer field indicates a malformed document.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated integer.

    Raises:
        ValueError: If the value is not an integer, or is a boolean.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"[{ERR_NOT_INT}] Field '{field}' must be an integer, got {type(value).__name__}"
        )
    return value


def _require_float(value: JSONValue, field: str) -> float:
    """Validate that a decoded JSON value is a number.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated value widened to ``float``.

    Raises:
        ValueError: If the value is not a number, or is a boolean.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"[{ERR_NOT_FLOAT}] Field '{field}' must be a number, got {type(value).__name__}"
        )
    return float(value)


def _require_bool(value: JSONValue, field: str) -> bool:
    """Validate that a decoded JSON value is a boolean.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated boolean.

    Raises:
        ValueError: If the value is not a boolean.
    """
    if not isinstance(value, bool):
        raise ValueError(
            f"[{ERR_NOT_BOOL}] Field '{field}' must be a boolean, got {type(value).__name__}"
        )
    return value


def _require_list(value: JSONValue, field: str) -> list[JSONValue]:
    """Validate that a decoded JSON value is an array.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated list.

    Raises:
        ValueError: If the value is not a JSON array.
    """
    if not isinstance(value, list):
        raise ValueError(
            f"[{ERR_NOT_LIST}] Field '{field}' must be an array, got {type(value).__name__}"
        )
    return value


def _require_float_list(value: JSONValue, field: str) -> list[float]:
    """Validate that a decoded JSON value is an array of numbers.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated numbers widened to ``float``.

    Raises:
        ValueError: If the value is not an array, or any element is not a
            number.
    """
    raw = _require_list(value, field)
    return [_require_float(item, f"{field}[{index}]") for index, item in enumerate(raw)]


def _require_int_list(value: JSONValue, field: str) -> list[int]:
    """Validate that a decoded JSON value is an array of integers.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated integers.

    Raises:
        ValueError: If the value is not an array, or any element is not an
            integer.
    """
    raw = _require_list(value, field)
    return [_require_int(item, f"{field}[{index}]") for index, item in enumerate(raw)]


def _require_model_name(value: JSONValue, field: str) -> BenchmarkModelName:
    """Validate that a decoded JSON value names one of the compared models.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated model name.

    Raises:
        ValueError: If the value is not one of :data:`BENCHMARK_MODEL_NAMES`.
    """
    name = _require_str(value, field)
    # Driven off the literal's own members rather than a hand-written chain,
    # so adding an arm cannot leave this validator a version behind.
    for candidate in BENCHMARK_MODEL_NAMES:
        if name == candidate:
            return candidate
    accepted = ", ".join(f"'{candidate}'" for candidate in BENCHMARK_MODEL_NAMES)
    raise ValueError(
        f"[{ERR_UNKNOWN_MODEL}] Field '{field}' must be one of {accepted}, got '{name}'"
    )


def _require_estimator(value: JSONValue, field: str) -> TimingEstimator:
    """Validate that a decoded JSON value names a supported timing estimator.

    Args:
        value: Value pulled from the parsed document.
        field: Dotted field path, used in the error message.

    Returns:
        The validated estimator name.

    Raises:
        ValueError: If the value is not ``"median"``.
    """
    name = _require_str(value, field)
    if name == "median":
        return "median"
    raise ValueError(f"[{ERR_UNKNOWN_ESTIMATOR}] Field '{field}' must be 'median', got '{name}'")


def encode_timing_summary(summary: TimingSummary) -> dict[str, JSONValue]:
    """Lower a timing summary to JSON.

    Args:
        summary: Summary to encode.

    Returns:
        JSON object mirroring :class:`TimingSummary`.
    """
    samples: list[JSONValue] = list(summary["samples_s"])
    return {
        "canonical_s": summary["canonical_s"],
        "min_s": summary["min_s"],
        "median_s": summary["median_s"],
        "mean_s": summary["mean_s"],
        "max_s": summary["max_s"],
        "samples_s": samples,
    }


def decode_timing_summary(data: dict[str, JSONValue], field: str) -> TimingSummary:
    """Raise a timing summary out of untrusted JSON.

    Args:
        data: Parsed JSON object.
        field: Dotted field path of ``data``, used in error messages.

    Returns:
        Validated :class:`TimingSummary`.

    Raises:
        ValueError: If any field is missing or of the wrong type.
    """
    return {
        "canonical_s": _require_float(data.get("canonical_s"), f"{field}.canonical_s"),
        "min_s": _require_float(data.get("min_s"), f"{field}.min_s"),
        "median_s": _require_float(data.get("median_s"), f"{field}.median_s"),
        "mean_s": _require_float(data.get("mean_s"), f"{field}.mean_s"),
        "max_s": _require_float(data.get("max_s"), f"{field}.max_s"),
        "samples_s": _require_float_list(data.get("samples_s"), f"{field}.samples_s"),
    }


def encode_quality_metrics(metrics: QualityMetrics) -> dict[str, JSONValue]:
    """Lower quality metrics to JSON.

    Args:
        metrics: Metrics to encode.

    Returns:
        JSON object mirroring :class:`QualityMetrics`.
    """
    return {
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "mean_pred": metrics["mean_pred"],
        "positive_rate": metrics["positive_rate"],
    }


def decode_quality_metrics(data: dict[str, JSONValue], field: str) -> QualityMetrics:
    """Raise quality metrics out of untrusted JSON.

    Args:
        data: Parsed JSON object.
        field: Dotted field path of ``data``, used in error messages.

    Returns:
        Validated :class:`QualityMetrics`.

    Raises:
        ValueError: If any field is missing or of the wrong type.
    """
    return {
        "auc_roc": _require_float(data.get("auc_roc"), f"{field}.auc_roc"),
        "auc_pr": _require_float(data.get("auc_pr"), f"{field}.auc_pr"),
        "log_loss": _require_float(data.get("log_loss"), f"{field}.log_loss"),
        "brier": _require_float(data.get("brier"), f"{field}.brier"),
        "mean_pred": _require_float(data.get("mean_pred"), f"{field}.mean_pred"),
        "positive_rate": _require_float(data.get("positive_rate"), f"{field}.positive_rate"),
    }


def encode_benchmark_config(config: BenchmarkConfig) -> dict[str, JSONValue]:
    """Lower a benchmark configuration to JSON.

    Args:
        config: Configuration to encode.

    Returns:
        JSON object mirroring :class:`BenchmarkConfig`.
    """
    return {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "max_bins": config["max_bins"],
        "min_data_in_leaf": config["min_data_in_leaf"],
        "num_leaves": config["num_leaves"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "n_jobs": config["n_jobs"],
        "repeats": config["repeats"],
        "warmups": config["warmups"],
    }


def decode_benchmark_config(data: dict[str, JSONValue], field: str) -> BenchmarkConfig:
    """Raise a benchmark configuration out of untrusted JSON.

    Args:
        data: Parsed JSON object.
        field: Dotted field path of ``data``, used in error messages.

    Returns:
        Validated :class:`BenchmarkConfig`.

    Raises:
        ValueError: If any field is missing or of the wrong type.
    """
    return {
        "n_estimators": _require_int(data.get("n_estimators"), f"{field}.n_estimators"),
        "max_depth": _require_int(data.get("max_depth"), f"{field}.max_depth"),
        "learning_rate": _require_float(data.get("learning_rate"), f"{field}.learning_rate"),
        "max_bins": _require_int(data.get("max_bins"), f"{field}.max_bins"),
        "min_data_in_leaf": _require_int(data.get("min_data_in_leaf"), f"{field}.min_data_in_leaf"),
        "num_leaves": _require_int(data.get("num_leaves"), f"{field}.num_leaves"),
        "reg_alpha": _require_float(data.get("reg_alpha"), f"{field}.reg_alpha"),
        "reg_lambda": _require_float(data.get("reg_lambda"), f"{field}.reg_lambda"),
        "n_jobs": _require_int(data.get("n_jobs"), f"{field}.n_jobs"),
        "repeats": _require_int(data.get("repeats"), f"{field}.repeats"),
        "warmups": _require_int(data.get("warmups"), f"{field}.warmups"),
    }


def encode_dataset_info(info: DatasetInfo) -> dict[str, JSONValue]:
    """Lower dataset identity to JSON.

    Args:
        info: Dataset identity to encode.

    Returns:
        JSON object mirroring :class:`DatasetInfo`.
    """
    return {
        "sha256": info["sha256"],
        "n_rows": info["n_rows"],
        "n_features": info["n_features"],
    }


def decode_dataset_info(data: dict[str, JSONValue], field: str) -> DatasetInfo:
    """Raise dataset identity out of untrusted JSON.

    Args:
        data: Parsed JSON object.
        field: Dotted field path of ``data``, used in error messages.

    Returns:
        Validated :class:`DatasetInfo`.

    Raises:
        ValueError: If any field is missing or of the wrong type.
    """
    return {
        "sha256": _require_str(data.get("sha256"), f"{field}.sha256"),
        "n_rows": _require_int(data.get("n_rows"), f"{field}.n_rows"),
        "n_features": _require_int(data.get("n_features"), f"{field}.n_features"),
    }


def encode_seed_result(result: SeedResult) -> dict[str, JSONValue]:
    """Lower one per-model per-seed record to JSON.

    Args:
        result: Record to encode.

    Returns:
        JSON object mirroring :class:`SeedResult`.
    """
    return {
        "model": result["model"],
        "seed": result["seed"],
        "position": result["position"],
        "timing": encode_timing_summary(result["timing"]),
        "quality": encode_quality_metrics(result["quality"]),
        "mean_leaves": result["mean_leaves"],
    }


def decode_seed_result(data: dict[str, JSONValue], field: str) -> SeedResult:
    """Raise one per-model per-seed record out of untrusted JSON.

    Args:
        data: Parsed JSON object.
        field: Dotted field path of ``data``, used in error messages.

    Returns:
        Validated :class:`SeedResult`.

    Raises:
        ValueError: If any field is missing or of the wrong type.
    """
    timing_raw = _require_mapping(data.get("timing"), f"{field}.timing")
    quality_raw = _require_mapping(data.get("quality"), f"{field}.quality")
    return {
        "model": _require_model_name(data.get("model"), f"{field}.model"),
        "seed": _require_int(data.get("seed"), f"{field}.seed"),
        "position": _require_int(data.get("position"), f"{field}.position"),
        "timing": decode_timing_summary(timing_raw, f"{field}.timing"),
        "quality": decode_quality_metrics(quality_raw, f"{field}.quality"),
        "mean_leaves": _require_float(data.get("mean_leaves"), f"{field}.mean_leaves"),
    }


def encode_benchmark_manifest(manifest: BenchmarkManifest) -> dict[str, JSONValue]:
    """Lower a complete manifest to JSON.

    Args:
        manifest: Manifest to encode.

    Returns:
        JSON object mirroring :class:`BenchmarkManifest`.
    """
    seeds: list[JSONValue] = list(manifest["seeds"])
    results: list[JSONValue] = [encode_seed_result(result) for result in manifest["results"]]
    return {
        "schema_version": manifest["schema_version"],
        "estimator": manifest["estimator"],
        "config": encode_benchmark_config(manifest["config"]),
        "dataset": encode_dataset_info(manifest["dataset"]),
        "seeds": seeds,
        "results": results,
    }


def decode_benchmark_manifest(data: dict[str, JSONValue]) -> BenchmarkManifest:
    """Raise a complete manifest out of untrusted JSON.

    Args:
        data: Parsed JSON object.

    Returns:
        Validated :class:`BenchmarkManifest`.

    Raises:
        ValueError: If the schema version does not match
            :data:`MANIFEST_SCHEMA_VERSION`, or any field is missing or of the
            wrong type.
    """
    schema_version = _require_int(data.get("schema_version"), "schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"[{ERR_SCHEMA_VERSION}] Field 'schema_version' must be "
            f"{MANIFEST_SCHEMA_VERSION}, got {schema_version}"
        )

    config_raw = _require_mapping(data.get("config"), "config")
    dataset_raw = _require_mapping(data.get("dataset"), "dataset")
    results_raw = _require_list(data.get("results"), "results")
    results: list[SeedResult] = []
    for index, item in enumerate(results_raw):
        item_mapping = _require_mapping(item, f"results[{index}]")
        results.append(decode_seed_result(item_mapping, f"results[{index}]"))

    return {
        "schema_version": schema_version,
        "estimator": _require_estimator(data.get("estimator"), "estimator"),
        "config": decode_benchmark_config(config_raw, "config"),
        "dataset": decode_dataset_info(dataset_raw, "dataset"),
        "seeds": _require_int_list(data.get("seeds"), "seeds"),
        "results": results,
    }


__all__ = [
    "decode_benchmark_config",
    "decode_benchmark_manifest",
    "decode_dataset_info",
    "decode_quality_metrics",
    "decode_seed_result",
    "decode_timing_summary",
    "encode_benchmark_config",
    "encode_benchmark_manifest",
    "encode_dataset_info",
    "encode_quality_metrics",
    "encode_seed_result",
    "encode_timing_summary",
]
