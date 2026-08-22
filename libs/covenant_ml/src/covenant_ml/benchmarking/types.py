"""Type definitions for the ClearGBM-versus-LightGBM benchmark.

Strict typing only. No Any, casts, stubs, or type-checking-only imports.

Every TypedDict in this module has a matching ``encode_*`` function that
lowers it to :data:`JSONValue`, and a matching ``decode_*`` function that
raises it back out of untrusted JSON through ``_require_*`` validators.
Decoding never softens a malformed document: the first invalid field raises
:class:`ValueError` carrying a traceable error code and a human message.
"""

from __future__ import annotations

from typing import Literal, TypedDict, get_args

from platform_core.json_utils import JSONValue

# Traceable error codes. Every decode failure carries exactly one of these so
# a malformed manifest can be triaged from a log line alone.
ERR_NOT_MAPPING = "CVML-BENCH-001"
ERR_NOT_STR = "CVML-BENCH-002"
ERR_NOT_FLOAT = "CVML-BENCH-003"
ERR_NOT_INT = "CVML-BENCH-004"
ERR_NOT_BOOL = "CVML-BENCH-005"
ERR_NOT_LIST = "CVML-BENCH-006"
ERR_UNKNOWN_MODEL = "CVML-BENCH-007"
ERR_UNKNOWN_ESTIMATOR = "CVML-BENCH-008"
ERR_SCHEMA_VERSION = "CVML-BENCH-009"
ERR_NO_TIMING_SAMPLES = "CVML-BENCH-010"
ERR_MISSING_COLUMN = "CVML-BENCH-011"
ERR_EMPTY_SPLIT = "CVML-BENCH-012"
ERR_LENGTH_MISMATCH = "CVML-BENCH-013"
ERR_NO_TREES = "CVML-BENCH-014"
ERR_INVALID_REPEATS = "CVML-BENCH-015"
ERR_NO_SEEDS = "CVML-BENCH-016"
ERR_NO_RESULTS = "CVML-BENCH-017"
ERR_TOO_FEW_TRAINERS = "CVML-BENCH-018"
ERR_DUPLICATE_TRAINER = "CVML-BENCH-019"
ERR_POWER_THROTTLING = "CVML-BENCH-020"

#: Schema version of :class:`BenchmarkManifest`. Bump on any field change so
#: an old manifest is rejected loudly instead of decoded into wrong types.
#: Bumped to 2 when the harness stopped being a two-model comparison:
#: ``ran_first: bool`` cannot describe an ordering over three or more arms, so
#: it became ``position: int``.
MANIFEST_SCHEMA_VERSION = 2

#: The arms this benchmark can compare.
#:
#: A closed literal, not an open string: an arm name is what a manifest is read
#: by, so a typo must fail at the boundary rather than silently produce a
#: fourth, nameless series. Variant arms are spelled ``<model>@<variant>``.
BenchmarkModelName = Literal["cleargbm", "cleargbm@leaf_wise", "lightgbm", "xgboost"]

#: Every accepted :data:`BenchmarkModelName`, for validation and iteration.
BENCHMARK_MODEL_NAMES: tuple[BenchmarkModelName, ...] = get_args(BenchmarkModelName)

#: The statistic taken as each seed's canonical fit time.
#:
#: ``median`` is the only permitted value. The minimum is deliberately not
#: offered: the first fits after an idle period run with full turbo headroom,
#: a different power regime rather than noise, so a minimum reports a
#: cold-start outlier as though it were the steady state that sustained
#: training actually experiences.
TimingEstimator = Literal["median"]


class TimingSummary(TypedDict, total=True):
    """Fit-time statistics over the timed repeats of one model at one seed.

    Args:
        canonical_s: The value callers should compare, in seconds. Always the
            median of ``samples_s``.
        min_s: Fastest timed repeat, in seconds.
        median_s: Median timed repeat, in seconds.
        mean_s: Arithmetic mean of timed repeats, in seconds.
        max_s: Slowest timed repeat, in seconds.
        samples_s: Every timed repeat, in seconds, in execution order.
    """

    canonical_s: float
    min_s: float
    median_s: float
    mean_s: float
    max_s: float
    samples_s: list[float]


class QualityMetrics(TypedDict, total=True):
    """Predictive-quality metrics on the held-out split.

    Recorded alongside timing so a change that trades accuracy for speed is
    visible in the same record rather than discovered later.

    Args:
        auc_roc: Area under the ROC curve.
        auc_pr: Area under the precision-recall curve (average precision).
        log_loss: Binary cross-entropy against the true labels.
        brier: Brier score (mean squared error of the probabilities).
        mean_pred: Mean predicted positive-class probability.
        positive_rate: Observed positive rate of the evaluation split.
    """

    auc_roc: float
    auc_pr: float
    log_loss: float
    brier: float
    mean_pred: float
    positive_rate: float


class BenchmarkConfig(TypedDict, total=True):
    """Hyperparameters held identical across both models.

    Args:
        n_estimators: Boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage applied to each tree's contribution.
        max_bins: Histogram bin count.
        min_data_in_leaf: Minimum samples required in a leaf.
        num_leaves: Leaf cap. Binds LightGBM's leaf-wise growth only;
            ClearGBM grows depth-wise and is bounded by ``max_depth``.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        n_jobs: Worker threads. One, so measurements are single-threaded.
        repeats: Timed fits per model per seed.
        warmups: Discarded fits before timing, which pull the data into cache
            and burn off the turbo window.
    """

    n_estimators: int
    max_depth: int
    learning_rate: float
    max_bins: int
    min_data_in_leaf: int
    num_leaves: int
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    repeats: int
    warmups: int


class DatasetInfo(TypedDict, total=True):
    """Identity of the benchmark input, so manifests are provably same-input.

    Args:
        sha256: SHA-256 of the source CSV.
        n_rows: Row count of the loaded frame.
        n_features: Feature-column count after dropping identifier columns.
    """

    sha256: str
    n_rows: int
    n_features: int


class SeedResult(TypedDict, total=True):
    """One model's outcome at one seed.

    Args:
        model: Which arm produced this record.
        seed: Split and model seed.
        position: Zero-based slot this arm occupied at this seed. The order
            rotates across seeds so no arm systematically occupies the
            cold-CPU slot. This replaced a ``ran_first`` boolean in schema 2:
            with three or more arms, "was it first" no longer describes where
            an arm ran, and averaging over an unrecorded position hides a
            systematic warm-up advantage rather than cancelling it.
        timing: Fit-time statistics.
        quality: Predictive-quality metrics.
        mean_leaves: Mean leaves per tree. The work-per-tree normalizer that
            makes a depth-wise model comparable to a leaf-wise one.
    """

    model: BenchmarkModelName
    seed: int
    position: int
    timing: TimingSummary
    quality: QualityMetrics
    mean_leaves: float


class BenchmarkManifest(TypedDict, total=True):
    """Complete machine-readable record of one benchmark invocation.

    Args:
        schema_version: Value of :data:`MANIFEST_SCHEMA_VERSION` at write time.
        estimator: Statistic used for each seed's canonical fit time.
        config: Hyperparameters shared by both models.
        dataset: Identity of the input data.
        seeds: Seeds measured, in execution order.
        results: Every per-model per-seed record.
    """

    schema_version: int
    estimator: TimingEstimator
    config: BenchmarkConfig
    dataset: DatasetInfo
    seeds: list[int]
    results: list[SeedResult]


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
    "BENCHMARK_MODEL_NAMES",
    "ERR_DUPLICATE_TRAINER",
    "ERR_EMPTY_SPLIT",
    "ERR_INVALID_REPEATS",
    "ERR_LENGTH_MISMATCH",
    "ERR_MISSING_COLUMN",
    "ERR_NOT_BOOL",
    "ERR_NOT_FLOAT",
    "ERR_NOT_INT",
    "ERR_NOT_LIST",
    "ERR_NOT_MAPPING",
    "ERR_NOT_STR",
    "ERR_NO_RESULTS",
    "ERR_NO_SEEDS",
    "ERR_NO_TIMING_SAMPLES",
    "ERR_NO_TREES",
    "ERR_POWER_THROTTLING",
    "ERR_SCHEMA_VERSION",
    "ERR_TOO_FEW_TRAINERS",
    "ERR_UNKNOWN_ESTIMATOR",
    "ERR_UNKNOWN_MODEL",
    "MANIFEST_SCHEMA_VERSION",
    "BenchmarkConfig",
    "BenchmarkManifest",
    "BenchmarkModelName",
    "DatasetInfo",
    "QualityMetrics",
    "SeedResult",
    "TimingEstimator",
    "TimingSummary",
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
