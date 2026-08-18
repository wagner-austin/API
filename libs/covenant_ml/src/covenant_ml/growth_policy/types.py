"""Record shapes for the growth-policy experiment, with their codecs.

Every record that can reach disk is a ``TypedDict`` with an ``encode_*`` and a
``decode_*`` function, and every decoded field passes a ``_require_*`` check
that names the field it failed on. Decoding is the only place untyped JSON
becomes typed, so a malformed report fails at the boundary with a traceable
code instead of surfacing later as a wrong number in a table.

Error codes are ``CVML-GROWTH-NNN``. They are distinct from the benchmarking
package's ``CVML-BENCH-NNN`` because these two harnesses answer different
questions and a reader tracing a code should land in exactly one of them.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONValue

ERR_NOT_MAPPING = "CVML-GROWTH-001"
ERR_NOT_STR = "CVML-GROWTH-002"
ERR_NOT_FLOAT = "CVML-GROWTH-003"
ERR_NOT_INT = "CVML-GROWTH-004"
ERR_NOT_LIST = "CVML-GROWTH-005"
ERR_SCHEMA_VERSION = "CVML-GROWTH-006"
ERR_EMPTY_DATASET = "CVML-GROWTH-007"
ERR_EMPTY_SPLIT = "CVML-GROWTH-008"
ERR_NO_SEEDS = "CVML-GROWTH-009"
ERR_NO_RESULTS = "CVML-GROWTH-010"
ERR_NO_TREES = "CVML-GROWTH-011"
ERR_UNKNOWN_GROWTH_POLICY = "CVML-GROWTH-012"
ERR_MISSING_COLUMN = "CVML-GROWTH-013"
ERR_RAGGED_ROWS = "CVML-GROWTH-014"
ERR_MISSING_VALUE = "CVML-GROWTH-015"
ERR_INVALID_REPEATS = "CVML-GROWTH-016"
ERR_NO_ARMS = "CVML-GROWTH-017"
ERR_DUPLICATE_ARM = "CVML-GROWTH-018"

#: Schema version of :class:`GrowthPolicyReport`. Bump on any field change so
#: an older report is rejected loudly rather than decoded into wrong types.
REPORT_SCHEMA_VERSION = 1


class ArmResult(TypedDict):
    """One arm measured at one seed.

    Args:
        arm: Display name of the arm, unique within a report.
        seed: Seed controlling both the split and the model's randomness.
        fit_seconds: Canonical fit time for this arm at this seed.
        auc_roc: Area under the ROC curve on the held-out fold.
        auc_pr: Area under the precision-recall curve on the held-out fold.
        log_loss: Log loss on the held-out fold.
        mean_leaves: Mean leaves per tree in the fitted ensemble.
    """

    arm: str
    seed: int
    fit_seconds: float
    auc_roc: float
    auc_pr: float
    log_loss: float
    mean_leaves: float


class ArmSummary(TypedDict):
    """One arm averaged across every seed it was measured at.

    Args:
        arm: Display name of the arm.
        seed_count: Number of seeds contributing to the means.
        fit_seconds: Mean fit time across seeds.
        auc_roc: Mean area under the ROC curve across seeds.
        auc_pr: Mean area under the precision-recall curve across seeds.
        log_loss: Mean log loss across seeds.
        mean_leaves: Mean leaves per tree across seeds.
    """

    arm: str
    seed_count: int
    fit_seconds: float
    auc_roc: float
    auc_pr: float
    log_loss: float
    mean_leaves: float


class ExperimentConfig(TypedDict):
    """Hyperparameters held identical across every arm.

    Only growth policy and its budgets vary between arms; everything here is
    shared, which is what makes a difference between arms attributable to the
    growth policy rather than to a config drift.

    Args:
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage applied to each tree's contribution.
        max_bins: Histogram bin count.
        min_leaf: Minimum-child constraint. XGBoost reads this as a hessian
            sum while LightGBM and ClearGBM read it as a sample count, which
            makes the XGBoost arms more heavily regularised at the same value.
        reg_alpha: L1 penalty.
        reg_lambda: L2 penalty.
        n_jobs: Worker threads, pinned to keep timings comparable.
        repeats: Timed fits per arm per seed.
        warmups: Discarded fits before timing begins.
    """

    n_estimators: int
    learning_rate: float
    max_bins: int
    min_leaf: int
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    repeats: int
    warmups: int


class DatasetInfo(TypedDict):
    """Shape of the dataset an experiment ran on.

    Args:
        name: Human-readable dataset name.
        row_count: Number of rows.
        feature_count: Number of feature columns.
        positive_rate: Fraction of rows whose label is 1.
    """

    name: str
    row_count: int
    feature_count: int
    positive_rate: float


class GrowthPolicyReport(TypedDict):
    """A complete growth-policy experiment result.

    Args:
        schema_version: Version of this record's shape.
        config: Hyperparameters every arm was held to.
        dataset: Shape of the dataset measured.
        seeds: Seeds the arms were measured at, in execution order.
        results: Every arm at every seed.
        summaries: Per-arm means across seeds, in arm order.
    """

    schema_version: int
    config: ExperimentConfig
    dataset: DatasetInfo
    seeds: list[int]
    results: list[ArmResult]
    summaries: list[ArmSummary]


def require_mapping(value: JSONValue, field: str) -> dict[str, JSONValue]:
    """Narrow a decoded JSON value to an object.

    Public because a caller holding a freshly parsed document needs the same
    narrowing before it can reach :func:`decode_growth_policy_report`.

    Args:
        value: Decoded JSON value.
        field: Field name, used in the error message.

    Returns:
        The value as a mapping.

    Raises:
        ValueError: If the value is not a JSON object.
    """
    return _require_mapping(value, field)


def _require_mapping(value: JSONValue, field: str) -> dict[str, JSONValue]:
    """Narrow a JSON value to an object.

    Args:
        value: Decoded JSON value.
        field: Field name, used in the error message.

    Returns:
        The value as a mapping.

    Raises:
        ValueError: If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise ValueError(f"[{ERR_NOT_MAPPING}] Field '{field}' must be an object")
    return value


def _require_str(value: JSONValue, field: str) -> str:
    """Narrow a JSON value to a string.

    Args:
        value: Decoded JSON value.
        field: Field name, used in the error message.

    Returns:
        The value as a string.

    Raises:
        ValueError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise ValueError(f"[{ERR_NOT_STR}] Field '{field}' must be a string")
    return value


def _require_int(value: JSONValue, field: str) -> int:
    """Narrow a JSON value to an integer.

    Booleans are rejected: ``bool`` is a subclass of ``int`` in Python, so an
    unchecked ``isinstance`` would silently admit ``True`` as ``1``.

    Args:
        value: Decoded JSON value.
        field: Field name, used in the error message.

    Returns:
        The value as an integer.

    Raises:
        ValueError: If the value is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"[{ERR_NOT_INT}] Field '{field}' must be an integer")
    return value


def _require_float(value: JSONValue, field: str) -> float:
    """Narrow a JSON value to a float.

    Integers are accepted and widened, because JSON writes ``1.0`` as ``1``.
    Booleans are rejected for the same reason as in :func:`_require_int`.

    Args:
        value: Decoded JSON value.
        field: Field name, used in the error message.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not a number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"[{ERR_NOT_FLOAT}] Field '{field}' must be a number")
    return float(value)


def _require_list(value: JSONValue, field: str) -> list[JSONValue]:
    """Narrow a JSON value to an array.

    Args:
        value: Decoded JSON value.
        field: Field name, used in the error message.

    Returns:
        The value as a list.

    Raises:
        ValueError: If the value is not a JSON array.
    """
    if not isinstance(value, list):
        raise ValueError(f"[{ERR_NOT_LIST}] Field '{field}' must be an array")
    return value


def encode_arm_result(result: ArmResult) -> dict[str, JSONValue]:
    """Lower one arm-seed result to JSON.

    Args:
        result: Result to encode.

    Returns:
        JSON object mirroring :class:`ArmResult`.
    """
    return {
        "arm": result["arm"],
        "seed": result["seed"],
        "fit_seconds": result["fit_seconds"],
        "auc_roc": result["auc_roc"],
        "auc_pr": result["auc_pr"],
        "log_loss": result["log_loss"],
        "mean_leaves": result["mean_leaves"],
    }


def decode_arm_result(data: dict[str, JSONValue]) -> ArmResult:
    """Raise one arm-seed result from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        The decoded result.

    Raises:
        ValueError: If any field is missing or has the wrong type.
    """
    return {
        "arm": _require_str(data["arm"], "arm"),
        "seed": _require_int(data["seed"], "seed"),
        "fit_seconds": _require_float(data["fit_seconds"], "fit_seconds"),
        "auc_roc": _require_float(data["auc_roc"], "auc_roc"),
        "auc_pr": _require_float(data["auc_pr"], "auc_pr"),
        "log_loss": _require_float(data["log_loss"], "log_loss"),
        "mean_leaves": _require_float(data["mean_leaves"], "mean_leaves"),
    }


def encode_arm_summary(summary: ArmSummary) -> dict[str, JSONValue]:
    """Lower one per-arm summary to JSON.

    Args:
        summary: Summary to encode.

    Returns:
        JSON object mirroring :class:`ArmSummary`.
    """
    return {
        "arm": summary["arm"],
        "seed_count": summary["seed_count"],
        "fit_seconds": summary["fit_seconds"],
        "auc_roc": summary["auc_roc"],
        "auc_pr": summary["auc_pr"],
        "log_loss": summary["log_loss"],
        "mean_leaves": summary["mean_leaves"],
    }


def decode_arm_summary(data: dict[str, JSONValue]) -> ArmSummary:
    """Raise one per-arm summary from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        The decoded summary.

    Raises:
        ValueError: If any field is missing or has the wrong type.
    """
    return {
        "arm": _require_str(data["arm"], "arm"),
        "seed_count": _require_int(data["seed_count"], "seed_count"),
        "fit_seconds": _require_float(data["fit_seconds"], "fit_seconds"),
        "auc_roc": _require_float(data["auc_roc"], "auc_roc"),
        "auc_pr": _require_float(data["auc_pr"], "auc_pr"),
        "log_loss": _require_float(data["log_loss"], "log_loss"),
        "mean_leaves": _require_float(data["mean_leaves"], "mean_leaves"),
    }


def encode_dataset_info(info: DatasetInfo) -> dict[str, JSONValue]:
    """Lower a dataset description to JSON.

    Args:
        info: Description to encode.

    Returns:
        JSON object mirroring :class:`DatasetInfo`.
    """
    return {
        "name": info["name"],
        "row_count": info["row_count"],
        "feature_count": info["feature_count"],
        "positive_rate": info["positive_rate"],
    }


def decode_dataset_info(data: dict[str, JSONValue]) -> DatasetInfo:
    """Raise a dataset description from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        The decoded description.

    Raises:
        ValueError: If any field is missing or has the wrong type.
    """
    return {
        "name": _require_str(data["name"], "name"),
        "row_count": _require_int(data["row_count"], "row_count"),
        "feature_count": _require_int(data["feature_count"], "feature_count"),
        "positive_rate": _require_float(data["positive_rate"], "positive_rate"),
    }


def encode_experiment_config(config: ExperimentConfig) -> dict[str, JSONValue]:
    """Lower the shared configuration to JSON.

    Args:
        config: Configuration to encode.

    Returns:
        JSON object mirroring :class:`ExperimentConfig`.
    """
    return {
        "n_estimators": config["n_estimators"],
        "learning_rate": config["learning_rate"],
        "max_bins": config["max_bins"],
        "min_leaf": config["min_leaf"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "n_jobs": config["n_jobs"],
        "repeats": config["repeats"],
        "warmups": config["warmups"],
    }


def decode_experiment_config(data: dict[str, JSONValue]) -> ExperimentConfig:
    """Raise the shared configuration from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        The decoded configuration.

    Raises:
        ValueError: If any field is missing or has the wrong type.
    """
    return {
        "n_estimators": _require_int(data["n_estimators"], "n_estimators"),
        "learning_rate": _require_float(data["learning_rate"], "learning_rate"),
        "max_bins": _require_int(data["max_bins"], "max_bins"),
        "min_leaf": _require_int(data["min_leaf"], "min_leaf"),
        "reg_alpha": _require_float(data["reg_alpha"], "reg_alpha"),
        "reg_lambda": _require_float(data["reg_lambda"], "reg_lambda"),
        "n_jobs": _require_int(data["n_jobs"], "n_jobs"),
        "repeats": _require_int(data["repeats"], "repeats"),
        "warmups": _require_int(data["warmups"], "warmups"),
    }


def encode_growth_policy_report(report: GrowthPolicyReport) -> dict[str, JSONValue]:
    """Lower a complete experiment report to JSON.

    Args:
        report: Report to encode.

    Returns:
        JSON object mirroring :class:`GrowthPolicyReport`.
    """
    seeds: list[JSONValue] = list(report["seeds"])
    results: list[JSONValue] = [encode_arm_result(result) for result in report["results"]]
    summaries: list[JSONValue] = [encode_arm_summary(summary) for summary in report["summaries"]]
    return {
        "schema_version": report["schema_version"],
        "config": encode_experiment_config(report["config"]),
        "dataset": encode_dataset_info(report["dataset"]),
        "seeds": seeds,
        "results": results,
        "summaries": summaries,
    }


def decode_growth_policy_report(data: dict[str, JSONValue]) -> GrowthPolicyReport:
    """Raise a complete experiment report from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        The decoded report.

    Raises:
        ValueError: If the schema version does not match, or if any field is
            missing or has the wrong type.
    """
    schema_version = _require_int(data["schema_version"], "schema_version")
    if schema_version != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"[{ERR_SCHEMA_VERSION}] Report schema version {schema_version} is not the "
            f"supported version {REPORT_SCHEMA_VERSION}"
        )
    seeds = [_require_int(seed, "seeds") for seed in _require_list(data["seeds"], "seeds")]
    results = [
        decode_arm_result(_require_mapping(entry, "results"))
        for entry in _require_list(data["results"], "results")
    ]
    summaries = [
        decode_arm_summary(_require_mapping(entry, "summaries"))
        for entry in _require_list(data["summaries"], "summaries")
    ]
    return {
        "schema_version": schema_version,
        "config": decode_experiment_config(_require_mapping(data["config"], "config")),
        "dataset": decode_dataset_info(_require_mapping(data["dataset"], "dataset")),
        "seeds": seeds,
        "results": results,
        "summaries": summaries,
    }


__all__ = [
    "ERR_DUPLICATE_ARM",
    "ERR_EMPTY_DATASET",
    "ERR_EMPTY_SPLIT",
    "ERR_INVALID_REPEATS",
    "ERR_MISSING_COLUMN",
    "ERR_MISSING_VALUE",
    "ERR_NOT_FLOAT",
    "ERR_NOT_INT",
    "ERR_NOT_LIST",
    "ERR_NOT_MAPPING",
    "ERR_NOT_STR",
    "ERR_NO_ARMS",
    "ERR_NO_RESULTS",
    "ERR_NO_SEEDS",
    "ERR_NO_TREES",
    "ERR_RAGGED_ROWS",
    "ERR_SCHEMA_VERSION",
    "ERR_UNKNOWN_GROWTH_POLICY",
    "REPORT_SCHEMA_VERSION",
    "ArmResult",
    "ArmSummary",
    "DatasetInfo",
    "ExperimentConfig",
    "GrowthPolicyReport",
    "decode_arm_result",
    "decode_arm_summary",
    "decode_dataset_info",
    "decode_experiment_config",
    "decode_growth_policy_report",
    "encode_arm_result",
    "encode_arm_summary",
    "encode_dataset_info",
    "encode_experiment_config",
    "encode_growth_policy_report",
    "require_mapping",
]
