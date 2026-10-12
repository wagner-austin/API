"""Encoders and strict decoders for the evaluation's record types.

Each ``decode_*`` refuses the first missing or mistyped field with
:class:`~platform_core.json_utils.JSONTypeError` naming it; nothing is
defaulted. The quality record is the benchmark's own, decoded by its own
decoder, so the two documents cannot disagree on what a quality record is.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_int,
    narrow_json_to_str,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)
from platform_core.members import require_member

from ..benchmarking.types_codec import decode_quality_metrics, encode_quality_metrics
from .types import (
    AblationGroup,
    AlertCosts,
    ArmResult,
    ComparisonMetric,
    IntervalEstimate,
    IntervalMethod,
    LearningCurvePoint,
    PairedComparison,
    PolicyOutcome,
    PooledComparison,
    ResamplingSettings,
    ResamplingUnit,
    SeedComparisons,
)


def encode_interval(interval: IntervalEstimate) -> JSONObject:
    """Lower an interval estimate to JSON.

    Args:
        interval: The estimate.

    Returns:
        A JSON object mirroring :class:`IntervalEstimate`.
    """
    return {
        "method": interval["method"].value,
        "estimate": interval["estimate"],
        "std_error": interval["std_error"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "confidence": interval["confidence"],
        "p_value": interval["p_value"],
        "n_units": interval["n_units"],
    }


def decode_interval(obj: JSONObject) -> IntervalEstimate:
    """Raise an interval estimate out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated estimate.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return {
        "method": require_member(obj, "method", IntervalMethod),
        "estimate": require_float(obj, "estimate"),
        "std_error": require_float(obj, "std_error"),
        "ci_low": require_float(obj, "ci_low"),
        "ci_high": require_float(obj, "ci_high"),
        "confidence": require_float(obj, "confidence"),
        "p_value": require_float(obj, "p_value"),
        "n_units": require_int(obj, "n_units"),
    }


def encode_comparison(comparison: PairedComparison) -> JSONObject:
    """Lower a paired comparison to JSON.

    Args:
        comparison: The comparison.

    Returns:
        A JSON object mirroring :class:`PairedComparison`.
    """
    return {
        "metric": comparison["metric"].value,
        "arm": comparison["arm"],
        "reference": comparison["reference"],
        "value_arm": comparison["value_arm"],
        "value_reference": comparison["value_reference"],
        "difference": encode_interval(comparison["difference"]),
        "n_rows": comparison["n_rows"],
    }


def decode_comparison(obj: JSONObject) -> PairedComparison:
    """Raise a paired comparison out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated comparison.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return {
        "metric": require_member(obj, "metric", ComparisonMetric),
        "arm": require_str(obj, "arm"),
        "reference": require_str(obj, "reference"),
        "value_arm": require_float(obj, "value_arm"),
        "value_reference": require_float(obj, "value_reference"),
        "difference": decode_interval(require_dict(obj, "difference")),
        "n_rows": require_int(obj, "n_rows"),
    }


def encode_policy(policy: PolicyOutcome) -> JSONObject:
    """Lower a policy outcome to JSON.

    Args:
        policy: The outcome.

    Returns:
        A JSON object mirroring :class:`PolicyOutcome`.
    """
    counts: list[JSONValue] = list(policy["action_counts"])
    return {
        "arm": policy["arm"],
        "action_counts": counts,
        "value": encode_interval(policy["value"]),
    }


def decode_policy(obj: JSONObject) -> PolicyOutcome:
    """Raise a policy outcome out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated outcome.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return {
        "arm": require_str(obj, "arm"),
        "action_counts": [narrow_json_to_int(v) for v in require_list(obj, "action_counts")],
        "value": decode_interval(require_dict(obj, "value")),
    }


def encode_resampling(settings: ResamplingSettings) -> JSONObject:
    """Lower resampling settings to JSON.

    Args:
        settings: The settings.

    Returns:
        A JSON object mirroring :class:`ResamplingSettings`.
    """
    return {
        "unit": settings["unit"].value,
        "block_length": settings["block_length"],
        "n_resamples": settings["n_resamples"],
        "resample_seed": settings["resample_seed"],
        "confidence": settings["confidence"],
    }


def decode_resampling(obj: JSONObject) -> ResamplingSettings:
    """Raise resampling settings out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated settings.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return {
        "unit": require_member(obj, "unit", ResamplingUnit),
        "block_length": require_int(obj, "block_length"),
        "n_resamples": require_int(obj, "n_resamples"),
        "resample_seed": require_int(obj, "resample_seed"),
        "confidence": require_float(obj, "confidence"),
    }


def encode_arm_result(result: ArmResult) -> JSONObject:
    """Lower one arm's quality at one seed to JSON.

    Args:
        result: The result.

    Returns:
        A JSON object mirroring :class:`ArmResult`.
    """
    return {
        "arm": result["arm"],
        "seed": result["seed"],
        "quality": encode_quality_metrics(result["quality"]),
    }


def decode_arm_result(obj: JSONObject) -> ArmResult:
    """Raise one arm's quality at one seed out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated result.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
        ValueError: If the quality record is malformed, from its own decoder.
    """
    return {
        "arm": require_str(obj, "arm"),
        "seed": require_int(obj, "seed"),
        "quality": decode_quality_metrics(require_dict(obj, "quality"), "quality"),
    }


def encode_seed_comparisons(entry: SeedComparisons) -> JSONObject:
    """Lower one seed's comparisons to JSON.

    Args:
        entry: The seed's comparisons.

    Returns:
        A JSON object mirroring :class:`SeedComparisons`.
    """
    comparisons: list[JSONValue] = [encode_comparison(c) for c in entry["comparisons"]]
    policies: list[JSONValue] = [encode_policy(p) for p in entry["policies"]]
    return {
        "seed": entry["seed"],
        "n_train": entry["n_train"],
        "n_test": entry["n_test"],
        "n_test_units": entry["n_test_units"],
        "comparisons": comparisons,
        "policies": policies,
    }


def decode_seed_comparisons(obj: JSONObject) -> SeedComparisons:
    """Raise one seed's comparisons out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated comparisons.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return {
        "seed": require_int(obj, "seed"),
        "n_train": require_int(obj, "n_train"),
        "n_test": require_int(obj, "n_test"),
        "n_test_units": require_int(obj, "n_test_units"),
        "comparisons": [
            decode_comparison(narrow_json_to_dict(v)) for v in require_list(obj, "comparisons")
        ],
        "policies": [decode_policy(narrow_json_to_dict(v)) for v in require_list(obj, "policies")],
    }


def encode_pooled(pooled: PooledComparison) -> JSONObject:
    """Lower a pooled comparison to JSON.

    Args:
        pooled: The comparison.

    Returns:
        A JSON object mirroring :class:`PooledComparison`.
    """
    per_seed: list[JSONValue] = list(pooled["per_seed"])
    return {
        "metric": pooled["metric"].value,
        "arm": pooled["arm"],
        "reference": pooled["reference"],
        "per_seed": per_seed,
        "difference": encode_interval(pooled["difference"]),
    }


def decode_pooled(obj: JSONObject) -> PooledComparison:
    """Raise a pooled comparison out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated comparison.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return {
        "metric": require_member(obj, "metric", ComparisonMetric),
        "arm": require_str(obj, "arm"),
        "reference": require_str(obj, "reference"),
        "per_seed": [narrow_json_to_float(v) for v in require_list(obj, "per_seed")],
        "difference": decode_interval(require_dict(obj, "difference")),
    }


def encode_alert_costs(costs: AlertCosts) -> JSONObject:
    """Lower alerting payoffs to JSON.

    Args:
        costs: The payoffs.

    Returns:
        A JSON object mirroring :class:`AlertCosts`.
    """
    return {
        "true_alert_value": costs["true_alert_value"],
        "false_alert_cost": costs["false_alert_cost"],
        "missed_cost": costs["missed_cost"],
    }


def decode_alert_costs(obj: JSONObject) -> AlertCosts:
    """Raise alerting payoffs out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated payoffs.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return {
        "true_alert_value": require_float(obj, "true_alert_value"),
        "false_alert_cost": require_float(obj, "false_alert_cost"),
        "missed_cost": require_float(obj, "missed_cost"),
    }


def encode_curve_point(point: LearningCurvePoint) -> JSONObject:
    """Lower a learning-curve point to JSON.

    Args:
        point: The point.

    Returns:
        A JSON object mirroring :class:`LearningCurvePoint`.
    """
    return {
        "arm": point["arm"],
        "seed": point["seed"],
        "train_fraction": point["train_fraction"],
        "n_train": point["n_train"],
        "quality": encode_quality_metrics(point["quality"]),
    }


def decode_curve_point(obj: JSONObject) -> LearningCurvePoint:
    """Raise a learning-curve point out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated point.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
        ValueError: If the quality record is malformed, from its own decoder.
    """
    return {
        "arm": require_str(obj, "arm"),
        "seed": require_int(obj, "seed"),
        "train_fraction": require_float(obj, "train_fraction"),
        "n_train": require_int(obj, "n_train"),
        "quality": decode_quality_metrics(require_dict(obj, "quality"), "quality"),
    }


def encode_ablation_group(group: AblationGroup) -> JSONObject:
    """Lower an ablation group to JSON.

    Args:
        group: The group.

    Returns:
        A JSON object mirroring :class:`AblationGroup`.
    """
    features: list[JSONValue] = list(group["features"])
    return {"name": group["name"], "features": features}


def decode_ablation_group(obj: JSONObject) -> AblationGroup:
    """Raise an ablation group out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated group.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    return {
        "name": require_str(obj, "name"),
        "features": [narrow_json_to_str(v) for v in require_list(obj, "features")],
    }


__all__ = [
    "decode_ablation_group",
    "decode_alert_costs",
    "decode_arm_result",
    "decode_comparison",
    "decode_curve_point",
    "decode_interval",
    "decode_policy",
    "decode_pooled",
    "decode_resampling",
    "decode_seed_comparisons",
    "encode_ablation_group",
    "encode_alert_costs",
    "encode_arm_result",
    "encode_comparison",
    "encode_curve_point",
    "encode_interval",
    "encode_policy",
    "encode_pooled",
    "encode_resampling",
    "encode_seed_comparisons",
]
