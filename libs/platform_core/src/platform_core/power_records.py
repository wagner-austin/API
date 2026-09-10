"""The JSON boundary for power records.

Separated from :mod:`platform_core.minimum_detectable_effect` by role: that
module decides what an instrument could have resolved, this one moves the
answer across a serialisation boundary and validates what comes back.

WHY DECODE VALIDATES RATHER THAN TRUSTS. These records are published beside
nulls, in wiki pages and run records that outlive the process that wrote
them. A verdict of ``TESTED`` read back from a file is a claim about whether
an experiment could see anything, so a typo in that field is a false
statement about the evidence, not a formatting nuisance. Every string field
is checked against its vocabulary and refused if it is outside it.

The instrument check is per-record-type rather than merely "is this a known
instrument": decoding a McNemar payload into a continuous record would
silently reinterpret ``discordant_pairs`` as replicates, so each decoder
demands its own instrument by name.
"""

from __future__ import annotations

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_float,
    require_int,
    require_str,
)
from platform_core.power_distributions import McNemarTest
from platform_core.power_types import (
    ClusteredPairedPower,
    McNemarDesignSize,
    McNemarDetectableEffect,
    McNemarPower,
    NetDifferencePower,
    PairedContinuousPower,
    PowerInstrument,
    PowerVerdict,
    RateFloorPower,
    RequiredReplicates,
    ZeroFailurePower,
)


def _require_verdict(obj: JSONObject, key: str) -> str:
    """Read a verdict string and check it against the vocabulary.

    Args:
        obj: Decoded JSON object.
        key: Field name.

    Returns:
        The verdict string.

    Raises:
        AppError: ``POWER_VERDICT_UNKNOWN`` when outside the vocabulary.
    """
    raw = require_str(obj, key)
    if raw not in {member.value for member in PowerVerdict}:
        raise AppError(
            StatisticalPowerErrorCode.POWER_VERDICT_UNKNOWN,
            f"{key} must be one of TESTED/NOT_TESTED; got {raw!r}",
        )
    return raw


def _require_mcnemar_test(obj: JSONObject, key: str) -> str:
    """Read a McNemar test name and check it against the vocabulary.

    Args:
        obj: Decoded JSON object.
        key: Field name.

    Returns:
        The test string.

    Raises:
        AppError: ``POWER_TEST_UNKNOWN`` when outside the vocabulary.
    """
    raw = require_str(obj, key)
    if raw not in {member.value for member in McNemarTest}:
        raise AppError(
            StatisticalPowerErrorCode.POWER_TEST_UNKNOWN,
            f"{key} must be one of exact/mid_p; got {raw!r}",
        )
    return raw


def _require_instrument(obj: JSONObject, key: str, expected: PowerInstrument) -> str:
    """Read an instrument string and check it is the expected one.

    Args:
        obj: Decoded JSON object.
        key: Field name.
        expected: The instrument this record must carry.

    Returns:
        The instrument string.

    Raises:
        AppError: ``POWER_INSTRUMENT_UNKNOWN`` when it is not ``expected``.
    """
    raw = require_str(obj, key)
    if raw != expected.value:
        raise AppError(
            StatisticalPowerErrorCode.POWER_INSTRUMENT_UNKNOWN,
            f"{key} must be {expected.value!r} for this record; got {raw!r}",
        )
    return raw


def encode_paired_continuous_power(record: PairedContinuousPower) -> JSONObject:
    """Encode a :class:`PairedContinuousPower` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "replicates": record["replicates"],
        "degrees_of_freedom": record["degrees_of_freedom"],
        "alpha": record["alpha"],
        "mean_difference": record["mean_difference"],
        "sample_sd": record["sample_sd"],
        "t_critical": record["t_critical"],
        "minimum_detectable_effect": record["minimum_detectable_effect"],
        "smallest_effect_of_interest": record["smallest_effect_of_interest"],
        "verdict": record["verdict"],
    }


def decode_paired_continuous_power(obj: JSONObject) -> PairedContinuousPower:
    """Decode a :class:`PairedContinuousPower` from a JSON object.

    Args:
        obj: JSON object as produced by
            :func:`encode_paired_continuous_power`.

    Returns:
        The validated record.

    Raises:
        AppError: On unknown verdict or instrument.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    return PairedContinuousPower(
        instrument=_require_instrument(obj, "instrument", PowerInstrument.PAIRED_CONTINUOUS),
        replicates=require_int(obj, "replicates"),
        degrees_of_freedom=require_int(obj, "degrees_of_freedom"),
        alpha=require_float(obj, "alpha"),
        mean_difference=require_float(obj, "mean_difference"),
        sample_sd=require_float(obj, "sample_sd"),
        t_critical=require_float(obj, "t_critical"),
        minimum_detectable_effect=require_float(obj, "minimum_detectable_effect"),
        smallest_effect_of_interest=require_float(obj, "smallest_effect_of_interest"),
        verdict=_require_verdict(obj, "verdict"),
    )


def encode_required_replicates(record: RequiredReplicates) -> JSONObject:
    """Encode a :class:`RequiredReplicates` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "observed_replicates": record["observed_replicates"],
        "observed_sample_sd": record["observed_sample_sd"],
        "alpha": record["alpha"],
        "smallest_effect_of_interest": record["smallest_effect_of_interest"],
        "required_replicates": record["required_replicates"],
        "additional_replicates": record["additional_replicates"],
    }


def decode_required_replicates(obj: JSONObject) -> RequiredReplicates:
    """Decode a :class:`RequiredReplicates` from a JSON object.

    Args:
        obj: JSON object as produced by :func:`encode_required_replicates`.

    Returns:
        The validated record.

    Raises:
        AppError: On an instrument outside this record's vocabulary.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    return RequiredReplicates(
        instrument=_require_instrument(obj, "instrument", PowerInstrument.PAIRED_CONTINUOUS),
        observed_replicates=require_int(obj, "observed_replicates"),
        observed_sample_sd=require_float(obj, "observed_sample_sd"),
        alpha=require_float(obj, "alpha"),
        smallest_effect_of_interest=require_float(obj, "smallest_effect_of_interest"),
        required_replicates=require_int(obj, "required_replicates"),
        additional_replicates=require_int(obj, "additional_replicates"),
    )


def encode_mcnemar_power(record: McNemarPower) -> JSONObject:
    """Encode a :class:`McNemarPower` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "test": record["test"],
        "discordant_pairs": record["discordant_pairs"],
        "alpha": record["alpha"],
        "smallest_attainable_p": record["smallest_attainable_p"],
        "can_ever_reject": record["can_ever_reject"],
        "most_balanced_rejecting_minority": record["most_balanced_rejecting_minority"],
    }


def decode_mcnemar_power(obj: JSONObject) -> McNemarPower:
    """Decode a :class:`McNemarPower` from a JSON object.

    Args:
        obj: JSON object as produced by :func:`encode_mcnemar_power`.

    Returns:
        The validated record.

    Raises:
        AppError: On unknown verdict, test or instrument.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    return McNemarPower(
        instrument=_require_instrument(obj, "instrument", PowerInstrument.MCNEMAR),
        test=_require_mcnemar_test(obj, "test"),
        discordant_pairs=require_int(obj, "discordant_pairs"),
        alpha=require_float(obj, "alpha"),
        smallest_attainable_p=require_float(obj, "smallest_attainable_p"),
        can_ever_reject=require_bool(obj, "can_ever_reject"),
        most_balanced_rejecting_minority=require_int(obj, "most_balanced_rejecting_minority"),
    )


def encode_net_difference_power(record: NetDifferencePower) -> JSONObject:
    """Encode a :class:`NetDifferencePower` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "test": record["test"],
        "net_difference": record["net_difference"],
        "total_pairs": record["total_pairs"],
        "alpha": record["alpha"],
        "best_case_p": record["best_case_p"],
        "smallest_resolvable_net_difference": record["smallest_resolvable_net_difference"],
        "net_could_ever_be_significant": record["net_could_ever_be_significant"],
    }


def decode_net_difference_power(obj: JSONObject) -> NetDifferencePower:
    """Decode a :class:`NetDifferencePower` from a JSON object.

    Args:
        obj: JSON object as produced by :func:`encode_net_difference_power`.

    Returns:
        The validated record.

    Raises:
        AppError: On unknown test or instrument.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    return NetDifferencePower(
        instrument=_require_instrument(obj, "instrument", PowerInstrument.NET_DIFFERENCE),
        test=_require_mcnemar_test(obj, "test"),
        net_difference=require_int(obj, "net_difference"),
        total_pairs=require_int(obj, "total_pairs"),
        alpha=require_float(obj, "alpha"),
        best_case_p=require_float(obj, "best_case_p"),
        smallest_resolvable_net_difference=require_int(obj, "smallest_resolvable_net_difference"),
        net_could_ever_be_significant=require_bool(obj, "net_could_ever_be_significant"),
    )


def encode_clustered_paired_power(record: ClusteredPairedPower) -> JSONObject:
    """Encode a :class:`ClusteredPairedPower` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "unit": record["unit"],
        "clusters": record["clusters"],
        "total_units": record["total_units"],
        "largest_cluster": record["largest_cluster"],
        "average_cluster_size": record["average_cluster_size"],
        "intracluster_correlation": record["intracluster_correlation"],
        "design_effect": record["design_effect"],
        "effective_sample_size": record["effective_sample_size"],
    }


def decode_clustered_paired_power(obj: JSONObject) -> ClusteredPairedPower:
    """Decode a :class:`ClusteredPairedPower` from a JSON object.

    The unit is checked for emptiness here as well as at construction. A
    record read back from a file is the form a reader acts on, and a blank
    unit makes the design effect beside it uninterpretable -- the same 875
    items grouped three ways give three different answers.

    Args:
        obj: JSON object as produced by :func:`encode_clustered_paired_power`.

    Returns:
        The validated record.

    Raises:
        AppError: On an unknown instrument, or on a blank clustering unit.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    unit = require_str(obj, "unit")
    if not unit.strip():
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            "field 'unit' is blank; a design effect without the grouping that "
            "produced it cannot be checked, because the same units grouped by "
            "directory and by package give different answers",
        )
    return ClusteredPairedPower(
        instrument=_require_instrument(obj, "instrument", PowerInstrument.CLUSTERED_PAIRED),
        unit=unit,
        clusters=require_int(obj, "clusters"),
        total_units=require_int(obj, "total_units"),
        largest_cluster=require_int(obj, "largest_cluster"),
        average_cluster_size=require_float(obj, "average_cluster_size"),
        intracluster_correlation=require_float(obj, "intracluster_correlation"),
        design_effect=require_float(obj, "design_effect"),
        effective_sample_size=require_float(obj, "effective_sample_size"),
    )


def encode_mcnemar_detectable_effect(record: McNemarDetectableEffect) -> JSONObject:
    """Encode a :class:`McNemarDetectableEffect` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "test": record["test"],
        "discordant_pairs": record["discordant_pairs"],
        "total_pairs": record["total_pairs"],
        "alpha": record["alpha"],
        "target_power": record["target_power"],
        "most_balanced_rejecting_minority": record["most_balanced_rejecting_minority"],
        "minimum_detectable_split": record["minimum_detectable_split"],
        "minimum_detectable_net_pairs": record["minimum_detectable_net_pairs"],
        "minimum_detectable_rate_difference": record["minimum_detectable_rate_difference"],
        "achieved_power": record["achieved_power"],
    }


def decode_mcnemar_detectable_effect(obj: JSONObject) -> McNemarDetectableEffect:
    """Decode a :class:`McNemarDetectableEffect` from a JSON object.

    Args:
        obj: JSON object as produced by
            :func:`encode_mcnemar_detectable_effect`.

    Returns:
        The validated record.

    Raises:
        AppError: On an unknown test or instrument.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    return McNemarDetectableEffect(
        instrument=_require_instrument(
            obj, "instrument", PowerInstrument.MCNEMAR_DETECTABLE_EFFECT
        ),
        test=_require_mcnemar_test(obj, "test"),
        discordant_pairs=require_int(obj, "discordant_pairs"),
        total_pairs=require_int(obj, "total_pairs"),
        alpha=require_float(obj, "alpha"),
        target_power=require_float(obj, "target_power"),
        most_balanced_rejecting_minority=require_int(obj, "most_balanced_rejecting_minority"),
        minimum_detectable_split=require_float(obj, "minimum_detectable_split"),
        minimum_detectable_net_pairs=require_float(obj, "minimum_detectable_net_pairs"),
        minimum_detectable_rate_difference=require_float(obj, "minimum_detectable_rate_difference"),
        achieved_power=require_float(obj, "achieved_power"),
    )


def encode_mcnemar_design_size(record: McNemarDesignSize) -> JSONObject:
    """Encode a :class:`McNemarDesignSize` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "test": record["test"],
        "discordant_rate": record["discordant_rate"],
        "split": record["split"],
        "alpha": record["alpha"],
        "target_power": record["target_power"],
        "search_ceiling": record["search_ceiling"],
        "first_reaching_pairs": record["first_reaching_pairs"],
        "durably_reaching_pairs": record["durably_reaching_pairs"],
        "sawtooth_gap_pairs": record["sawtooth_gap_pairs"],
        "power_at_first_reaching": record["power_at_first_reaching"],
        "power_at_durably_reaching": record["power_at_durably_reaching"],
        "expected_discordant_pairs": record["expected_discordant_pairs"],
    }


def decode_mcnemar_design_size(obj: JSONObject) -> McNemarDesignSize:
    """Decode a :class:`McNemarDesignSize` from a JSON object.

    Args:
        obj: JSON object as produced by :func:`encode_mcnemar_design_size`.

    Returns:
        The validated record.

    Raises:
        AppError: On an unknown test or instrument.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    return McNemarDesignSize(
        instrument=_require_instrument(obj, "instrument", PowerInstrument.MCNEMAR_DESIGN_SIZE),
        test=_require_mcnemar_test(obj, "test"),
        discordant_rate=require_float(obj, "discordant_rate"),
        split=require_float(obj, "split"),
        alpha=require_float(obj, "alpha"),
        target_power=require_float(obj, "target_power"),
        search_ceiling=require_int(obj, "search_ceiling"),
        first_reaching_pairs=require_int(obj, "first_reaching_pairs"),
        durably_reaching_pairs=require_int(obj, "durably_reaching_pairs"),
        sawtooth_gap_pairs=require_int(obj, "sawtooth_gap_pairs"),
        power_at_first_reaching=require_float(obj, "power_at_first_reaching"),
        power_at_durably_reaching=require_float(obj, "power_at_durably_reaching"),
        expected_discordant_pairs=require_float(obj, "expected_discordant_pairs"),
    )


def encode_rate_floor_power(record: RateFloorPower) -> JSONObject:
    """Encode a :class:`RateFloorPower` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "successes": record["successes"],
        "trials": record["trials"],
        "observed_rate": record["observed_rate"],
        "floor": record["floor"],
        "alpha": record["alpha"],
        "p_value": record["p_value"],
        "perfect_record_trials": record["perfect_record_trials"],
        "rate_significantly_exceeds_floor": record["rate_significantly_exceeds_floor"],
        "design_can_clear_floor": record["design_can_clear_floor"],
    }


def decode_rate_floor_power(obj: JSONObject) -> RateFloorPower:
    """Decode a :class:`RateFloorPower` from a JSON object.

    Args:
        obj: JSON object as produced by :func:`encode_rate_floor_power`.

    Returns:
        The validated record.

    Raises:
        AppError: On unknown verdict or mismatched instrument.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    return RateFloorPower(
        instrument=_require_instrument(obj, "instrument", PowerInstrument.RATE_FLOOR),
        successes=require_int(obj, "successes"),
        trials=require_int(obj, "trials"),
        observed_rate=require_float(obj, "observed_rate"),
        floor=require_float(obj, "floor"),
        alpha=require_float(obj, "alpha"),
        p_value=require_float(obj, "p_value"),
        perfect_record_trials=require_int(obj, "perfect_record_trials"),
        rate_significantly_exceeds_floor=require_bool(obj, "rate_significantly_exceeds_floor"),
        design_can_clear_floor=require_bool(obj, "design_can_clear_floor"),
    )


def encode_zero_failure_power(record: ZeroFailurePower) -> JSONObject:
    """Encode a :class:`ZeroFailurePower` to a JSON object.

    Args:
        record: The record to encode.

    Returns:
        A JSON object carrying every field.
    """
    return {
        "instrument": record["instrument"],
        "trials": record["trials"],
        "confidence": record["confidence"],
        "upper_bound": record["upper_bound"],
        "largest_rate_of_interest": record["largest_rate_of_interest"],
        "verdict": record["verdict"],
    }


def decode_zero_failure_power(obj: JSONObject) -> ZeroFailurePower:
    """Decode a :class:`ZeroFailurePower` from a JSON object.

    Args:
        obj: JSON object as produced by :func:`encode_zero_failure_power`.

    Returns:
        The validated record.

    Raises:
        AppError: On unknown verdict or instrument.
        JSONTypeError: On a missing or wrongly-typed field.
    """
    return ZeroFailurePower(
        instrument=_require_instrument(obj, "instrument", PowerInstrument.ZERO_FAILURE_PROPORTION),
        trials=require_int(obj, "trials"),
        confidence=require_float(obj, "confidence"),
        upper_bound=require_float(obj, "upper_bound"),
        largest_rate_of_interest=require_float(obj, "largest_rate_of_interest"),
        verdict=_require_verdict(obj, "verdict"),
    )


__all__ = [
    "decode_clustered_paired_power",
    "decode_mcnemar_design_size",
    "decode_mcnemar_detectable_effect",
    "decode_mcnemar_power",
    "decode_net_difference_power",
    "decode_paired_continuous_power",
    "decode_rate_floor_power",
    "decode_required_replicates",
    "decode_zero_failure_power",
    "encode_clustered_paired_power",
    "encode_mcnemar_design_size",
    "encode_mcnemar_detectable_effect",
    "encode_mcnemar_power",
    "encode_net_difference_power",
    "encode_paired_continuous_power",
    "encode_rate_floor_power",
    "encode_required_replicates",
    "encode_zero_failure_power",
]
