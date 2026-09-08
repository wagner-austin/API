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
from platform_core.minimum_detectable_effect import (
    McNemarPower,
    PairedContinuousPower,
    PowerInstrument,
    PowerVerdict,
    ZeroFailurePower,
)
from platform_core.power_distributions import McNemarTest


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
        "verdict": record["verdict"],
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
        verdict=_require_verdict(obj, "verdict"),
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
    "decode_mcnemar_power",
    "decode_paired_continuous_power",
    "decode_zero_failure_power",
    "encode_mcnemar_power",
    "encode_paired_continuous_power",
    "encode_zero_failure_power",
]
