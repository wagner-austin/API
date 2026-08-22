"""Shared field decoders for the optimization codecs."""

from __future__ import annotations

from typing import Literal

from covenant_ml.features import FeaturePreset
from covenant_ml.types import BackendName
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
)


def _require_backend_name(raw: JSONObject) -> BackendName:
    """Extract and validate backend name from JSON object.

    Args:
        raw: JSON object.

    Returns:
        Validated BackendName.

    Raises:
        JSONTypeError: If backend field is missing or invalid.
    """
    val = raw.get("backend")
    if val is None:
        raise JSONTypeError("Missing required field 'backend'")
    if not isinstance(val, str):
        raise JSONTypeError("Field 'backend' must be a string")
    valid: tuple[str, ...] = (
        "xgboost",
        "mlp",
        "lstm",
        "lightgbm",
        "cleargbm",
        "logreg",
        "random_forest",
    )
    if val not in valid:
        raise JSONTypeError(f"Field 'backend' must be one of: {', '.join(valid)} (got {val})")
    # Narrow to BackendName via explicit matching
    if val == "xgboost":
        return "xgboost"
    if val == "mlp":
        return "mlp"
    if val == "lstm":
        return "lstm"
    if val == "lightgbm":
        return "lightgbm"
    if val == "cleargbm":
        return "cleargbm"
    if val == "logreg":
        return "logreg"
    return "random_forest"


def _require_device(raw: JSONObject) -> Literal["cpu", "cuda", "auto"]:
    """Extract and validate device from JSON object.

    Args:
        raw: JSON object.

    Returns:
        Device literal.

    Raises:
        JSONTypeError: If device field is invalid.
    """
    val = raw.get("device")
    if val is None:
        raise JSONTypeError("Missing required field 'device'")
    if val == "cpu":
        return "cpu"
    if val == "cuda":
        return "cuda"
    if val == "auto":
        return "auto"
    raise JSONTypeError("Field 'device' must be one of: cpu, cuda, auto")


def _require_feature_preset(raw: JSONObject) -> FeaturePreset:
    """Extract and validate feature_preset from JSON object.

    Args:
        raw: JSON object.

    Returns:
        FeaturePreset literal.

    Raises:
        JSONTypeError: If feature_preset field is invalid.
    """
    val = raw.get("feature_preset")
    if val is None:
        raise JSONTypeError("Missing required field 'feature_preset'")
    if val == "none":
        return "none"
    if val == "log_only":
        return "log_only"
    if val == "ratios_only":
        return "ratios_only"
    if val == "full":
        return "full"
    if val == "temporal":
        return "temporal"
    raise JSONTypeError(
        "Field 'feature_preset' must be one of: none, log_only, ratios_only, full, temporal"
    )


def _require_precision(raw: JSONObject) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Extract and validate precision from JSON object.

    Args:
        raw: JSON object.

    Returns:
        Precision literal.

    Raises:
        JSONTypeError: If precision field is invalid.
    """
    val = raw.get("precision")
    if val is None:
        raise JSONTypeError("Missing required field 'precision'")
    if val == "fp32":
        return "fp32"
    if val == "fp16":
        return "fp16"
    if val == "bf16":
        return "bf16"
    if val == "auto":
        return "auto"
    raise JSONTypeError("Field 'precision' must be one of: fp32, fp16, bf16, auto")


def _require_nn_optimizer(raw: JSONObject) -> Literal["adamw", "adam", "sgd"]:
    """Extract and validate nn_optimizer from JSON object.

    Args:
        raw: JSON object.

    Returns:
        NN optimizer literal.

    Raises:
        JSONTypeError: If nn_optimizer field is invalid.
    """
    val = raw.get("nn_optimizer")
    if val is None:
        raise JSONTypeError("Missing required field 'nn_optimizer'")
    if val == "adamw":
        return "adamw"
    if val == "adam":
        return "adam"
    if val == "sgd":
        return "sgd"
    raise JSONTypeError("Field 'nn_optimizer' must be one of: adamw, adam, sgd")


def _require_int(raw: JSONObject, key: str) -> int:
    """Extract required int from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        Integer value.

    Raises:
        JSONTypeError: If field is missing or not an integer.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, int):
        raise JSONTypeError(f"Field '{key}' must be an integer")
    return val


def _require_bool(raw: JSONObject, key: str) -> bool:
    """Extract required bool from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If field is missing or not a boolean.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, bool):
        raise JSONTypeError(f"Field '{key}' must be a boolean")
    return val


def _require_str(raw: JSONObject, key: str) -> str:
    """Extract required string from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        String value.

    Raises:
        JSONTypeError: If field is missing or not a string.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, str):
        raise JSONTypeError(f"Field '{key}' must be a string")
    return val


def _require_float(raw: JSONObject, key: str) -> float:
    """Extract required float from JSON object.

    Args:
        raw: JSON object.
        key: Field name.

    Returns:
        Float value.

    Raises:
        JSONTypeError: If field is missing or not a number.
    """
    val = raw.get(key)
    if val is None:
        raise JSONTypeError(f"Missing required field '{key}'")
    if not isinstance(val, (int, float)):
        raise JSONTypeError(f"Field '{key}' must be a number")
    return float(val)
