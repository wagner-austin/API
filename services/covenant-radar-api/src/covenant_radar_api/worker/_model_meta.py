"""Model metadata decoding for inference loading."""

from __future__ import annotations

from pathlib import Path

from covenant_ml.types import (
    LogRegPenalty,
    LogRegSolver,
)
from covenant_ml.types_model_meta import (
    LightGBMModelMeta,
    LogRegModelMeta,
    LSTMModelMeta,
    MLPModelMeta,
    ModelMeta,
    RandomForestModelMeta,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_float,
    require_int,
    require_list,
    require_str,
)
from platform_core.members import require_member


def _decode_mlp_meta(raw: JSONObject) -> MLPModelMeta:
    """Decode and validate MLP model metadata from JSON object.

    Args:
        raw: Parsed JSON object with MLP metadata fields.

    Returns:
        Validated MLPModelMeta TypedDict.

    Raises:
        JSONTypeError: If any field is missing or has wrong type.
    """
    backend = require_str(raw, "backend")
    if backend != "mlp":
        raise JSONTypeError(f"Expected backend 'mlp', got '{backend}'")

    n_features = require_int(raw, "n_features")
    dropout = require_float(raw, "dropout")

    # Parse hidden_sizes as list of ints
    hidden_sizes_raw = require_list(raw, "hidden_sizes")
    hidden_sizes: list[int] = []
    for i, val in enumerate(hidden_sizes_raw):
        if isinstance(val, bool) or not isinstance(val, int):
            raise JSONTypeError(f"hidden_sizes[{i}] must be an integer, got {type(val).__name__}")
        hidden_sizes.append(val)

    return {
        "backend": "mlp",
        "n_features": n_features,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
    }


def _decode_lstm_meta(raw: JSONObject) -> LSTMModelMeta:
    """Decode and validate LSTM model metadata from JSON object.

    Args:
        raw: Parsed JSON object with LSTM metadata fields.

    Returns:
        Validated LSTMModelMeta TypedDict.

    Raises:
        JSONTypeError: If any field is missing or has wrong type.
    """
    backend = require_str(raw, "backend")
    if backend != "lstm":
        raise JSONTypeError(f"Expected backend 'lstm', got '{backend}'")

    return {
        "backend": "lstm",
        "n_features": require_int(raw, "n_features"),
        "sequence_length": require_int(raw, "sequence_length"),
        "hidden_size": require_int(raw, "hidden_size"),
        "num_layers": require_int(raw, "num_layers"),
        "bidirectional": require_bool(raw, "bidirectional"),
        "dropout": require_float(raw, "dropout"),
    }


def _decode_lightgbm_meta(raw: JSONObject) -> LightGBMModelMeta:
    """Decode and validate LightGBM model metadata from JSON object.

    Args:
        raw: Parsed JSON object with LightGBM metadata fields.

    Returns:
        Validated LightGBMModelMeta TypedDict.

    Raises:
        JSONTypeError: If backend field is missing or wrong.
    """
    backend = require_str(raw, "backend")
    if backend != "lightgbm":
        raise JSONTypeError(f"Expected backend 'lightgbm', got '{backend}'")

    return {"backend": "lightgbm"}


def _decode_logreg_meta(raw: JSONObject) -> LogRegModelMeta:
    """Decode and validate Logistic Regression model metadata from JSON object.

    Args:
        raw: Parsed JSON object with LogReg metadata fields.

    Returns:
        Validated LogRegModelMeta TypedDict.

    Raises:
        JSONTypeError: If any field is missing or has wrong type.
    """
    backend = require_str(raw, "backend")
    if backend != "logreg":
        raise JSONTypeError(f"Expected backend 'logreg', got '{backend}'")

    return {
        "backend": "logreg",
        "n_features": require_int(raw, "n_features"),
        "penalty": require_member(raw, "penalty", LogRegPenalty),
        "solver": require_member(raw, "solver", LogRegSolver),
    }


def _decode_random_forest_meta(raw: JSONObject) -> RandomForestModelMeta:
    """Decode and validate Random Forest model metadata from JSON object.

    Args:
        raw: Parsed JSON object with Random Forest metadata fields.

    Returns:
        Validated RandomForestModelMeta TypedDict.

    Raises:
        JSONTypeError: If any field is missing or has wrong type.
    """
    backend = require_str(raw, "backend")
    if backend != "random_forest":
        raise JSONTypeError(f"Expected backend 'random_forest', got '{backend}'")

    n_features = require_int(raw, "n_features")
    n_estimators = require_int(raw, "n_estimators")

    # max_depth can be None or int
    max_depth_raw = raw.get("max_depth")
    max_depth: int | None = None
    if max_depth_raw is not None:
        if isinstance(max_depth_raw, bool) or not isinstance(max_depth_raw, int):
            raise JSONTypeError("max_depth must be an integer or null")
        max_depth = max_depth_raw

    return {
        "backend": "random_forest",
        "n_features": n_features,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }


def _load_model_metadata(meta_path: Path) -> ModelMeta:
    """Load and decode model metadata from JSON file.

    Args:
        meta_path: Path to the metadata JSON file.

    Returns:
        Decoded ModelMeta (one of MLPModelMeta, LSTMModelMeta, LightGBMModelMeta,
        LogRegModelMeta, RandomForestModelMeta).

    Raises:
        FileNotFoundError: If metadata file doesn't exist.
        JSONTypeError: If metadata is invalid.
    """
    content = meta_path.read_text(encoding="utf-8")
    parsed = load_json_str(content)
    raw = narrow_json_to_dict(parsed)

    backend = require_str(raw, "backend")

    if backend == "mlp":
        return _decode_mlp_meta(raw)
    if backend == "lstm":
        return _decode_lstm_meta(raw)
    if backend == "lightgbm":
        return _decode_lightgbm_meta(raw)
    if backend == "logreg":
        return _decode_logreg_meta(raw)
    if backend == "random_forest":
        return _decode_random_forest_meta(raw)
    raise JSONTypeError(f"Unknown backend '{backend}' in metadata")


# =============================================================================
# MLP Model Loading - Constructor Protocols
# =============================================================================
