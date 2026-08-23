"""Decode a native ClearGBM model (via its Rust JSON) into the Python model."""

from __future__ import annotations

import types
from typing import Literal, Protocol

from cleargbm.types import (
    DecisionTree,
    GradientBoostingConfig,
    GradientBoostingModel,
    TreeNode,
    require_growth_strategy,
    require_objective,
)
from platform_core.json_utils import (
    JSONValue,
    load_json_str,
    narrow_json_to_bool,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)


class _PyGbmModelProto(Protocol):
    """Opaque native model handle produced by the Rust training loop."""

    ...


class _ToJsonProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_to_json_rs``."""

    def __call__(self, model: _PyGbmModelProto) -> str:
        """Serialize a native model to JSON.

        Args:
            model: Trained native model handle.

        Returns:
            JSON representation.
        """
        ...


_native_mod: types.ModuleType = __import__("cleargbm_rs")


_py_gbm_model_to_json: _ToJsonProto = _native_mod.py_gbm_model_to_json_rs


def _optional_int(raw: JSONValue) -> int | None:
    """Coerce an optional decoded JSON value to ``int | None``.

    Args:
        raw: Value decoded from JSON (may be ``None``).

    Returns:
        ``None`` if the value is JSON null, else the value as ``int``.

    Raises:
        TypeError: If the value is present but not an int.
    """
    if raw is None:
        return None
    return narrow_json_to_int(raw)


def _optional_float(raw: JSONValue) -> float | None:
    """Coerce an optional decoded JSON value to ``float | None``.

    Args:
        raw: Value decoded from JSON (may be ``None``).

    Returns:
        ``None`` if the value is JSON null, else the value as ``float``.

    Raises:
        TypeError: If the value is present but not a number.
    """
    if raw is None:
        return None
    return narrow_json_to_float(raw)


def _decode_rust_node(
    raw: JSONValue,
    feature_names: tuple[str, ...],
) -> TreeNode:
    """Translate a single Rust-shape JSON node dict into a Python TreeNode.

    Field-level differences bridged:

    - Rust ``nan_goes_left: bool`` → Python
      ``nan_direction: Literal["left", "right"]``.
    - Rust does not carry ``feature_name`` per node; it is looked up from the
      model-level ``feature_names`` tuple via ``feature_index`` (``None`` for
      leaf nodes).

    Args:
        raw: Decoded Rust JSON node.
        feature_names: Model-level feature names (indexed by ``feature_index``).

    Returns:
        A ``TreeNode`` TypedDict populated from the Rust dict.

    Raises:
        TypeError: On any field-shape mismatch.
        ValueError: If ``feature_index`` is out of bounds for
            ``feature_names``.
    """
    node = narrow_json_to_dict(raw)
    # The path explainer walks threshold splits; a set-membership split has
    # no threshold to walk, so a categorical model is refused loudly rather
    # than silently mis-attributed.
    if node.get("categories_goes_left") is not None:
        raise ValueError(
            "the SHAP path explainer does not support categorical splits; "
            f"node {node.get('node_id')} routes by category membership"
        )
    feature_index = _optional_int(node.get("feature_index"))
    feature_name: str | None
    if feature_index is None:
        feature_name = None
    else:
        if feature_index < 0 or feature_index >= len(feature_names):
            raise ValueError(
                f"feature_index {feature_index} out of range for {len(feature_names)} feature names"
            )
        feature_name = feature_names[feature_index]
    nan_goes_left = narrow_json_to_bool(node["nan_goes_left"])
    nan_direction: Literal["left", "right"] = "left" if nan_goes_left else "right"
    return TreeNode(
        node_id=narrow_json_to_int(node["node_id"]),
        is_leaf=narrow_json_to_bool(node["is_leaf"]),
        feature_index=feature_index,
        feature_name=feature_name,
        threshold=_optional_float(node.get("threshold")),
        nan_direction=nan_direction,
        value=narrow_json_to_float(node["value"]),
        n_samples=narrow_json_to_int(node["n_samples"]),
        left_child=_optional_int(node.get("left_child")),
        right_child=_optional_int(node.get("right_child")),
    )


def _decode_rust_tree(
    raw: JSONValue,
    feature_names: tuple[str, ...],
) -> DecisionTree:
    """Translate a single Rust-shape JSON tree into a Python DecisionTree.

    Injects the model-level ``feature_names`` into each tree (Python
    ``DecisionTree`` duplicates them per tree; Rust stores them once at
    the model level).

    Args:
        raw: Decoded Rust JSON tree.
        feature_names: Model-level feature names.

    Returns:
        A ``DecisionTree`` TypedDict populated from the Rust dict.

    Raises:
        TypeError: On any field-shape mismatch.
        ValueError: If any node has an out-of-range ``feature_index``.
    """
    tree = narrow_json_to_dict(raw)
    nodes_raw = narrow_json_to_list(tree["nodes"])
    nodes: tuple[TreeNode, ...] = tuple(_decode_rust_node(n, feature_names) for n in nodes_raw)
    return DecisionTree(
        nodes=nodes,
        max_depth=narrow_json_to_int(tree["max_depth"]),
        n_leaves=narrow_json_to_int(tree["n_leaves"]),
        feature_names=feature_names,
    )


_MONOTONIC_STRING_TO_INT: dict[str, int] = {
    "None": 0,
    "Increasing": 1,
    "Decreasing": -1,
}


def _decode_rust_categorical_features(raw: JSONValue) -> tuple[int, ...] | None:
    """Decode the stored categorical feature indices.

    Args:
        raw: The config's ``categorical_features`` entry (list of ints or
            null; absent only in pre-categorical payloads, which the Rust
            loader already refuses).

    Returns:
        The indices as a tuple, or None.

    Raises:
        TypeError: If the value is neither null nor a list of ints.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError(f"categorical_features must be list or null, got {type(raw).__name__}")
    return tuple(narrow_json_to_int(v) for v in raw)


def _decode_rust_monotonic_constraints(raw: JSONValue) -> tuple[int, ...] | None:
    """Translate Rust-shape monotonic constraints into Python integers.

    Rust serializes each constraint as one of ``"None"``, ``"Increasing"``, or
    ``"Decreasing"``. Python's ``GradientBoostingConfig`` stores them as
    integers in ``{-1, 0, 1}``. Missing constraints are ``null`` on both sides.

    Args:
        raw: JSON value (``None`` or a list of strings).

    Returns:
        A tuple of ints, or ``None`` if constraints are not set.

    Raises:
        TypeError: If ``raw`` is not None and not a list.
        ValueError: If a list entry is not one of the three known variants.
    """
    if raw is None:
        return None
    items = narrow_json_to_list(raw)
    result: list[int] = []
    for item in items:
        label = narrow_json_to_str(item)
        if label not in _MONOTONIC_STRING_TO_INT:
            raise ValueError(f"unknown monotonic constraint variant: {label!r}")
        result.append(_MONOTONIC_STRING_TO_INT[label])
    return tuple(result)


def _decode_rust_config(raw: JSONValue) -> GradientBoostingConfig:
    """Translate a Rust-shape ``GradientBoostingConfig`` JSON dict into the Python TypedDict.

    Fills in a default for the one field the Rust core does not carry:

    - ``n_jobs`` defaults to ``1`` (a runtime knob, deliberately not part of
      the serialized model config).

    Args:
        raw: Decoded Rust JSON config object.

    Returns:
        A ``GradientBoostingConfig`` TypedDict populated from the Rust JSON.

    Raises:
        TypeError: On field shape mismatches.
        ValueError: On unknown monotonic constraint variants.
    """
    cfg = narrow_json_to_dict(raw)
    return GradientBoostingConfig(
        n_estimators=narrow_json_to_int(cfg["n_estimators"]),
        max_depth=narrow_json_to_int(cfg["max_depth"]),
        learning_rate=narrow_json_to_float(cfg["learning_rate"]),
        min_samples_split=narrow_json_to_int(cfg["min_samples_split"]),
        min_samples_leaf=narrow_json_to_int(cfg["min_samples_leaf"]),
        # Read from the payload: the serialized config records the budget
        # the model actually trained under.
        max_features=_optional_int(cfg.get("max_features")),
        colsample_bytree=_optional_float(cfg.get("colsample_bytree")),
        categorical_features=_decode_rust_categorical_features(cfg.get("categorical_features")),
        max_bins=narrow_json_to_int(cfg["max_bins"]),
        subsample=narrow_json_to_float(cfg["subsample"]),
        random_state=narrow_json_to_int(cfg["random_state"]),
        monotonic_constraints=_decode_rust_monotonic_constraints(cfg.get("monotonic_constraints")),
        reg_alpha=narrow_json_to_float(cfg["reg_alpha"]),
        reg_lambda=narrow_json_to_float(cfg["reg_lambda"]),
        n_jobs=1,
        early_stopping_rounds=_optional_int(cfg.get("early_stopping_rounds")),
        # Read from the payload rather than hardcoded: the Rust config
        # serializes the policy it actually trained under, and a decoder that
        # asserted "depth_wise" would silently relabel any future arm.
        growth_strategy=require_growth_strategy(
            narrow_json_to_str(cfg["growth_strategy"]), "growth_strategy"
        ),
        num_leaves=_optional_int(cfg.get("num_leaves")),
        # Read from the payload for the same reason as growth_strategy: the
        # serialized config records the loss and weight the model actually
        # trained under. The weight is None exactly under squared_error.
        objective=require_objective(narrow_json_to_str(cfg["objective"]), "objective"),
        scale_pos_weight=_optional_float(cfg.get("scale_pos_weight")),
    )


def _decode_rust_json_to_python_model(json_str: str) -> GradientBoostingModel:
    """Decode Rust-shape ``PyGbmModel`` JSON into a Python ``GradientBoostingModel``.

    Used at the SHAP-explainer boundary: SHAP's tree walker consumes the
    Python TypedDict shape, so a native ``PyGbmModel`` must be serialized and
    re-parsed into that shape before walking. The ``config`` sub-object stays
    as an opaque ``dict[str, object]`` — the SHAP walker never reads it, so
    strict validation there would be dead weight.

    Args:
        json_str: JSON produced by ``py_gbm_model_to_json_rs``.

    Returns:
        A ``GradientBoostingModel`` TypedDict populated from the Rust JSON.

    Raises:
        TypeError: On any field-shape mismatch at the model, tree, or node
            level.
        ValueError: If any node references a feature index outside the
            model-level ``feature_names`` list.
    """
    raw = narrow_json_to_dict(load_json_str(json_str))
    feature_names_raw = narrow_json_to_list(raw["feature_names"])
    feature_names: tuple[str, ...] = tuple(narrow_json_to_str(name) for name in feature_names_raw)
    trees_raw = narrow_json_to_list(raw["trees"])
    trees: tuple[DecisionTree, ...] = tuple(_decode_rust_tree(t, feature_names) for t in trees_raw)
    return GradientBoostingModel(
        trees=trees,
        base_prediction=narrow_json_to_float(raw["base_prediction"]),
        learning_rate=narrow_json_to_float(raw["learning_rate"]),
        feature_names=feature_names,
        config=_decode_rust_config(raw["config"]),
    )
