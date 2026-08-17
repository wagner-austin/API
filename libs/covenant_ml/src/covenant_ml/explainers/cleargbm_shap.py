"""ClearGBM SHAP adapter for TreeExplainer compatibility.

Consumes a native ClearGBM model handle (``cleargbm_rs.PyGbmModel``) and
converts it to a format that SHAP's TreeExplainer can parse. Internally, the
native model is serialized to JSON via ``py_gbm_model_to_json_rs`` and decoded
into the Python-side ``GradientBoostingModel`` TypedDict shape that the tree
walker consumes.

The decode step bridges the small format differences between the Rust struct
serialization and the historical Python TypedDict:

- Rust nodes carry ``nan_goes_left: bool``; Python nodes carry
  ``nan_direction: "left" | "right" | None``.
- Rust trees do not carry ``feature_names`` (stored once at model level);
  Python ``DecisionTree`` duplicates them per tree.
- Rust nodes do not carry ``feature_name`` on internal splits; Python
  ``TreeNode`` does.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import types
from typing import Literal, Protocol, TypedDict

import numpy as np
from cleargbm.types import (
    DecisionTree,
    GradientBoostingConfig,
    GradientBoostingModel,
    TreeNode,
)
from numpy.typing import NDArray
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

# =============================================================================
# Native model access — Protocol-typed getattr onto the Rust extension
# =============================================================================


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

# -----------------------------------------------------------------------------
# TypedDicts for SHAP tree format (internal representation)
# -----------------------------------------------------------------------------


class ShapTreeArrays(TypedDict):
    """Arrays representing a single tree in SHAP format.

    Each array is indexed by node_id, where -1 indicates a leaf node
    for children arrays, and -2 indicates the feature is not used (leaf).

    Args:
        children_left: Left child node IDs (-1 for leaves).
        children_right: Right child node IDs (-1 for leaves).
        children_default: Default child for NaN values.
        features: Feature indices for splits (-2 for leaves).
        thresholds: Split thresholds (0.0 for leaves).
        values: Leaf values, shape (n_nodes, 1) for binary classification.
        node_sample_weight: Number of samples at each node.
    """

    children_left: NDArray[np.int64]
    children_right: NDArray[np.int64]
    children_default: NDArray[np.int64]
    features: NDArray[np.int64]
    thresholds: NDArray[np.float64]
    values: NDArray[np.float64]
    node_sample_weight: NDArray[np.float64]


class ShapModelFormat(TypedDict):
    """Complete SHAP-compatible model format.

    This format can be passed directly to shap.TreeExplainer.

    Args:
        trees: List of ShapTreeArrays, one per boosting iteration.
        num_outputs: Number of output classes (1 for binary classification).
        base_offset: Base prediction (log-odds for binary classification).
        objective: Loss function identifier.
        tree_output: Type of tree output ("raw" for log-odds).
        input_dtype: Expected input data type.
    """

    trees: list[ShapTreeArrays]
    num_outputs: int
    base_offset: float
    objective: str
    tree_output: str
    input_dtype: type[np.float64]


# -----------------------------------------------------------------------------
# Conversion functions
# -----------------------------------------------------------------------------


class _ShapArrays:
    """Mutable container for SHAP tree arrays during construction.

    Provides a clean interface for populating arrays while converting
    ClearGBM tree nodes to SHAP format.
    """

    def __init__(self, n_nodes: int) -> None:
        """Initialize arrays with default values.

        Args:
            n_nodes: Number of nodes in the tree.
        """
        self.children_left: NDArray[np.int64] = np.full(n_nodes, -1, dtype=np.int64)
        self.children_right: NDArray[np.int64] = np.full(n_nodes, -1, dtype=np.int64)
        self.children_default: NDArray[np.int64] = np.full(n_nodes, -1, dtype=np.int64)
        self.features: NDArray[np.int64] = np.full(n_nodes, -2, dtype=np.int64)
        self.thresholds: NDArray[np.float64] = np.zeros(n_nodes, dtype=np.float64)
        # SHAP expects values shape (n_nodes, n_outputs)
        # For binary classification: (n_nodes, 1)
        self.values: NDArray[np.float64] = np.zeros((n_nodes, 1), dtype=np.float64)
        self.node_sample_weight: NDArray[np.float64] = np.zeros(n_nodes, dtype=np.float64)

    def to_typed_dict(self) -> ShapTreeArrays:
        """Convert to immutable ShapTreeArrays TypedDict.

        Returns:
            ShapTreeArrays with all arrays.
        """
        return ShapTreeArrays(
            children_left=self.children_left,
            children_right=self.children_right,
            children_default=self.children_default,
            features=self.features,
            thresholds=self.thresholds,
            values=self.values,
            node_sample_weight=self.node_sample_weight,
        )


def _get_default_child_idx(
    node: TreeNode,
    node_id_to_idx: dict[int, int],
) -> int:
    """Determine default child index for NaN handling.

    Args:
        node: TreeNode with split information.
        node_id_to_idx: Mapping from node_id to array index.

    Returns:
        Array index of default child, or -1 if none.
    """
    nan_direction = node["nan_direction"]
    left_id = node["left_child"]
    right_id = node["right_child"]

    if nan_direction == "left" and left_id is not None:
        return node_id_to_idx[left_id]
    if nan_direction == "right" and right_id is not None:
        return node_id_to_idx[right_id]
    if left_id is not None:
        # Default to left if no nan_direction specified
        return node_id_to_idx[left_id]
    return -1


def _populate_internal_node(
    arrays: _ShapArrays,
    idx: int,
    node: TreeNode,
    node_id_to_idx: dict[int, int],
) -> None:
    """Populate SHAP arrays for an internal (split) node.

    Args:
        arrays: Mutable SHAP arrays container.
        idx: Array index for this node.
        node: TreeNode with split information.
        node_id_to_idx: Mapping from node_id to array index.
    """
    left_id = node["left_child"]
    right_id = node["right_child"]
    feature_idx = node["feature_index"]
    threshold = node["threshold"]

    if left_id is not None:
        arrays.children_left[idx] = np.int64(node_id_to_idx[left_id])
    if right_id is not None:
        arrays.children_right[idx] = np.int64(node_id_to_idx[right_id])
    if feature_idx is not None:
        arrays.features[idx] = np.int64(feature_idx)
    if threshold is not None:
        arrays.thresholds[idx] = threshold

    # Set default child for NaN handling
    default_idx = _get_default_child_idx(node, node_id_to_idx)
    if default_idx >= 0:
        arrays.children_default[idx] = np.int64(default_idx)

    # Internal nodes also have values (for partial dependence)
    arrays.values[idx, 0] = node["value"]


def _convert_tree_node_to_arrays(
    nodes: tuple[TreeNode, ...],
) -> ShapTreeArrays:
    """Convert ClearGBM tree nodes to SHAP array format.

    Args:
        nodes: Tuple of TreeNode from ClearGBM DecisionTree.

    Returns:
        ShapTreeArrays with numpy arrays indexed by node_id.

    Raises:
        ValueError: If nodes tuple is empty.
    """
    n_nodes = len(nodes)
    if n_nodes == 0:
        raise ValueError("Cannot convert empty tree nodes")

    # Initialize arrays
    arrays = _ShapArrays(n_nodes)

    # Build node_id to index mapping (in case node_ids are not sequential)
    node_id_to_idx: dict[int, int] = {node["node_id"]: idx for idx, node in enumerate(nodes)}

    # Fill arrays
    for node in nodes:
        idx = node_id_to_idx[node["node_id"]]
        arrays.node_sample_weight[idx] = float(node["n_samples"])
        arrays.values[idx, 0] = node["value"]

        if not node["is_leaf"]:
            _populate_internal_node(arrays, idx, node, node_id_to_idx)

    return arrays.to_typed_dict()


def _convert_decision_tree(tree: DecisionTree) -> ShapTreeArrays:
    """Convert a ClearGBM DecisionTree to SHAP array format.

    Args:
        tree: ClearGBM DecisionTree TypedDict.

    Returns:
        ShapTreeArrays for SHAP TreeExplainer.
    """
    return _convert_tree_node_to_arrays(tree["nodes"])


def convert_cleargbm_to_shap_format(model: GradientBoostingModel) -> ShapModelFormat:
    """Convert ClearGBM GradientBoostingModel to SHAP-compatible format.

    This function transforms the entire ClearGBM model into a dictionary
    format that can be passed to shap.TreeExplainer.

    Args:
        model: Trained ClearGBM GradientBoostingModel.

    Returns:
        ShapModelFormat dictionary compatible with SHAP TreeExplainer.

    Raises:
        ValueError: If model has no trees.
    """
    if len(model["trees"]) == 0:
        raise ValueError("Cannot convert model with no trees")

    # Convert each tree
    shap_trees: list[ShapTreeArrays] = []
    for tree in model["trees"]:
        shap_tree = _convert_decision_tree(tree)
        shap_trees.append(shap_tree)

    return ShapModelFormat(
        trees=shap_trees,
        num_outputs=1,  # Binary classification
        base_offset=model["base_prediction"],
        objective="binary:logistic",
        tree_output="raw",
        input_dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Native (Rust) model → Python TypedDict decode
# -----------------------------------------------------------------------------


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

    Fills in defaults for the three fields the Rust core does not carry:

    - ``max_features`` defaults to ``None`` (Rust always uses all features).
    - ``track_contributions`` defaults to ``False`` (Python-only bookkeeping).
    - ``n_jobs`` defaults to ``1`` (Rust core handles parallelism internally).

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
        max_features=None,
        max_bins=narrow_json_to_int(cfg["max_bins"]),
        subsample=narrow_json_to_float(cfg["subsample"]),
        random_state=narrow_json_to_int(cfg["random_state"]),
        track_contributions=False,
        monotonic_constraints=_decode_rust_monotonic_constraints(cfg.get("monotonic_constraints")),
        reg_alpha=narrow_json_to_float(cfg["reg_alpha"]),
        reg_lambda=narrow_json_to_float(cfg["reg_lambda"]),
        n_jobs=1,
        early_stopping_rounds=_optional_int(cfg.get("early_stopping_rounds")),
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
        n_classes=narrow_json_to_int(raw["n_classes"]),
        config=_decode_rust_config(raw["config"]),
    )


# -----------------------------------------------------------------------------
# SHAP TreeExplainer Protocol
# -----------------------------------------------------------------------------


class ShapExplainerProtocol(Protocol):
    """Protocol for shap.TreeExplainer instance."""

    @property
    def expected_value(self) -> float | NDArray[np.float64]:
        """Base value (bias) of the model."""
        ...

    def shap_values(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        tree_limit: int | None = None,
        approximate: bool = False,
        check_additivity: bool = True,
        from_call: bool = False,
    ) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """Compute Shapley values."""
        ...


class TreeExplainerConstructor(Protocol):
    """Protocol for shap.TreeExplainer constructor."""

    def __call__(
        self,
        model: ShapModelFormat,
        data: NDArray[np.float64] | None = None,
        model_output: str = "raw",
        feature_perturbation: str = "interventional",
    ) -> ShapExplainerProtocol:
        """Create a new TreeExplainer."""
        ...


# -----------------------------------------------------------------------------
# ClearGBM SHAP Wrapper
# -----------------------------------------------------------------------------


class ClearGBMShapWrapper:
    """Wrapper for computing SHAP values on ClearGBM models.

    Accepts a native ClearGBM model handle (``cleargbm_rs.PyGbmModel``),
    serializes it to JSON via ``py_gbm_model_to_json_rs``, decodes the JSON
    into the Python-side ``GradientBoostingModel`` TypedDict shape, and
    forwards to ``convert_cleargbm_to_shap_format`` for SHAP consumption.

    Example:
        >>> from cleargbm.ensemble import train_gradient_boosting
        >>> native_model = train_gradient_boosting(...)
        >>> wrapper = ClearGBMShapWrapper(native_model)
        >>> explanations = wrapper.explain_local(x_data, feature_names)
    """

    def __init__(self, model: _PyGbmModelProto) -> None:
        """Initialize wrapper with a native ClearGBM model handle.

        Args:
            model: Native ``PyGbmModel`` returned by
                ``train_gradient_boosting``.

        Raises:
            ValueError: If the decoded model has no trees.
            TypeError: On any Rust JSON field-shape mismatch during decode.
        """
        json_str = _py_gbm_model_to_json(model)
        python_model = _decode_rust_json_to_python_model(json_str)
        self._shap_format = convert_cleargbm_to_shap_format(python_model)
        self._base_prediction = python_model["base_prediction"]

        # Dynamically import shap and create explainer
        shap_mod = __import__("shap")
        tree_explainer_cls: TreeExplainerConstructor = shap_mod.TreeExplainer
        self._explainer: ShapExplainerProtocol = tree_explainer_cls(self._shap_format)

    def explain_local(
        self,
        x: NDArray[np.float64],
        feature_names: list[str],
    ) -> list[LocalExplanation]:
        """Compute local SHAP values for the provided instances.

        Args:
            x: Feature matrix (n_samples, n_features).
            feature_names: List of feature names corresponding to columns.

        Returns:
            List of LocalExplanation dicts, one per sample.

        Raises:
            ValueError: If feature count doesn't match x columns.
        """
        n_features = int(x.shape[1])
        if n_features != len(feature_names):
            raise ValueError(f"Feature count mismatch: x={n_features}, names={len(feature_names)}")

        # Compute SHAP values
        raw_values = self._explainer.shap_values(x)

        # Handle different output formats
        # SHAP may return (n_samples, n_features) or list of arrays
        # Take positive class for binary classification if list
        values_array: NDArray[np.float64] = (
            raw_values[-1] if isinstance(raw_values, list) else raw_values
        )

        # Get expected value (base value)
        # SHAP returns array for our model format - convert to scalar
        ev = self._explainer.expected_value
        ev_array: NDArray[np.float64] = np.atleast_1d(np.asarray(ev, dtype=np.float64))
        # For binary classification, take last element (positive class)
        base_value: float = float(ev_array.flat[ev_array.size - 1])

        # Build results
        results: list[LocalExplanation] = []
        n_samples = int(values_array.shape[0])

        for i in range(n_samples):
            row_vals: NDArray[np.float64] = values_array[i]
            values_list: list[float] = [float(v) for v in row_vals.flat]

            explanation: LocalExplanation = {
                "base_value": base_value,
                "feature_names": feature_names,
                "values": values_list,
            }
            results.append(explanation)

        return results


# -----------------------------------------------------------------------------
# TypedDict for local explanations (matches platform_ml)
# -----------------------------------------------------------------------------


class LocalExplanation(TypedDict):
    """Type-safe container for a single instance's SHAP explanation.

    Args:
        base_value: Model's base prediction (expected value).
        feature_names: List of feature names.
        values: SHAP values for each feature.
    """

    base_value: float
    feature_names: list[str]
    values: list[float]


__all__ = [
    "ClearGBMShapWrapper",
    "LocalExplanation",
    "ShapModelFormat",
    "ShapTreeArrays",
    "convert_cleargbm_to_shap_format",
]
