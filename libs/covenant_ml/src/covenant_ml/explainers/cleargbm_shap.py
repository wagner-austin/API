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

from typing import Protocol, TypedDict

import numpy as np
from cleargbm.types import (
    DecisionTree,
    GradientBoostingModel,
    TreeNode,
)
from numpy.typing import NDArray

from covenant_ml.explainers.cleargbm_shap_decode import (
    _decode_rust_json_to_python_model,
    _py_gbm_model_to_json,
    _PyGbmModelProto,
)

# =============================================================================
# Native model access — Protocol-typed getattr onto the Rust extension
# =============================================================================


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
