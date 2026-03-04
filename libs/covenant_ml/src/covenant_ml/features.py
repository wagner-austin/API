"""Feature engineering for bankruptcy prediction models.

Provides functions to create derived features from raw financial ratios:
- Pairwise ratios: Xi/Xj for capturing relative relationships
- Pairwise products: Xi*Xj for interaction effects
- Log transforms: log(1 + |x|) * sign(x) for handling skewed distributions
"""

from __future__ import annotations

import math
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray


def _int_sqrt(value: int) -> int:
    """Return integer square root (floor)."""
    return int(math.sqrt(value))


def _get_index_value(arr: NDArray[np.intp], idx: int) -> int:
    """Get integer value from index array."""
    for i, val in enumerate(arr.flat):
        if i == idx:
            return int(val)
    raise IndexError(f"Index {idx} out of bounds for array of size {arr.size}")


class FeatureEngineeringConfig(TypedDict):
    """Configuration for feature engineering transforms."""

    use_ratios: bool
    use_products: bool
    use_log_transforms: bool
    use_temporal: bool
    max_ratio_features: int  # Limit to avoid explosion (0 = no limit)
    max_product_features: int  # Limit to avoid explosion (0 = no limit)


class EngineeredFeatures(TypedDict):
    """Result of feature engineering."""

    x: NDArray[np.float64]
    feature_names: list[str]
    n_original: int
    n_ratios: int
    n_products: int
    n_log: int
    n_temporal: int


def default_feature_config() -> FeatureEngineeringConfig:
    """Return default feature engineering configuration."""
    return {
        "use_ratios": True,
        "use_products": False,  # Products can cause multicollinearity
        "use_log_transforms": True,
        "use_temporal": False,
        "max_ratio_features": 500,  # Limit ratio features
        "max_product_features": 200,  # Limit product features
    }


def compute_pairwise_ratios(
    x: NDArray[np.float64],
    feature_names: list[str],
    max_features: int = 0,
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute pairwise ratios Xi/Xj for all feature pairs.

    Args:
        x: Feature matrix (n_samples, n_features)
        feature_names: Names of original features
        max_features: Maximum number of ratio features (0 = no limit)

    Returns:
        Tuple of (ratio_features, ratio_names)
    """
    n_samples: int = int(x.shape[0])
    n_features: int = int(x.shape[1])

    # Calculate number of possible pairs
    n_pairs: int = n_features * (n_features - 1)  # Xi/Xj and Xj/Xi are different

    if max_features > 0 and n_pairs > max_features:
        # Select top pairs based on variance of original features
        variances: NDArray[np.float64] = np.var(x, axis=0)
        limit: int = _int_sqrt(max_features * 2)
        top_indices: NDArray[np.intp] = np.argsort(variances)[::-1][:limit]
        selected_features: int = len(top_indices)
    else:
        top_indices = np.arange(n_features)
        selected_features = n_features

    # Compute ratios only for selected features
    ratio_list: list[NDArray[np.float64]] = []
    ratio_names: list[str] = []

    for i in range(selected_features):
        for j in range(selected_features):
            if i != j:
                idx_i: int = _get_index_value(top_indices, i)
                idx_j: int = _get_index_value(top_indices, j)

                # Safe division with small epsilon to avoid division by zero
                denominator: NDArray[np.float64] = x[:, idx_j].copy()
                abs_denom: NDArray[np.float64] = np.abs(denominator)
                small_mask: NDArray[np.bool_] = abs_denom < 1e-10
                denominator[small_mask] = 1e-10
                ratio: NDArray[np.float64] = x[:, idx_i] / denominator

                # Clip extreme values
                ratio = np.clip(ratio, -1e6, 1e6)

                # Replace inf/nan with 0
                ratio = np.nan_to_num(ratio, nan=0.0, posinf=1e6, neginf=-1e6)

                ratio_list.append(ratio.reshape(-1, 1))
                ratio_names.append(f"{feature_names[idx_i]}/{feature_names[idx_j]}")

                if max_features > 0 and len(ratio_list) >= max_features:
                    break
        if max_features > 0 and len(ratio_list) >= max_features:
            break

    if not ratio_list:
        return np.zeros((n_samples, 0), dtype=np.float64), []

    ratios: NDArray[np.float64] = np.hstack(ratio_list)
    return ratios.astype(np.float64), ratio_names


def compute_pairwise_products(
    x: NDArray[np.float64],
    feature_names: list[str],
    max_features: int = 0,
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute pairwise products Xi*Xj for feature pairs.

    Args:
        x: Feature matrix (n_samples, n_features)
        feature_names: Names of original features
        max_features: Maximum number of product features (0 = no limit)

    Returns:
        Tuple of (product_features, product_names)
    """
    n_samples: int = int(x.shape[0])
    n_features: int = int(x.shape[1])

    # Calculate number of possible pairs (Xi*Xj = Xj*Xi, so n*(n-1)/2)
    n_pairs: int = n_features * (n_features - 1) // 2

    if max_features > 0 and n_pairs > max_features:
        # Select top features based on variance
        variances: NDArray[np.float64] = np.var(x, axis=0)
        limit: int = _int_sqrt(max_features * 2)
        top_indices: NDArray[np.intp] = np.argsort(variances)[::-1][:limit]
        selected_features: int = len(top_indices)
    else:
        top_indices = np.arange(n_features)
        selected_features = n_features

    product_list: list[NDArray[np.float64]] = []
    product_names: list[str] = []

    for i in range(selected_features):
        for j in range(i + 1, selected_features):
            idx_i: int = _get_index_value(top_indices, i)
            idx_j: int = _get_index_value(top_indices, j)

            product: NDArray[np.float64] = x[:, idx_i] * x[:, idx_j]

            # Clip extreme values
            product = np.clip(product, -1e12, 1e12)
            product = np.nan_to_num(product, nan=0.0, posinf=1e12, neginf=-1e12)

            product_list.append(product.reshape(-1, 1))
            product_names.append(f"{feature_names[idx_i]}*{feature_names[idx_j]}")

            if max_features > 0 and len(product_list) >= max_features:
                break
        if max_features > 0 and len(product_list) >= max_features:
            break

    if not product_list:
        return np.zeros((n_samples, 0), dtype=np.float64), []

    products: NDArray[np.float64] = np.hstack(product_list)
    return products.astype(np.float64), product_names


def compute_log_transforms(
    x: NDArray[np.float64],
    feature_names: list[str],
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute log transforms for all features.

    Uses signed log: sign(x) * log(1 + |x|) to handle negative values.

    Args:
        x: Feature matrix (n_samples, n_features)
        feature_names: Names of original features

    Returns:
        Tuple of (log_features, log_names)
    """
    # Signed log transform: sign(x) * log(1 + |x|)
    signs: NDArray[np.float64] = np.sign(x)
    abs_x: NDArray[np.float64] = np.abs(x)
    log_abs: NDArray[np.float64] = np.log1p(abs_x)
    log_values: NDArray[np.float64] = signs * log_abs

    # Handle any remaining inf/nan
    log_values = np.nan_to_num(log_values, nan=0.0, posinf=20.0, neginf=-20.0)

    log_names = [f"log({name})" for name in feature_names]

    return log_values.astype(np.float64), log_names


def engineer_features(
    x: NDArray[np.float64],
    feature_names: list[str],
    config: FeatureEngineeringConfig,
    temporal_features: NDArray[np.float64] | None = None,
    temporal_feature_names: tuple[str, ...] = (),
) -> EngineeredFeatures:
    """Apply feature engineering transforms based on configuration.

    Temporal features are computed upstream (via fit/transform pattern) and
    passed in as pre-computed columns. The ``use_temporal`` flag in *config*
    controls whether they are appended to the output.

    Args:
        x: Original feature matrix (n_samples, n_features).
        feature_names: Names of original features.
        config: Feature engineering configuration.
        temporal_features: Pre-computed temporal feature matrix
            (n_samples, n_temporal). Required when ``use_temporal`` is True.
        temporal_feature_names: Ordered names for temporal feature columns.
            Required when ``use_temporal`` is True.

    Returns:
        EngineeredFeatures with combined feature matrix and metadata.

    Raises:
        ValueError: If ``use_temporal`` is True but *temporal_features* or
            *temporal_feature_names* is missing, or if the sample count does
            not match.
    """
    n_original: int = int(x.shape[1])
    n_samples: int = int(x.shape[0])

    # Start with original features
    all_features: list[NDArray[np.float64]] = [x]
    all_names: list[str] = list(feature_names)

    n_ratios = 0
    n_products = 0
    n_log = 0
    n_temporal = 0

    # Add ratio features
    if config["use_ratios"]:
        ratios, ratio_names = compute_pairwise_ratios(
            x, feature_names, config["max_ratio_features"]
        )
        n_ratio_cols: int = int(ratios.shape[1])
        if n_ratio_cols > 0:
            all_features.append(ratios)
            all_names.extend(ratio_names)
            n_ratios = len(ratio_names)

    # Add product features
    if config["use_products"]:
        products, product_names = compute_pairwise_products(
            x, feature_names, config["max_product_features"]
        )
        n_product_cols: int = int(products.shape[1])
        if n_product_cols > 0:
            all_features.append(products)
            all_names.extend(product_names)
            n_products = len(product_names)

    # Add log transforms
    if config["use_log_transforms"]:
        logs, log_names = compute_log_transforms(x, feature_names)
        all_features.append(logs)
        all_names.extend(log_names)
        n_log = len(log_names)

    # Add temporal features
    if config["use_temporal"]:
        if temporal_features is None:
            raise ValueError("use_temporal is True but temporal_features was not provided")
        if len(temporal_feature_names) == 0:
            raise ValueError("use_temporal is True but temporal_feature_names is empty")
        temporal_samples: int = int(temporal_features.shape[0])
        if temporal_samples != n_samples:
            raise ValueError(
                f"temporal_features has {temporal_samples} samples but x has {n_samples}"
            )
        all_features.append(temporal_features)
        all_names.extend(temporal_feature_names)
        n_temporal = len(temporal_feature_names)

    # Combine all features
    combined: NDArray[np.float64] = np.hstack(all_features)

    return {
        "x": combined.astype(np.float64),
        "feature_names": all_names,
        "n_original": n_original,
        "n_ratios": n_ratios,
        "n_products": n_products,
        "n_log": n_log,
        "n_temporal": n_temporal,
    }


# Feature engineering presets for Optuna
FeaturePreset = Literal["none", "log_only", "ratios_only", "full", "temporal"]


def get_feature_config_for_preset(preset: FeaturePreset) -> FeatureEngineeringConfig:
    """Get feature engineering config for a named preset.

    Args:
        preset: One of "none", "log_only", "ratios_only", "full", "temporal".

    Returns:
        FeatureEngineeringConfig for the preset.
    """
    if preset == "none":
        return {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": False,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }
    if preset == "log_only":
        return {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": True,
            "use_temporal": False,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }
    if preset == "ratios_only":
        return {
            "use_ratios": True,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": False,
            "max_ratio_features": 500,
            "max_product_features": 0,
        }
    if preset == "temporal":
        return {
            "use_ratios": False,
            "use_products": False,
            "use_log_transforms": False,
            "use_temporal": True,
            "max_ratio_features": 0,
            "max_product_features": 0,
        }
    # "full"
    return {
        "use_ratios": True,
        "use_products": True,
        "use_log_transforms": True,
        "use_temporal": False,
        "max_ratio_features": 500,
        "max_product_features": 200,
    }
