//! Shared pre-boosting setup for both trainers.
//!
//! The single-score and multiclass trainers consume identical preparation:
//! feature-count-dependent config validation, categorical resolution,
//! binning, and the tree-build configuration. One implementation here
//! keeps the two boosting loops free of duplicated setup that could
//! drift.

use crate::binning::{precompute_feature_bins, FeatureBinning, FeatureBins};
use crate::error::ClearGbmError;
use crate::tree::{tree_column_budget, CategoricalLayout, TreeBuildConfig};
use crate::types::SplitConfig;

use super::config::GradientBoostingConfig;

/// Everything the boosting loops need that is computed once per run.
pub(super) struct PreparedTraining {
    /// The binning result: per-feature binning + flat bin assignments.
    pub feature_bins: FeatureBins,
    /// Per-feature threshold table for finalizing threshold splits.
    pub bin_thresholds: Vec<Vec<f64>>,
    /// Per-feature category tables, when the categorical axis is on.
    pub categorical_layout: Option<CategoricalLayout>,
    /// The validated tree-build configuration.
    pub tree_build_config: TreeBuildConfig,
    /// The per-tree column budget from `colsample_bytree`, when set.
    pub tree_column_budget: Option<usize>,
}

/// Runs the feature-count-dependent validations and the one-time
/// preparation shared by both trainers.
///
/// # Args
///
/// * `x_train` - Training feature matrix, already shape-validated.
/// * `n_features` - The validated feature count.
/// * `config` - Training hyperparameters.
///
/// # Errors
///
/// * `ClearGbmError::ShapeMismatch` if the monotonic-constraint count
///   disagrees with the feature count.
/// * `ClearGbmError::InvalidParameter` if `max_features` exceeds the
///   feature count, a categorical index is out of range or carries a
///   monotonic constraint, or a categorical column fails binning
///   validation.
pub(super) fn prepare_training(
    x_train: &[&[f64]],
    n_features: usize,
    config: &GradientBoostingConfig,
) -> Result<PreparedTraining, ClearGbmError> {
    // Validate monotonic constraints length if provided.
    if let Some(mc) = config.monotonic_constraints() {
        if mc.len() != n_features {
            return Err(ClearGbmError::ShapeMismatch {
                expected: format!("{n_features} monotonic constraints"),
                got: format!("{} monotonic constraints", mc.len()),
            });
        }
    }

    // Validate the per-split feature budget against the feature count
    // (the config layer cannot: it does not know n_features).
    if let Some(k) = config.max_features() {
        if k > n_features {
            return Err(ClearGbmError::InvalidParameter {
                name: "max_features".to_string(),
                reason: format!("must be <= n_features ({n_features}), got {k}"),
            });
        }
    }

    // Resolve the per-tree column budget: k_tree = max(1,
    // floor(colsample_bytree * n_features)), the row-subsampling
    // convention. The count lives on [1, n_features] by construction (the
    // fraction is validated in (0, 1) exclusive).
    let column_budget: Option<usize> = match config.colsample_bytree() {
        Some(fraction) => Some(propagate!(tree_column_budget(fraction, n_features))),
        None => None,
    };

    // Resolve the categorical axis, then precompute feature bins. The
    // config lists categorical feature indices; binning needs a
    // per-feature flag, and the bounds plus the no-monotonic-constraint
    // pairing are checked here, where n_features is known.
    let categorical_mask: Option<Vec<bool>> = match config.categorical_features() {
        Some(indices) => {
            let mut mask = vec![false; n_features];
            for &idx in indices {
                if idx >= n_features {
                    return Err(ClearGbmError::InvalidParameter {
                        name: "categorical_features".to_string(),
                        reason: format!("index {idx} is out of range for {n_features} features"),
                    });
                }
                if let Some(mc) = config.monotonic_constraints() {
                    let constrained = mc.get(idx).copied().is_some_and(|c| !c.is_none());
                    if constrained {
                        return Err(ClearGbmError::InvalidParameter {
                            name: "categorical_features".to_string(),
                            reason: format!(
                                "feature {idx} is categorical but carries a monotonic \
                                 constraint; category codes have no order to constrain"
                            ),
                        });
                    }
                }
                mask[idx] = true;
            }
            Some(mask)
        }
        None => None,
    };
    let feature_bins = propagate!(precompute_feature_bins(
        x_train,
        config.max_bins(),
        categorical_mask.as_deref()
    ));
    let bin_thresholds = feature_bins.bin_thresholds();

    // The per-feature category tables the tree layer consults: which
    // features are categorical (split search) and the bin -> code mapping
    // (node finalization). None when the axis is off, which keeps every
    // numeric-only run bit-identical to history.
    let categorical_layout: Option<CategoricalLayout> = categorical_mask.as_ref().map(|_| {
        CategoricalLayout::new(
            feature_bins
                .per_feature()
                .iter()
                .map(|binning| match binning {
                    FeatureBinning::Categorical(map) => Some(map.codes().to_vec()),
                    FeatureBinning::Numeric(_) => None,
                })
                .collect(),
        )
    });

    let split_config = propagate!(SplitConfig::new(
        config.min_samples_split(),
        config.min_samples_leaf(),
        config.max_bins(),
        config.reg_lambda(),
        0.0_f64,
    ));

    // Under depth-wise growth the leaf count is left unbounded (0) and
    // `max_depth` does the bounding, which is what every manifest recorded
    // before the growth axis existed. Under leaf-wise there is no depth to
    // bound the shape, so the validated `num_leaves` becomes the budget.
    let max_leaves = config.num_leaves().unwrap_or_default();

    let tree_build_config = propagate!(TreeBuildConfig::new(
        config.max_depth(),
        max_leaves,
        config.reg_alpha(),
        config.reg_lambda(),
        split_config,
    ));

    Ok(PreparedTraining {
        feature_bins,
        bin_thresholds,
        categorical_layout,
        tree_build_config,
        tree_column_budget: column_budget,
    })
}
