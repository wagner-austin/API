//! Shared fixtures for the leaf-wise growth tests: the 8x2 dataset,
//! its binning, and the grow/predict drivers that
//! [`super::leafwise_tests`] and [`super::leafwise_error_tests`] use.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::predict::predict_tree;
use crate::tree::{
    build_tree, build_tree_leaf_wise_with_leaf_assignment, BuildTreeInput, TreeBuildConfig,
};
use crate::types::SplitConfig;

/// Sample count of the shared fixture's root node.
pub(super) const ROOT_SAMPLES: usize = 8_usize;

/// Eight samples over two features, both informative but at different
/// strengths, so the frontier holds several candidates with distinct gains and
/// the growth order is observable rather than forced.
pub(super) fn fixture_bins() -> Vec<u8> {
    // Column-major: feature 0 then feature 1, 8 samples each.
    vec![
        // feature 0: a clean 4/4 break
        0_u8, 0_u8, 1_u8, 1_u8, 2_u8, 2_u8, 3_u8, 3_u8, //
        // feature 1: a different partition
        0_u8, 1_u8, 2_u8, 3_u8, 0_u8, 1_u8, 2_u8, 3_u8,
    ]
}

/// Row-major features matching [`fixture_bins`], for prediction.
pub(super) fn fixture_rows() -> Vec<Vec<f64>> {
    vec![
        vec![0.1_f64, 0.1_f64],
        vec![0.1_f64, 0.4_f64],
        vec![0.4_f64, 0.6_f64],
        vec![0.4_f64, 0.9_f64],
        vec![0.6_f64, 0.1_f64],
        vec![0.6_f64, 0.4_f64],
        vec![0.9_f64, 0.6_f64],
        vec![0.9_f64, 0.9_f64],
    ]
}

pub(super) fn fixture_gradients() -> Vec<f64> {
    vec![
        1.0_f64, 0.9_f64, 0.6_f64, 0.4_f64, -0.4_f64, -0.6_f64, -0.9_f64, -1.0_f64,
    ]
}

pub(super) fn fixture_hessians() -> Vec<f64> {
    vec![1.0_f64; 8_usize]
}

pub(super) fn fixture_indices() -> Vec<u32> {
    vec![0_u32, 1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32]
}

pub(super) fn fixture_thresholds() -> Vec<Vec<f64>> {
    vec![
        vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64],
        vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64],
    ]
}

/// Builds a config with the given depth bound and leaf budget.
pub(super) fn make_config(
    max_depth: usize,
    max_leaves: usize,
) -> Result<TreeBuildConfig, ClearGbmError> {
    let split_config = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    TreeBuildConfig::new(max_depth, max_leaves, 0.0_f64, 0.0_f64, split_config)
}

/// Runs the leaf-wise builder over the shared fixture.
pub(super) fn grow_leaf_wise(
    config: &TreeBuildConfig,
) -> Result<(crate::tree::Tree, Vec<f64>), ClearGbmError> {
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    build_tree_leaf_wise_with_leaf_assignment(&input, &Hooks::default())
}

/// Runs the depth-wise builder over the same fixture.
pub(super) fn grow_depth_wise(
    config: &TreeBuildConfig,
) -> Result<crate::tree::Tree, ClearGbmError> {
    let sample_indices = fixture_indices();
    let gradients = fixture_gradients();
    let hessians = fixture_hessians();
    let bins = fixture_bins();
    let bin_thresholds = fixture_thresholds();

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 8_usize, 2_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 8_usize,
        n_features: 2_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    build_tree(&input, &Hooks::default())
}

/// Predicts every fixture row through a tree.
pub(super) fn predict_fixture(tree: &crate::tree::Tree) -> Result<Vec<f64>, ClearGbmError> {
    let rows = fixture_rows();
    let row_slices: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    predict_tree(tree, &row_slices)
}
