//! Tests for `build_tree` behavior on well-formed inputs: structure,
//! determinism, constraints and stopping conditions.

use crate::error::ClearGbmError;
use crate::hooks::Hooks;
use crate::tree::{build_tree, BuildTreeInput, TreeBuildConfig};
use crate::types::SplitConfig;

// =========================================================================
// build_tree tests
// =========================================================================

#[test]
fn test_build_tree_single_leaf() -> Result<(), ClearGbmError> {
    // Create simple data that results in a single leaf
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(1_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![0.1_f64, 0.1_f64, 0.1_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    // 3 samples, 1 feature, all bin 0 (column-major flat).
    let bins: Vec<u8> = vec![0_u8, 0_u8, 0_u8];
    let bin_thresholds = vec![vec![0.5_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 3_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 3_usize,
        n_features: 1_usize,
        n_regular_bins: 1_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Should be a single leaf (max_depth = 1, samples in same bin)
    assert_eq!(tree.n_leaves(), 1_usize);
    let _ = match tree.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    Ok(())
}

#[test]
fn test_build_tree_with_split() -> Result<(), ClearGbmError> {
    // Create data with clear split
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32];
    // Left samples (bins 0,1) have positive gradients
    // Right samples (bins 2,3) have negative gradients
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    // 4 samples, 1 feature, column-major flat.
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 4_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Should have split
    assert!(tree.n_nodes() >= 3_usize);
    assert!(tree.n_leaves() >= 2_usize);

    // Root should not be a leaf
    let root = match tree.root() {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    assert!(!root.is_leaf());
    Ok(())
}

#[test]
fn test_build_tree_empty_input() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices: Vec<u32> = vec![];
    let gradients: Vec<f64> = vec![];
    let hessians: Vec<f64> = vec![];
    let bins: Vec<u8> = vec![];
    let bin_thresholds: Vec<Vec<f64>> = vec![];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 0_usize, 4_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 0_usize,
        n_features: 4_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let result = build_tree(&input, &Hooks::default());
    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::EmptyInput { .. })
    ));
    Ok(())
}

#[test]
fn test_build_tree_max_depth_constraint() -> Result<(), ClearGbmError> {
    // max_depth = 1 should create root + 2 leaves max
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(1_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 4_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    // Max depth = 1, so max 3 nodes (root + 2 leaves)
    assert!(tree.n_nodes() <= 3_usize);
    assert!(tree.max_depth() <= 1_usize);
    Ok(())
}

#[test]
fn test_build_tree_max_leaves_constraint() -> Result<(), ClearGbmError> {
    // max_leaves = 2
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(10_usize, 2_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32, 3_u32];
    let gradients = vec![1.0_f64, 1.0_f64, -1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8, 3_u8];
    let bin_thresholds = vec![vec![0.25_f64, 0.5_f64, 0.75_f64, 1.0_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 4_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 4_usize,
        n_features: 1_usize,
        n_regular_bins: 4_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let tree = match build_tree(&input, &Hooks::default()) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    assert!(tree.n_leaves() <= 2_usize);
    Ok(())
}

#[test]
fn test_build_tree_gradients_too_short() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![1.0_f64]; // Too short!
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8];
    let bin_thresholds = vec![vec![0.5_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 3_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 3_usize,
        n_features: 1_usize,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let result = build_tree(&input, &Hooks::default());
    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::ShapeMismatch { .. })
    ));
    Ok(())
}

#[test]
fn test_build_tree_hessians_too_short() -> Result<(), ClearGbmError> {
    let sc = match SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    let cfg = match TreeBuildConfig::new(3_usize, 0_usize, 0.0_f64, 0.0_f64, sc) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };

    let sample_indices = vec![0_u32, 1_u32, 2_u32];
    let gradients = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let hessians = vec![1.0_f64]; // Too short!
    let bins: Vec<u8> = vec![0_u8, 1_u8, 2_u8];
    let bin_thresholds = vec![vec![0.5_f64]];

    let bins_rows = crate::testkit::binning::transpose_cols_to_rows(&bins, 3_usize, 1_usize);
    let input = BuildTreeInput {
        sample_indices: &sample_indices,
        gradients: &gradients,
        hessians: &hessians,
        bins_rows: &bins_rows,
        n_samples: 3_usize,
        n_features: 1_usize,
        n_regular_bins: 3_usize,
        bin_thresholds: &bin_thresholds,
        config: &cfg,
        monotonic_constraints: None,
        feature_subsample: None,
        tree_feature_mask: None,
    };

    let result = build_tree(&input, &Hooks::default());
    assert!(result.is_err());
    assert!(matches!(
        result.err(),
        Some(ClearGbmError::ShapeMismatch { .. })
    ));
    Ok(())
}
