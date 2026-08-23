//! Tests for [`super::super::categorical::CategoricalLayout`]: the bin ->
//! code translation node finalization depends on, including the loud
//! failures when the split search and layout disagree.

use crate::error::ClearGbmError;
use crate::split::CategoryBinSet;
use crate::tree::CategoricalLayout;

#[test]
fn test_layout_reports_category_counts() {
    let layout = CategoricalLayout::new(vec![
        None,
        Some(vec![0.0_f64, 3.0_f64, 9.0_f64]),
        Some(vec![1.0_f64]),
    ]);
    assert_eq!(layout.n_categories(0_usize), None);
    assert_eq!(layout.n_categories(1_usize), Some(3_usize));
    assert_eq!(layout.n_categories(2_usize), Some(1_usize));
    assert_eq!(layout.n_categories(9_usize), None);
}

#[test]
fn test_left_codes_translates_bins_in_code_order() -> Result<(), ClearGbmError> {
    let layout = CategoricalLayout::new(vec![Some(vec![0.0_f64, 3.0_f64, 9.0_f64])]);
    let mut set = CategoryBinSet::new();
    set.insert(2_usize);
    set.insert(0_usize);
    let codes = propagate!(layout.left_codes(0_usize, set));
    assert_eq!(codes, vec![0.0_f64, 9.0_f64]);
    Ok(())
}

#[test]
fn test_left_codes_rejects_a_numeric_feature() -> Result<(), ClearGbmError> {
    let layout = CategoricalLayout::new(vec![None]);
    let mut set = CategoryBinSet::new();
    set.insert(0_usize);
    match layout.left_codes(0_usize, set) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a numeric feature must be rejected".to_string(),
        }),
        Err(ClearGbmError::TreeConstructionFailed { reason }) => {
            assert!(reason.contains("not mark as categorical"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_left_codes_rejects_a_bin_beyond_the_code_table() -> Result<(), ClearGbmError> {
    let layout = CategoricalLayout::new(vec![Some(vec![0.0_f64, 3.0_f64])]);
    let mut set = CategoryBinSet::new();
    set.insert(5_usize);
    match layout.left_codes(0_usize, set) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an out-of-table bin must be rejected".to_string(),
        }),
        Err(ClearGbmError::TreeConstructionFailed { reason }) => {
            assert!(reason.contains("only 2 categories"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_dispatcher_propagates_a_categorical_search_error() -> Result<(), ClearGbmError> {
    // A layout claiming more categories than the regular-bin budget makes
    // the categorical finder error; the dispatcher must surface it intact.
    use crate::types::{HistogramBuffer, SplitConfig};

    let histograms = vec![HistogramBuffer::new(3_usize)];
    let config = propagate!(SplitConfig::new(
        2_usize, 1_usize, 8_usize, 0.0_f64, 0.0_f64
    ));
    let layout = CategoricalLayout::new(vec![Some(vec![0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64])]);
    match super::super::histograms::find_best_split_across_features_internal(
        &histograms,
        &config,
        2_usize,
        None,
        None,
        Some(&layout),
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an oversized category table must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_categories");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_finalize_rejects_a_categorical_node_without_a_layout() -> Result<(), ClearGbmError> {
    use crate::hooks::Hooks;
    use crate::split::SplitDecision;
    use crate::tree::nodes::{finalize_nodes, BuildNode};

    let mut left_bins = CategoryBinSet::new();
    left_bins.insert(0_usize);
    let build_nodes = vec![BuildNode {
        node_id: 0_usize,
        is_leaf: false,
        feature_index: Some(0_usize),
        decision: Some(SplitDecision::CategorySubset { left_bins }),
        value: 0.0_f64,
        n_samples: 4_usize,
        nan_goes_left: true,
    }];
    let child_pointers = vec![(Some(1_usize), Some(2_usize))];
    let bin_thresholds: Vec<Vec<f64>> = vec![Vec::new()];
    match finalize_nodes(
        &build_nodes,
        &child_pointers,
        &bin_thresholds,
        None,
        &Hooks::default(),
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a categorical node without a layout must be rejected".to_string(),
        }),
        Err(ClearGbmError::TreeConstructionFailed { reason }) => {
            assert!(reason.contains("no categorical"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_finalize_propagates_a_layout_disagreement() -> Result<(), ClearGbmError> {
    use crate::hooks::Hooks;
    use crate::split::SplitDecision;
    use crate::tree::nodes::{finalize_nodes, BuildNode};

    let mut left_bins = CategoryBinSet::new();
    left_bins.insert(0_usize);
    let build_nodes = vec![BuildNode {
        node_id: 0_usize,
        is_leaf: false,
        feature_index: Some(0_usize),
        decision: Some(SplitDecision::CategorySubset { left_bins }),
        value: 0.0_f64,
        n_samples: 4_usize,
        nan_goes_left: true,
    }];
    let child_pointers = vec![(Some(1_usize), Some(2_usize))];
    let bin_thresholds: Vec<Vec<f64>> = vec![Vec::new()];
    // The layout marks feature 0 numeric: left_codes must refuse and
    // finalize must propagate its error.
    let layout = CategoricalLayout::new(vec![None]);
    match finalize_nodes(
        &build_nodes,
        &child_pointers,
        &bin_thresholds,
        Some(&layout),
        &Hooks::default(),
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a layout disagreement must be rejected".to_string(),
        }),
        Err(ClearGbmError::TreeConstructionFailed { reason }) => {
            assert!(reason.contains("not mark as categorical"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}
