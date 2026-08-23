//! Tests for categorical binning: one bin per distinct code, strict
//! validation, and the shared NaN bin for missing values.

use crate::binning::feature_bins::{precompute_feature_bins, FeatureBinning, FeatureBins};
use crate::error::ClearGbmError;

/// Extracts a categorical feature's code table, failing on a numeric one.
fn category_codes(fb: &FeatureBins, feature_index: usize) -> &[f64] {
    match &fb.per_feature()[feature_index] {
        FeatureBinning::Categorical(map) => map.codes(),
        FeatureBinning::Numeric(_) => &[],
    }
}

#[test]
fn test_each_distinct_code_gets_its_own_bin_ascending() -> Result<(), ClearGbmError> {
    // Codes arrive out of order; bins follow ascending code order, and each
    // row's bin points back at its code.
    let rows: Vec<Vec<f64>> = vec![
        vec![7.0_f64, 0.5_f64],
        vec![2.0_f64, 0.6_f64],
        vec![7.0_f64, 0.7_f64],
        vec![0.0_f64, 0.8_f64],
    ];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true, false];
    let fb = propagate!(precompute_feature_bins(&refs, 4_usize, Some(&mask)));

    assert_eq!(category_codes(&fb, 0_usize), &[0.0_f64, 2.0_f64, 7.0_f64]);
    match &fb.per_feature()[0] {
        FeatureBinning::Categorical(map) => assert_eq!(map.n_categories(), 3_usize),
        FeatureBinning::Numeric(_) => assert!(category_codes(&fb, 0_usize).is_empty()),
    }
    // Row bins for feature 0: 7 -> bin 2, 2 -> bin 1, 7 -> bin 2, 0 -> bin 0.
    let bins = fb.bins();
    let n_features = fb.n_features();
    let feature0: Vec<u8> = (0_usize..4_usize).map(|s| bins[s * n_features]).collect();
    assert_eq!(feature0, vec![2_u8, 1_u8, 2_u8, 0_u8]);
    // Feature 1 stayed numeric.
    assert!(matches!(fb.per_feature()[1], FeatureBinning::Numeric(_)));
    Ok(())
}

#[test]
fn test_missing_values_keep_the_nan_bin() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![f64::NAN], vec![3.0_f64]];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true];
    let fb = propagate!(precompute_feature_bins(&refs, 4_usize, Some(&mask)));
    // NaN bin sits at n_regular_bins = 4.
    assert_eq!(fb.bins(), &[0_u8, 4_u8, 1_u8]);
    Ok(())
}

#[test]
fn test_negative_zero_normalizes_into_the_zero_bin() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![vec![-0.0_f64], vec![0.0_f64], vec![1.0_f64]];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true];
    let fb = propagate!(precompute_feature_bins(&refs, 4_usize, Some(&mask)));
    assert_eq!(category_codes(&fb, 0_usize), &[0.0_f64, 1.0_f64]);
    assert_eq!(fb.bins(), &[0_u8, 0_u8, 1_u8]);
    Ok(())
}

#[test]
fn test_non_integer_code_is_rejected_naming_feature_and_row() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![2.5_f64]];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true];
    match precompute_feature_bins(&refs, 4_usize, Some(&mask)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a non-integer categorical value must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "categorical_features");
            assert!(reason.contains("feature 0 row 1"), "got: {reason}");
            assert!(reason.contains("2.5"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_negative_code_is_rejected() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![-3.0_f64]];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true];
    match precompute_feature_bins(&refs, 4_usize, Some(&mask)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a negative categorical value must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "categorical_features");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_infinite_code_is_rejected() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![f64::INFINITY]];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true];
    match precompute_feature_bins(&refs, 4_usize, Some(&mask)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an infinite categorical value must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "categorical_features");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_too_many_categories_is_rejected_not_grouped() -> Result<(), ClearGbmError> {
    // 5 distinct codes against max_bins 4: no silent rare-category
    // grouping exists, the error names both counts.
    let rows: Vec<Vec<f64>> = (0_usize..5_usize)
        .map(|i| {
            let mut code = 0.0_f64;
            for _ in 0_usize..i {
                code += 1.0_f64;
            }
            vec![code]
        })
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true];
    match precompute_feature_bins(&refs, 4_usize, Some(&mask)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an over-budget category count must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "categorical_features");
            assert!(reason.contains("5 distinct categories"), "got: {reason}");
            assert!(reason.contains("max_bins (4)"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_mask_length_mismatch_is_rejected() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![vec![1.0_f64, 2.0_f64]];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true];
    match precompute_feature_bins(&refs, 4_usize, Some(&mask)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a short categorical mask must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "categorical_features");
            assert!(reason.contains("covers 1 features"), "got: {reason}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_code_beyond_u32_is_rejected() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![4_294_967_296.0_f64]];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let mask = vec![true];
    match precompute_feature_bins(&refs, 4_usize, Some(&mask)) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a code beyond u32::MAX must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "categorical_features");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_precompute_rejects_an_empty_matrix() -> Result<(), ClearGbmError> {
    let refs: Vec<&[f64]> = Vec::new();
    match precompute_feature_bins(&refs, 4_usize, None) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "an empty matrix must be rejected".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_precompute_rejects_ragged_rows() -> Result<(), ClearGbmError> {
    let rows: Vec<Vec<f64>> = vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64]];
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    match precompute_feature_bins(&refs, 4_usize, None) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "ragged rows must be rejected".to_string(),
        }),
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn test_precompute_rejects_zero_features() -> Result<(), ClearGbmError> {
    let row: Vec<f64> = Vec::new();
    let refs: Vec<&[f64]> = vec![row.as_slice()];
    match precompute_feature_bins(&refs, 4_usize, None) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a zero-feature matrix must be rejected".to_string(),
        }),
        Err(ClearGbmError::EmptyInput { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}
