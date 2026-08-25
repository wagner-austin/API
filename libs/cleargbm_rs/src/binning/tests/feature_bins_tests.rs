//! Tests for FeatureBins and precompute_feature_bins.

use crate::binning::feature_bins::{precompute_feature_bins, FeatureBinning, FeatureBins};
use crate::error::ClearGbmError;

/// Returns a numeric feature's edges, failing the assertion if the feature
/// is categorical (these tests only bin numerically).
fn numeric_edges(fb: &FeatureBins, feature_index: usize) -> &[f64] {
    match &fb.per_feature()[feature_index] {
        FeatureBinning::Numeric(be) => be.edges(),
        FeatureBinning::Categorical(_) => &[],
    }
}

#[test]
fn test_precompute_min_data_in_bin_zero_is_refused() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![2.0_f64]];
    let rows: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    match precompute_feature_bins(&rows, 4_usize, 0_usize, None) {
        Ok(_) => Err(ClearGbmError::InvalidParameter {
            name: "min_data_in_bin".to_string(),
            reason: "a zero floor must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "min_data_in_bin");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_precompute_basic() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![
        vec![1.0_f64, 10.0_f64],
        vec![2.0_f64, 20.0_f64],
        vec![3.0_f64, 30.0_f64],
        vec![4.0_f64, 40.0_f64],
    ];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 4_usize, 1_usize, None) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    assert_eq!(fb.per_feature().len(), 2_usize);
    assert_eq!(fb.n_regular_bins(), 4_usize);
    assert_eq!(fb.n_samples(), 4_usize);
    assert_eq!(fb.n_features(), 2_usize);
    // Flat storage length = n_samples * n_features
    assert_eq!(fb.bins().len(), 8_usize);
    // Per-sample rows are contiguous n_features-long views
    assert_eq!(fb.bins_for_sample(0_usize).len(), 2_usize);
    assert_eq!(fb.bins_for_sample(3_usize).len(), 2_usize);
    Ok(())
}

#[test]
fn test_precompute_bin_thresholds_format() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![2.0_f64], vec![3.0_f64], vec![4.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 4_usize, 1_usize, None) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    let thresholds = fb.bin_thresholds();

    assert_eq!(thresholds.len(), 1_usize);
    assert_eq!(thresholds[0].len(), 4_usize);
    let actual_edge_count = numeric_edges(&fb, 0_usize).len();
    for (i, &t) in thresholds[0].iter().enumerate() {
        if i < actual_edge_count {
            assert!(t.is_finite());
        } else {
            assert_eq!(t, f64::INFINITY);
        }
    }
    Ok(())
}

#[test]
fn test_precompute_with_nan() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![f64::NAN], vec![3.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 4_usize, 1_usize, None) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    let nan_bin_u8 = match u8::try_from(fb.n_regular_bins()) {
        Ok(v) => v,
        Err(_) => {
            return Err(ClearGbmError::IntegerConversion {
                context: "nan_bin_u8 test setup".to_string(),
            })
        }
    };
    // Feature 0, sample 1 has NaN → NaN bin.
    assert_eq!(fb.bins_for_sample(1_usize)[0_usize], nan_bin_u8);
    // Samples 0 and 2 are non-NaN → regular bin (< n_regular_bins).
    assert!(fb.bins_for_sample(0_usize)[0_usize] < nan_bin_u8);
    assert!(fb.bins_for_sample(2_usize)[0_usize] < nan_bin_u8);
    Ok(())
}

#[test]
fn test_precompute_all_nan_feature() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![f64::NAN], vec![f64::NAN]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 4_usize, 1_usize, None) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    let nan_bin_u8 = match u8::try_from(fb.n_regular_bins()) {
        Ok(v) => v,
        Err(_) => {
            return Err(ClearGbmError::IntegerConversion {
                context: "nan_bin_u8 test setup".to_string(),
            })
        }
    };
    for sample_idx in 0_usize..fb.n_samples() {
        assert_eq!(fb.bins_for_sample(sample_idx)[0_usize], nan_bin_u8);
    }
    let thresholds = fb.bin_thresholds();
    for &t in &thresholds[0] {
        assert_eq!(t, f64::INFINITY);
    }
    Ok(())
}

#[test]
fn test_precompute_roundtrip_with_build_tree_input() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![
        vec![1.0_f64, 100.0_f64],
        vec![2.0_f64, 200.0_f64],
        vec![3.0_f64, 300.0_f64],
    ];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 8_usize, 1_usize, None) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    assert_eq!(fb.n_samples(), 3_usize);
    assert_eq!(fb.n_features(), 2_usize);
    let nan_bin_u8 = match u8::try_from(fb.n_regular_bins()) {
        Ok(v) => v,
        Err(_) => {
            return Err(ClearGbmError::IntegerConversion {
                context: "nan_bin_u8 test setup".to_string(),
            })
        }
    };
    // Every value is within `0..=n_regular_bins`
    for &b in fb.bins() {
        assert!(b <= nan_bin_u8);
    }

    let thresholds = fb.bin_thresholds();
    assert_eq!(thresholds.len(), 2_usize);
    for feat_thresholds in &thresholds {
        assert_eq!(feat_thresholds.len(), fb.n_regular_bins());
    }
    Ok(())
}

#[test]
fn test_precompute_max_bins_2() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![2.0_f64], vec![3.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 2_usize, 1_usize, None) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    assert_eq!(fb.n_regular_bins(), 2_usize);
    assert!(numeric_edges(&fb, 0_usize).len() <= 1_usize);
    Ok(())
}

#[test]
fn test_precompute_error_max_bins_1() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let result = precompute_feature_bins(&refs, 1_usize, 1_usize, None);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_precompute_error_max_bins_over_255() -> Result<(), ClearGbmError> {
    // The u8 bin-index invariant caps max_bins at 255.
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let result = precompute_feature_bins(&refs, 256_usize, 1_usize, None);
    assert!(matches!(
        result,
        Err(ClearGbmError::InvalidParameter { .. })
    ));
    Ok(())
}

#[test]
fn test_precompute_error_empty() -> Result<(), ClearGbmError> {
    let refs: Vec<&[f64]> = Vec::new();
    let result = precompute_feature_bins(&refs, 4_usize, 1_usize, None);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_bin_thresholds_padding_with_fewer_edges() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![0.0_f64], vec![0.0_f64], vec![1.0_f64], vec![1.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 8_usize, 1_usize, None) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    let thresholds = fb.bin_thresholds();
    assert_eq!(thresholds[0].len(), 8_usize);
    let n_actual_edges = numeric_edges(&fb, 0_usize).len();
    assert!(n_actual_edges >= 1_usize);
    assert!(thresholds[0][0].is_finite());
    for item in thresholds[0].iter().skip(n_actual_edges) {
        assert_eq!(*item, f64::INFINITY);
    }
    Ok(())
}

#[test]
fn test_bins_for_sample_out_of_range_returns_empty() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![2.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 4_usize, 1_usize, None) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    assert!(fb.bins_for_sample(999_usize).is_empty());
    Ok(())
}

#[test]
fn test_precompute_rejects_max_bins_above_u8_range() -> Result<(), ClearGbmError> {
    // Bin indices are packed into u8 for cache density. The training config
    // caps max_bins at 255, but `precompute_feature_bins` is public, so it
    // enforces the same ceiling itself rather than trusting the caller — a
    // truncated bin index would silently corrupt every downstream histogram.
    let data: Vec<Vec<f64>> = (0_usize..300_usize)
        .map(|i| {
            let value = f64::from(u32::try_from(i).unwrap_or(0_u32));
            vec![value]
        })
        .collect();
    let rows: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();

    match precompute_feature_bins(&rows, 300_usize, 1_usize, None) {
        Ok(_) => Err(ClearGbmError::InvalidParameter {
            name: "max_bins".to_string(),
            reason: "max_bins of 300 must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "max_bins");
            assert!(
                reason.contains("255"),
                "rejection should name the u8 ceiling, got: {reason}"
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

#[test]
fn test_precompute_accepts_the_maximum_supported_bin_count() -> Result<(), ClearGbmError> {
    // The boundary the training config enforces: 255 regular bins plus the
    // NaN bin at index 255 still fits in u8.
    let data: Vec<Vec<f64>> = (0_usize..300_usize)
        .map(|i| {
            let value = f64::from(u32::try_from(i).unwrap_or(0_u32));
            vec![value]
        })
        .collect();
    let rows: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();

    let bins = match precompute_feature_bins(&rows, 255_usize, 1_usize, None) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };
    assert_eq!(bins.n_samples(), 300_usize);
    assert_eq!(bins.n_features(), 1_usize);
    Ok(())
}
