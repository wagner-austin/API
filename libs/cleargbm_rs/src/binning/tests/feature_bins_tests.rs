//! Tests for FeatureBins and precompute_feature_bins.

use crate::binning::feature_bins::precompute_feature_bins;
use crate::error::ClearGbmError;

#[test]
fn test_precompute_basic() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![
        vec![1.0_f64, 10.0_f64],
        vec![2.0_f64, 20.0_f64],
        vec![3.0_f64, 30.0_f64],
        vec![4.0_f64, 40.0_f64],
    ];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 4_usize) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    assert_eq!(fb.bin_edges().len(), 2_usize);
    assert_eq!(fb.n_regular_bins(), 4_usize);
    assert_eq!(fb.sample_bins().len(), 4_usize);
    for row in fb.sample_bins() {
        assert_eq!(row.len(), 2_usize);
    }
    Ok(())
}

#[test]
fn test_precompute_bin_thresholds_format() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![2.0_f64], vec![3.0_f64], vec![4.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 4_usize) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    let thresholds = fb.bin_thresholds();

    assert_eq!(thresholds.len(), 1_usize);
    assert_eq!(thresholds[0].len(), 4_usize);
    let actual_edge_count = fb.bin_edges()[0].edges().len();
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
    let fb = match precompute_feature_bins(&refs, 4_usize) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    assert_eq!(fb.sample_bins()[1][0], fb.n_regular_bins());
    assert!(fb.sample_bins()[0][0] < fb.n_regular_bins());
    assert!(fb.sample_bins()[2][0] < fb.n_regular_bins());
    Ok(())
}

#[test]
fn test_precompute_all_nan_feature() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![f64::NAN], vec![f64::NAN]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 4_usize) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    for row in fb.sample_bins() {
        assert_eq!(row[0], fb.n_regular_bins());
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
    let fb = match precompute_feature_bins(&refs, 8_usize) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    assert_eq!(fb.sample_bins().len(), 3_usize);
    for row in fb.sample_bins() {
        assert_eq!(row.len(), 2_usize);
        for &bin in row {
            assert!(bin <= fb.n_regular_bins());
        }
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
    let fb = match precompute_feature_bins(&refs, 2_usize) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    assert_eq!(fb.n_regular_bins(), 2_usize);
    assert!(fb.bin_edges()[0].edges().len() <= 1_usize);
    Ok(())
}

#[test]
fn test_precompute_error_max_bins_1() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let result = precompute_feature_bins(&refs, 1_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_precompute_error_empty() -> Result<(), ClearGbmError> {
    let refs: Vec<&[f64]> = Vec::new();
    let result = precompute_feature_bins(&refs, 4_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_bin_thresholds_padding_with_fewer_edges() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![0.0_f64], vec![0.0_f64], vec![1.0_f64], vec![1.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let fb = match precompute_feature_bins(&refs, 8_usize) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };

    let thresholds = fb.bin_thresholds();
    assert_eq!(thresholds[0].len(), 8_usize);
    let n_actual_edges = fb.bin_edges()[0].edges().len();
    assert!(n_actual_edges >= 1_usize);
    assert!(thresholds[0][0].is_finite());
    for item in thresholds[0].iter().skip(n_actual_edges) {
        assert_eq!(*item, f64::INFINITY);
    }
    Ok(())
}
