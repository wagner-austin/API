//! Tests for sample bin assignment.

use crate::binning::assignment::bin_samples;
use crate::binning::edges::BinEdges;
use crate::error::ClearGbmError;

/// Helper to create BinEdges from a vec.
fn make_edges(vals: Vec<f64>) -> Result<BinEdges, ClearGbmError> {
    BinEdges::new(vals)
}

#[test]
fn test_bin_samples_basic() -> Result<(), ClearGbmError> {
    // edges = [2.0, 4.0] → 3 regular bins: [≤2], (2,4], (>4)
    let edges = vec![match make_edges(vec![2.0_f64, 4.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let data: Vec<Vec<f64>> = vec![
        vec![1.0_f64], // bin 0 (≤ 2.0)
        vec![2.0_f64], // bin 0 (≤ 2.0)
        vec![3.0_f64], // bin 1 (2.0 < x ≤ 4.0)
        vec![4.0_f64], // bin 1 (≤ 4.0)
        vec![5.0_f64], // bin 2 (> 4.0)
    ];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let bins = match bin_samples(&refs, &edges, 4_usize) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    assert_eq!(bins.len(), 5_usize);
    assert_eq!(bins[0][0], 0_usize);
    assert_eq!(bins[1][0], 0_usize);
    assert_eq!(bins[2][0], 1_usize);
    assert_eq!(bins[3][0], 1_usize);
    assert_eq!(bins[4][0], 2_usize);
    Ok(())
}

#[test]
fn test_bin_samples_nan_goes_to_nan_bin() -> Result<(), ClearGbmError> {
    let edges = vec![match make_edges(vec![1.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let data: Vec<Vec<f64>> = vec![vec![f64::NAN], vec![0.5_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let n_regular = 4_usize;
    let bins = match bin_samples(&refs, &edges, n_regular) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    assert_eq!(bins[0][0], n_regular);
    assert_eq!(bins[1][0], 0_usize);
    Ok(())
}

#[test]
fn test_bin_samples_two_features() -> Result<(), ClearGbmError> {
    let e0 = match make_edges(vec![5.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    let e1 = match make_edges(vec![10.0_f64, 20.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    let edges = vec![e0, e1];
    let data: Vec<Vec<f64>> = vec![vec![3.0_f64, 15.0_f64], vec![7.0_f64, 5.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let bins = match bin_samples(&refs, &edges, 4_usize) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    assert_eq!(bins[0][0], 0_usize);
    assert_eq!(bins[0][1], 1_usize);
    assert_eq!(bins[1][0], 1_usize);
    assert_eq!(bins[1][1], 0_usize);
    Ok(())
}

#[test]
fn test_bin_samples_empty_edges() -> Result<(), ClearGbmError> {
    let edges = vec![match make_edges(Vec::new()) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![999.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let bins = match bin_samples(&refs, &edges, 4_usize) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    assert_eq!(bins[0][0], 0_usize);
    assert_eq!(bins[1][0], 0_usize);
    Ok(())
}

#[test]
fn test_bin_samples_boundary_exact() -> Result<(), ClearGbmError> {
    let edges = vec![match make_edges(vec![1.0_f64, 2.0_f64, 3.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let data: Vec<Vec<f64>> = vec![
        vec![1.0_f64], // exactly on first edge → bin 0
        vec![2.0_f64], // exactly on second edge → bin 1
        vec![3.0_f64], // exactly on third edge → bin 2
    ];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let bins = match bin_samples(&refs, &edges, 4_usize) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    assert_eq!(bins[0][0], 0_usize);
    assert_eq!(bins[1][0], 1_usize);
    assert_eq!(bins[2][0], 2_usize);
    Ok(())
}

#[test]
fn test_bin_samples_empty_x() -> Result<(), ClearGbmError> {
    let edges = vec![match make_edges(vec![1.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let refs: Vec<&[f64]> = Vec::new();
    let result = bin_samples(&refs, &edges, 4_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_bin_samples_edges_count_mismatch() -> Result<(), ClearGbmError> {
    let edges = vec![match make_edges(vec![1.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64, 2.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let result = bin_samples(&refs, &edges, 4_usize);
    match result {
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected ShapeMismatch, got {other:?}"),
        }),
    }
}

#[test]
fn test_bin_samples_inconsistent_rows() -> Result<(), ClearGbmError> {
    let edges = vec![match make_edges(vec![1.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let data1 = vec![1.0_f64];
    let data2 = vec![1.0_f64, 2.0_f64];
    let refs: Vec<&[f64]> = vec![data1.as_slice(), data2.as_slice()];
    let result = bin_samples(&refs, &edges, 4_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_bin_samples_zero_features() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![Vec::new()];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let result = bin_samples(&refs, &[], 4_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_bin_samples_value_below_all_edges() -> Result<(), ClearGbmError> {
    let edges = vec![match make_edges(vec![10.0_f64, 20.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let data: Vec<Vec<f64>> = vec![vec![-100.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let bins = match bin_samples(&refs, &edges, 4_usize) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };
    assert_eq!(bins[0][0], 0_usize);
    Ok(())
}

#[test]
fn test_bin_samples_value_above_all_edges() -> Result<(), ClearGbmError> {
    let edges = vec![match make_edges(vec![10.0_f64, 20.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    }];
    let data: Vec<Vec<f64>> = vec![vec![100.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let bins = match bin_samples(&refs, &edges, 4_usize) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };
    assert_eq!(bins[0][0], 2_usize);
    Ok(())
}
