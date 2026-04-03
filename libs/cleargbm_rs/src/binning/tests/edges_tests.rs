//! Tests for bin edge computation.

use crate::binning::edges::{compute_bin_edges, f64_to_usize_checked, quantile_position, BinEdges};
use crate::error::ClearGbmError;

// ── BinEdges construction ──────────────────────────────────────────

#[test]
fn test_bin_edges_empty() -> Result<(), ClearGbmError> {
    let edges = match BinEdges::new(Vec::new()) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert!(edges.edges().is_empty());
    assert_eq!(edges.n_regular_bins(), 1_usize);
    Ok(())
}

#[test]
fn test_bin_edges_single_edge() -> Result<(), ClearGbmError> {
    let edges = match BinEdges::new(vec![1.5_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges.edges(), &[1.5_f64]);
    assert_eq!(edges.n_regular_bins(), 2_usize);
    Ok(())
}

#[test]
fn test_bin_edges_multiple_sorted() -> Result<(), ClearGbmError> {
    let edges = match BinEdges::new(vec![1.0_f64, 2.0_f64, 3.0_f64]) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges.edges().len(), 3_usize);
    assert_eq!(edges.n_regular_bins(), 4_usize);
    Ok(())
}

#[test]
fn test_bin_edges_not_sorted() -> Result<(), ClearGbmError> {
    let result = BinEdges::new(vec![2.0_f64, 1.0_f64]);
    match result {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "edges");
            assert!(reason.contains("not strictly sorted"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter, got {other:?}"),
        }),
    }
}

#[test]
fn test_bin_edges_duplicate_values() -> Result<(), ClearGbmError> {
    let result = BinEdges::new(vec![1.0_f64, 1.0_f64]);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_bin_edges_nan() -> Result<(), ClearGbmError> {
    let result = BinEdges::new(vec![f64::NAN]);
    match result {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "edges");
            assert!(reason.contains("not finite"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter, got {other:?}"),
        }),
    }
}

#[test]
fn test_bin_edges_infinity() -> Result<(), ClearGbmError> {
    let result = BinEdges::new(vec![f64::INFINITY]);
    assert!(result.is_err());
    Ok(())
}

// ── quantile_position ────────────────────────────────────────────

#[test]
fn test_quantile_position_basic() -> Result<(), ClearGbmError> {
    // pos = floor(1/3 * 9) = 3
    assert_eq!(quantile_position(1_usize, 9_usize, 3_usize), 3_usize);
    // pos = floor(2/3 * 9) = 6
    assert_eq!(quantile_position(2_usize, 9_usize, 3_usize), 6_usize);
    Ok(())
}

#[test]
fn test_quantile_position_small_values() -> Result<(), ClearGbmError> {
    // edge_idx=1, n_valid-1=3, max_bins=4 → pos = 1*3/4 = 0
    assert_eq!(quantile_position(1_usize, 3_usize, 4_usize), 0_usize);
    // edge_idx=2, n_valid-1=3, max_bins=4 → pos = 2*3/4 = 1
    assert_eq!(quantile_position(2_usize, 3_usize, 4_usize), 1_usize);
    // edge_idx=3, n_valid-1=3, max_bins=4 → pos = 3*3/4 = 2
    assert_eq!(quantile_position(3_usize, 3_usize, 4_usize), 2_usize);
    Ok(())
}

#[test]
fn test_quantile_position_overflow_uses_fallback() -> Result<(), ClearGbmError> {
    // Create values where a * b overflows usize.
    // a=2, b=usize::MAX/2 + 1, c=3 → a*b = usize::MAX + 1 → overflows.
    let a = 2_usize;
    let b = usize::MAX / 2_usize + 1_usize;
    let c = 3_usize;
    let result = quantile_position(a, b, c);

    // Verify using the fallback formula: a*(b/c) + a*(b%c)/c
    let expected = a * (b / c) + a * (b % c) / c;
    assert_eq!(result, expected);
    assert!(result > 0_usize);
    Ok(())
}

#[test]
fn test_quantile_position_edge_one() -> Result<(), ClearGbmError> {
    // a=1, any b, c: pos = b/c
    assert_eq!(quantile_position(1_usize, 99_usize, 10_usize), 9_usize);
    assert_eq!(quantile_position(1_usize, 0_usize, 10_usize), 0_usize);
    Ok(())
}

// ── compute_bin_edges ──────────────────────────────────────────────

#[test]
fn test_compute_bin_edges_basic() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![2.0_f64], vec![3.0_f64], vec![4.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 3_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges.len(), 1_usize);
    let feature_edges = edges[0].edges();
    assert!(!feature_edges.is_empty());
    for &e in feature_edges {
        assert!(e >= 1.0_f64);
        assert!(e <= 4.0_f64);
    }
    Ok(())
}

#[test]
fn test_compute_bin_edges_two_features() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![
        vec![1.0_f64, 10.0_f64],
        vec![2.0_f64, 20.0_f64],
        vec![3.0_f64, 30.0_f64],
    ];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 4_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges.len(), 2_usize);
    Ok(())
}

#[test]
fn test_compute_bin_edges_all_nan_feature() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![f64::NAN], vec![f64::NAN]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 4_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges.len(), 1_usize);
    assert!(edges[0].edges().is_empty());
    assert_eq!(edges[0].n_regular_bins(), 1_usize);
    Ok(())
}

#[test]
fn test_compute_bin_edges_single_unique_value() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![5.0_f64], vec![5.0_f64], vec![5.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 4_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert!(edges[0].edges().is_empty());
    Ok(())
}

#[test]
fn test_compute_bin_edges_with_nan_and_valid() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![
        vec![1.0_f64],
        vec![f64::NAN],
        vec![3.0_f64],
        vec![f64::NAN],
        vec![5.0_f64],
    ];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 4_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert!(!edges[0].edges().is_empty());
    Ok(())
}

#[test]
fn test_compute_bin_edges_max_bins_too_small() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let result = compute_bin_edges(&refs, 1_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_compute_bin_edges_max_bins_exceeds_u32() -> Result<(), ClearGbmError> {
    // On 64-bit, create a max_bins value that exceeds u32::MAX.
    let u32_max_usize = match usize::try_from(u32::MAX) {
        Ok(v) => v,
        Err(_) => return Ok(()), // skip on platforms where usize < u32
    };
    let large_max_bins = u32_max_usize + 2_usize;
    let data: Vec<Vec<f64>> = vec![vec![1.0_f64], vec![2.0_f64]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let result = compute_bin_edges(&refs, large_max_bins);
    match result {
        Err(ClearGbmError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "max_bins");
            assert!(reason.contains("exceeds"));
            Ok(())
        }
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected InvalidParameter, got {other:?}"),
        }),
    }
}

#[test]
fn test_compute_bin_edges_empty_x() -> Result<(), ClearGbmError> {
    let refs: Vec<&[f64]> = Vec::new();
    let result = compute_bin_edges(&refs, 4_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_compute_bin_edges_zero_features() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = vec![Vec::new()];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let result = compute_bin_edges(&refs, 4_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_compute_bin_edges_inconsistent_row_lengths() -> Result<(), ClearGbmError> {
    let data1 = vec![1.0_f64, 2.0_f64];
    let data2 = vec![3.0_f64];
    let refs: Vec<&[f64]> = vec![data1.as_slice(), data2.as_slice()];
    let result = compute_bin_edges(&refs, 4_usize);
    match result {
        Err(ClearGbmError::ShapeMismatch { .. }) => Ok(()),
        other => Err(ClearGbmError::TreeConstructionFailed {
            reason: format!("expected ShapeMismatch, got {other:?}"),
        }),
    }
}

#[test]
fn test_compute_bin_edges_deduplication() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = (0_usize..100_usize).map(|_| vec![42.0_f64]).collect();
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 256_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert!(edges[0].edges().is_empty());
    Ok(())
}

#[test]
fn test_compute_bin_edges_many_bins() -> Result<(), ClearGbmError> {
    let data: Vec<Vec<f64>> = (0_u32..100_u32).map(|i| vec![f64::from(i)]).collect();
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 10_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert!(edges[0].edges().len() <= 9_usize);
    assert!(!edges[0].edges().is_empty());
    let e = edges[0].edges();
    for i in 1_usize..e.len() {
        assert!(e[i] > e[i - 1_usize]);
    }
    Ok(())
}

// ── f64_to_usize_checked ──────────────────────────────────────────

#[test]
fn test_f64_to_usize_zero() -> Result<(), ClearGbmError> {
    let result = match f64_to_usize_checked(0.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, 0_usize);
    Ok(())
}

#[test]
fn test_f64_to_usize_positive() -> Result<(), ClearGbmError> {
    let r1 = match f64_to_usize_checked(42.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(r1, 42_usize);
    let r2 = match f64_to_usize_checked(1000.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(r2, 1000_usize);
    Ok(())
}

#[test]
fn test_f64_to_usize_powers_of_two() -> Result<(), ClearGbmError> {
    // Test powers of 2 to exercise both is_odd branches thoroughly
    let r1 = match f64_to_usize_checked(1.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(r1, 1_usize);
    let r2 = match f64_to_usize_checked(2.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(r2, 2_usize);
    let r3 = match f64_to_usize_checked(256.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(r3, 256_usize);
    Ok(())
}

#[test]
fn test_f64_to_usize_large() -> Result<(), ClearGbmError> {
    let val = f64::from(u32::MAX);
    let expected = match usize::try_from(u32::MAX) {
        Ok(v) => v,
        Err(_) => {
            return Err(ClearGbmError::IntegerConversion {
                context: "test platform does not support u32::MAX as usize".to_string(),
            })
        }
    };
    let result = match f64_to_usize_checked(val, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(result, expected);
    Ok(())
}

#[test]
fn test_f64_to_usize_negative() -> Result<(), ClearGbmError> {
    let result = f64_to_usize_checked(-1.0_f64, "test");
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_f64_to_usize_non_integer() -> Result<(), ClearGbmError> {
    let result = f64_to_usize_checked(1.5_f64, "test");
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_f64_to_usize_nan() -> Result<(), ClearGbmError> {
    let result = f64_to_usize_checked(f64::NAN, "test");
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_f64_to_usize_infinity() -> Result<(), ClearGbmError> {
    let result = f64_to_usize_checked(f64::INFINITY, "test");
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_f64_to_usize_exceeds_u32_max() -> Result<(), ClearGbmError> {
    let val = f64::from(u32::MAX) + 1.0_f64;
    let result = f64_to_usize_checked(val, "test");
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_f64_to_usize_odd_values() -> Result<(), ClearGbmError> {
    // Odd values exercise the is_odd >= 0.5 branch in the first loop iteration
    let r1 = match f64_to_usize_checked(3.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(r1, 3_usize);
    let r2 = match f64_to_usize_checked(255.0_f64, "test") {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(r2, 255_usize);
    Ok(())
}
