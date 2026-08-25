//! Tests for bin edge computation.

use crate::binning::edges::{compute_bin_edges, f64_to_usize_checked, BinEdges};
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

// ── count-aware binning ────────────────────────────────────────────

#[test]
fn test_per_value_bins_when_distinct_fits_budget() -> Result<(), ClearGbmError> {
    // 5 ones, one 2, one 3: three distinct values under a 64 budget →
    // every distinct value gets its own bin, edges at midpoints.
    let mut data: Vec<Vec<f64>> = (0_usize..5_usize).map(|_| vec![1.0_f64]).collect();
    data.push(vec![2.0_f64]);
    data.push(vec![3.0_f64]);
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 64_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges[0].edges(), &[1.5_f64, 2.5_f64]);
    assert_eq!(edges[0].n_regular_bins(), 3_usize);
    Ok(())
}

#[test]
fn test_zero_inflated_feature_keeps_tail_resolution() -> Result<(), ClearGbmError> {
    // 90 zeros plus 1..=10 once each, 8-bin budget. The heavy zero takes
    // one bin and the tail keeps the rest: 8 bins, 7 edges. The replaced
    // quantile-of-multiset rule put every quantile position on the zero
    // and produced 2 bins from this shape.
    let mut data: Vec<Vec<f64>> = (0_usize..90_usize).map(|_| vec![0.0_f64]).collect();
    for i in 1_u32..=10_u32 {
        data.push(vec![f64::from(i)]);
    }
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 8_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(
        edges[0].edges(),
        &[0.5_f64, 2.5_f64, 4.5_f64, 6.5_f64, 7.5_f64, 8.5_f64, 9.5_f64]
    );
    assert_eq!(edges[0].n_regular_bins(), 8_usize);
    Ok(())
}

#[test]
fn test_heavy_value_closes_a_bin_on_arrival() -> Result<(), ClearGbmError> {
    // 50 fives among singles 1,2,3,4,6,7,8,9 under a 4-bin budget. The
    // heavy five closes its bin the moment it arrives; the singles bins
    // form around it at the running mean.
    let mut data: Vec<Vec<f64>> = (0_usize..50_usize).map(|_| vec![5.0_f64]).collect();
    for v in [
        1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 6.0_f64, 7.0_f64, 8.0_f64, 9.0_f64,
    ] {
        data.push(vec![v]);
    }
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 4_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges[0].edges(), &[3.5_f64, 5.5_f64, 8.5_f64]);
    assert_eq!(edges[0].n_regular_bins(), 4_usize);
    Ok(())
}

#[test]
fn test_rest_budget_exhausts_before_late_heavy_values() -> Result<(), ClearGbmError> {
    // Singles 1..6 spend both rest bins before the two heavy values
    // (8, 9, ten copies each) arrive; once the rest budget hits zero
    // the running mean stops refreshing and the heavies still close
    // their own bins. Budget 4: total 26, mean 6.5, both heavies big;
    // rest mean 6/2 = 3 → closes at 3, at 6 (budget spent), at 8.
    let mut data: Vec<Vec<f64>> = (1_u32..=6_u32).map(|v| vec![f64::from(v)]).collect();
    for _ in 0_usize..10_usize {
        data.push(vec![8.0_f64]);
        data.push(vec![9.0_f64]);
    }
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 4_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges[0].edges(), &[3.5_f64, 7.0_f64, 8.5_f64]);
    assert_eq!(edges[0].n_regular_bins(), 4_usize);
    Ok(())
}

#[test]
fn test_adjacent_double_midpoint_stays_below_upper_value() -> Result<(), ClearGbmError> {
    // For adjacent doubles the midpoint can round onto the upper value;
    // the edge must still route the lower value left and the upper right.
    // An odd-mantissa lower value forces round-half-to-even UP onto the
    // (even-mantissa) upper value, exercising the collapse guard.
    let a = f64::from_bits(1.0_f64.to_bits() + 1_u64);
    let b = f64::from_bits(1.0_f64.to_bits() + 2_u64);
    let data: Vec<Vec<f64>> = vec![vec![a], vec![b]];
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let edges = match compute_bin_edges(&refs, 4_usize) {
        Ok(e) => e,
        Err(e) => return Err(e),
    };
    assert_eq!(edges[0].edges().len(), 1_usize);
    let edge = edges[0].edges()[0];
    assert!(edge >= a);
    assert!(edge < b);
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
