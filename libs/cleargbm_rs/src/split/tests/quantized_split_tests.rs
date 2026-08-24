//! Tests for the integer threshold scan (`split::threshold_quantized`).
//!
//! The scan's contract: exact integer prefix sums, conversion to
//! gradient/hessian space only at the candidate boundary, and — for
//! sums that floats represent exactly — the same split the float scan
//! chooses under identity scales.

use crate::error::ClearGbmError;
use crate::split::{
    find_best_split_from_histogram, find_best_split_from_quantized_histogram, MonotonicConstraint,
    NanDirection, QuantizedScanScales, SplitDecision,
};
use crate::types::{HistogramBuffer, SplitConfig};

/// Identity scales: integer sums ARE the gradient/hessian sums.
const UNIT_SCALES: QuantizedScanScales = QuantizedScanScales {
    grad_scale: 1.0_f64,
    hess_scale: 1.0_f64,
};

fn split_config() -> Result<SplitConfig, ClearGbmError> {
    SplitConfig::new(2_usize, 1_usize, 64_usize, 0.0_f64, 0.0_f64)
}

/// Identity unpack: the tests state bins as bare integer triples.
fn unpack_triple(entry: (i64, i64, usize)) -> (i64, i64, usize) {
    entry
}

#[test]
fn test_matches_the_float_scan_on_exact_integer_sums() -> Result<(), ClearGbmError> {
    // Three regular bins plus a NaN bin, all sums small integers, unit
    // scales: both scans see identical numbers and must choose the same
    // bin, direction, and gain.
    let bins: Vec<(i64, i64, usize)> = vec![
        (8_i64, 10_i64, 10_usize),
        (-3_i64, 10_i64, 10_usize),
        (-9_i64, 12_i64, 12_usize),
        (2_i64, 3_i64, 3_usize),
    ];
    let mut float_hist = HistogramBuffer::new(4_usize);
    for (bin_idx, &(g, h, n)) in bins.iter().enumerate() {
        for _ in 0_usize..n {
            let g_share = grad_share(g, n);
            let h_share = hess_share(h, n);
            propagate!(float_hist.accumulate(bin_idx, g_share, h_share));
        }
    }
    let config = propagate!(split_config());
    let quant = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        3_usize,
        MonotonicConstraint::None,
        UNIT_SCALES,
    ));
    let float = propagate!(find_best_split_from_histogram(
        &float_hist,
        0_usize,
        &config,
        3_usize,
        MonotonicConstraint::None,
    ));
    let (Some(q), Some(f)) = (quant, float) else {
        return Err(ClearGbmError::TreeConstructionFailed {
            reason: "both scans must find a split".to_string(),
        });
    };
    assert_eq!(q.decision(), f.decision());
    assert_eq!(q.nan_goes_left(), f.nan_goes_left());
    assert!((q.gain() - f.gain()).abs() < 1e-9_f64);
    assert_eq!(q.left_count(), f.left_count());
    assert_eq!(q.right_count(), f.right_count());
    Ok(())
}

/// Splits an integer gradient sum into n equal float shares.
fn grad_share(g: i64, n: usize) -> f64 {
    let n_f = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let g_f = f64::from(i32::try_from(g).unwrap_or(i32::MAX));
    g_f / n_f
}

/// Splits an integer hessian sum into n equal float shares.
fn hess_share(h: i64, n: usize) -> f64 {
    let n_f = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let h_f = f64::from(i32::try_from(h).unwrap_or(i32::MAX));
    h_f / n_f
}

#[test]
fn test_scales_convert_sums_into_gradient_space() -> Result<(), ClearGbmError> {
    // grad_scale 2, hess_scale 0.5: the reported split sums must be the
    // integer sums multiplied through.
    let bins: Vec<(i64, i64, usize)> = vec![(6_i64, 8_i64, 4_usize), (-6_i64, 8_i64, 4_usize)];
    let config = propagate!(split_config());
    let result = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        2_usize,
        MonotonicConstraint::None,
        QuantizedScanScales {
            grad_scale: 2.0_f64,
            hess_scale: 0.5_f64,
        },
    ));
    let Some(split) = result else {
        return Err(ClearGbmError::TreeConstructionFailed {
            reason: "a symmetric two-bin histogram must split".to_string(),
        });
    };
    assert!((split.left_gradient_sum() - 12.0_f64).abs() < 1e-12_f64);
    assert!((split.left_hessian_sum() - 4.0_f64).abs() < 1e-12_f64);
    assert!((split.right_gradient_sum() + 12.0_f64).abs() < 1e-12_f64);
    assert!((split.right_hessian_sum() - 4.0_f64).abs() < 1e-12_f64);
    assert_eq!(split.left_count(), 4_usize);
    assert_eq!(split.right_count(), 4_usize);
    Ok(())
}

#[test]
fn test_nan_bin_tries_both_directions() -> Result<(), ClearGbmError> {
    // A NaN bin whose gradient reinforces the right side: sending NaN
    // right must win, and the reported direction must say so.
    let bins: Vec<(i64, i64, usize)> = vec![
        (10_i64, 5_i64, 5_usize),
        (-10_i64, 5_i64, 5_usize),
        (-10_i64, 5_i64, 5_usize),
    ];
    let config = propagate!(split_config());
    let result = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        2_usize,
        MonotonicConstraint::None,
        UNIT_SCALES,
    ));
    let Some(split) = result else {
        return Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected a split".to_string(),
        });
    };
    assert!(matches!(
        split.decision(),
        SplitDecision::Threshold { split_bin: 0_usize }
    ));
    assert_eq!(split.nan_direction(), NanDirection::Right);
    assert_eq!(split.left_count(), 5_usize);
    assert_eq!(split.right_count(), 10_usize);
    Ok(())
}

#[test]
fn test_min_samples_leaf_uses_exact_counts() -> Result<(), ClearGbmError> {
    // With min_samples_leaf 6, the only candidate (5 | 5) is refused on
    // counts even though its gain is positive.
    let bins: Vec<(i64, i64, usize)> = vec![(10_i64, 5_i64, 5_usize), (-10_i64, 5_i64, 5_usize)];
    let config = propagate!(SplitConfig::new(
        2_usize, 6_usize, 64_usize, 0.0_f64, 0.0_f64
    ));
    let result = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        2_usize,
        MonotonicConstraint::None,
        UNIT_SCALES,
    ));
    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_monotonic_constraint_filters_candidates() -> Result<(), ClearGbmError> {
    // Left mean prediction below right mean violates a Decreasing
    // constraint; with gradients arranged so left is negative-mean and
    // right positive-mean, Increasing passes and Decreasing refuses.
    let bins: Vec<(i64, i64, usize)> = vec![(10_i64, 5_i64, 5_usize), (-10_i64, 5_i64, 5_usize)];
    let config = propagate!(split_config());
    let increasing = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        2_usize,
        MonotonicConstraint::Increasing,
        UNIT_SCALES,
    ));
    let decreasing = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        2_usize,
        MonotonicConstraint::Decreasing,
        UNIT_SCALES,
    ));
    assert!(increasing.is_some());
    assert!(decreasing.is_none());
    Ok(())
}

#[test]
fn test_min_gain_threshold_refuses_flat_histograms() -> Result<(), ClearGbmError> {
    // Identical bins produce zero gain, which is not above min_gain 0.
    let bins: Vec<(i64, i64, usize)> = vec![(4_i64, 4_i64, 4_usize), (4_i64, 4_i64, 4_usize)];
    let config = propagate!(split_config());
    let result = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        2_usize,
        MonotonicConstraint::None,
        UNIT_SCALES,
    ));
    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_zero_regular_bins_returns_none() -> Result<(), ClearGbmError> {
    let bins: Vec<(i64, i64, usize)> = vec![(1_i64, 1_i64, 1_usize)];
    let config = propagate!(split_config());
    let result = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        0_usize,
        MonotonicConstraint::None,
        UNIT_SCALES,
    ));
    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_oversized_n_regular_bins_is_an_error() -> Result<(), ClearGbmError> {
    let bins: Vec<(i64, i64, usize)> = vec![(1_i64, 1_i64, 1_usize)];
    let config = propagate!(split_config());
    match find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        5_usize,
        MonotonicConstraint::None,
        UNIT_SCALES,
    ) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "n_regular_bins beyond the histogram must be refused".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "n_regular_bins");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_too_few_samples_returns_none() -> Result<(), ClearGbmError> {
    // n_total below 2 * min_samples_leaf exits before any scan.
    let bins: Vec<(i64, i64, usize)> = vec![(1_i64, 1_i64, 1_usize), (0_i64, 0_i64, 0_usize)];
    let config = propagate!(split_config());
    let result = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        2_usize,
        MonotonicConstraint::None,
        UNIT_SCALES,
    ));
    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_no_nan_bin_treats_all_bins_as_regular() -> Result<(), ClearGbmError> {
    // bins.len() == n_regular_bins: the NaN arm contributes zeros.
    let bins: Vec<(i64, i64, usize)> = vec![(10_i64, 5_i64, 5_usize), (-10_i64, 5_i64, 5_usize)];
    let config = propagate!(split_config());
    let result = propagate!(find_best_split_from_quantized_histogram(
        &bins,
        unpack_triple,
        0_usize,
        &config,
        2_usize,
        MonotonicConstraint::None,
        UNIT_SCALES,
    ));
    let Some(split) = result else {
        return Err(ClearGbmError::TreeConstructionFailed {
            reason: "expected a split".to_string(),
        });
    };
    assert_eq!(split.left_count() + split.right_count(), 10_usize);
    Ok(())
}
