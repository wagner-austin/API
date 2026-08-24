//! Threshold split search over packed integer histograms.
//!
//! The split side of quantized training, as LightGBM ships it
//! (`FindBestThresholdSequentiallyInt` in `feature_histogram.hpp` @
//! 3ec5b99b, pinned in the tech-wiki): the prefix scan stays integer —
//! left and right sums are exact, with no float reassociation anywhere
//! in the loop — and the conversion to gradient/hessian space happens at
//! the candidate boundary, where the integer sums are multiplied by the
//! round's scales and fed into the SAME f64 gain formula, monotonicity
//! check, and NaN-direction trial the float path uses. Sample counts are
//! exact integers throughout, so `min_samples_leaf` means what it says.

use crate::error::ClearGbmError;
use crate::types::SplitConfig;

use super::{
    check_monotonicity_constraint, compute_split_gain, MonotonicConstraint, NanDirection,
    SplitDecision, SplitResult, SplitResultConfig,
};

/// The round's decode scales, as the scan needs them.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedScanScales {
    /// Integer gradient sums multiply by this.
    pub grad_scale: f64,
    /// Integer hessian sums multiply by this.
    pub hess_scale: f64,
}

/// Internal struct to track the best split candidate during search.
///
/// Sums are kept as the exact integers they were found at; the
/// conversion to `SplitResult`'s f64 sums happens once, at the end.
#[derive(Debug, Clone, Copy)]
struct QuantizedSplitCandidate {
    /// Bin index for the split.
    bin_index: usize,
    /// Gain from this split.
    gain: f64,
    /// Integer gradient sum going left (including NaN if applicable).
    g_left: i64,
    /// Integer hessian sum going left (including NaN if applicable).
    h_left: i64,
    /// Count of samples going left (including NaN if applicable).
    n_left: usize,
    /// Direction for NaN values.
    nan_direction: NanDirection,
}

/// Converts a signed integer gradient sum to f64.
///
/// Bounded to `|v| < 2^31` by the histogram width invariant, so the
/// magnitude fits u32 exactly and the conversion is lossless (the
/// saturating arm is the crate's dead-arm idiom).
fn grad_int_to_f64(v: i64) -> f64 {
    let magnitude = f64::from(u32::try_from(v.unsigned_abs()).unwrap_or(u32::MAX));
    if v < 0_i64 {
        -magnitude
    } else {
        magnitude
    }
}

/// Converts a non-negative integer hessian sum to f64.
///
/// Bounded to `v < 2^32` by the histogram width invariant; lossless
/// (the saturating arm is the crate's dead-arm idiom).
fn hess_int_to_f64(v: i64) -> f64 {
    f64::from(u32::try_from(v).unwrap_or(u32::MAX))
}

/// Finds the best split for a single feature from its quantized histogram.
///
/// The structural mirror of the float path's
/// [`super::threshold::find_best_split_from_histogram`]: same candidate
/// enumeration (split after each regular bin, both NaN directions), same
/// constraints, same gain formula — with the prefix sums kept as exact
/// integers and rescaled to f64 only per candidate.
///
/// Generic over the packed entry type: the caller supplies the width's
/// unpack function and the scan reads entries in place — no materialized
/// tuple vector per feature per node (that intermediate was measured as
/// real per-node churn on deep trees).
///
/// # Args
///
/// * `bins` - The feature's packed bins, including the NaN bin as the
///   last entry when present.
/// * `unpack` - Decodes one entry to `(gradient, hessian, count)`.
/// * `feature_index` - Index of this feature.
/// * `config` - Split configuration (`min_samples_leaf`, `reg_lambda`,
///   `min_gain`).
/// * `n_regular_bins` - Number of regular bins (excluding the NaN bin).
/// * `monotonic_constraint` - Monotonicity constraint for this feature.
/// * `scales` - The round's decode scales.
///
/// # Returns
///
/// * `Ok(Some(SplitResult))` - Best split if one meets all constraints.
/// * `Ok(None)` - No valid split found.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` if `n_regular_bins` exceeds
/// the histogram's bin count.
pub fn find_best_split_from_quantized_histogram<T: Copy>(
    bins: &[T],
    unpack: fn(T) -> (i64, i64, usize),
    feature_index: usize,
    config: &SplitConfig,
    n_regular_bins: usize,
    monotonic_constraint: MonotonicConstraint,
    scales: QuantizedScanScales,
) -> Result<Option<SplitResult>, ClearGbmError> {
    let n_bins = bins.len();

    if n_regular_bins == 0_usize {
        return Ok(None);
    }
    if n_regular_bins > n_bins {
        return Err(ClearGbmError::InvalidParameter {
            name: "n_regular_bins".to_string(),
            reason: format!(
                "n_regular_bins ({n_regular_bins}) cannot exceed histogram n_bins ({n_bins})"
            ),
        });
    }

    let has_nan_bin = n_bins > n_regular_bins;
    let (g_nan, h_nan, n_nan) = if has_nan_bin {
        unpack(bins[n_regular_bins])
    } else {
        (0_i64, 0_i64, 0_usize)
    };

    let mut g_regular = 0_i64;
    let mut h_regular = 0_i64;
    let mut n_regular = 0_usize;
    for &entry in bins.iter().take(n_regular_bins) {
        let (g, h, n) = unpack(entry);
        g_regular += g;
        h_regular += h;
        n_regular += n;
    }

    let g_total = g_regular + g_nan;
    let h_total = h_regular + h_nan;
    let n_total = n_regular + n_nan;

    let min_samples_leaf = config.min_samples_leaf();
    if n_total < 2_usize * min_samples_leaf {
        return Ok(None);
    }

    let reg_lambda = config.reg_lambda();
    let min_gain = config.min_gain();
    let g_total_f = grad_int_to_f64(g_total) * scales.grad_scale;
    let h_total_f = hess_int_to_f64(h_total) * scales.hess_scale;

    let mut best: Option<QuantizedSplitCandidate> = None;

    let mut g_left_base = 0_i64;
    let mut h_left_base = 0_i64;
    let mut n_left_base = 0_usize;

    for (bin_idx, &entry) in bins
        .iter()
        .enumerate()
        .take(n_regular_bins.saturating_sub(1_usize))
    {
        let (g, h, n) = unpack(entry);
        g_left_base += g;
        h_left_base += h;
        n_left_base += n;

        for nan_dir in [NanDirection::Left, NanDirection::Right] {
            let (g_left, h_left, n_left) = if nan_dir.goes_left() {
                (
                    g_left_base + g_nan,
                    h_left_base + h_nan,
                    n_left_base + n_nan,
                )
            } else {
                (g_left_base, h_left_base, n_left_base)
            };

            let n_right = n_total.saturating_sub(n_left);
            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            let g_right = g_total - g_left;
            let h_right = h_total - h_left;

            // The boundary conversion: exact integer sums become
            // gradient/hessian-space doubles here, and nowhere earlier.
            let g_left_f = grad_int_to_f64(g_left) * scales.grad_scale;
            let h_left_f = hess_int_to_f64(h_left) * scales.hess_scale;
            let g_right_f = grad_int_to_f64(g_right) * scales.grad_scale;
            let h_right_f = hess_int_to_f64(h_right) * scales.hess_scale;

            if !check_monotonicity_constraint(
                monotonic_constraint,
                g_left_f,
                h_left_f,
                g_right_f,
                h_right_f,
            ) {
                continue;
            }

            let gain = compute_split_gain(
                g_left_f, h_left_f, g_right_f, h_right_f, g_total_f, h_total_f, reg_lambda,
            );

            if gain <= min_gain {
                continue;
            }

            let dominated = if let Some(ref current_best) = best {
                gain <= current_best.gain
            } else {
                false
            };

            if !dominated {
                best = Some(QuantizedSplitCandidate {
                    bin_index: bin_idx,
                    gain,
                    g_left,
                    h_left,
                    n_left,
                    nan_direction: nan_dir,
                });
            }
        }
    }

    if let Some(candidate) = best {
        let g_right = g_total - candidate.g_left;
        let h_right = h_total - candidate.h_left;
        let n_right = n_total.saturating_sub(candidate.n_left);

        Ok(Some(SplitResult::new(SplitResultConfig {
            feature_index,
            decision: SplitDecision::Threshold {
                split_bin: candidate.bin_index,
            },
            gain: candidate.gain,
            left_gradient_sum: grad_int_to_f64(candidate.g_left) * scales.grad_scale,
            left_hessian_sum: hess_int_to_f64(candidate.h_left) * scales.hess_scale,
            left_count: candidate.n_left,
            right_gradient_sum: grad_int_to_f64(g_right) * scales.grad_scale,
            right_hessian_sum: hess_int_to_f64(h_right) * scales.hess_scale,
            right_count: n_right,
            nan_direction: candidate.nan_direction,
        })))
    } else {
        Ok(None)
    }
}
