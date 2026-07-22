//! Narrowing conversions for the perf-critical histogram inputs.
//!
//! LightGBM's `score_t` defaults to `float` (32-bit); the histogram
//! accumulator (`hist_t`) is `double` (64-bit). That asymmetric shape halves
//! the memory bandwidth on the two hottest input streams — see
//! `~/PROJECTS/tech-wiki/pages/lightgbm-score-t-float.md`. This module is the
//! ONE gated site where `f64 → f32` narrowing is allowed; everywhere else in
//! the crate the `as_conversions` + `cast_precision_loss` lints forbid it.

/// Narrows an `f64` intermediate value to the `f32` score representation
/// used by the histogram input streams.
///
/// # Precision
///
/// Binary-log-loss gradients live in `[-1, 1]` and hessians in `[0, 0.25]`.
/// `f32`'s 7-digit mantissa is already better than the analytic uncertainty
/// in the boosted probability at those magnitudes; the LightGBM analysis in
/// `~/PROJECTS/tech-wiki/pages/lightgbm-score-t-float.md` walks through the
/// precision accounting in more detail.
///
/// The histogram accumulator (`HistogramBuffer.gradient_sums` /
/// `hessian_sums`) stays `f64`; the write site widens via `f64::from(x)`
/// before the accumulate, preserving 15-digit precision across billions of
/// summed values (the asymmetric shape LightGBM uses).
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "grad/hess narrowed to f32 for cache-line bandwidth per lightgbm-score-t-float; binary-log-loss range is [-1, 1] where f32's 7-digit precision exceeds analytic uncertainty"
)]
#[inline]
#[must_use]
pub const fn score_narrow(x: f64) -> f32 {
    x as f32
}

/// Widens a `u32` sample index to `usize` for slice/vec indexing.
///
/// `sample_indices` are stored as `u32` to halve cache-line pressure vs
/// `usize` on 64-bit targets (LightGBM's `data_size_t = int32` pattern).
/// Widening back to `usize` at every array access is infallible on every
/// supported target (x86_64, aarch64 — `usize` is 64-bit) since `u32`'s
/// range trivially fits. The alternative — `usize::try_from(idx)` in the
/// hot loop — introduces a branch per element that the optimizer keeps
/// live in profile-guided compiles.
#[expect(
    clippy::as_conversions,
    reason = "u32 -> usize is infallible widening on every supported target; try_from would add a per-iteration branch to the histogram hot loop"
)]
#[inline]
#[must_use]
pub const fn index_widen(idx: u32) -> usize {
    idx as usize
}

#[cfg(test)]
mod tests {
    use super::score_narrow;

    #[test]
    fn narrows_zero_to_zero() {
        assert!(score_narrow(0.0_f64).to_bits() == 0_u32);
    }

    #[test]
    fn narrows_representable_binary_log_loss_gradient() {
        // -0.5 is exactly representable in both f32 and f64
        let g_f64 = -0.5_f64;
        let g_f32 = score_narrow(g_f64);
        assert!(f64::from(g_f32) == g_f64);
    }

    #[test]
    fn narrows_representable_binary_log_loss_hessian() {
        // 0.25 is exactly representable in both f32 and f64
        let h_f64 = 0.25_f64;
        let h_f32 = score_narrow(h_f64);
        assert!(f64::from(h_f32) == h_f64);
    }

    #[test]
    fn narrowing_rounds_high_precision_input_within_f32_ulp() {
        // A value that needs more than f32's mantissa precision
        let x_f64 = 0.123_456_789_012_345_67_f64;
        let x_f32 = score_narrow(x_f64);
        let widened_back = f64::from(x_f32);
        // Difference must be within one f32 ULP at this magnitude
        let f32_ulp_at_1 = f64::from(f32::EPSILON);
        assert!((widened_back - x_f64).abs() < f32_ulp_at_1);
    }

    #[test]
    fn narrows_one_to_one() {
        assert!(f64::from(score_narrow(1.0_f64)) == 1.0_f64);
    }

    #[test]
    fn narrows_negative_one_to_negative_one() {
        assert!(f64::from(score_narrow(-1.0_f64)) == -1.0_f64);
    }

    #[test]
    fn index_widen_zero() {
        assert!(super::index_widen(0_u32) == 0_usize);
    }

    #[test]
    fn index_widen_max_u32() {
        // Every supported target has usize >= 32 bits, so u32::MAX widens exactly.
        // On the (unsupported) 16-bit-usize target this test would be skipped.
        if let Ok(want) = usize::try_from(u32::MAX) {
            assert!(super::index_widen(u32::MAX) == want);
        }
    }

    #[test]
    fn index_widen_mid_range() {
        assert!(super::index_widen(1_000_000_u32) == 1_000_000_usize);
    }
}
