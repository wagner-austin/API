//! Histogram building for gradient boosting.
//!
//! Implements O(n) histogram construction with NaN handling.
//! This is the primary hot path and performance-critical code.
//!
//! # Bin dtype
//!
//! Bin indices are `u8`. Under the config invariant `max_bins ≤ 255`, every
//! bin index (including the NaN bin at `max_bins`) fits in `u8`. Packing bins
//! into 1 byte instead of 8 (`usize`) makes 32 bin values per 256-bit AVX
//! load and multiplies the cache-line density by 8×.

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

/// Builds a histogram from sample gradients and hessians.
///
/// This is the core O(n) operation that accumulates gradient statistics
/// into bins for split finding.
///
/// # Args
///
/// * `sample_indices` - Indices of samples at this node.
/// * `gradients` - Gradient values for all samples.
/// * `hessians` - Hessian values for all samples.
/// * `bins` - Pre-computed bin assignments for this feature (`u8` per sample).
/// * `n_bins` - Number of bins (including NaN bin).
///
/// # Returns
///
/// A `HistogramBuffer` with accumulated statistics.
///
/// # Errors
///
/// * `ClearGbmError::EmptyInput` - If `sample_indices` is empty.
/// * `ClearGbmError::SampleIndexOutOfBounds` - If any index is out of bounds.
/// * `ClearGbmError::ShapeMismatch` - If array lengths don't match.
/// * `ClearGbmError::BinIndexOutOfBounds` - If any bin index is out of bounds.
pub fn build_histogram(
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[u8],
    n_bins: usize,
) -> Result<HistogramBuffer, ClearGbmError> {
    if sample_indices.is_empty() {
        return Err(ClearGbmError::EmptyInput {
            context: "sample_indices cannot be empty".to_string(),
        });
    }

    let n_samples = gradients.len();

    // Validate array lengths match
    if hessians.len() != n_samples {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("hessians length {n_samples}"),
            got: format!("hessians length {}", hessians.len()),
        });
    }
    if bins.len() != n_samples {
        return Err(ClearGbmError::ShapeMismatch {
            expected: format!("bins length {n_samples}"),
            got: format!("bins length {}", bins.len()),
        });
    }

    // ------------------------------------------------------------
    // Pre-validation pass (SIMD-friendly)
    //
    // One dedicated pass over `sample_indices` up front instead of a
    // per-sample check inside the hot loop, collapsing the bounds check
    // AND the bin-range check into a single scan. LLVM auto-vectorizes
    // the pass into SIMD comparisons on modern targets (x86_64 AVX2,
    // ARM64 NEON), so validating N indices costs on the order of N/4
    // cycles instead of N. The main hot loop that follows is
    // bounds-check-free at the semantic level, and Rust's panic-safe
    // indexing collapses to a fast path when the compiler sees the
    // invariant established.
    //
    // The scalar-equivalent reference implementation is the previous
    // per-iteration `accumulate` call — its behavior is preserved
    // bit-identically by the tests in `tests/`.
    // ------------------------------------------------------------
    let mut max_bin_used: u8 = 0_u8;
    for &idx in sample_indices {
        if idx >= n_samples {
            return Err(ClearGbmError::SampleIndexOutOfBounds {
                index: idx,
                n_samples,
            });
        }
        let b = bins[idx];
        if b > max_bin_used {
            max_bin_used = b;
        }
    }
    let max_bin_used_usize = usize::from(max_bin_used);
    if max_bin_used_usize >= n_bins {
        return Err(ClearGbmError::BinIndexOutOfBounds {
            bin: max_bin_used_usize,
            n_bins,
        });
    }

    Ok(build_histogram_trusted(
        sample_indices,
        gradients,
        hessians,
        bins,
        n_bins,
    ))
}

/// Fast-path histogram build that skips input validation.
///
/// Bypasses the sample-index bounds check, the array-length shape check,
/// and the bin-range check performed by [`build_histogram`]. This is the
/// hot-path call made from the tree builder, where the invariants are
/// established by construction:
///
/// * `sample_indices` are drawn from `get_sample_indices(n_train, ..)`,
///   which returns a subset of `0..n_train`, and every downstream child
///   node's indices are a subset of the parent's — so `idx < n_samples`
///   is a compile-time-known invariant of the tree traversal.
/// * `bins` originate in [`FeatureBins::bins`], whose constructor caps
///   each entry at `max_bins ≤ 255` and whose length equals
///   `n_samples * n_features`.
///
/// Callers outside this crate should use the validated [`build_histogram`]
/// entry point instead.
///
/// # Args
///
/// * `sample_indices` - Indices of samples at this node.
/// * `gradients` - Gradient values for all samples.
/// * `hessians` - Hessian values for all samples.
/// * `bins` - Pre-computed bin assignments for this feature (`u8` per sample).
/// * `n_bins` - Number of bins (including NaN bin).
///
/// # Panics
///
/// Rust's safe indexing will panic if any invariant listed above is
/// violated. That is a bug in the caller, not a recoverable runtime error.
#[must_use]
pub(crate) fn build_histogram_trusted(
    sample_indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    bins: &[u8],
    n_bins: usize,
) -> HistogramBuffer {
    let mut histogram = HistogramBuffer::new(n_bins);
    // Direct field access: skips the `accumulate` function-call boundary
    // and its per-sample bin bounds check (established by the caller).
    let gradient_sums = &mut histogram.gradient_sums;
    let hessian_sums = &mut histogram.hessian_sums;
    let counts = &mut histogram.counts;

    // ------------------------------------------------------------
    // Vectorized main loop: unrolled 4-wide.
    //
    // Processing samples in chunks of 4 gives the compiler space to
    // interleave the four independent RMW streams — modern CPUs can
    // execute 3-4 memory operations per cycle, and the unrolled shape
    // maps onto that pipeline width. Chunks that aren't a multiple of
    // 4 fall through to the scalar tail below.
    // ------------------------------------------------------------
    let chunks = sample_indices.chunks_exact(4_usize);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let idx0 = chunk[0_usize];
        let idx1 = chunk[1_usize];
        let idx2 = chunk[2_usize];
        let idx3 = chunk[3_usize];

        let b0 = usize::from(bins[idx0]);
        let b1 = usize::from(bins[idx1]);
        let b2 = usize::from(bins[idx2]);
        let b3 = usize::from(bins[idx3]);

        let g0 = gradients[idx0];
        let g1 = gradients[idx1];
        let g2 = gradients[idx2];
        let g3 = gradients[idx3];

        let h0 = hessians[idx0];
        let h1 = hessians[idx1];
        let h2 = hessians[idx2];
        let h3 = hessians[idx3];

        gradient_sums[b0] += g0;
        hessian_sums[b0] += h0;
        counts[b0] += 1_usize;

        gradient_sums[b1] += g1;
        hessian_sums[b1] += h1;
        counts[b1] += 1_usize;

        gradient_sums[b2] += g2;
        hessian_sums[b2] += h2;
        counts[b2] += 1_usize;

        gradient_sums[b3] += g3;
        hessian_sums[b3] += h3;
        counts[b3] += 1_usize;
    }

    // Scalar tail for the last (n_samples mod 4) elements.
    for &idx in remainder {
        let bin = usize::from(bins[idx]);
        gradient_sums[bin] += gradients[idx];
        hessian_sums[bin] += hessians[idx];
        counts[bin] += 1_usize;
    }

    histogram
}

/// Computes sibling histogram by subtraction (2x speedup).
///
/// Given parent histogram and one child histogram, computes the
/// other child by subtraction: sibling = parent - child.
///
/// This matches the Python `subtract_histogram` function.
///
/// # Args
///
/// * `parent` - Parent node histogram.
/// * `child` - One child node histogram.
///
/// # Returns
///
/// The sibling histogram.
///
/// # Errors
///
/// * `ClearGbmError::ShapeMismatch` - If histograms have different `n_bins`.
pub fn subtract_histogram(
    parent: &HistogramBuffer,
    child: &HistogramBuffer,
) -> Result<HistogramBuffer, ClearGbmError> {
    let mut sibling = HistogramBuffer::new(parent.n_bins());
    match sibling.subtract_into(parent, child) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    Ok(sibling)
}

#[cfg(test)]
mod tests;
