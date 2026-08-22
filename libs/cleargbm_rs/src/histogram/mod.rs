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
//!
//! # Entry points
//!
//! The tree builder calls [`build_node_histograms_single_pass`] once per
//! node after one node-scoped [`reorder_grad_hess_into`] pass; both
//! functions establish invariants by construction (see the doc on each).
//! There is no validated entry point — validation happens at the top-level
//! pyo3 boundary in `pyo3_module::training_fns`, not at the per-histogram
//! level.

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

/// The inputs one single-pass all-features node build needs.
///
/// The row-major sibling of [`HistogramRequest`]: instead of one feature's
/// contiguous bin column, it carries the whole row-major bin matrix and
/// builds every feature's histogram in one walk over the node's samples.
#[derive(Debug, Clone, Copy)]
pub struct NodeHistogramRequest<'a> {
    /// Indices of samples at this node, used for the bin-row gather; NOT
    /// indexed into `ordered_gradients` / `ordered_hessians`.
    pub sample_indices: &'a [u32],

    /// Pre-permuted gradient stream, as in [`HistogramRequest`].
    pub ordered_gradients: &'a [f64],

    /// Pre-permuted hessian stream, same shape.
    pub ordered_hessians: &'a [f64],

    /// Row-major bin matrix: sample `i`'s bins for all features are the
    /// `n_features` contiguous bytes at `bins_rows[i * n_features..]`.
    pub bins_rows: &'a [u8],

    /// Number of features (the row stride of `bins_rows`).
    pub n_features: usize,

    /// Number of bins per feature, including the NaN bin.
    pub n_bins: usize,
}

/// Builds every feature's histogram for one node in a single sample walk.
///
/// The per-feature path reads each node's samples once per feature: the
/// index, the bin byte, and both ordered streams are re-read `n_features`
/// times per node (~378 bytes of reads per sample at 18 features). This
/// walk reads the index, the 18-byte bin row, and each stream value once
/// (~38 bytes per sample), updating all feature histograms as it goes —
/// LightGBM's `ConstructHistograms` shape.
///
/// Bit-identity with the per-feature path holds by construction: for every
/// (feature, bin) pair the additions happen in `sample_indices` order in
/// both shapes, so the floating-point sums are identical.
///
/// The accumulators live in one flat `n_features * n_bins` block during the
/// walk (better locality than `n_features` separate buffers) and are carved
/// into per-feature [`HistogramBuffer`]s at the end.
///
/// # Args
///
/// * `request` - The node-scoped inputs.
///
/// # Returns
///
/// One populated [`HistogramBuffer`] per feature, in feature order.
///
/// # Panics
///
/// Rust's safe indexing will panic if any invariant is violated (a row
/// extending past `bins_rows`, a bin byte at or above `n_bins`, or stream
/// lengths disagreeing with `sample_indices`). That is a bug in the caller,
/// not a recoverable runtime error — the same trust contract as
/// [`build_histogram_ordered_trusted`].
#[must_use]
pub(crate) fn build_node_histograms_single_pass(
    request: NodeHistogramRequest<'_>,
) -> Vec<crate::types::HistogramBuffer> {
    let NodeHistogramRequest {
        sample_indices,
        ordered_gradients,
        ordered_hessians,
        bins_rows,
        n_features,
        n_bins,
    } = request;

    let mut flat: Vec<crate::types::BinAccumulator> =
        vec![crate::types::BinAccumulator::ZERO; n_features * n_bins];

    for (pos, &idx) in sample_indices.iter().enumerate() {
        let row_start = crate::narrow::index_widen(idx) * n_features;
        let row = &bins_rows[row_start..row_start + n_features];
        let g = ordered_gradients[pos];
        let h = ordered_hessians[pos];
        let mut base = 0_usize;
        for &bin in row {
            let acc = &mut flat[base + usize::from(bin)];
            acc.gradient_sum += g;
            acc.hessian_sum += h;
            acc.count += 1_usize;
            base += n_bins;
        }
    }

    let mut out = Vec::with_capacity(n_features);
    for feat_idx in 0_usize..n_features {
        let mut histogram = crate::types::HistogramBuffer::new(n_bins);
        histogram
            .bins
            .copy_from_slice(&flat[feat_idx * n_bins..(feat_idx + 1_usize) * n_bins]);
        out.push(histogram);
    }
    out
}

/// Fills the ordered scratch buffers `ordered_gradients[i] = gradients[sample_indices[i]]`
/// (and the same for hessians) in one pass.
///
/// This is the amortization step for [`build_histogram_ordered_trusted`]: one
/// scan over `sample_indices` produces two sequential-access-shaped arrays
/// that all per-feature histogram builds for this node reuse.
///
/// # Args
///
/// * `sample_indices` - Sample indices at this node.
/// * `gradients` - Full gradient array (indexed by sample index).
/// * `hessians` - Full hessian array (indexed by sample index).
/// * `ordered_gradients` - Output scratch, length must equal `sample_indices.len()`.
/// * `ordered_hessians` - Output scratch, length must equal `sample_indices.len()`.
///
/// # Panics
///
/// Panics if any output length disagrees with `sample_indices.len()` or if any
/// sample index is out of bounds for `gradients` / `hessians`. The tree builder
/// establishes both invariants by construction.
#[inline]
pub(crate) fn reorder_grad_hess_into(
    sample_indices: &[u32],
    gradients: &[f64],
    hessians: &[f64],
    ordered_gradients: &mut [f64],
    ordered_hessians: &mut [f64],
) {
    assert!(ordered_gradients.len() == sample_indices.len());
    assert!(ordered_hessians.len() == sample_indices.len());
    // No software prefetch here, and that is a measured decision: a
    // black-box forced-read 16 iterations ahead (the safe spelling of
    // `_mm_prefetch` under this crate's `unsafe_code = "forbid"`) was
    // measured 2026-08-21 at +2.5% fit time against this plain loop. The
    // sample indices are sorted ascending, so the hardware prefetcher
    // already covers the walk and the warming loads are pure overhead. See
    // `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-21_interleave.md`.
    for (i, &idx) in sample_indices.iter().enumerate() {
        let idx_usize = crate::narrow::index_widen(idx);
        ordered_gradients[i] = gradients[idx_usize];
        ordered_hessians[i] = hessians[idx_usize];
    }
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
