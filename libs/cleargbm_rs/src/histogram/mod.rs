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
//! The tree builder calls [`build_histogram_ordered_trusted`] per feature
//! after one node-scoped [`reorder_grad_hess_into`] pass; both functions
//! establish invariants by construction (see the doc on each). There is no
//! validated entry point — validation happens at the top-level pyo3
//! boundary in `pyo3_module::training_fns`, not at the per-histogram level.

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

/// The inputs one per-feature histogram build needs.
///
/// Grouped into a struct rather than passed as five positional arguments,
/// matching the parameter-struct shape already used by
/// [`crate::tree::histograms::BuildHistogramConfig`]. This keeps the
/// dependency-injection hook in [`crate::hooks::Hooks`] a simple
/// single-argument function pointer instead of a five-parameter one.
///
/// Borrowed, not owned: the caller's node-scoped buffers outlive every
/// per-feature build, so this is a view and never copies sample data.
#[derive(Debug, Clone, Copy)]
pub struct HistogramRequest<'a> {
    /// Indices of samples at this node, used only for the bin gather; NOT
    /// indexed into `ordered_gradients` / `ordered_hessians`.
    pub sample_indices: &'a [u32],

    /// Pre-permuted gradient stream: `ordered_gradients[i]` equals
    /// `gradients[sample_indices[i]]`. Length equals `sample_indices.len()`.
    pub ordered_gradients: &'a [f64],

    /// Pre-permuted hessian stream, same shape as `ordered_gradients`.
    pub ordered_hessians: &'a [f64],

    /// Pre-computed bin assignments (`u8` per sample) for one feature.
    pub bins: &'a [u8],

    /// Number of bins, including the NaN bin.
    pub n_bins: usize,
}

/// Histogram build fast path that reads gradients + hessians sequentially.
///
/// Assumes `ordered_gradients[i]` is the gradient for the sample at
/// `sample_indices[i]` — i.e. the caller has pre-permuted the gradient and
/// hessian streams into position-space instead of sample-space. The hot loop
/// then reads `ordered_gradients[i]` and `ordered_hessians[i]` with pure
/// sequential access; the bin lookup is the only random-index access (per
/// LightGBM's own shape — see the wiki page `lightgbm-construct-histogram-inner`).
///
/// Amortization: the caller does one reorder pass per node (cost:
/// `sample_indices.len()` gathers on each of `gradients` + `hessians`), then
/// reuses the two ordered arrays across all `n_features` histogram builds
/// for that node. For 18 features per node the effective gather count on
/// gradients + hessians drops 18×.
///
/// # Args
///
/// * `request` - The node-scoped inputs for this feature's build.
///
/// # Returns
///
/// A populated [`HistogramBuffer`].
///
/// # Panics
///
/// Rust's safe indexing will panic if any invariant is violated. That is a
/// bug in the caller, not a recoverable runtime error.
#[must_use]
#[inline]
pub(crate) fn build_histogram_ordered_trusted(request: HistogramRequest<'_>) -> HistogramBuffer {
    let HistogramRequest {
        sample_indices,
        ordered_gradients,
        ordered_hessians,
        bins,
        n_bins,
    } = request;
    let mut histogram = HistogramBuffer::new(n_bins);
    let gradient_sums = &mut histogram.gradient_sums;
    let hessian_sums = &mut histogram.hessian_sums;
    let counts = &mut histogram.counts;

    // ------------------------------------------------------------
    // Vectorized main loop: unrolled 8-wide, sequential reads on the
    // pre-permuted gradient + hessian streams. The `bins[idx]` gather is
    // the only random-index access — grouping all 8 gathers before the
    // dependent RMW gives the compiler room to schedule them in parallel
    // on modern out-of-order cores. Chunks not a multiple of 8 fall to
    // the scalar tail.
    //
    // Grad/hess and the accumulator are both f64 — no widening at the write
    // site. LightGBM's asymmetric `hist_t += score_t` shape (f32 in, f64
    // accumulator) was implemented here and then reverted: narrowing the two
    // input streams measured slower on this workload, because at the node
    // sizes reached here both widths already fit in L2, so there is no
    // bandwidth to save and each element pays a widening conversion before
    // its accumulate. See the wiki page
    // `cleargbm-f32-score-narrowing-reverted`.
    // ------------------------------------------------------------
    let chunks = sample_indices.chunks_exact(8_usize);
    let remainder = chunks.remainder();
    // Zip in the ordered streams; both are length == sample_indices.len().
    let mut pos: usize = 0_usize;
    for chunk in chunks {
        // sample_indices are u32; widen for slice-indexing via the
        // infallible `crate::narrow::index_widen` (see wiki page
        // `lightgbm-score-t-float` and the `data_size_t = int32` pattern).
        let idx0 = crate::narrow::index_widen(chunk[0_usize]);
        let idx1 = crate::narrow::index_widen(chunk[1_usize]);
        let idx2 = crate::narrow::index_widen(chunk[2_usize]);
        let idx3 = crate::narrow::index_widen(chunk[3_usize]);
        let idx4 = crate::narrow::index_widen(chunk[4_usize]);
        let idx5 = crate::narrow::index_widen(chunk[5_usize]);
        let idx6 = crate::narrow::index_widen(chunk[6_usize]);
        let idx7 = crate::narrow::index_widen(chunk[7_usize]);

        let b0 = usize::from(bins[idx0]);
        let b1 = usize::from(bins[idx1]);
        let b2 = usize::from(bins[idx2]);
        let b3 = usize::from(bins[idx3]);
        let b4 = usize::from(bins[idx4]);
        let b5 = usize::from(bins[idx5]);
        let b6 = usize::from(bins[idx6]);
        let b7 = usize::from(bins[idx7]);

        let g0 = ordered_gradients[pos];
        let g1 = ordered_gradients[pos + 1_usize];
        let g2 = ordered_gradients[pos + 2_usize];
        let g3 = ordered_gradients[pos + 3_usize];
        let g4 = ordered_gradients[pos + 4_usize];
        let g5 = ordered_gradients[pos + 5_usize];
        let g6 = ordered_gradients[pos + 6_usize];
        let g7 = ordered_gradients[pos + 7_usize];

        let h0 = ordered_hessians[pos];
        let h1 = ordered_hessians[pos + 1_usize];
        let h2 = ordered_hessians[pos + 2_usize];
        let h3 = ordered_hessians[pos + 3_usize];
        let h4 = ordered_hessians[pos + 4_usize];
        let h5 = ordered_hessians[pos + 5_usize];
        let h6 = ordered_hessians[pos + 6_usize];
        let h7 = ordered_hessians[pos + 7_usize];

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

        gradient_sums[b4] += g4;
        hessian_sums[b4] += h4;
        counts[b4] += 1_usize;

        gradient_sums[b5] += g5;
        hessian_sums[b5] += h5;
        counts[b5] += 1_usize;

        gradient_sums[b6] += g6;
        hessian_sums[b6] += h6;
        counts[b6] += 1_usize;

        gradient_sums[b7] += g7;
        hessian_sums[b7] += h7;
        counts[b7] += 1_usize;

        pos += 8_usize;
    }

    for &idx in remainder {
        let idx_usize = crate::narrow::index_widen(idx);
        let bin = usize::from(bins[idx_usize]);
        gradient_sums[bin] += ordered_gradients[pos];
        hessian_sums[bin] += ordered_hessians[pos];
        counts[bin] += 1_usize;
        pos += 1_usize;
    }

    histogram
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
