//! Histogram building for gradient boosting.
//!
//! Implements O(n) histogram construction with NaN handling.
//! This is the primary hot path and performance-critical code.

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
/// * `bins` - Pre-computed bin assignments for this feature.
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
    bins: &[usize],
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

    let mut histogram = HistogramBuffer::new(n_bins);

    // Core hot loop - this is where Rust shines
    for &idx in sample_indices {
        if idx >= n_samples {
            return Err(ClearGbmError::SampleIndexOutOfBounds {
                index: idx,
                n_samples,
            });
        }

        let bin = bins[idx];
        let grad = gradients[idx];
        let hess = hessians[idx];

        match histogram.accumulate(bin, grad, hess) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
    }

    Ok(histogram)
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
