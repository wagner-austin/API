//! Vectorized sigmoid for converting log-odds to probabilities.

use crate::predict::sigmoid;

/// Applies sigmoid to each element of a slice.
///
/// Uses [`crate::predict::sigmoid`] for each element, which provides
/// numerical stability via input clipping to `[-500, 500]`.
///
/// # Args
///
/// * `x` - Input values (typically log-odds).
///
/// # Returns
///
/// Probabilities in `(0, 1)` for each input.
#[must_use]
pub fn sigmoid_array(x: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(x.len());
    for &val in x {
        result.push(sigmoid(val));
    }
    result
}
