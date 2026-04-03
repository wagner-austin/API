//! Row subsampling for gradient boosting training.
//!
//! Selects a random subset of training samples for each boosting iteration
//! to reduce overfitting and improve training speed.

use crate::error::ClearGbmError;

use super::rng::SimpleRng;

/// Converts a non-negative integer-valued f64 to usize without `as` casts.
///
/// Decomposes the f64 into binary digits and builds the usize bit-by-bit.
/// O(32) constant time for values up to u32::MAX.
///
/// # Args
///
/// * `x` - Non-negative integer-valued f64 in `[0, u32::MAX]`.
/// * `context` - Description for error messages.
///
/// # Errors
///
/// Returns `ClearGbmError::IntegerConversion` if `x` is negative, non-integer,
/// NaN, infinite, or exceeds u32::MAX.
pub(crate) fn f64_to_usize_checked(x: f64, context: &str) -> Result<usize, ClearGbmError> {
    if x < 0.0_f64 || !x.is_finite() {
        return Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {x} is not a valid non-negative finite number"),
        });
    }
    if x != x.floor() {
        return Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {x} is not an integer"),
        });
    }
    if x > f64::from(u32::MAX) {
        return Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {x} exceeds u32::MAX"),
        });
    }

    // x == 0.0 handled separately (loop body requires x >= 1.0).
    if x == 0.0_f64 {
        return Ok(0_usize);
    }

    // Decompose x into binary digits and build the usize.
    // x is a non-negative integer f64 in [1.0, u32::MAX].
    // f64 represents integers ≤ 2^53 exactly; u32::MAX < 2^53, so all
    // intermediate values (halves) are exact.
    let mut remaining = x;
    let mut result = 0_usize;
    let mut place = 1_usize;

    loop {
        let half = (remaining / 2.0_f64).floor();
        let is_odd = remaining - 2.0_f64 * half;
        if is_odd >= 0.5_f64 {
            result += place;
        }
        remaining = half;
        if remaining < 1.0_f64 {
            break;
        }
        place *= 2_usize;
    }

    Ok(result)
}

/// Returns sample indices for one boosting iteration.
///
/// If `subsample >= 1.0`, returns all indices `0..n_samples`.
/// Otherwise, selects a random subset of size `max(1, floor(n_samples * subsample))`.
///
/// # Args
///
/// * `n_samples` - Total number of training samples.
/// * `subsample` - Fraction of samples to use (in (0.0, 1.0]).
/// * `rng` - Random number generator for reproducibility.
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` if `n_samples == 0`.
/// Returns `ClearGbmError::IntegerConversion` if subsample calculation overflows.
pub(crate) fn get_sample_indices(
    n_samples: usize,
    subsample: f64,
    rng: &mut SimpleRng,
) -> Result<Vec<usize>, ClearGbmError> {
    if n_samples == 0_usize {
        return Err(ClearGbmError::InvalidParameter {
            name: "n_samples".to_string(),
            reason: "must be > 0".to_string(),
        });
    }

    if subsample >= 1.0_f64 {
        return Ok((0_usize..n_samples).collect());
    }

    // Compute subsample count: max(1, floor(n_samples * subsample))
    let n_samples_f64 = f64::from(match u32::try_from(n_samples) {
        Ok(v) => v,
        Err(_) => {
            return Err(ClearGbmError::IntegerConversion {
                context: format!("n_samples = {n_samples} exceeds u32::MAX"),
            })
        }
    });
    let n_sub_f64 = (n_samples_f64 * subsample).floor();
    let n_sub_raw = propagate!(f64_to_usize_checked(n_sub_f64, "subsample count"));
    let n_sub = if n_sub_raw < 1_usize {
        1_usize
    } else {
        n_sub_raw
    };

    // Create index array and partially shuffle
    let mut indices: Vec<usize> = (0_usize..n_samples).collect();
    propagate!(rng.shuffle_partial(&mut indices, n_sub));
    indices.truncate(n_sub);

    Ok(indices)
}
