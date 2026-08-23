//! Deterministic pseudo-random number generator for reproducible training.
//!
//! Uses xorshift64 — no external dependencies, deterministic given seed.
//! All index arithmetic goes through u32 to ensure testable error paths
//! on all platforms (u32 → usize is infallible; usize → u32 is testable).

use crate::error::ClearGbmError;

/// Xorshift64 pseudo-random number generator.
///
/// Provides deterministic random sequences for row subsampling and
/// other training randomization. Different random sequences from Python's
/// Mersenne Twister, but the same API semantics.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SimpleRng {
    /// Internal xorshift64 state (always nonzero).
    state: u64,
}

/// Default nonzero seed for xorshift64 (seed 0 is mapped to this).
const DEFAULT_NONZERO_SEED: u64 = 0x5EED_CAFE_BABE_D00D_u64;

/// Converts a u64 value that is known to be < 2^32 to usize via u32.
///
/// The intermediate u32 conversion ensures a testable error path for
/// values that exceed u32::MAX. The u32 → usize conversion is infallible
/// on all Rust platforms (usize ≥ 32 bits).
///
/// # Args
///
/// * `val` - Value to convert (must be < 2^32 for success).
/// * `context` - Description for error messages.
///
/// # Errors
///
/// Returns `ClearGbmError::IntegerConversion` if `val > u32::MAX`.
pub(crate) fn u64_to_usize_via_u32(val: u64, context: &str) -> Result<usize, ClearGbmError> {
    let v_u32 = match u32::try_from(val) {
        Ok(v) => v,
        Err(_) => {
            return Err(ClearGbmError::IntegerConversion {
                context: format!("{context}: {val} exceeds u32::MAX"),
            })
        }
    };
    // u32 → usize: infallible on all Rust platforms (usize ≥ 32 bits).
    // Since From<u32> for usize is not in std and `as` is forbidden,
    // decompose through le_bytes → From<u8> for usize.
    let bytes = v_u32.to_le_bytes();
    Ok(usize::from(bytes[0_usize])
        + usize::from(bytes[1_usize]) * 256_usize
        + usize::from(bytes[2_usize]) * 65536_usize
        + usize::from(bytes[3_usize]) * 16_777_216_usize)
}

/// Converts a usize to u32.
///
/// # Args
///
/// * `val` - Value to convert.
/// * `context` - Description for error messages.
///
/// # Errors
///
/// Returns `ClearGbmError::IntegerConversion` if `val > u32::MAX`.
pub(crate) fn usize_to_u32(val: usize, context: &str) -> Result<u32, ClearGbmError> {
    match u32::try_from(val) {
        Ok(v) => Ok(v),
        Err(_) => Err(ClearGbmError::IntegerConversion {
            context: format!("{context}: {val} exceeds u32::MAX"),
        }),
    }
}

impl SimpleRng {
    /// Creates a new PRNG with the given seed.
    ///
    /// Seed 0 is mapped to a nonzero constant (xorshift requires nonzero state).
    ///
    /// # Args
    ///
    /// * `seed` - Random seed. 0 is valid and mapped to a default nonzero value.
    pub(crate) fn new(seed: u64) -> Self {
        let state = if seed == 0_u64 {
            DEFAULT_NONZERO_SEED
        } else {
            seed
        };
        Self { state }
    }

    /// Generates the next random u64 using xorshift64.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13_u32;
        x ^= x >> 7_u32;
        x ^= x << 17_u32;
        self.state = x;
        x
    }

    /// Generates a uniform random f64 in `[0, 1)`.
    ///
    /// Takes the xorshift word's high 32 bits (its strongest bits) and
    /// divides by 2^32; `u32 -> f64` is exact, so the result is an
    /// unbiased 32-bit-resolution uniform draw. The `try_from` error arm
    /// is statically dead after the shift (the crate's dead-arm idiom).
    pub(crate) fn next_f64(&mut self) -> f64 {
        let hi = u32::try_from(self.next_u64() >> 32_u32).unwrap_or(u32::MAX);
        f64::from(hi) / 4_294_967_296.0_f64
    }

    /// Generates a uniform random usize in `[0, n)`.
    ///
    /// Converts `n` through u32 for portable, testable index arithmetic.
    /// The random u64 is reduced modulo `n` (via u32), then the remainder
    /// (which is < n ≤ u32::MAX) is converted back to usize infallibly.
    ///
    /// # Args
    ///
    /// * `n` - Exclusive upper bound, must be in `1..=u32::MAX`.
    ///
    /// # Errors
    ///
    /// * `ClearGbmError::InvalidParameter` if `n == 0`.
    /// * `ClearGbmError::IntegerConversion` if `n > u32::MAX`.
    pub(crate) fn next_usize_below(&mut self, n: usize) -> Result<usize, ClearGbmError> {
        if n == 0_usize {
            return Err(ClearGbmError::InvalidParameter {
                name: "n".to_string(),
                reason: "upper bound must be > 0".to_string(),
            });
        }
        let n_u32 = match usize_to_u32(n, "next_usize_below bound") {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let val = self.next_u64();
        let remainder = val % u64::from(n_u32);
        // remainder < u64::from(n_u32) ≤ u64::from(u32::MAX), always fits in u32.
        // u32 → usize is infallible on all platforms (usize ≥ 32 bits).
        u64_to_usize_via_u32(remainder, "next_usize_below remainder")
    }

    /// Performs a partial Fisher-Yates shuffle.
    ///
    /// After this call, the first `n_take` elements of `slice` are a
    /// uniformly random sample (without replacement) from the original slice.
    ///
    /// # Args
    ///
    /// * `slice` - Mutable slice to shuffle.
    /// * `n_take` - Number of elements to select (must be ≤ `slice.len()`).
    ///
    /// # Errors
    ///
    /// Returns `ClearGbmError::InvalidParameter` if `n_take > slice.len()`.
    pub(crate) fn shuffle_partial(
        &mut self,
        slice: &mut [u32],
        n_take: usize,
    ) -> Result<(), ClearGbmError> {
        let len = slice.len();
        if n_take > len {
            return Err(ClearGbmError::InvalidParameter {
                name: "n_take".to_string(),
                reason: format!("n_take ({n_take}) > slice length ({len})"),
            });
        }
        for i in 0_usize..n_take {
            let remaining = len - i;
            let j = propagate!(self.next_usize_below(remaining));
            slice.swap(i, i + j);
        }
        Ok(())
    }
}
