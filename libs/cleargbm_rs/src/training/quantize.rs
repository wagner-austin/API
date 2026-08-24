//! Gradient discretization for quantized training.
//!
//! Implements the CPU discretizer of Shi 2022 as LightGBM ships it
//! (`gradient_discretizer.cpp` @ 3ec5b99b, pinned in the tech-wiki): one
//! global max scan per round sets the scales — `max|gradient| /
//! (bins / 2)` for gradients (signed values get half the bins per side)
//! and `max|hessian| / bins` for hessians — then every row's pair is
//! stochastically rounded into one interleaved `int8` stream, hessian at
//! `2i` and gradient at `2i + 1`.
//!
//! Stated divergences from the shipped code, all in determinism's favor:
//!
//! - The per-row rounding randoms are one [`SimpleRng`] stream seeded by
//!   a pure function of `random_state` (LightGBM seeds a `mt19937` per
//!   512-row thread block), and the per-round rotation offset is a pure
//!   function of `(random_state, global round)` rather than a draw from
//!   a stateful engine — so a continuation's round `n + k` discretizes
//!   exactly as the fresh run's round `n + k` did, which is what makes
//!   split training exact under quantization.
//! - Quantized values clamp to the stated range (`±bins/2` for
//!   gradients, `[0, bins]` for hessians) where LightGBM lets a
//!   floating-point epsilon write one value past it; the clamp makes the
//!   histogram width invariants provable instead of probabilistic.
//! - There is no constant-hessian special case. Under unit hessians the
//!   general scaling quantizes every row to exactly `bins` (`1.0 *
//!   bins / 1.0` is exact, and the added random is truncated away), so
//!   the relative weighting the special case preserves is preserved
//!   here too, without a second code path.
//! - Stochastic rounding is always on: it is the paper's unbiased
//!   choice and LightGBM's own default; nearest rounding is not offered.

use crate::error::ClearGbmError;

use super::rng::SimpleRng;

/// Seed-mixing constant for the rounding-randoms stream, so the stream
/// never collides with the row-sampling RNG seeded from `random_state`
/// directly.
const QUANT_RANDOMS_MIX: u64 = 0xA24B_AED4_963E_E407_u64;

/// Seed-mixing constant for the per-round rotation offset derivation.
const QUANT_OFFSET_MIX: u64 = 0x9FB2_1C65_1E98_DF25_u64;

/// Per-round mixing multiplier for the rotation offset (the golden-ratio
/// constant the feature-subsample derivation also uses).
const QUANT_ROUND_MIX: u64 = 0x9E37_79B9_7F4A_7C15_u64;

/// The pre-generated per-row rounding randoms, one uniform in `[0, 1)`
/// per row per stream, generated once per training run.
#[derive(Debug, Clone)]
pub(crate) struct QuantRoundingRandoms {
    /// Gradient-stream randoms, one per row.
    pub(crate) grad: Vec<f64>,
    /// Hessian-stream randoms, one per row.
    pub(crate) hess: Vec<f64>,
}

/// The scales one round's discretization used, needed to convert packed
/// integer histogram sums back to gradient/hessian space at split time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuantizedScales {
    /// Multiplying an integer gradient sum by this recovers the
    /// gradient-space value.
    pub grad_scale: f64,
    /// Multiplying an integer hessian sum by this recovers the
    /// hessian-space value.
    pub hess_scale: f64,
}

/// One round's discretized streams: the interleaved `int8` pairs plus
/// the scales that decode them.
#[derive(Debug, Clone)]
pub(crate) struct QuantizedGradients {
    /// Interleaved stream, length `2 * n_rows`: hessian at `2i`,
    /// gradient at `2i + 1` (LightGBM's layout).
    pub packed_int8: Vec<i8>,
    /// The decode scales.
    pub scales: QuantizedScales,
}

/// The inputs one discretization pass needs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiscretizeRequest<'a> {
    /// This round's gradients (post-GOSS when GOSS is active).
    pub gradients: &'a [f64],
    /// This round's hessians, same length.
    pub hessians: &'a [f64],
    /// The quantization bin count (`quantized_gradient_bins`, even, in
    /// `[2, 126]` by config validation).
    pub n_quant_bins: usize,
    /// The run's pre-generated rounding randoms.
    pub randoms: &'a QuantRoundingRandoms,
    /// This round's rotation offset into the randoms.
    pub offset: usize,
}

/// Generates the run's per-row rounding randoms.
///
/// One dedicated stream seeded by a pure function of `random_state`, so
/// the vectors are a function of `(config, n_rows)` alone — the property
/// continuation's split-training exactness rests on.
///
/// # Args
///
/// * `random_state` - The config seed.
/// * `n_rows` - Training row count.
///
/// # Returns
///
/// The gradient- and hessian-stream randoms, `n_rows` each.
pub(crate) fn generate_rounding_randoms(random_state: u64, n_rows: usize) -> QuantRoundingRandoms {
    let mut rng = SimpleRng::new(random_state ^ QUANT_RANDOMS_MIX);
    let grad: Vec<f64> = (0_usize..n_rows).map(|_| rng.next_f64()).collect();
    let hess: Vec<f64> = (0_usize..n_rows).map(|_| rng.next_f64()).collect();
    QuantRoundingRandoms { grad, hess }
}

/// Derives one round's rotation offset into the rounding randoms.
///
/// A pure function of `(random_state, global_round)` — fresh randomness
/// per round without regenerating the vectors, and without a stateful
/// engine that a continuation could not resume.
///
/// # Args
///
/// * `random_state` - The config seed.
/// * `global_round` - The boosting round plus any continuation offset.
/// * `n_rows` - Training row count (the modulus).
///
/// # Errors
///
/// Returns `ClearGbmError::InvalidParameter` if `n_rows` is zero and
/// `ClearGbmError::IntegerConversion` past the u32 index ceiling — both
/// already excluded by training validation.
pub(crate) fn rotation_offset(
    random_state: u64,
    global_round: u64,
    n_rows: usize,
) -> Result<usize, ClearGbmError> {
    let seed =
        (random_state ^ QUANT_OFFSET_MIX).wrapping_add(global_round.wrapping_mul(QUANT_ROUND_MIX));
    let mut rng = SimpleRng::new(seed);
    rng.next_usize_below(n_rows)
}

/// Returns the digit-extraction start bit for a magnitude bound.
///
/// The smallest power of two at or above `bound` (capped at 64), so the
/// per-value extraction loop in [`trunc_to_i8`] runs `log2(bins)` steps
/// instead of a fixed seven — the discretizer runs this loop twice per
/// row per round, so the bound matters.
fn start_bit_for(bound: usize) -> u8 {
    let mut bit = 64_u8;
    while bit > 1_u8 && usize::from(bit / 2_u8) >= bound {
        bit /= 2_u8;
    }
    bit
}

/// Truncates toward zero an f64 whose magnitude is at most `start_bit`'s
/// coverage (`2 * start_bit - 1`) into i8.
///
/// The bound holds by construction at both call sites: the scaling maps
/// gradients into `±bins/2` and hessians into `[0, bins]`, the added
/// random is below 1, the clamp caps the result at the stated range
/// (`bins <= 126` by config validation), and the caller derives
/// `start_bit` from `bins` via [`start_bit_for`]. Digit extraction by
/// descending powers of two — no `as` cast, no error arm.
fn trunc_to_i8(value: f64, start_bit: u8) -> i8 {
    let t = value.trunc();
    let neg = t < 0.0_f64;
    let mut remaining = t.abs();
    let mut magnitude = 0_u8;
    let mut bit = start_bit;
    loop {
        let bit_f = f64::from(bit);
        if remaining >= bit_f {
            remaining -= bit_f;
            magnitude += bit;
        }
        if bit == 1_u8 {
            break;
        }
        bit /= 2_u8;
    }
    // magnitude <= 127 by the loop's construction, so the arm is
    // statically dead (the crate's dead-arm idiom).
    let signed = i8::try_from(magnitude).unwrap_or(i8::MAX);
    if neg {
        -signed
    } else {
        signed
    }
}

/// Converts a bin count bounded by config validation (`<= 126`) to f64.
///
/// The saturating arm is statically dead (the crate's dead-arm idiom).
fn bins_to_f64(n_quant_bins: usize) -> f64 {
    f64::from(u32::try_from(n_quant_bins).unwrap_or(u32::MAX))
}

/// Discretizes one round's gradients and hessians into the interleaved
/// `int8` stream.
///
/// The scales come from a global max scan over ALL rows (LightGBM's
/// shape — under GOSS the unsampled rows still participate in the scan
/// and are still discretized; the tree simply never reads them). A zero
/// gradient max (every gradient exactly zero, e.g. squared error on
/// constant labels at the base prediction) quantizes every gradient to
/// zero with a zero scale; the same guard covers a zero hessian max.
///
/// # Args
///
/// * `request` - The discretization inputs.
///
/// # Returns
///
/// The packed stream and its decode scales.
#[must_use]
pub(crate) fn discretize_gradients(request: DiscretizeRequest<'_>) -> QuantizedGradients {
    let DiscretizeRequest {
        gradients,
        hessians,
        n_quant_bins,
        randoms,
        offset,
    } = request;
    let n_rows = gradients.len();

    let mut max_gradient = 0.0_f64;
    let mut max_hessian = 0.0_f64;
    for (&g, &h) in gradients.iter().zip(hessians.iter()) {
        max_gradient = max_gradient.max(g.abs());
        max_hessian = max_hessian.max(h.abs());
    }

    let half_bins = bins_to_f64(n_quant_bins / 2_usize);
    let bins_f = bins_to_f64(n_quant_bins);
    let grad_scale = max_gradient / half_bins;
    let hess_scale = max_hessian / bins_f;
    let inverse_grad_scale = if grad_scale > 0.0_f64 {
        1.0_f64 / grad_scale
    } else {
        0.0_f64
    };
    let inverse_hess_scale = if hess_scale > 0.0_f64 {
        1.0_f64 / hess_scale
    } else {
        0.0_f64
    };

    let start_bit = start_bit_for(n_quant_bins);
    let mut packed_int8: Vec<i8> = vec![0_i8; 2_usize * n_rows];
    // The random position walks sequentially from the offset and wraps —
    // the same values as `(i + offset) % n_rows` (the offset is below
    // `n_rows` by construction) without a modulo per row.
    let mut position = offset;
    for i in 0_usize..n_rows {
        let g = gradients[i];
        let scaled_g = g * inverse_grad_scale;
        let quantized_g = if g >= 0.0_f64 {
            trunc_to_i8(
                (scaled_g + randoms.grad[position]).min(half_bins),
                start_bit,
            )
        } else {
            trunc_to_i8(
                (scaled_g - randoms.grad[position]).max(-half_bins),
                start_bit,
            )
        };
        let scaled_h = hessians[i] * inverse_hess_scale;
        let quantized_h = trunc_to_i8((scaled_h + randoms.hess[position]).min(bins_f), start_bit);
        packed_int8[2_usize * i] = quantized_h;
        packed_int8[2_usize * i + 1_usize] = quantized_g;
        position += 1_usize;
        if position == n_rows {
            position = 0_usize;
        }
    }

    QuantizedGradients {
        packed_int8,
        scales: QuantizedScales {
            grad_scale,
            hess_scale,
        },
    }
}
