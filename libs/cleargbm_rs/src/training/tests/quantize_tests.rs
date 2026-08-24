//! Unit tests for the gradient discretizer (`training::quantize`).
//!
//! Hand-computed expectations throughout: zeroed rounding randoms make
//! stochastic rounding a pure truncation, and near-one randoms force the
//! round-up, so every packed value is checkable on paper.

use crate::error::ClearGbmError;
use crate::training::quantize::{
    discretize_gradients, generate_rounding_randoms, rotation_offset, DiscretizeRequest,
    QuantRoundingRandoms,
};

/// Randoms that make stochastic rounding a pure truncation.
fn zero_randoms(n: usize) -> QuantRoundingRandoms {
    QuantRoundingRandoms {
        grad: vec![0.0_f64; n],
        hess: vec![0.0_f64; n],
    }
}

#[test]
fn test_scales_follow_the_shipped_formulas() -> Result<(), ClearGbmError> {
    // max|g| = 4 over bins/2 = 2 -> grad_scale 2; max h = 2 over bins = 4
    // -> hess_scale 0.5.
    let gradients = vec![1.0_f64, -2.0_f64, 4.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 2.0_f64];
    let randoms = zero_randoms(3_usize);
    let out = discretize_gradients(DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 4_usize,
        randoms: &randoms,
        offset: 0_usize,
    });
    assert!((out.scales.grad_scale - 2.0_f64).abs() < 1e-15_f64);
    assert!((out.scales.hess_scale - 0.5_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_truncation_rounding_hand_values() -> Result<(), ClearGbmError> {
    // With zero randoms: q_g = trunc(g / 2) = [0, -1, 2];
    // q_h = trunc(h * 2) = [2, 2, 4]. Layout: hessian at 2i, gradient
    // at 2i + 1.
    let gradients = vec![1.0_f64, -2.0_f64, 4.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 2.0_f64];
    let randoms = zero_randoms(3_usize);
    let out = discretize_gradients(DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 4_usize,
        randoms: &randoms,
        offset: 0_usize,
    });
    assert_eq!(out.packed_int8, vec![2_i8, 0_i8, 2_i8, -1_i8, 4_i8, 2_i8]);
    Ok(())
}

#[test]
fn test_near_one_randoms_round_magnitudes_up() -> Result<(), ClearGbmError> {
    // With max |g| = 1 and bins 4 the scale is 0.5, so g = 0.25 scales
    // to 0.5; adding a 0.9 random crosses 1 and the quantized gradient
    // becomes 1 (and -1 on the mirrored side). The max rows anchor the
    // scale.
    let gradients = vec![0.25_f64, -0.25_f64, 1.0_f64, -1.0_f64];
    let hessians = vec![1.0_f64; 4_usize];
    let randoms = QuantRoundingRandoms {
        grad: vec![0.9_f64; 4_usize],
        hess: vec![0.0_f64; 4_usize],
    };
    let out = discretize_gradients(DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 4_usize,
        randoms: &randoms,
        offset: 0_usize,
    });
    assert_eq!(out.packed_int8[1_usize], 1_i8);
    assert_eq!(out.packed_int8[3_usize], -1_i8);
    Ok(())
}

#[test]
fn test_rotation_offset_shifts_the_random_positions() -> Result<(), ClearGbmError> {
    // Row 0 (g = 0.25, scaled 0.5) reads random position `offset`.
    // Position 0 holds 0.0 (truncate to 0) and position 1 holds 0.9
    // (round up to 1), so the offset is visible in row 0's output.
    let gradients = vec![0.25_f64, 1.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64];
    let randoms = QuantRoundingRandoms {
        grad: vec![0.0_f64, 0.9_f64],
        hess: vec![0.0_f64, 0.0_f64],
    };
    let request = DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 4_usize,
        randoms: &randoms,
        offset: 0_usize,
    };
    let at_zero = discretize_gradients(request);
    let at_one = discretize_gradients(DiscretizeRequest {
        offset: 1_usize,
        ..request
    });
    assert_eq!(at_zero.packed_int8[1_usize], 0_i8);
    assert_eq!(at_one.packed_int8[1_usize], 1_i8);
    Ok(())
}

#[test]
fn test_max_magnitude_rows_clamp_to_the_stated_range() -> Result<(), ClearGbmError> {
    // The max-|g| row scales to exactly bins/2; even a 0.999 random must
    // not push it past the stated range (the clamp divergence).
    let gradients = vec![8.0_f64, -8.0_f64];
    let hessians = vec![3.0_f64, 3.0_f64];
    let randoms = QuantRoundingRandoms {
        grad: vec![0.999_f64, 0.999_f64],
        hess: vec![0.999_f64, 0.999_f64],
    };
    let out = discretize_gradients(DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 4_usize,
        randoms: &randoms,
        offset: 0_usize,
    });
    assert_eq!(out.packed_int8[1_usize], 2_i8);
    assert_eq!(out.packed_int8[3_usize], -2_i8);
    assert_eq!(out.packed_int8[0_usize], 4_i8);
    assert_eq!(out.packed_int8[2_usize], 4_i8);
    Ok(())
}

#[test]
fn test_all_zero_gradients_quantize_to_zero() -> Result<(), ClearGbmError> {
    // Squared error on constant labels at the base prediction: every
    // gradient is exactly zero, the scale guard zeroes the inverse, and
    // every quantized gradient is zero.
    let gradients = vec![0.0_f64, 0.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64];
    let randoms = zero_randoms(2_usize);
    let out = discretize_gradients(DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 4_usize,
        randoms: &randoms,
        offset: 0_usize,
    });
    assert!((out.scales.grad_scale - 0.0_f64).abs() < 1e-15_f64);
    assert_eq!(out.packed_int8[1_usize], 0_i8);
    assert_eq!(out.packed_int8[3_usize], 0_i8);
    Ok(())
}

#[test]
fn test_all_zero_hessians_quantize_to_zero() -> Result<(), ClearGbmError> {
    // The same guard on the hessian side, exercised directly.
    let gradients = vec![1.0_f64, -1.0_f64];
    let hessians = vec![0.0_f64, 0.0_f64];
    let randoms = zero_randoms(2_usize);
    let out = discretize_gradients(DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 4_usize,
        randoms: &randoms,
        offset: 0_usize,
    });
    assert!((out.scales.hess_scale - 0.0_f64).abs() < 1e-15_f64);
    assert_eq!(out.packed_int8[0_usize], 0_i8);
    assert_eq!(out.packed_int8[2_usize], 0_i8);
    Ok(())
}

#[test]
fn test_unit_hessians_quantize_to_exactly_bins() -> Result<(), ClearGbmError> {
    // The documented reason the constant-hessian special case is not
    // needed: 1.0 * (bins / 1.0) = bins exactly, and truncation removes
    // the added random, so every unit hessian lands on `bins`.
    let gradients = vec![1.0_f64, -1.0_f64, 0.5_f64];
    let hessians = vec![1.0_f64, 1.0_f64, 1.0_f64];
    let randoms = QuantRoundingRandoms {
        grad: vec![0.5_f64; 3_usize],
        hess: vec![0.5_f64; 3_usize],
    };
    let out = discretize_gradients(DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 6_usize,
        randoms: &randoms,
        offset: 0_usize,
    });
    assert_eq!(out.packed_int8[0_usize], 6_i8);
    assert_eq!(out.packed_int8[2_usize], 6_i8);
    assert_eq!(out.packed_int8[4_usize], 6_i8);
    Ok(())
}

#[test]
fn test_negative_zero_gradient_takes_the_positive_branch() -> Result<(), ClearGbmError> {
    // IEEE: -0.0 >= 0.0 is true, matching LightGBM's `gradient >= 0.0f`
    // branch condition.
    let gradients = vec![-0.0_f64, 2.0_f64];
    let hessians = vec![1.0_f64, 1.0_f64];
    let randoms = QuantRoundingRandoms {
        grad: vec![0.9_f64, 0.0_f64],
        hess: vec![0.0_f64, 0.0_f64],
    };
    let out = discretize_gradients(DiscretizeRequest {
        gradients: &gradients,
        hessians: &hessians,
        n_quant_bins: 4_usize,
        randoms: &randoms,
        offset: 0_usize,
    });
    // -0.0 scales to -0.0; +0.9 truncates to 0 (not -0 via the negative
    // branch, which would subtract the random instead).
    assert_eq!(out.packed_int8[1_usize], 0_i8);
    Ok(())
}

#[test]
fn test_generated_randoms_are_deterministic_per_seed() -> Result<(), ClearGbmError> {
    let a = generate_rounding_randoms(42_u64, 8_usize);
    let b = generate_rounding_randoms(42_u64, 8_usize);
    let c = generate_rounding_randoms(43_u64, 8_usize);
    assert_eq!(a.grad, b.grad);
    assert_eq!(a.hess, b.hess);
    assert!(a.grad != c.grad);
    assert_eq!(a.grad.len(), 8_usize);
    assert_eq!(a.hess.len(), 8_usize);
    for &r in a.grad.iter().chain(a.hess.iter()) {
        assert!((0.0_f64..1.0_f64).contains(&r));
    }
    Ok(())
}

#[test]
fn test_rotation_offset_is_a_pure_function_of_seed_and_round() -> Result<(), ClearGbmError> {
    let a = propagate!(rotation_offset(42_u64, 3_u64, 100_usize));
    let b = propagate!(rotation_offset(42_u64, 3_u64, 100_usize));
    assert_eq!(a, b);
    assert!(a < 100_usize);
    // Different rounds draw different offsets somewhere in a small span.
    let mut offsets: Vec<usize> = Vec::new();
    for round in 0_u64..8_u64 {
        offsets.push(propagate!(rotation_offset(42_u64, round, 1_000_000_usize)));
    }
    let first = offsets[0_usize];
    assert!(offsets.iter().any(|&o| o != first));
    Ok(())
}

#[test]
fn test_rotation_offset_rejects_zero_rows() -> Result<(), ClearGbmError> {
    match rotation_offset(42_u64, 0_u64, 0_usize) {
        Ok(_) => Err(ClearGbmError::TreeConstructionFailed {
            reason: "a zero-row modulus must be rejected".to_string(),
        }),
        Err(ClearGbmError::InvalidParameter { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}
