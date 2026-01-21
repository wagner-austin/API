//! Tests for compute_split_gain and check_monotonicity_constraint functions.

use super::helpers::EPSILON;
use crate::error::ClearGbmError;
use crate::split::{check_monotonicity_constraint, compute_split_gain, MonotonicConstraint};

// =========================================================================
// compute_split_gain tests
// =========================================================================

#[test]
fn test_compute_split_gain_basic() -> Result<(), ClearGbmError> {
    // Simple case: equal split, should have positive gain
    let gain = compute_split_gain(
        1.0_f64,  // g_left
        10.0_f64, // h_left
        1.0_f64,  // g_right
        10.0_f64, // h_right
        2.0_f64,  // g_total
        20.0_f64, // h_total
        0.0_f64,  // reg_lambda
    );
    // Gain = 1^2/10 + 1^2/10 - 2^2/20 = 0.1 + 0.1 - 0.2 = 0
    assert!(gain.abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_split_gain_asymmetric() -> Result<(), ClearGbmError> {
    // Asymmetric split with clear gain
    let gain = compute_split_gain(
        2.0_f64,  // g_left
        10.0_f64, // h_left
        0.0_f64,  // g_right
        10.0_f64, // h_right
        2.0_f64,  // g_total
        20.0_f64, // h_total
        0.0_f64,  // reg_lambda
    );
    // Gain = 4/10 + 0/10 - 4/20 = 0.4 - 0.2 = 0.2
    assert!((gain - 0.2_f64).abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_split_gain_with_regularization() -> Result<(), ClearGbmError> {
    // With L2 regularization
    let gain = compute_split_gain(
        2.0_f64,  // g_left
        10.0_f64, // h_left
        0.0_f64,  // g_right
        10.0_f64, // h_right
        2.0_f64,  // g_total
        20.0_f64, // h_total
        1.0_f64,  // reg_lambda = 1.0
    );
    // Gain = 4/11 + 0/11 - 4/21 ≈ 0.3636 - 0.1905 ≈ 0.173
    assert!(gain > 0.0_f64);
    assert!(gain < 0.2_f64); // Less than without regularization
    Ok(())
}

#[test]
fn test_compute_split_gain_zero_hessian() -> Result<(), ClearGbmError> {
    // Zero hessian should return 0 gain
    let gain = compute_split_gain(
        1.0_f64, // g_left
        0.0_f64, // h_left = 0
        1.0_f64, // g_right
        1.0_f64, // h_right
        2.0_f64, // g_total
        1.0_f64, // h_total
        0.0_f64, // reg_lambda
    );
    assert!(gain.abs() < EPSILON);
    Ok(())
}

// =========================================================================
// check_monotonicity_constraint tests
// =========================================================================

#[test]
fn test_check_monotonicity_none() -> Result<(), ClearGbmError> {
    let result = check_monotonicity_constraint(
        MonotonicConstraint::None,
        1.0_f64,  // g_left
        10.0_f64, // h_left
        -1.0_f64, // g_right
        10.0_f64, // h_right
    );
    assert!(result); // No constraint, always passes
    Ok(())
}

#[test]
fn test_check_monotonicity_increasing_satisfied() -> Result<(), ClearGbmError> {
    // Left value = -g_left/h_left = -1/10 = -0.1
    // Right value = -g_right/h_right = -(-1)/10 = 0.1
    // -0.1 <= 0.1, so increasing constraint is satisfied
    let result = check_monotonicity_constraint(
        MonotonicConstraint::Increasing,
        1.0_f64,  // g_left
        10.0_f64, // h_left
        -1.0_f64, // g_right
        10.0_f64, // h_right
    );
    assert!(result);
    Ok(())
}

#[test]
fn test_check_monotonicity_increasing_violated() -> Result<(), ClearGbmError> {
    // Left value = -g_left/h_left = -(-1)/10 = 0.1
    // Right value = -g_right/h_right = -(1)/10 = -0.1
    // 0.1 > -0.1, so increasing constraint is violated
    let result = check_monotonicity_constraint(
        MonotonicConstraint::Increasing,
        -1.0_f64, // g_left
        10.0_f64, // h_left
        1.0_f64,  // g_right
        10.0_f64, // h_right
    );
    assert!(!result);
    Ok(())
}

#[test]
fn test_check_monotonicity_decreasing_satisfied() -> Result<(), ClearGbmError> {
    // Left value = -g_left/h_left = -(-1)/10 = 0.1
    // Right value = -g_right/h_right = -(1)/10 = -0.1
    // 0.1 >= -0.1, so decreasing constraint is satisfied
    let result = check_monotonicity_constraint(
        MonotonicConstraint::Decreasing,
        -1.0_f64, // g_left
        10.0_f64, // h_left
        1.0_f64,  // g_right
        10.0_f64, // h_right
    );
    assert!(result);
    Ok(())
}

#[test]
fn test_check_monotonicity_decreasing_violated() -> Result<(), ClearGbmError> {
    // Left value = -g_left/h_left = -(1)/10 = -0.1
    // Right value = -g_right/h_right = -(-1)/10 = 0.1
    // -0.1 < 0.1, so decreasing constraint is violated
    let result = check_monotonicity_constraint(
        MonotonicConstraint::Decreasing,
        1.0_f64,  // g_left
        10.0_f64, // h_left
        -1.0_f64, // g_right
        10.0_f64, // h_right
    );
    assert!(!result);
    Ok(())
}

#[test]
fn test_check_monotonicity_near_zero_hessian_left() -> Result<(), ClearGbmError> {
    // When h_left is very small (< EPSILON), use EPSILON as safe value
    // Left value = -1.0 / EPSILON (large negative)
    // Right value = -(-1.0) / 10.0 = 0.1
    // With increasing constraint: large_neg <= 0.1 should be true
    let result = check_monotonicity_constraint(
        MonotonicConstraint::Increasing,
        1.0_f64,  // g_left
        0.0_f64,  // h_left (near zero)
        -1.0_f64, // g_right
        10.0_f64, // h_right
    );
    assert!(result);
    Ok(())
}

#[test]
fn test_check_monotonicity_near_zero_hessian_right() -> Result<(), ClearGbmError> {
    // When h_right is very small (< EPSILON), use EPSILON as safe value
    // Left value = -1.0 / 10.0 = -0.1
    // Right value = -1.0 / EPSILON (large negative)
    // With increasing constraint: -0.1 <= large_neg should be false
    let result = check_monotonicity_constraint(
        MonotonicConstraint::Increasing,
        1.0_f64,  // g_left
        10.0_f64, // h_left
        1.0_f64,  // g_right
        0.0_f64,  // h_right (near zero)
    );
    assert!(!result);
    Ok(())
}

#[test]
fn test_check_monotonicity_both_hessians_near_zero() -> Result<(), ClearGbmError> {
    // When both hessians are near zero, both use EPSILON
    // Left value = -1.0 / EPSILON, Right value = -(-1.0) / EPSILON
    // With decreasing: -1/EPSILON >= 1/EPSILON should be false
    let result = check_monotonicity_constraint(
        MonotonicConstraint::Decreasing,
        1.0_f64,  // g_left
        0.0_f64,  // h_left (near zero)
        -1.0_f64, // g_right
        0.0_f64,  // h_right (near zero)
    );
    assert!(!result);
    Ok(())
}

#[test]
fn test_check_monotonicity_zero_hessian() -> Result<(), ClearGbmError> {
    // Edge case: both hessians exactly zero with None constraint
    let result = check_monotonicity_constraint(
        MonotonicConstraint::None,
        1.0_f64, // g_left
        0.0_f64, // h_left = 0
        1.0_f64, // g_right
        0.0_f64, // h_right = 0
    );
    assert!(result); // None constraint always passes
    Ok(())
}

#[test]
fn test_compute_split_gain_negative_result_clamped() -> Result<(), ClearGbmError> {
    // Test the branch where computed gain is negative and gets clamped to 0.
    // This can happen with certain numerical edge cases.
    // Use inputs where the parent term dominates.
    let gain = compute_split_gain(
        0.0_f64,    // g_left = 0
        0.0001_f64, // h_left very small
        0.0_f64,    // g_right = 0
        0.0001_f64, // h_right very small
        10.0_f64,   // g_total large
        0.0002_f64, // h_total = h_left + h_right
        0.0_f64,    // reg_lambda
    );
    // Parent term: 100 / 0.0002 = 500000
    // Child terms: 0/0.0001 + 0/0.0001 = 0
    // Gain = 0 - 500000 = -500000, clamped to 0
    assert!(gain.abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_split_gain_zero_total_hessian() -> Result<(), ClearGbmError> {
    // Cover the h_total_reg < EPSILON branch
    let gain = compute_split_gain(
        1.0_f64, // g_left
        1.0_f64, // h_left
        1.0_f64, // g_right
        1.0_f64, // h_right
        2.0_f64, // g_total
        0.0_f64, // h_total = 0 (near epsilon)
        0.0_f64, // reg_lambda
    );
    // h_total + reg_lambda = 0, which is < EPSILON, so returns 0
    assert!(gain.abs() < EPSILON);
    Ok(())
}

#[test]
fn test_compute_split_gain_zero_right_hessian() -> Result<(), ClearGbmError> {
    // Cover the h_right_reg < EPSILON branch
    let gain = compute_split_gain(
        1.0_f64, // g_left
        1.0_f64, // h_left
        1.0_f64, // g_right
        0.0_f64, // h_right = 0
        2.0_f64, // g_total
        1.0_f64, // h_total
        0.0_f64, // reg_lambda
    );
    // h_right = 0, which is < EPSILON, so returns 0
    assert!(gain.abs() < EPSILON);
    Ok(())
}
