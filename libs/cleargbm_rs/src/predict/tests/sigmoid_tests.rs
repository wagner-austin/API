//! Tests for the sigmoid function.

use crate::error::ClearGbmError;
use crate::predict::sigmoid;

#[test]
fn test_sigmoid_zero_returns_half() -> Result<(), ClearGbmError> {
    let result = sigmoid(0.0_f64);
    assert!((result - 0.5_f64).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_sigmoid_large_positive_near_one() -> Result<(), ClearGbmError> {
    let result = sigmoid(100.0_f64);
    assert!(result > 0.999_f64);
    // exp(-100) is so small it rounds to 0 in f64, so sigmoid(100) == 1.0
    assert!(result <= 1.0_f64);
    Ok(())
}

#[test]
fn test_sigmoid_large_negative_near_zero() -> Result<(), ClearGbmError> {
    let result = sigmoid(-100.0_f64);
    assert!(result < 0.001_f64);
    assert!(result > 0.0_f64);
    Ok(())
}

#[test]
fn test_sigmoid_extreme_positive_clips() -> Result<(), ClearGbmError> {
    // Beyond clip range (500), should still produce valid output
    let result = sigmoid(1000.0_f64);
    assert!(result > 0.0_f64);
    assert!(result <= 1.0_f64);
    // Should equal sigmoid(500)
    let at_boundary = sigmoid(500.0_f64);
    assert!((result - at_boundary).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_sigmoid_extreme_negative_clips() -> Result<(), ClearGbmError> {
    let result = sigmoid(-1000.0_f64);
    assert!(result >= 0.0_f64);
    assert!(result < 1.0_f64);
    let at_boundary = sigmoid(-500.0_f64);
    assert!((result - at_boundary).abs() < 1e-15_f64);
    Ok(())
}

#[test]
fn test_sigmoid_symmetry() -> Result<(), ClearGbmError> {
    // sigmoid(x) + sigmoid(-x) == 1.0
    let values = [0.5_f64, 1.0_f64, 2.0_f64, 5.0_f64, 10.0_f64, 50.0_f64];
    for x in values {
        let sum = sigmoid(x) + sigmoid(-x);
        assert!(
            (sum - 1.0_f64).abs() < 1e-14_f64,
            "symmetry failed for x={x}: sum={sum}"
        );
    }
    Ok(())
}

#[test]
fn test_sigmoid_monotonic() -> Result<(), ClearGbmError> {
    assert!(sigmoid(1.0_f64) > sigmoid(0.0_f64));
    assert!(sigmoid(0.0_f64) > sigmoid(-1.0_f64));
    assert!(sigmoid(10.0_f64) > sigmoid(5.0_f64));
    Ok(())
}

#[test]
fn test_sigmoid_known_value() -> Result<(), ClearGbmError> {
    // sigmoid(1) = 1 / (1 + e^(-1)) = 0.7310585786300049...
    let result = sigmoid(1.0_f64);
    assert!((result - 0.731_058_578_630_004_9_f64).abs() < 1e-12_f64);
    Ok(())
}

#[test]
fn test_sigmoid_at_clip_boundaries() -> Result<(), ClearGbmError> {
    let at_max = sigmoid(500.0_f64);
    let at_min = sigmoid(-500.0_f64);
    // At boundaries, still valid probabilities
    assert!(at_max > 0.0_f64);
    assert!(at_max <= 1.0_f64);
    assert!(at_min >= 0.0_f64);
    assert!(at_min < 1.0_f64);
    // Symmetry holds at boundaries
    assert!((at_max + at_min - 1.0_f64).abs() < 1e-14_f64);
    Ok(())
}
